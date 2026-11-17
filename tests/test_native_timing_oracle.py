"""CPU execution proves timing-oracle work shrinks without changing numerical coverage."""

import io
import json
import sys

import pytest

from kernel_envolve import native
from kernel_envolve.protocol import order, post_seed, timing_seed

pytest_plugins = ["test_native"]


@pytest.fixture(autouse=True)
def cpu_only(monkeypatch):
    monkeypatch.setenv("KERNEL_ENVOLVE_NATIVE_DEVICE", "cpu")


@pytest.mark.parametrize("deterministic", [True, False])
def test_real_timing_oracle_preserves_schedule_and_checks(native_task, monkeypatch, deterministic):
    task, _ = native_task
    cfg = json.loads((task / "native.json").read_text())
    cfg.update(quick_trials=3, warmup=2, iterations=3, deterministic=deterministic)
    (task / "native.json").write_text(json.dumps(cfg))
    original = native.Worker.call
    commands = []

    def call(self, kind, case, seed):
        meta, observed = original(self, kind, case, seed)
        commands.append((self.folder.name, kind, case["id"], seed, len(observed), meta))
        return meta, observed

    monkeypatch.setattr(native.Worker, "call", call)
    _, result = native.evaluate_native(task, timeout=60)
    assert result["correctness"] == "pass", result.get("error")
    full_count = 3 if deterministic else 2
    for role in ("oracle", "candidate", "comparator"):
        checks = [c for c in commands if c[0] == role and c[1] == "check"]
        assert [(c[3], c[4]) for c in checks] == [
            (0, full_count),
            (post_seed("padded", [0]), full_count),
        ]  # Both precheck and postcheck retain the original check command.
    timing = [c for c in commands if c[1] == "timing_oracle"]
    expected_seeds = [timing_seed("padded", 0, t, r, [0]) for t in range(3) for r in range(2)]
    assert [(c[0], c[3], c[4]) for c in timing] == [("oracle", s, 1) for s in expected_seeds]
    expected_samples = []
    for trial in range(3):
        repeats = {"candidate": 0, "comparator": 0}
        for role in order(["candidate", "comparator"], 0, trial):
            repeat = repeats[role]
            expected_samples.append(
                (role, trial, repeat, timing_seed("padded", 0, trial, repeat, [0]))
            )
            repeats[role] += 1
    assert [
        (s["role"], s["trial"], s["repeat"], s["seed"]) for s in result["samples"]
    ] == expected_samples
    measured = [c for c in commands if c[1] == "measure"]
    assert len(measured) == 12
    assert all(c[5]["validated_iterations"] == 5 for c in measured)
    assert all(c[5]["input_guard"]["check_calls"] == 10 for c in measured)
    assert all(len(c[5]["samples_us"]) == 3 for c in measured)
    kinds = {"fresh", "reuse", "repeat"} if deterministic else {"fresh", "reuse"}
    assert {c["kind"] for c in result["checks"]} == kinds | {"post", "timed_output"}
    assert not result["decision"]["eligible_for_best"]


def test_wrong_timing_oracle_output_is_still_compared(native_task, monkeypatch):
    task, _ = native_task
    original = native.Worker.call
    changed = []

    def call(self, kind, case, seed):
        meta, observed = original(self, kind, case, seed)
        if kind == "timing_oracle":
            changed.append(seed)
            # Corrupt the captured output, leaving metadata and gradients intact.
            observed[0]["output"]["output"].add_(1)
        return meta, observed

    monkeypatch.setattr(native.Worker, "call", call)
    _, result = native.evaluate_native(task, timeout=60)
    assert len(changed) == 1
    assert result["correctness"] == "fail", result.get("error")
    assert result["failed_phase"] == "measurement"
    assert not result["samples"]
    assert not result["decision"]["eligible_for_best"]


@pytest.mark.parametrize("count", [0, 2])
def test_timing_oracle_requires_exactly_one_observation(native_task, monkeypatch, count):
    task, _ = native_task
    original = native.Worker.call

    def call(self, kind, case, seed):
        meta, observed = original(self, kind, case, seed)
        return meta, observed * count if kind == "timing_oracle" else observed

    monkeypatch.setattr(native.Worker, "call", call)
    _, result = native.evaluate_native(task, timeout=60)
    assert result["correctness"] == "fail", result.get("error")
    assert result["failed_phase"] == "measurement"
    assert result["error"]["message"] == "Expected one timing oracle observation"
    assert not result["samples"]


@pytest.mark.parametrize(
    "kind,expected_runs,expected_builds", [("timing_oracle", 1, 1), ("check", 3, 2)]
)
def test_real_worker_fresh_count_and_guard_execution(
    tmp_path, monkeypatch, kind, expected_runs, expected_builds
):
    torch = pytest.importorskip("torch")
    from kernel_envolve.native_worker import serve

    counts = tmp_path / "counts.jsonl"
    adapter = tmp_path / "adapter.py"
    adapter.write_text(
        "import torch\nfrom pathlib import Path\nfrom kernel_envolve.training import Workload\n"
        f"def record(x):\n with Path({str(counts)!r}).open('a') as f: f.write(x+'\\n')\n"
        "def create(case,seed,device,oracle):\n"
        " record('build'); x=torch.ones(4) * (seed+1)\n"
        " def run():\n  record('run'); return x*2\n"
        " def validate(out):\n  record('validate'); assert out.shape == x.shape\n"
        " return Workload({'x':x},run,validate=validate)\n"
    )
    request, reply, output = (
        tmp_path / name for name in ("request.json", "reply.json", "output.pt")
    )
    request.write_text(
        json.dumps(
            {
                "config": {"device": "cpu", "env": {}, "imports": {}, "deterministic": True},
                "roots": {},
                "role": "oracle",
                "adapter": str(adapter),
            }
        )
    )
    command = {
        "id": 1,
        "kind": kind,
        "case": {},
        "seed": 0,
        "reply": str(reply),
        "output": str(output),
    }
    monkeypatch.setattr(sys, "stdin", io.StringIO(json.dumps(command) + "\n"))
    threads, deterministic = torch.get_num_threads(), torch.are_deterministic_algorithms_enabled()
    try:
        assert serve(request) == 0
    finally:
        torch.set_num_threads(threads)
        torch.use_deterministic_algorithms(deterministic)
    events = counts.read_text().splitlines()
    assert events.count("run") == events.count("validate") == expected_runs
    assert events.count("build") == expected_builds
    assert len(torch.load(output, weights_only=True)) == expected_runs
    assert json.loads(reply.read_text())["input_guard"]["check_calls"] == expected_runs * 2


@pytest.mark.parametrize("failure", ["mutation", "validation", "runtime"])
def test_timing_oracle_faults_abort_worker_before_next_command(tmp_path, monkeypatch, failure):
    torch = pytest.importorskip("torch")
    from kernel_envolve.native_worker import serve

    adapter = tmp_path / "adapter.py"
    action = {
        "mutation": "x.add_(1); return x*2",
        "validation": "return x[:1]",
        "runtime": "torch.set_grad_enabled(False); return x*2",
    }[failure]
    adapter.write_text(
        "import torch\nfrom kernel_envolve.training import Workload\n"
        "def create(case,seed,device,oracle):\n x=torch.ones(4)\n"
        f" def run():\n  {action}\n"
        " def validate(out):\n  assert out.shape == x.shape, 'Invalid output shape'\n"
        " return Workload({'x':x},run,validate=validate)\n"
    )
    request = tmp_path / "request.json"
    request.write_text(
        json.dumps(
            {
                "config": {"device": "cpu", "env": {}, "imports": {}, "deterministic": True},
                "roots": {},
                "role": "oracle",
                "adapter": str(adapter),
            }
        )
    )
    commands = [
        {
            "id": i,
            "kind": "timing_oracle",
            "case": {},
            "seed": 0,
            "reply": str(tmp_path / f"reply{i}.json"),
            "output": str(tmp_path / f"output{i}.pt"),
        }
        for i in (1, 2)
    ]
    monkeypatch.setattr(sys, "stdin", io.StringIO("".join(json.dumps(c) + "\n" for c in commands)))
    state = (
        torch.get_num_threads(),
        torch.are_deterministic_algorithms_enabled(),
        torch.is_grad_enabled(),
    )
    try:
        assert serve(request) == 1
    finally:
        torch.set_num_threads(state[0])
        torch.use_deterministic_algorithms(state[1])
        torch.set_grad_enabled(state[2])
    reply = json.loads((tmp_path / "reply1.json").read_text())
    expected = {
        "mutation": "Forbidden input",
        "validation": "Invalid output shape",
        "runtime": "Persistent runtime setting",
    }
    assert expected[failure] in reply["error"]
    assert not (tmp_path / "reply2.json").exists()
    assert not (tmp_path / "output1.pt").exists()


def test_timing_oracle_cannot_replace_candidate_checks(tmp_path, monkeypatch):
    torch = pytest.importorskip("torch")
    from kernel_envolve.native_worker import serve

    adapter = tmp_path / "adapter.py"
    adapter.write_text("def create(*args): raise AssertionError('must not build candidate')\n")
    request = tmp_path / "request.json"
    request.write_text(
        json.dumps(
            {
                "config": {"device": "cpu", "env": {}, "imports": {}, "deterministic": True},
                "roots": {},
                "role": "candidate",
                "adapter": str(adapter),
            }
        )
    )
    reply = tmp_path / "reply.json"
    command = {
        "id": 1,
        "kind": "timing_oracle",
        "case": {},
        "seed": 0,
        "reply": str(reply),
        "output": str(tmp_path / "output.pt"),
    }
    monkeypatch.setattr(sys, "stdin", io.StringIO(json.dumps(command) + "\n"))
    threads, deterministic = torch.get_num_threads(), torch.are_deterministic_algorithms_enabled()
    try:
        assert serve(request) == 1
    finally:
        torch.set_num_threads(threads)
        torch.use_deterministic_algorithms(deterministic)
    assert "require the oracle role" in json.loads(reply.read_text())["error"]
    assert not (tmp_path / "output.pt").exists()
