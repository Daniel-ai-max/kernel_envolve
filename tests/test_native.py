import json
import os
import subprocess

import pytest

from kernel_envolve.native import compare, evaluate_native, junit, promote_native


def git_repo(path, body):
    path.mkdir()
    (path / "toyop.py").write_text(body)
    (path / "test_native_contract.py").write_text(
        "from toyop import run\nimport torch\n"
        "def test_contract():\n assert torch.equal(run(torch.ones(2)), torch.ones(2)*2)\n"
    )
    for argv in (
        ["git", "init", "-q"],
        ["git", "add", "."],
        [
            "git",
            "-c",
            "user.name=Test",
            "-c",
            "user.email=test@example.test",
            "commit",
            "-qm",
            "fixture",
        ],
    ):
        subprocess.run(argv, cwd=path, check=True, capture_output=True)


@pytest.fixture
def native_task(tmp_path):
    pytest.importorskip("torch")
    baseline, candidate, task = (tmp_path / n for n in ("baseline", "candidate", "task"))
    git_repo(baseline, "import time\ndef run(x):\n time.sleep(0.004)\n return x*2\n")
    git_repo(candidate, "def run(x): return x*2\n")
    task.mkdir()
    (task / "adapter.py").write_text("""import torch
from toyop import run
from kernel_envolve.training import Workload, forward_backward

def create(case, seed, device, oracle):
    generator = torch.Generator(device=device).manual_seed(seed)
    base = torch.randn(2, 11, generator=generator, device=device)
    x = base[:, :4].detach().requires_grad_()
    go = torch.randn(2, 4, generator=generator, device=device)
    fn = (lambda x: x*2) if oracle else run
    return Workload({"x": x, "go": go}, lambda: forward_backward(fn, (x,), (go,)))
""")
    checks = [
        {
            "id": scope,
            "kind": "pytest",
            "scope": scope,
            "root": "core",
            "argv": [
                "{python}",
                "-m",
                "pytest",
                "test_native_contract.py",
                "-q",
                "--junitxml={reports}/junit.xml",
            ],
            "reports": ["junit.xml"],
            "required_tests": ["*::test_contract"],
        }
        for scope in ("operator", "numerical", "integration")
    ]
    cfg = {
        "name": "fixture",
        "objective": "forward backward",
        "device": os.environ.get("KERNEL_ENVOLVE_NATIVE_DEVICE", "cpu"),
        "sources": {"candidate": {"core": str(candidate)}, "baseline": {"core": str(baseline)}},
        "imports": {"toyop": "core"},
        "cases": [{"id": "padded", "args": {}, "weight": 1}],
        "quick_trials": 1,
        "full_trials": 3,
        "warmup": 1,
        "iterations": 1,
        "quick_seeds": [0],
        "full_seeds": [0],
        "checks": checks,
        "rtol": 0,
        "atol": 0,
        "bitwise": True,
    }
    (task / "native.json").write_text(json.dumps(cfg))
    return task, candidate


def test_native_full_autograd_frozen_best(native_task):
    task, candidate = native_task
    run, result = evaluate_native(task, "full", timeout=90)
    assert result["correctness"] == "pass", result.get("error")
    assert result["decision"]["eligible_for_best"], result["decision"]
    assert len(result["native_checks"]) == 3
    promote_native(run)
    # Changing the live package cannot alter the frozen best worker.
    (candidate / "toyop.py").write_text("def run(x): return x*2 + 0\n")
    _, again = evaluate_native(task, "quick", timeout=60)
    assert again["correctness"] == "pass", again.get("error")
    assert again["comparator"]["kind"] == "best"
    assert not again["decision"]["eligible_for_best"]
    with pytest.raises(ValueError, match="Best changed"):
        promote_native(run)


@pytest.mark.parametrize(
    "body",
    [
        "def run(x): return (x*2).unsqueeze(0)\n",
        "def run(x): return (x*2).double()\n",
        "import torch\ndef run(x):\n with torch.no_grad(): x.zero_()\n return x*2\n",
        "import torch\nclass Bad(torch.autograd.Function):\n @staticmethod\n def forward(ctx,x): return x*2\n @staticmethod\n def backward(ctx,g): return g*3\ndef run(x): return Bad.apply(x)\n",
        "cache={}\ndef run(x):\n k=id(x)\n if k not in cache: cache[k]=x*2\n return cache[k]\n",
    ],
)
def test_native_rejects_contract_and_backward_faults(native_task, body):
    task, candidate = native_task
    (candidate / "toyop.py").write_text(body)
    run, result = evaluate_native(task, "quick", timeout=60)
    assert result["correctness"] != "pass", result
    assert not result["decision"]["eligible_for_best"]
    assert not result["samples"]
    with pytest.raises(ValueError, match="Not eligible"):
        promote_native(run)


def test_junit_requires_each_rank_each_test_and_no_skips(tmp_path):
    valid = '<testsuite><testcase classname="mod" name="test_x[a]"/></testsuite>'
    (tmp_path / "rank0.xml").write_text(valid)
    with pytest.raises(ValueError, match="Missing"):
        junit(tmp_path, ["rank0.xml", "rank1.xml"], ["mod::test_x*"])
    (tmp_path / "rank1.xml").write_text(valid.replace("test_x", "test_y"))
    with pytest.raises(ValueError, match="required"):
        junit(tmp_path, ["rank0.xml", "rank1.xml"], ["mod::test_x*"])
    (tmp_path / "rank1.xml").write_text(valid.replace("/>", "><skipped/></testcase>"))
    with pytest.raises(ValueError, match="skipped"):
        junit(tmp_path, ["rank0.xml", "rank1.xml"], ["mod::test_x*"])


def test_native_timeout_and_no_cuda_fallback(native_task):
    task, candidate = native_task
    (candidate / "toyop.py").write_text("import time\ndef run(x):\n time.sleep(60)\n return x*2\n")
    _, result = evaluate_native(task, "quick", timeout=4)
    assert result["execution"] == "error"
    assert not result["decision"]["eligible_for_best"]


def test_native_receipt_tamper_and_source_tamper(native_task):
    task, _ = native_task
    run, result = evaluate_native(task, "full", timeout=90)
    assert result["decision"]["eligible_for_best"], result
    log = run / "native_checks/operator/command.log"
    original = log.read_text()
    log.write_text(original + "altered")
    with pytest.raises(ValueError, match="evidence changed"):
        promote_native(run)
    log.write_text(original)
    source = run / "sources/candidate/core/toyop.py"
    source.write_text("def run(x): return x*3\n")
    with pytest.raises(ValueError, match="source changed"):
        promote_native(run)


def test_session_preserves_padding_aliases_and_state_reset():
    torch = pytest.importorskip("torch")
    from kernel_envolve.training import Session, Workload

    base = torch.arange(22.0).reshape(2, 11)
    x, y = base[:, :4], base[:, 4:8]
    state = torch.tensor([7.0])

    def run():
        state.add_(1)
        return {"value": x + y, "state": state}

    session = Session(Workload({"x": x, "y": y, "state": state}, run, ("state",)), "cpu")
    assert session.specs["x"]["stride"] == [11, 1]
    assert session.aliases["x"] == ["x", "y"]
    for _ in range(3):
        session.reset()
        observed = session.observe(run())
        assert observed["state"]["state"].item() == 8
    x.add_(1)
    with pytest.raises(AssertionError, match="mutation"):
        session.check_state()


def test_comparison_detects_allowed_update_wrong_values():
    torch = pytest.importorskip("torch")
    from kernel_envolve.training import Session, Workload

    def observed(delta):
        state = torch.tensor([7.0])

        def run():
            state.add_(delta)
            return torch.tensor([1.0])

        session = Session(Workload({"state": state}, run, ("state",)), "cpu")
        return session.observe(run())

    with pytest.raises(AssertionError):
        compare(
            observed(2),
            observed(1),
            {
                "rtol": 0,
                "atol": 0,
                "equal_nan": False,
                "bitwise": True,
                "check_output_layout": False,
            },
        )


def test_comparison_requires_bit_identical_inputs():
    torch = pytest.importorskip("torch")
    from kernel_envolve.training import Session, Workload

    def observe(value):
        x = torch.tensor([value])
        return Session(Workload({"x": x}, lambda: x.abs()), "cpu").observe(x.abs())

    cfg = {"rtol": 0, "atol": 0, "equal_nan": True, "bitwise": False, "check_output_layout": False}
    with pytest.raises(AssertionError, match="Input values differ"):
        compare(observe(-0.0), observe(0.0), cfg)


@pytest.mark.parametrize("mode", ["pass", "missing", "nonzero", "stale", "fail"])
def test_native_gate_owns_verdict_and_fresh_complete_records(tmp_path, mode):
    import time

    from kernel_envolve.native import native_checks

    root = tmp_path / "core"
    method = root / "apodex_core/verification/method"
    method.mkdir(parents=True)
    for folder in (root / "apodex_core", method.parent, method):
        (folder / "__init__.py").touch()
    history = root / "apodex_core/verification/results/gate_history.json"
    history.parent.mkdir()
    history.write_text('{"op": {"overall": "PASS", "R": 4096}}')
    code = """from pathlib import Path
import json
history=Path('apodex_core/verification/results/gate_history.json')
history.parent.mkdir(exist_ok=True)
"""
    if mode != "stale":
        record = (
            {}
            if mode == "missing"
            else {"op": {"overall": "FAIL" if mode == "fail" else "PASS", "R": 4096}}
        )
        code += f"history.write_text({json.dumps(record)!r})\n"
    code += "print('GATE RESULT: PASS (no blocking verdicts)')\n"
    if mode == "nonzero":
        code += "raise SystemExit(1)\n"
    (method / "gate.py").write_text(code)
    run = tmp_path / "run"
    run.mkdir()
    cfg = {
        "env": {},
        "checks": [
            {
                "id": "numerics",
                "kind": "gate",
                "scope": "numerical",
                "root": "core",
                "operators": ["op"],
            }
        ],
    }
    if mode == "pass":
        records = native_checks(run, cfg, {"core": str(root)}, time.monotonic() + 20)
        assert records[0]["status"] == "pass"
    else:
        with pytest.raises((ValueError, FileNotFoundError)):
            native_checks(run, cfg, {"core": str(root)}, time.monotonic() + 20)


def test_cuda_request_does_not_fall_back_to_cpu(native_task):
    import torch

    if torch.cuda.is_available():
        pytest.skip("This rejection probe needs a CPU-only host")
    task, _ = native_task
    cfg = json.loads((task / "native.json").read_text())
    cfg["device"] = "cuda:0"
    (task / "native.json").write_text(json.dumps(cfg))
    _, result = evaluate_native(task, "quick", timeout=30)
    assert result["execution"] == "error"
    assert not result["samples"]
    assert not result["decision"]["eligible_for_best"]


def test_native_source_runtime_setting_change_rejected(native_task):
    task, candidate = native_task
    (candidate / "toyop.py").write_text(
        "import torch\ntorch.set_float32_matmul_precision('medium')\ndef run(x): return x*2\n"
    )
    _, result = evaluate_native(task, "quick", timeout=40)
    assert result["execution"] == "error"
    assert "environments differ" in result["error"]["message"]
