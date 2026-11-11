"""Exact guard semantics run on CPU; CUDA allocation/transfer tests are separate."""

import pytest

torch = pytest.importorskip("torch")

from kernel_envolve import training  # noqa: E402
from kernel_envolve.training import Session, Workload  # noqa: E402


def workload(x, validate=lambda result: None):
    return Workload({"x": x}, lambda: x * 2, validate=validate)


def test_cache_avoids_host_comparison_but_checks_both_sides(monkeypatch):
    x = torch.arange(12.0).reshape(3, 4).t()
    session = Session(workload(x), "cpu", device_guard_limit=48)
    monkeypatch.setattr(training, "equal", lambda *a: pytest.fail("host comparison"))
    session.observe(x * 2)
    assert session.input_guard_stats()["device_inputs"] == ["x"]
    # Direct storage writes do not increment PyTorch's version counter.
    version = x._version
    x.numpy()[1, 1] += 1
    assert x._version == version
    with pytest.raises(AssertionError, match="Forbidden input"):
        session.observe(x * 2)
    assert session.input_check_calls == 3
    assert session.input_check_seconds > 0


def test_validator_mutation_is_rejected_after_validation():
    x = torch.ones(4)

    def mutate(_):
        x.numpy()[0] = 2

    session = Session(workload(x, mutate), "cpu", device_guard_limit=16)
    with pytest.raises(AssertionError, match="Forbidden input"):
        session.observe(x * 2)


@pytest.mark.parametrize("bits,new_bits", [(0, -2147483648), (2143289345, 2143289346)])
def test_signed_zero_and_nan_payload_remain_bitwise(bits, new_bits):
    raw = torch.tensor([bits], dtype=torch.int32)
    x = raw.view(torch.float32)
    session = Session(workload(x), "cpu", device_guard_limit=4)
    session.check_state()
    raw.numpy()[0] = new_bits
    with pytest.raises(AssertionError, match="Forbidden input"):
        session.check_state()


def test_readonly_aliases_and_rebinding_are_checked():
    backing = torch.arange(16.0)
    x, y = backing[1:9:2], backing[3:11:2]
    w = Workload({"x": x, "y": y}, lambda: x + y)
    session = Session(w, "cpu", device_guard_limit=32)
    assert session.aliases["x"] == ["x", "y"]
    session.observe(w.run())
    backing.numpy()[3] = -1
    with pytest.raises(AssertionError, match="Forbidden input"):
        session.check_state()
    session = Session(w, "cpu", device_guard_limit=32)
    w.tensors["y"] = y.clone()
    with pytest.raises(AssertionError, match="binding/layout"):
        session.check_state()


def test_refresh_replaces_guard_bytes_and_preserves_live_objects():
    x = torch.arange(6.0)
    session = Session(workload(x), "cpu", device_guard_limit=24)
    session.refresh(workload(torch.arange(6.0) + 10))
    assert session.workload.tensors["x"] is x
    assert torch.equal(x, torch.arange(6.0) + 10)
    session.observe(session.workload.run())
    x.numpy()[0] = 0  # Old seed content must not become an accepted cache hit.
    with pytest.raises(AssertionError, match="Forbidden input"):
        session.check_state()


def test_budget_fallback_and_mutable_inputs_keep_original_semantics():
    x, y, state = torch.ones(4), torch.ones(8), torch.zeros(2)
    w = Workload({"x": x, "y": y, "state": state}, lambda: x + 1, mutable=("state",))
    session = Session(w, "cpu", device_guard_limit=16)
    assert session.input_guard_stats()["device_inputs"] == ["x"]
    assert session.input_guard_stats()["host_inputs"] == ["y"]
    assert session.input_guard_stats()["fallbacks"] == {"budget": 1}
    state.add_(1)
    session.check_state()
    session.reset()
    assert torch.equal(state, torch.zeros(2))
    y.numpy()[0] = 2
    with pytest.raises(AssertionError, match="Forbidden input"):
        session.check_state()


def test_snapshot_oom_falls_back_but_driver_errors_propagate(monkeypatch):
    class FailedCopy:
        def to(self, *args, **kwargs):
            raise torch.cuda.OutOfMemoryError("guard allocation")

    monkeypatch.setattr(training, "byte_view", lambda tensor: FailedCopy())
    session = Session(workload(torch.ones(4)), "cpu", device_guard_limit=16)
    assert session.input_guard_stats()["device_inputs"] == []
    assert session.input_guard_stats()["fallbacks"] == {"allocation_oom": 1}
    session.check_state()

    def fail(tensor):
        raise RuntimeError("CUDA illegal memory access")

    monkeypatch.setattr(training, "byte_view", fail)
    with pytest.raises(RuntimeError, match="illegal memory access"):
        Session(workload(torch.ones(4)), "cpu", device_guard_limit=16)


def test_comparison_oom_fallback_still_detects_mutation(monkeypatch):
    x = torch.ones(4)
    session = Session(workload(x), "cpu", device_guard_limit=16)

    def fail(tensor):
        raise torch.cuda.OutOfMemoryError("contiguous temporary")

    monkeypatch.setattr(training, "byte_view", fail)
    x.numpy()[0] = 2
    with pytest.raises(AssertionError, match="Forbidden input"):
        session.check_state()
    assert session.input_guard_stats()["device_inputs"] == []
    assert session.input_guard_stats()["fallbacks"] == {"comparison_oom": 1}


def test_worker_failure_reply_retains_guard_statistics(tmp_path, monkeypatch):
    import io
    import json
    import sys

    from kernel_envolve.native_worker import serve

    adapter = tmp_path / "adapter.py"
    adapter.write_text(
        "import torch\nfrom kernel_envolve.training import Workload\n"
        "def create(case, seed, device, oracle):\n"
        " x=torch.ones(4)\n return Workload({'x':x}, lambda: x.add_(1))\n"
    )
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
        "kind": "check",
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
    value = json.loads(reply.read_text())
    assert "Forbidden input" in value["error"]
    assert value["input_guard"]["check_calls"] == 1
    assert value["input_guard"]["check_seconds"] > 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA device required")
def test_cuda_exact_cache_and_actual_allocator_accounting(monkeypatch):
    x = torch.arange(4096.0, device="cuda").reshape(64, 64).t()
    before = torch.cuda.memory_allocated()
    session = Session(workload(x), "cuda:0", device_guard_limit=1024**2)
    assert session.private_allocated_bytes == torch.cuda.memory_allocated() - before
    assert session.private_allocated_bytes > 0
    monkeypatch.setattr(training, "equal", lambda *a: pytest.fail("D2H input comparison"))
    session.check_state()
    torch.cuda.reset_peak_memory_stats()
    extra = torch.empty(8192, device="cuda")
    adjusted = torch.cuda.max_memory_allocated() - session.private_allocated_bytes
    assert adjusted >= before + extra.numel() * extra.element_size()
    x.data[0, 0] = -1
    with pytest.raises(AssertionError, match="Forbidden input"):
        session.check_state()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA device required")
def test_cuda_guard_respects_live_free_memory(monkeypatch):
    x = torch.ones(4096, device="cuda")
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda device: (8192, 16384))
    session = Session(workload(x), "cuda:0", device_guard_limit=2 * 1024**3)
    assert session.input_guard_stats()["device_inputs"] == []
    assert session.private_allocated_bytes == 0
    session.check_state()


@pytest.mark.parametrize("capture", [False, True])
def test_intermediate_observation_keeps_dispatch_and_state_guards(capture):
    x = torch.ones(4)

    def mutate(_):
        x.numpy()[0] = 2

    session = Session(workload(x, mutate), "cpu", device_guard_limit=16)
    with pytest.raises(AssertionError, match="Forbidden input"):
        session.observe(x * 2, capture=capture)
    session = Session(workload(torch.ones(4)), "cpu")
    with pytest.raises(ValueError, match="nonempty tensor tree"):
        session.observe({"output": None}, capture=capture)


def test_intermediate_observation_avoids_output_and_mutable_host_copy(monkeypatch):
    x, state = torch.ones(4), torch.zeros(4)
    validated = []
    session = Session(
        Workload(
            {"x": x, "state": state},
            lambda: x * 2,
            mutable=("state",),
            validate=lambda output: validated.append(output),
        ),
        "cpu",
    )
    copies = []
    real_host = training.host

    def copied(tensor):
        copies.append(tensor)
        return real_host(tensor)

    monkeypatch.setattr(training, "host", copied)
    output = {"output": x * 2, "gradients": (x.clone(),)}
    state.add_(1)
    for _ in range(7):
        assert session.observe(output, capture=False) is None
    assert not copies
    captured = session.observe(output)
    assert len(copies) == 3  # final output, gradient and declared mutable state
    assert len(validated) == 8 and session.input_check_calls == 16
    assert torch.equal(captured["output"]["gradients"][0], x)
    assert torch.equal(captured["state"]["state"], torch.ones(4))
    session.reset()
    assert torch.equal(state, torch.zeros(4))


@pytest.mark.parametrize("capture", [False, True])
def test_intermediate_observation_propagates_validator_failure(capture):
    def reject(output):
        raise AssertionError("expected candidate dispatch missing")

    session = Session(workload(torch.ones(4), reject), "cpu")
    with pytest.raises(AssertionError, match="candidate dispatch missing"):
        session.observe(torch.ones(4), capture=capture)
