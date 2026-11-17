"""Real CPU worker transport, durable failure evidence and stop-before-delete."""

import gc
import shutil
import time
from pathlib import Path

import pytest

import kernel_envolve.native as native
from kernel_envolve.storage import file_hash, read_json

pytest.importorskip("torch")
pytest_plugins = ["test_native"]


@pytest.fixture(autouse=True)
def cpu_only(monkeypatch):
    monkeypatch.setenv("KERNEL_ENVOLVE_NATIVE_DEVICE", "cpu")


@pytest.fixture
def worker_run(tmp_path, monkeypatch):
    scratch = tmp_path / "host-temporary"
    scratch.mkdir()
    monkeypatch.setattr(native.tempfile, "tempdir", str(scratch))
    monkeypatch.delenv("KERNEL_ENVOLVE_NATIVE_TMPDIR", raising=False)
    run = tmp_path / "run"
    run.mkdir()
    root = run / "sources" / "core"
    root.mkdir(parents=True)
    (root / "toyop.py").write_text("def run(x): return x * 2\n")
    adapter = """import torch
from toyop import run
from kernel_envolve.training import Workload

def create(case, seed, device, oracle):
    x = torch.randn(2, 7, generator=torch.Generator().manual_seed(seed))[:, ::2]
    return Workload({'x': x}, lambda: run(x))
"""
    (run / "adapter.py").write_text(adapter)
    cfg = {
        "device": "cpu",
        "threads": 1,
        "deterministic": True,
        "env": {"KERNEL_ENVOLVE_NATIVE_TMPDIR": str(run / "task-selected")},
        "imports": {"toyop": "core"},
        "warmup": 1,
        "iterations": 2,
        "check_output_layout": True,
        "bitwise": True,
        "rtol": 0,
        "atol": 0,
        "equal_nan": False,
    }
    return run, cfg, {"core": str(root)}, scratch


def worker(worker_run, role="candidate"):
    run, cfg, roots, _ = worker_run
    return native.Worker(run, cfg, roots, role, time.monotonic() + 30)


def test_real_cpu_worker_local_exchange_preserves_contract(worker_run, monkeypatch):
    run, cfg, _, scratch = worker_run
    # A relative host selection is resolved; the task's env cannot select it.
    monkeypatch.chdir(scratch.parent)
    monkeypatch.setenv("KERNEL_ENVOLVE_NATIVE_TMPDIR", scratch.name)
    oracle, candidate = worker(worker_run, "oracle"), worker(worker_run)
    directories = (oracle.temporary, candidate.temporary)
    assert directories[0] != directories[1]
    real_stop = native.stop
    stopped = []

    def stop_before_delete(process):
        assert all(p.exists() for p in directories if p not in stopped)
        real_stop(process)
        stopped.append(oracle.temporary if process is oracle.process else candidate.temporary)

    monkeypatch.setattr(native, "stop", stop_before_delete)
    try:
        _, expected = oracle.call("check", {"id": "strided", "args": {}}, 31)
        result, actual = candidate.call("check", {"id": "strided", "args": {}}, 31)
        assert len(actual) == len(expected) == 3
        for left, right in zip(actual, expected):
            native.compare(left, right, cfg)
        assert result["imports"]["toyop"].startswith(str(run / "sources"))
        timed, observed = candidate.call("timing", {"id": "strided", "args": {}}, 31)
        assert len(observed) == 1 and timed["validated_iterations"] == 3
        assert len(timed["samples_us"]) == 2
        assert timed["input_guard"]["check_calls"] == 6
        for current in (oracle, candidate):
            assert current.temporary.parent == scratch.resolve()
            assert not current.temporary.is_relative_to(run)
            assert list(current.temporary.iterdir()) == []
            command = read_json(current.folder / "1.command.json")
            assert Path(command["output"]) == current.temporary / "1.pt"
            assert Path(command["reply"]).parent == current.folder
            assert (current.folder / "1.json").is_file()
        assert not list(run.rglob("*.pt"))
    finally:
        oracle.close()
        candidate.close()
    for current in (oracle, candidate):
        assert not current.temporary.exists()
        transport = read_json(current.folder / "transport.json")
        assert transport["state"] == "closed"
        assert transport["preserved_outputs"] == []
        assert transport["selected_parent"] == str(scratch.resolve())


@pytest.mark.parametrize("failure", ["load", "save"])
def test_failed_transport_preserves_only_residual_output(worker_run, monkeypatch, failure):
    import torch

    run, _, _, _ = worker_run
    if failure == "save":
        with (run / "adapter.py").open("a") as stream:
            stream.write("""
def failed_save(value, path):
    from pathlib import Path
    Path(path).write_bytes(b'partial tensor')
    raise OSError('injected tensor save failure')
torch.save = failed_save
""")
    current = worker(worker_run)
    if failure == "load":

        def failed_load(*args, **kwargs):
            raise OSError("injected tensor load failure")

        monkeypatch.setattr(torch, "load", failed_load)
    try:
        with pytest.raises((OSError, RuntimeError), match="injected tensor"):
            current.call("check", {"id": "failure", "args": {}}, 7)
        residual = current.temporary / "1.pt"
        expected_hash = file_hash(residual)
        reply = (current.folder / "1.json").read_bytes()
        command = (current.folder / "1.command.json").read_bytes()
    finally:
        current.close()
    assert not current.temporary.exists()
    transport = read_json(current.folder / "transport.json")
    assert transport["state"] == "closed"
    assert len(transport["preserved_outputs"]) == 1
    evidence = transport["preserved_outputs"][0]
    saved = current.folder / evidence["path"]
    assert saved.name == "1.pt" and file_hash(saved) == expected_hash == evidence["sha256"]
    assert saved.stat().st_size == evidence["bytes"]
    assert evidence["temporary_path"] == str(residual)
    assert (current.folder / "1.json").read_bytes() == reply
    assert (current.folder / "1.command.json").read_bytes() == command
    assert current.process.poll() is not None


def test_stop_failure_never_deletes_live_writer_directory(worker_run, monkeypatch):
    run, _, _, _ = worker_run
    with (run / "adapter.py").open("a") as stream:
        stream.write("""
original_save = torch.save
saves = 0
def slow_save(value, path):
    global saves
    saves += 1
    if saves == 1:
        return original_save(value, path)
    import time
    from pathlib import Path
    Path(path).write_bytes(b'writer is still active')
    time.sleep(60)
torch.save = slow_save
""")
    current = worker(worker_run)
    current.call("check", {"id": "warm", "args": {}}, 7)
    current.deadline = time.monotonic() + 1
    process, temporary, folder = current.process, current.temporary, current.folder
    real_stop = native.stop
    try:
        with pytest.raises(TimeoutError):
            current.call("check", {"id": "slow", "args": {}}, 7)
        assert (temporary / "2.pt").read_bytes() == b"writer is still active"

        def failed_stop(process):
            raise OSError("stop not confirmed")

        monkeypatch.setattr(native, "stop", failed_stop)
        with pytest.raises(OSError, match="stop not confirmed"):
            current.close()
        assert process.poll() is None
        del current
        gc.collect()  # No automatic TemporaryDirectory finalizer may erase it.
        assert (temporary / "2.pt").exists()
        transport = read_json(folder / "transport.json")
        assert transport["state"] == "retained"
        assert "stop not confirmed" in transport["cleanup_errors"][0]
    finally:
        real_stop(process)
        shutil.rmtree(temporary)


def test_failed_evidence_copy_retains_original_after_stop(worker_run, monkeypatch):
    current = worker(worker_run)
    (current.temporary / "1.pt").write_bytes(b"diagnostic output")

    def failed_copy(*args, **kwargs):
        raise OSError("evidence storage unavailable")

    monkeypatch.setattr(native.shutil, "copy2", failed_copy)
    try:
        with pytest.raises(OSError, match="evidence storage unavailable"):
            current.close()
        assert current.process.poll() is not None
        assert (current.temporary / "1.pt").read_bytes() == b"diagnostic output"
        transport = read_json(current.folder / "transport.json")
        assert transport["state"] == "retained"
        assert "storage unavailable" in transport["cleanup_errors"][0]
    finally:
        shutil.rmtree(current.temporary)


def test_uncertain_launch_preserves_directory_for_possible_writer(worker_run, monkeypatch):
    run, _, _, scratch = worker_run
    processes = []
    real_spawn = native.spawn

    def failed_spawn(*args, **kwargs):
        processes.append(real_spawn(*args, **kwargs))
        raise OSError("registration failed; stop unconfirmed")

    monkeypatch.setattr(native, "spawn", failed_spawn)
    try:
        with pytest.raises(OSError, match="registration failed"):
            worker(worker_run)
        gc.collect()
        transport = read_json(run / "workers/candidate/transport.json")
        assert transport["state"] == "retained"
        assert Path(transport["directory"]).exists()
        assert "stop unconfirmed" in transport["launch_error"]
    finally:
        for process in processes:
            native.stop(process)
            process.stdin.close()
        for path in scratch.iterdir():
            shutil.rmtree(path)


def test_explicit_temporary_parent_inside_run_is_rejected(worker_run, monkeypatch):
    run, _, _, _ = worker_run
    monkeypatch.setenv("KERNEL_ENVOLVE_NATIVE_TMPDIR", str(run))
    with pytest.raises(ValueError, match="outside the run"):
        worker(worker_run)
    assert not list(run.glob("kernel-envolve-*"))


def test_failed_tensor_manifest_is_covered_by_native_receipt(native_task, monkeypatch):
    import torch

    task, _ = native_task

    def failed_load(*args, **kwargs):
        raise OSError("parent tensor load failed")

    monkeypatch.setattr(torch, "load", failed_load)
    run, result = native.evaluate_native(task, "quick", timeout=60)
    assert result["execution"] == "error"
    manifest_path = run / "workers/oracle/transport.json"
    transport = read_json(manifest_path)
    assert transport["state"] == "closed"
    assert len(transport["preserved_outputs"]) == 1
    evidence = transport["preserved_outputs"][0]
    output = manifest_path.parent / evidence["path"]
    receipt = read_json(run / "receipt.json")
    assert receipt["evidence"][str(manifest_path.relative_to(run))] == file_hash(manifest_path)
    assert str(output.relative_to(run)) not in receipt["evidence"]
    assert file_hash(output) == evidence["sha256"]
