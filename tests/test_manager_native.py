"""Native evidence integration uses the same frozen autograd/project-gate fixture as the evaluator."""

import json

import pytest

from kernel_envolve.manager.cli import task_contract
from kernel_envolve.manager.config import validate
from kernel_envolve.manager.evidence import inspect_run, refresh
from kernel_envolve.manager.store import Store
from kernel_envolve.manager.views import detail
from kernel_envolve.native import evaluate_native, promote_native
from kernel_envolve.storage import digest

pytest_plugins = ["test_native"]


def native_spec(task):
    return {
        "id": "native",
        "name": "Native autograd",
        "workspace": str(task),
        "task_dir": str(task),
        "kind": "native",
        "resource": "cpu",
        "command": ["true"],
        "baseline": {
            "repository": "fixture",
            "revision": "baseline",
            "entrypoint": "toyop:run",
            "introduced_on": None,
            "introduction_url": None,
            "frozen_on": "2026-11-10",
            "hardware": "CPU",
            "contract": task_contract(task, "native"),
        },
    }


def test_native_manager_imports_full_evidence_and_detects_gate_tampering(native_task, tmp_path):
    task, _ = native_task
    spec = native_spec(task)
    store = Store(tmp_path / "manager")
    try:
        store.register(
            validate(
                {"version": 1, "resources": [{"id": "cpu", "device": "cpu"}], "tasks": [spec]},
                tmp_path,
            )
        )
        run, result = evaluate_native(task, "full", timeout=90)
        assert result["decision"]["eligible_for_best"], result.get("error")
        assert inspect_run(run, spec)["contract_verified"] is True
        original_receipt = (run / "receipt.json").read_text()
        incomplete = {key: value for key, value in result.items() if key != "initial_best_hash"}
        (run / "result.json").write_text(json.dumps(incomplete))
        receipt = json.loads(original_receipt)
        receipt["result_hash"] = digest(incomplete)
        (run / "receipt.json").write_text(json.dumps(receipt))
        with pytest.raises(KeyError, match="initial_best_hash"):
            inspect_run(run, spec)
        (run / "result.json").write_text(json.dumps(result))
        (run / "receipt.json").write_text(original_receipt)
        promote_native(run)
        refresh(store)
        data = detail(store, "native")
        assert data["best"], data["evidence_error"]
        assert data["best"]["baseline_speedup"] > 1
        assert len(data["best"]["native_checks"]) == 3
        receipt = json.loads((run / "receipt.json").read_text())
        evidence = next(n for n in receipt["evidence"] if n.endswith("junit.xml"))
        (run / evidence).write_text("<testsuite />")
        refresh(store)
        assert detail(store, "native")["best"] is None
        assert "Native evidence changed" in store.task("native")["evidence_error"]
    finally:
        store.close()


@pytest.fixture
def snapshot_failure(native_task, monkeypatch):
    from kernel_envolve import native

    task, _ = native_task
    spec = native_spec(task)

    def expire(*args, **kwargs):
        raise TimeoutError("Native evaluation time budget exhausted")

    with monkeypatch.context() as patch:
        patch.setattr(native, "snapshot", expire)
        run, result = evaluate_native(task, "quick")
    assert result["execution"] == "error" and result["failed_phase"] == "snapshot"
    return task, run, result, spec


def test_manager_preserves_snapshot_timeout_as_unverified_diagnosis(snapshot_failure, tmp_path):
    task, run, result, spec = snapshot_failure
    data = inspect_run(run, spec)
    assert data["contract_verified"] is False
    assert data["contract"] is data["source"] is data["initial_best_hash"] is None
    assert not data["decision"]["eligible_for_best"]
    assert data["error"] == result["error"] and data["failed_phase"] == "snapshot"
    assert data["phase_wall_seconds"] == result["phase_wall_seconds"]
    store = Store(tmp_path / "manager")
    try:
        store.register(
            validate(
                {"version": 1, "resources": [{"id": "cpu", "device": "cpu"}], "tasks": [spec]},
                tmp_path,
            )
        )
        refresh(store)
        view = detail(store, "native")
        assert view["evidence_error"] is None and view["best"] is None
        latest = view["latest_evaluation"]
        assert latest["execution"] == "error" and latest["error"]["type"] == "TimeoutError"
        assert latest["failed_phase"] == "snapshot" and not latest["contract_verified"]
        assert latest["phase_wall_seconds"] == result["phase_wall_seconds"]
        # A self-written best pointer cannot turn the diagnostic into performance evidence.
        (task / ".kernel_native/best.json").write_text(
            json.dumps({"run": f"runs/{run.name}", "result_hash": digest(result), "previous": None})
        )
        refresh(store)
        assert detail(store, "native")["best"] is None
        assert "eligible, verified full" in store.task("native")["evidence_error"]
    finally:
        store.close()


@pytest.mark.parametrize("target", ["result.json", "config.json", "adapter.py", "evidence"])
def test_snapshot_diagnosis_still_requires_untampered_receipts(snapshot_failure, target):
    _, run, _, spec = snapshot_failure
    if target == "evidence":
        path = run / "workers/diagnostic.log"
        path.parent.mkdir()
        path.write_text("recorded diagnostic")
        from kernel_envolve.storage import file_hash

        receipt = json.loads((run / "receipt.json").read_text())
        receipt["evidence"]["workers/diagnostic.log"] = file_hash(path)
        (run / "receipt.json").write_text(json.dumps(receipt))
        assert not inspect_run(run, spec)["contract_verified"]
    else:
        path = run / target
    path.write_text("{}" if path.suffix == ".json" else "changed")
    with pytest.raises((ValueError, KeyError)):
        inspect_run(run, spec)


@pytest.mark.parametrize(
    "changes",
    [
        {"mode": "full", "execution": "completed", "correctness": "pass"},
        {"execution": "completed"},
        {"correctness": "pass"},
        {"failed_phase": "correctness"},
        {"contract": "different-frozen-contract"},
        {"checks": [{"status": "pass"}]},
        {"samples": [{"us": 1}]},
        {"native_checks": [{"status": "pass"}]},
    ],
)
def test_only_empty_precontract_snapshot_failures_are_diagnostic(snapshot_failure, changes):
    _, run, result, spec = snapshot_failure
    result.update(changes)
    (run / "result.json").write_text(json.dumps(result))
    receipt = json.loads((run / "receipt.json").read_text())
    receipt["result_hash"] = digest(result)
    (run / "receipt.json").write_text(json.dumps(receipt))
    with pytest.raises((ValueError, KeyError)):
        inspect_run(run, spec)
