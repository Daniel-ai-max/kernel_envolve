"""Native evidence integration uses the same frozen autograd/project-gate fixture as the evaluator."""

import json

from kernel_envolve.manager.cli import task_contract
from kernel_envolve.manager.config import validate
from kernel_envolve.manager.evidence import refresh
from kernel_envolve.manager.store import Store
from kernel_envolve.manager.views import detail
from kernel_envolve.native import evaluate_native, promote_native

pytest_plugins = ["test_native"]


def test_native_manager_imports_full_evidence_and_detects_gate_tampering(native_task, tmp_path):
    task, _ = native_task
    spec = {
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
