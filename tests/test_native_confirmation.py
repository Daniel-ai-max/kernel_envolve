import copy
from pathlib import Path

import pytest
from test_native import native_task as native_task

from kernel_envolve.native import evaluate_native, promote_native
from kernel_envolve.native_config import load_config
from kernel_envolve.native_confirmation import confirm_native, verify_confirmation
from kernel_envolve.storage import digest, file_hash, read_json, write_json


@pytest.fixture
def confirmation_task(native_task):
    task, candidate = native_task
    cfg = read_json(task / "native.json")
    cfg["cases"][0]["args"] = {"n": 4}
    cfg["confirmation"] = {
        "cases": [{"id": "unseen", "args": {"n": 7}, "weight": 1}],
        "seeds": [83],
        "trials": 3,
    }
    write_json(task / "native.json", cfg)
    adapter = (task / "adapter.py").read_text()
    adapter = adapter.replace(
        "base = torch.randn(2, 11", "n = case['args']['n']\n    base = torch.randn(2, n + 7"
    )
    adapter = adapter.replace("base[:, :4]", "base[:, :n]").replace(
        "torch.randn(2, 4,", "torch.randn(2, n,"
    )
    (task / "adapter.py").write_text(adapter)
    return task, candidate


@pytest.mark.parametrize(
    "change",
    [
        lambda c: c.update(seeds=[0]),
        lambda c: c.update(trials=2),
        lambda c: c.update(cases=[{"id": "unseen", "args": {"n": 4}, "weight": 1}]),
        lambda c: c.update(cases=[{"id": "padded", "args": {"n": 7}, "weight": 1}]),
        lambda c: c.update(seeds=[]),
        lambda c: c.update(trials=True),
        lambda c: c.update(extra=True),
    ],
)
def test_confirmation_contract_rejects_reused_inputs(confirmation_task, change):
    task, _ = confirmation_task
    cfg = read_json(task / "native.json")
    change(cfg["confirmation"])
    write_json(task / "native.json", cfg)
    with pytest.raises(ValueError):
        load_config(task)


def test_real_full_confirmation_promotion_and_integrity(confirmation_task):
    from kernel_envolve.manager.evidence import inspect_run
    from kernel_envolve.native import _best

    task, candidate = confirmation_task
    run, result = evaluate_native(task, "full", timeout=90)
    assert result["decision"]["eligible_for_best"], result
    cfg = read_json(run / "config.json")
    with pytest.raises(ValueError, match="confirmation receipt is required"):
        promote_native(run)
    spec = {"task_dir": str(task), "kind": "native", "baseline": {"contract": result["contract"]}}
    pending = inspect_run(run, spec)
    assert pending["correctness"] == "pass" and pending["full_eligible_for_best"]
    assert pending["confirmation_status"] == "pending"
    assert not pending["decision"]["eligible_for_best"]
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr("kernel_envolve.native_confirmation.evaluator_hash", lambda: "changed")
        with pytest.raises(ValueError, match="Evaluator changed"):
            confirm_native(run, 90)
    original_config = (task / "native.json").read_bytes()
    changed_config = read_json(task / "native.json")
    changed_config["confirmation"]["seeds"] = [89]
    write_json(task / "native.json", changed_config)
    try:
        with pytest.raises(ValueError, match="Live task changed"):
            confirm_native(run, 90)
    finally:
        (task / "native.json").write_bytes(original_config)
    original = (candidate / "toyop.py").read_text()
    (candidate / "toyop.py").write_text(original + "\n# changed after full\n")
    with pytest.raises(ValueError, match="Live confirmation source changed"):
        confirm_native(run, 90)
    assert not (task / ".kernel_native/confirmation-attempt.json").exists()
    (candidate / "toyop.py").write_text(original)
    receipt = confirm_native(run, 90)
    assert receipt["status"] == "passed", receipt
    phases = receipt["phase_wall_seconds"]
    assert set(phases) == {
        "lock_wait",
        "parent_audit",
        "live_sources_before",
        "child_full",
        "live_sources_after",
        "posterior_audit",
    }
    assert all(seconds >= 0 for seconds in phases.values())
    assert receipt["seconds"] >= sum(phases.values())
    confirmation = verify_confirmation(run, result, cfg)
    child = Path(confirmation["run"])
    held = read_json(child / "result.json")
    assert held["mode"] == "full" and held["correctness"] == "pass"
    assert {check["case"] for check in held["checks"]} == {"unseen"}
    assert len(held["native_checks"]) == 3
    spec = {"task_dir": str(task), "kind": "native", "baseline": {"contract": result["contract"]}}
    assert inspect_run(run, spec)["confirmation"] == confirmation

    # Every independent part remains bound to the original full before promotion.
    raw_evidence = next(iter(read_json(child / "receipt.json")["evidence"]))
    targets = [
        child / raw_evidence,
        child.parents[2] / "native.json",
        run / "confirmation/receipt.json",
        child / "config.json",
        child / "receipt.json",
        child / "adapter.py",
        child / "sources/candidate/core/toyop.py",
        task / ".kernel_native/confirmation-attempt.json",
    ]
    for target in targets:
        saved = target.read_bytes()
        target.write_bytes(b"{}\n" if target.suffix == ".json" else b"# altered\n")
        try:
            with pytest.raises((ValueError, KeyError)):
                verify_confirmation(run, result, cfg)
            with pytest.raises((ValueError, KeyError)):
                inspect_run(run, spec)
        finally:
            target.write_bytes(saved)

    # Even internally consistent receipts cannot confirm on another runtime.
    env_targets = [child / "result.json", child / "receipt.json", run / "confirmation/receipt.json"]
    saved_env = {path: path.read_bytes() for path in env_targets}
    changed_held = copy.deepcopy(held)
    for environment in changed_held["environments"].values():
        environment["gpu"] = "another-device"
    child_receipt = read_json(child / "receipt.json")
    child_receipt["result_hash"] = digest(changed_held)
    write_json(child / "result.json", changed_held)
    write_json(child / "receipt.json", child_receipt)
    changed_confirmation = read_json(run / "confirmation/receipt.json")
    changed_confirmation["result_hash"] = digest(changed_held)
    changed_confirmation["receipt_hash"] = file_hash(child / "receipt.json")
    write_json(run / "confirmation/receipt.json", changed_confirmation)
    try:
        with pytest.raises(ValueError, match="runtime environments differ"):
            verify_confirmation(run, result, cfg)
        with pytest.raises(ValueError, match="runtime environments differ"):
            promote_native(run)
    finally:
        for path, data in saved_env.items():
            path.write_bytes(data)

    pointer = promote_native(run)
    assert pointer["confirmation"] == confirmation
    assert _best(task / ".kernel_native")[0] == pointer
    saved_config = (run / "config.json").read_bytes()
    changed = dict(cfg)
    del changed["confirmation"]
    write_json(run / "config.json", changed)
    try:
        with pytest.raises(ValueError, match="Best confirmation config/receipt changed"):
            _best(task / ".kernel_native")
    finally:
        (run / "config.json").write_bytes(saved_config)

    # The task-level consumption record is outside the run subtree: cached
    # manager evidence must still invalidate immediately when it disappears.
    from kernel_envolve.manager.cli import task_contract
    from kernel_envolve.manager.config import validate
    from kernel_envolve.manager.evidence import refresh
    from kernel_envolve.manager.store import Store
    from kernel_envolve.manager.views import detail

    registered = {
        "id": "held",
        "name": "Confirmed",
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
            "frozen_on": "2026-11-11",
            "hardware": "CPU",
            "contract": task_contract(task, "native"),
        },
    }
    store = Store(task.parent / "manager")
    claim = task / ".kernel_native/confirmation-attempt.json"
    saved_claim = claim.read_bytes()
    try:
        store.register(
            validate(
                {
                    "version": 1,
                    "resources": [{"id": "cpu", "device": "cpu"}],
                    "tasks": [registered],
                },
                task.parent,
            )
        )
        refresh(store)
        assert detail(store, "held")["best"]
        claim.unlink()
        refresh(store)
        assert detail(store, "held")["best"] is None
        claim.write_bytes(saved_claim)
        refresh(store)
        assert detail(store, "held")["best"]
    finally:
        claim.write_bytes(saved_claim)
        store.close()
    with pytest.raises(ValueError):
        confirm_native(run, 90)
    saved = (run / "confirmation/receipt.json").read_bytes()
    (run / "confirmation/receipt.json").unlink()
    try:
        with pytest.raises(ValueError, match="confirmation receipt is required"):
            _best(task / ".kernel_native")
        assert not inspect_run(run, spec)["decision"]["eligible_for_best"]
    finally:
        (run / "confirmation/receipt.json").write_bytes(saved)


@pytest.mark.parametrize(
    "body",
    [
        "def run(x): return x * (3 if x.shape[-1] == 7 else 2)\n",
        "import time\ndef run(x):\n if x.shape[-1] == 7: time.sleep(.02)\n return x * 2\n",
    ],
)
def test_failed_confirmation_consumes_task_even_for_new_candidate(confirmation_task, body):
    task, candidate = confirmation_task
    # Correct on development shape, incorrect only on the reserved shape.
    (candidate / "toyop.py").write_text(body)
    run, result = evaluate_native(task, "full", timeout=90)
    assert result["decision"]["eligible_for_best"], result
    receipt = confirm_native(run, 90)
    assert receipt["status"] == "failed"
    with pytest.raises(ValueError, match="already consumed"):
        confirm_native(run, 90)
    with pytest.raises(ValueError, match="confirmation failed"):
        promote_native(run)
    (candidate / "toyop.py").write_text("def run(x): return x * 2\n")
    second, result = evaluate_native(task, "full", timeout=90)
    assert result["decision"]["eligible_for_best"], result
    with pytest.raises(ValueError, match="already consumed"):
        confirm_native(second, 90)
    assert not (second / "confirmation").exists()


def test_confirmation_interruption_is_not_retryable(confirmation_task, monkeypatch):
    import kernel_envolve.native as native

    task, _ = confirmation_task
    run, result = evaluate_native(task, "full", timeout=90)
    assert result["decision"]["eligible_for_best"], result

    def interrupted(*args, **kwargs):
        raise KeyboardInterrupt("simulated process interruption")

    with monkeypatch.context() as patch:
        patch.setattr(native, "evaluate_native", interrupted)
        with pytest.raises(KeyboardInterrupt):
            confirm_native(run, 90)
    assert read_json(run / "confirmation/receipt.json")["status"] == "failed"
    with pytest.raises(ValueError, match="already consumed"):
        confirm_native(run, 90)
