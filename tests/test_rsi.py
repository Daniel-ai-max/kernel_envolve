"""RSI failure isolation and complete control flow using real CPU evaluators."""

import copy
import importlib.util
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

from kernel_envolve.manager.process import exclusive
from kernel_envolve.manager.store import Store as Manager
from kernel_envolve.manager.worker import checkpoint
from kernel_envolve.rsi.cli import audit, main
from kernel_envolve.rsi.config import initialize
from kernel_envolve.rsi.context import for_task
from kernel_envolve.rsi.learning import collect, learn, stage
from kernel_envolve.rsi.model import context, read
from kernel_envolve.rsi.process import invoke
from kernel_envolve.rsi.store import Store
from kernel_envolve.rsi.trials import adopt, rollback, validate, verdict
from kernel_envolve.storage import digest, write_json


@pytest.fixture
def rsi(tmp_path):
    path = Path(__file__).parents[1] / "examples/rsi/demo.py"
    spec = importlib.util.spec_from_file_location("rsi_demo", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    cfg = module.create(tmp_path / "demo")
    root = tmp_path / "rsi"
    initialize(root, cfg)
    store = Store(root)
    yield store
    store.close()


def proposal(store):
    collect(store)
    ref = store.db.execute("SELECT id FROM evidence ORDER BY seq LIMIT 1").fetchone()[0]
    return {
        "scope": "demo",
        "parent": "base",
        "hypothesis": "Improve synthetic fixture",
        "upsert": [
            {
                "id": "avoid-sleep",
                "when": "Synthetic demo only",
                "action": "Remove artificial sleep",
                "unless": "Never change actual task semantics",
                "evidence": [ref],
            }
        ],
        "remove": [],
    }


@pytest.mark.parametrize(
    "damage", ["missing", "empty", "missing_config_hash", "missing_operations"]
)
def test_open_damaged_state_cannot_reset_budget_or_recreate_schema(rsi, damage):
    import sqlite3

    rsi.reserve("learn", 2)
    rsi.close()
    database = rsi.root / "rsi.sqlite"
    if damage == "missing":
        database.unlink()
    elif damage == "empty":
        database.write_bytes(b"")
    else:
        with sqlite3.connect(database) as db:
            if damage == "missing_config_hash":
                db.execute("DELETE FROM meta WHERE key='config_hash'")
            else:
                db.execute("DROP TABLE operations")
    before = database.read_bytes() if database.exists() else None
    with pytest.raises((ValueError, sqlite3.Error)):
        Store(rsi.root)
    assert (database.read_bytes() if database.exists() else None) == before


def test_complete_learn_validate_adopt_pin_rollback(rsi, monkeypatch):
    # Exercise real subprocesses/evaluators, but keep this lifecycle test independent
    # of 25 ms process-poll timing jitter. Cost thresholds have explicit verdict tests.
    def measured_adapter(*args, **kwargs):
        elapsed = invoke(*args, **kwargs)
        return 0.1 if "kind" in args[2] else elapsed

    monkeypatch.setattr("kernel_envolve.rsi.trials.invoke", measured_adapter)
    collect(rsi)
    before = files_of_production(rsi)
    learned = learn(rsi)
    assert learned["status"] == "proposed"
    identity = learned["candidates"][0]
    assert rsi.active("demo") == "base"
    assert learn(rsi)["status"] == "insufficient_new_evidence"
    result = validate(rsi, identity)
    assert result["accepted"], result
    assert adopt(rsi, result["receipt"]) == identity
    manager = Manager(rsi.config["manager"], readonly=True)
    try:
        task = manager.task("demo")
        snapshot = checkpoint(manager, task, {"id": "fixture-round"})
        assert identity in snapshot and "Remove artificial sleep" in snapshot
        rollback(rsi, "demo", "Operator observed a counterexample")
        assert for_task(manager.root, task) == ""
        assert identity in snapshot  # Running sessions keep their already materialized prompt.
        with pytest.raises(ValueError, match="not validated"):
            adopt(rsi, result["receipt"])
    finally:
        manager.close()
    assert files_of_production(rsi) == before


def files_of_production(store):
    manager = Manager(store.config["manager"], readonly=True)
    try:
        path = Path(manager.task("demo")["spec"]["workspace"])
        return {p.name: p.read_bytes() for p in path.iterdir()}
    finally:
        manager.close()


def test_collector_incremental_and_lifecycle_not_metrics(rsi):
    assert collect(rsi) == 1
    assert collect(rsi) == 0
    row = json.loads(rsi.db.execute("SELECT data FROM evidence").fetchone()[0])
    assert row["authority"].startswith("agent interpretation")
    assert "speedup" not in row
    assert rsi.get("event_cursor") > 0


@pytest.mark.parametrize(
    "mutation",
    [
        lambda p: p.update(scope="other"),
        lambda p: p.update(parent="1" * 64),
        lambda p: p.update(code="print('run me')"),
        lambda p: p["upsert"][0].update(evidence=["a" * 64]),
        lambda p: p["upsert"][0].update(action="x" * 401),
        lambda p: p["upsert"].append(copy.deepcopy(p["upsert"][0])),
        lambda p: p.update(remove=[{"id": "missing", "reason": "unused"}]),
    ],
)
def test_reject_invalid_or_unproven_proposals(rsi, mutation):
    value = proposal(rsi)
    mutation(value)
    with pytest.raises(ValueError):
        stage(rsi, value)
    assert rsi.active("demo") == "base"


def test_delta_preserves_unmentioned_rules_and_explicit_removal(rsi):
    value = proposal(rsi)
    identity = stage(rsi, value)
    rsi.db.execute("INSERT INTO active VALUES('demo',?)", (identity,))
    next_value = copy.deepcopy(value)
    next_value["parent"] = identity
    next_value["upsert"][0]["id"] = "another"
    new = stage(rsi, next_value)
    assert len(rsi.candidate(new)["rules"]) == 2
    with pytest.raises(ValueError, match="No change"):
        stage(rsi, {**value, "parent": identity})


def test_budget_reservations_survive_crash_and_restart(rsi):
    total = rsi.config["learn_budget_seconds"]
    key = rsi.reserve("learn", total)
    another = Store(rsi.root)
    try:
        with pytest.raises(ValueError, match="budget exhausted"):
            another.reserve("learn", 1)
        another.finish(key, 2, "failed")
        another.reserve("learn", total - 2)
    finally:
        another.close()


@pytest.mark.parametrize("kind", ["learn", "validation"])
def test_disabled_state_cannot_reserve_new_work(rsi, kind):
    rsi.enable(False)
    with pytest.raises(ValueError, match="disabled"):
        rsi.reserve(kind, 1)
    assert rsi.db.execute("SELECT count(*) FROM operations").fetchone()[0] == 0


@pytest.mark.parametrize("elapsed", [-1, float("nan"), float("inf"), None, True])
def test_invalid_elapsed_cannot_release_or_corrupt_reserved_budget(rsi, elapsed):
    key = rsi.reserve("learn", 2)
    before = list(rsi.db.iterdump())
    with pytest.raises(ValueError):
        rsi.finish(key, elapsed)
    assert list(rsi.db.iterdump()) == before
    rsi.finish(key, 0)
    assert rsi.db.execute("SELECT charged FROM operations WHERE id=?", (key,)).fetchone()[0] == 0


@pytest.mark.parametrize("enabled", [1, "false"])
def test_enable_requires_boolean_without_changing_published_state(rsi, enabled):
    before = list(rsi.db.iterdump())
    with pytest.raises(ValueError):
        rsi.enable(enabled)
    assert list(rsi.db.iterdump()) == before


def test_learner_failure_consumes_batch_without_retry(rsi):
    collect(rsi)
    rsi.config["learner_command"] = [sys.executable, "-c", "raise SystemExit(7)"]
    with pytest.raises(ValueError, match="exited 7"):
        learn(rsi)
    assert learn(rsi)["status"] == "insufficient_new_evidence"
    row = rsi.db.execute("SELECT charged,error FROM operations").fetchone()
    assert row[0] > 0 and "exited 7" in row[1]


def test_no_llm_without_new_evidence_or_during_cooldown(rsi, monkeypatch):
    monkeypatch.setattr(
        "kernel_envolve.rsi.process.invoke", lambda *a, **k: pytest.fail("Unexpected LLM")
    )
    assert learn(rsi)["status"] == "insufficient_new_evidence"
    collect(rsi)
    rsi.set("last_learn", time.time())
    assert learn(rsi)["status"] == "cooldown"


def test_context_fail_open_and_contract_binding(rsi, capsys):
    manager = Manager(rsi.config["manager"], readonly=True)
    try:
        task = manager.task("demo")
        assert for_task(manager.root, task) == ""
        rsi.publish()
        path = manager.root / "rsi-context/demo.json"
        payload = read(path)
        payload["text"] = "test advice"
        payload["hash"] = digest({k: v for k, v in payload.items() if k != "hash"})
        write_json(path, payload)
        assert "test advice" in for_task(manager.root, task)
        task["spec"]["identity"] = "wrong"
        assert for_task(manager.root, task) == ""
        path.write_text("broken")
        assert for_task(manager.root, task) == ""
        path.write_bytes(b"x" * 8193)
        assert for_task(manager.root, task) == ""
        path.unlink()
        os.mkfifo(path)
        start = time.monotonic()
        assert for_task(manager.root, task) == ""
        assert time.monotonic() - start < 0.1
        assert "original workflow" in capsys.readouterr().err
    finally:
        manager.close()


def test_context_bound_limits_prompt_overhead(rsi):
    value = proposal(rsi)["upsert"][0]
    rules = [{**value, "id": f"r{i}", "action": "字" * 400} for i in range(6)]
    with pytest.raises(ValueError, match="2400"):
        context(rules)


@pytest.mark.parametrize("raw", ["null", "42", '"not an object"', "true", "[" * 4000 + "]" * 4000])
def test_malformed_advice_never_breaks_manager_checkpoint(rsi, raw, capsys):
    manager = Manager(rsi.config["manager"], readonly=True)
    try:
        task = manager.task("demo")
        expected = checkpoint(manager, task, {"id": "readiness-round"})
        directory = manager.root / "rsi-context"
        directory.mkdir(exist_ok=True)
        (directory / "demo.json").write_text(raw)
        assert checkpoint(manager, task, {"id": "readiness-round"}) == expected
        assert "original workflow" in capsys.readouterr().err
    finally:
        manager.close()


def test_advice_recursion_error_preserves_manager_checkpoint(rsi, monkeypatch, capsys):
    from kernel_envolve.rsi import context as reader

    manager = Manager(rsi.config["manager"], readonly=True)
    try:
        task = manager.task("demo")
        expected = checkpoint(manager, task, {"id": "readiness-round"})

        def recursive_json(*args):
            raise RecursionError("JSON nesting exceeds this interpreter's limit")

        monkeypatch.setattr(reader, "read", recursive_json)
        assert checkpoint(manager, task, {"id": "readiness-round"}) == expected
        assert "original workflow" in capsys.readouterr().err
    finally:
        manager.close()


def test_corrupted_supporting_event_revokes_advice(rsi):
    identity = stage(rsi, proposal(rsi))
    rsi.db.execute("INSERT INTO active VALUES('demo',?)", (identity,))
    rsi.db.execute("UPDATE candidates SET status='validated'")
    rsi.publish()
    manager = Manager(rsi.config["manager"])
    manager.db.execute("UPDATE events SET data='{}' WHERE kind='report'")
    manager.close()
    assert audit(rsi) == [identity]
    assert rsi.active("demo") == "base"


def test_immutable_config_and_duplicate_publisher(rsi, tmp_path):
    cfg = read(rsi.root / "config.json")
    cfg["learn_budget_seconds"] += 10
    write_json(rsi.root / "config.json", cfg)
    with pytest.raises(ValueError, match="configuration changed"):
        Store(rsi.root)
    with pytest.raises(ValueError, match="already has"):
        initialize(tmp_path / "other", Path(rsi.config["manager"]).parent / "rsi.json")


def test_protected_task_changes_fail_before_evaluation(rsi):
    identity = stage(rsi, proposal(rsi))
    suite = rsi.config["suites"]["demo"]
    suite["command"] = [
        sys.executable,
        "-c",
        "import json,sys; from pathlib import Path; p=json.load(sys.stdin); (Path(p['task'])/'task.json').write_text('{}')",
    ]
    with pytest.raises(ValueError, match="protected files"):
        validate(rsi, identity)
    assert rsi.active("demo") == "base"
    assert rsi.candidate(identity)["status"] == "rejected"


def test_template_tampering_prevents_validation(rsi):
    identity = stage(rsi, proposal(rsi))
    path = Path(rsi.config["suites"]["demo"]["tasks"][0]["path"]) / "task/task.py"
    path.write_text("changed")
    with pytest.raises(ValueError, match="fixture changed"):
        validate(rsi, identity)


def test_gate_preservation_and_repeated_benefit():
    suite = {
        "repeats": 2,
        "min_quality_gain": 0.02,
        "max_quality_regression": 0.01,
        "min_cost_gain": 0.1,
        "max_cost_regression": 0.05,
        "tasks": [{}, {}],
    }
    base = {"passed": True, "quality": 0.2, "solver_seconds": 10, "seconds": 12}
    pairs = [
        {"repeat": r, "task": t, "control": dict(base), "candidate": {**base, "quality": 0.3}}
        for r in range(2)
        for t in range(2)
    ]
    assert verdict(pairs, suite)["accepted"]
    pairs[1]["candidate"]["passed"] = False
    assert not verdict(pairs, suite)["accepted"]
    pairs[1]["candidate"] = {**base, "quality": 0.1, "solver_seconds": 1}
    assert not verdict(pairs, suite)["accepted"]
    pairs[1]["candidate"] = dict(base)
    pairs[0]["candidate"] = dict(base)
    assert not verdict(pairs, suite)["accepted"]


def gate_fixture(tasks=2):
    suite = {
        "repeats": 2,
        "tasks": [{} for _ in range(tasks)],
        "min_quality_gain": 0.01,
        "max_quality_regression": 0.02,
        "min_cost_gain": 0.1,
        "max_cost_regression": 0.05,
    }
    base = {"passed": True, "quality": 0.4, "solver_seconds": 10, "seconds": 12}
    pairs = [
        {"repeat": r, "task": t, "control": dict(base), "candidate": dict(base)}
        for r in range(2)
        for t in range(tasks)
    ]
    return suite, pairs


def test_gate_rejects_local_win_hiding_aggregate_quality_loss():
    suite, pairs = gate_fixture(16)
    for pair in pairs:
        pair["candidate"]["quality"] = 0.411 if pair["task"] == 0 else 0.381
    result = verdict(pairs, suite)
    assert not result["accepted"]
    assert "Aggregate quality regression" in result["reasons"]


@pytest.mark.parametrize("field", ["solver_seconds", "seconds"])
def test_gate_rejects_cost_explosion_despite_quality_and_local_cost_win(field):
    suite, pairs = gate_fixture(16)
    for pair in pairs:
        pair["candidate"]["quality"] = 0.5
        pair["candidate"][field] = 5 if pair["task"] == 0 else 100
    assert not verdict(pairs, suite)["accepted"]


def test_gate_accepts_whole_suite_cost_saving_and_reports_aggregates():
    suite, pairs = gate_fixture()
    for pair in pairs:
        pair["candidate"]["solver_seconds"] = 8
        pair["candidate"]["seconds"] = 10
    result = verdict(pairs, suite)
    assert result["accepted"]
    assert result["repeats"][0]["candidate_solver_seconds"] == 16
    assert result["repeats"][0]["quality_gain"] == 0


def test_gate_does_not_reward_fast_zero_yield_or_failure_score_arithmetic():
    suite, pairs = gate_fixture()
    for pair in pairs:
        pair["control"]["quality"] = 0
        pair["candidate"].update(quality=0, solver_seconds=1, seconds=3)
    assert not verdict(pairs, suite)["accepted"]
    # Fixing correctness is useful, but cannot mask losing valid speedups elsewhere.
    for pair in pairs:
        if pair["task"] == 0:
            pair["control"].update(passed=False, quality=-1)
        else:
            pair["control"]["quality"] = 0.02
            pair["candidate"]["quality"] = 0.001
    assert not verdict(pairs, suite)["accepted"]
    for pair in pairs:
        if pair["task"] == 1:
            pair["candidate"]["quality"] = 0.02
    assert verdict(pairs, suite)["accepted"]


@pytest.mark.parametrize("defect", ["missing", "duplicate", "nan", "infinity", "negative"])
def test_gate_fails_closed_on_invalid_trial_metrics(defect):
    suite, pairs = gate_fixture()
    for pair in pairs:
        pair["candidate"]["quality"] = 0.5
    if defect == "missing":
        pairs.pop()
    elif defect == "duplicate":
        pairs[1] = pairs[0]
    elif defect == "nan":
        pairs[0]["candidate"]["quality"] = float("nan")
    elif defect == "infinity":
        pairs[0]["candidate"]["solver_seconds"] = float("inf")
    else:
        pairs[0]["candidate"]["seconds"] = -1
    assert not verdict(pairs, suite)["accepted"]


def test_busy_operation_fails_without_duplicate_work(rsi):
    with exclusive(rsi.root / "operation.lock"):
        assert main(["--state", str(rsi.root), "learn"]) == 2
        assert main(["--state", str(rsi.root), "status"]) == 0
    assert rsi.db.execute("SELECT COUNT(*) FROM operations").fetchone()[0] == 0


def test_timeout_cleans_up_child_processes(tmp_path):
    code = "import subprocess,sys,time; p=subprocess.Popen([sys.executable,'-c','import time;time.sleep(90)']); open('child','w').write(str(p.pid)); time.sleep(90)"
    with pytest.raises(TimeoutError):
        invoke([sys.executable, "-c", code], tmp_path, {}, 0.2, device="cpu")
    pid = int((tmp_path / "child").read_text())
    # A reparented child may briefly be a zombie; it must not remain running.
    state = subprocess.run(
        ["ps", "-o", "stat=", "-p", str(pid)], text=True, capture_output=True
    ).stdout.strip()
    assert not state or state.startswith("Z")


def test_daemon_signal_during_adapter_exits_and_charges(rsi):
    # Explicitly reseal this synthetic process fixture; production state is immutable.
    cfg = read(rsi.root / "config.json")
    cfg["learner_command"] = [
        sys.executable,
        "-c",
        "import time;open('ready','w').write('1');time.sleep(60)",
    ]
    write_json(rsi.root / "config.json", cfg)
    rsi.set("config_hash", digest(cfg))
    write_json(
        Path(cfg["manager"]) / "rsi-owner.json",
        {"state": str(rsi.root), "config_hash": digest(cfg)},
    )
    process = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "kernel_envolve.cli",
            "rsi",
            "--state",
            str(rsi.root),
            "run",
            "--interval",
            "1",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    try:
        deadline = time.monotonic() + 10
        while not list((rsi.root / "learning").glob("*/ready")) and time.monotonic() < deadline:
            time.sleep(0.05)
        assert list((rsi.root / "learning").glob("*/ready"))
        process.send_signal(signal.SIGTERM)
        process.communicate(timeout=5)
        assert process.returncode == 0
        assert rsi.db.execute("SELECT charged FROM operations").fetchone()[0] is not None
    finally:
        if process.poll() is None:
            process.kill()
            process.wait()


def test_disable_is_available_during_learning_and_preserves_lineage(rsi):
    identity = stage(rsi, proposal(rsi))
    rsi.db.execute("INSERT INTO active VALUES('demo',?)", (identity,))
    rsi.publish()
    manager = Manager(rsi.config["manager"], readonly=True)
    try:
        task = manager.task("demo")
        assert identity in for_task(manager.root, task)
        with exclusive(rsi.root / "operation.lock"):
            assert main(["--state", str(rsi.root), "disable"]) == 0
        assert for_task(manager.root, task) == ""
        assert rsi.active("demo") == identity
        assert learn(rsi)["status"] == "disabled"
        assert main(["--state", str(rsi.root), "enable"]) == 0
        assert identity in for_task(manager.root, task)
    finally:
        manager.close()


def test_rejected_behavior_cannot_be_retried_with_new_explanation(rsi):
    value = proposal(rsi)
    identity = stage(rsi, value)
    rsi.db.execute("UPDATE candidates SET status='rejected'")
    value["hypothesis"] = "Different story about the same policy"
    value["upsert"][0]["id"] = "renamed"
    assert stage(rsi, value) == identity
    with pytest.raises(ValueError, match="already been evaluated"):
        validate(rsi, identity)


def test_validation_reserves_no_budget_when_device_busy(rsi, monkeypatch):
    import contextlib

    identity = stage(rsi, proposal(rsi))
    rsi.config["validation_device"] = "GPU-123"
    monkeypatch.setattr(
        "kernel_envolve.rsi.trials.exclusive", lambda path: contextlib.nullcontext(None)
    )
    assert validate(rsi, identity)["status"] == "resource_busy"
    assert rsi.db.execute("SELECT COUNT(*) FROM operations").fetchone()[0] == 0


def test_real_evaluator_records_collected_and_invalidated(rsi):
    from kernel_envolve.cli import evaluate
    from kernel_envolve.manager.evidence import refresh

    manager = Manager(rsi.config["manager"])
    try:
        task = manager.task("demo")
        run, _ = evaluate(task["spec"]["task_dir"], "full")
        refresh(manager)
        assert collect(rsi) == 2  # existing report + new evaluator run
        assert collect(rsi) == 0
        ref = next(
            r[0]
            for r in rsi.db.execute(
                "SELECT id FROM evidence WHERE json_extract(data,'$.type')='run'"
            )
        )
        evidence = json.loads(
            rsi.db.execute("SELECT data FROM evidence WHERE id=?", (ref,)).fetchone()[0]
        )
        indexed = json.loads(manager.one("SELECT data FROM runs WHERE path=?", (str(run),))["data"])
        assert evidence["source_digest"] == digest(indexed["source"])
        assert evidence["failed_checks"] == indexed["failed_checks"]
        assert evidence["metrics"]["baseline_speedup"] == indexed["baseline_speedup"]
        assert evidence["metrics"]["incremental_speedup"] == indexed["incremental_speedup"]
        assert (
            evidence["metrics"]["fixed_baseline_gain_by_batch"]
            == indexed["decision"]["fixed_baseline_gain_by_batch"]
        )
        assert evidence["metrics"]["eligible_for_best"] == indexed["decision"]["eligible_for_best"]
        assert not evidence["metrics"]["truncated"]
        value = proposal(rsi)
        value["upsert"][0]["evidence"] = [ref]
        stage(rsi, value)
        result = read(run / "result.json")
        result["correctness"] = "fail"
        write_json(run / "result.json", result)
        with pytest.raises(ValueError, match="receipt mismatch"):
            stage(rsi, {**value, "hypothesis": "Another explanation"})
    finally:
        manager.close()


def test_failed_check_conditions_reach_background_learner(rsi):
    from kernel_envolve.cli import evaluate
    from kernel_envolve.manager.evidence import refresh

    manager = Manager(rsi.config["manager"])
    try:
        task = Path(manager.task("demo")["spec"]["task_dir"])
        (task / "candidate.py").write_text("def run(x): return x+2\n")
        _, result = evaluate(task, "full")
        assert result["correctness"] == "fail"
        refresh(manager)
        collect(rsi)
        outcome = learn(rsi)
        packet = read(rsi.root / "learning" / outcome["operation"] / "input.json")
        observed = next(row for row in packet["evidence"] if row["type"] == "run")
        assert observed["failed_checks"]["count"] == 1
        check = observed["failed_checks"]["checks"][0]
        assert check["role"] == "candidate"
        assert check["case"] == result["checks"][-1]["case"]
        assert check["args"] is not None
        assert "unlisted cases are unknown" in observed["failed_checks"]["scope"]
    finally:
        manager.close()


def test_detach_requires_disable_and_prevents_old_publisher(rsi):
    with pytest.raises(ValueError, match="Disable"):
        rsi.detach()
    rsi.enable(False)
    assert main(["--state", str(rsi.root), "detach"]) == 0
    with pytest.raises((ValueError, OSError)):
        rsi.publish()
    assert not (Path(rsi.config["manager"]) / "rsi-owner.json").exists()


def test_background_cycle_keeps_current_validated_strategy(rsi, monkeypatch):
    from kernel_envolve.rsi.cli import cycle

    identity = stage(rsi, proposal(rsi))
    rsi.db.execute("UPDATE candidates SET status='validated'")
    rsi.db.execute("INSERT INTO active VALUES('demo',?)", (identity,))
    rsi.set("last_audit", time.time())
    rsi.set("learned_seq", 10000)
    monkeypatch.setattr(
        "kernel_envolve.rsi.cli.validate", lambda *args: pytest.fail("Active strategy revalidated")
    )
    cycle(rsi)
    assert rsi.candidate(identity)["status"] == "validated"


def test_adoption_rejects_tampered_receipt_and_changed_parent(rsi, monkeypatch):
    import kernel_envolve.rsi.trials as trials

    identity = stage(rsi, proposal(rsi))
    rsi.db.execute(
        "INSERT INTO validations VALUES(?,?,?)",
        ("bad", identity, json.dumps({"evaluator_hash": trials.evaluator_hash()})),
    )
    with pytest.raises(ValueError, match="receipt mismatch"):
        adopt(rsi, "bad")
    candidate = rsi.candidate(identity)
    monkeypatch.setattr(trials, "checked_receipt", lambda *args: candidate)
    rsi.db.execute("INSERT INTO active VALUES('demo','other')")
    with pytest.raises(ValueError, match="Active strategy changed"):
        adopt(rsi, "fixture")
    assert rsi.active("demo") == "other"


def test_superseded_stable_version_remains_available_for_rollback(rsi, monkeypatch):
    from kernel_envolve.rsi.cli import cycle

    first = stage(rsi, proposal(rsi))
    rsi.db.execute("UPDATE candidates SET status='validated'")
    rsi.db.execute("INSERT INTO active VALUES('demo',?)", (first,))
    rsi.db.execute(
        "INSERT INTO history(at,scope,previous,current,reason) VALUES(0,'demo','base',?,'fixture')",
        (first,),
    )
    update = proposal(rsi)
    update["parent"] = first
    update["upsert"][0]["action"] = "Second distinct strategy"
    second = stage(rsi, update)
    rsi.db.execute("UPDATE candidates SET status='validated' WHERE id=?", (second,))
    rsi.db.execute("UPDATE active SET id=?", (second,))
    rsi.db.execute(
        "INSERT INTO history(at,scope,previous,current,reason) VALUES(0,'demo',?,?,'fixture')",
        (first, second),
    )
    rsi.set("last_audit", time.time())
    rsi.set("learned_seq", 10000)
    monkeypatch.setattr(
        "kernel_envolve.rsi.cli.adopt", lambda *args: pytest.fail("Ancestor readopted")
    )
    cycle(rsi)
    assert rsi.candidate(first)["status"] == "validated"
    assert rollback(rsi, "demo", "test") == first


def test_learning_packet_is_bounded_without_dropping_pending_evidence(rsi, monkeypatch):
    from kernel_envolve.rsi.learning import learn

    rsi.config["max_evidence"] = 48
    for n in range(48):
        rsi.add_evidence({"type": "event", "task": "demo", "n": n, "data": "x" * 9000})
    observed = []

    def fake_adapter(command, directory, packet, timeout, **kwargs):
        observed.append(packet)
        write_json(directory / "proposals.json", [])

    monkeypatch.setattr("kernel_envolve.rsi.process.invoke", fake_adapter)
    assert learn(rsi)["status"] == "proposed"
    assert len(json.dumps(observed[0], ensure_ascii=False).encode()) <= 131072
    assert 0 < rsi.get("learned_seq") < 48
    assert rsi.db.execute("SELECT COUNT(*) FROM evidence").fetchone()[0] == 48


@pytest.mark.parametrize(
    "changed", ["research.py", "episodes.py", "reuse.py", "allocation.py", "artifact_snapshot.py"]
)
def test_validator_identity_binds_owned_research_dependencies(monkeypatch, changed):
    from kernel_envolve.rsi import trials

    original = trials.validator_identity()
    original_hash = trials.file_hash
    monkeypatch.setattr(
        trials, "file_hash", lambda path: "0" * 64 if path.name == changed else original_hash(path)
    )
    assert trials.validator_identity() != original


def test_numeric_evidence_is_bounded_and_does_not_invent_missing_metrics():
    from kernel_envolve.rsi.learning import numeric_metrics

    data = {
        "baseline_speedup": float("inf"),
        "incremental_speedup": True,
        "decision": {"fixed_baseline_gain_by_batch": [0.25] * 20, "eligible_for_best": False},
    }
    metrics = numeric_metrics(data)
    assert metrics["baseline_speedup"] is None
    assert metrics["incremental_speedup"] is None
    assert metrics["fixed_baseline_gain_by_batch"] == [0.25] * 8
    assert metrics["batch_count"] == 20 and metrics["truncated"]
    missing = numeric_metrics({"decision": {}})
    assert missing["baseline_speedup"] is None and missing["eligible_for_best"] is None
    assert missing["fixed_baseline_gain_by_batch"] == []


def test_case_latency_summary_retains_comparator_identity_and_bounds():
    from kernel_envolve.rsi.learning import numeric_metrics

    case = {
        "case": "strided",
        "batch": 0,
        "candidate_us": 40.0,
        "comparator_us": 20.0,
        "speedup": 0.5,
    }
    metrics = numeric_metrics({"decision": {"cases": [case] * 20}})
    assert metrics["cases"] == [case] * 8
    assert metrics["case_count"] == 20 and metrics["cases_truncated"]
    # This comparator can be current best; do not relabel it as fixed baseline.
    assert "baseline_us" not in metrics["cases"][0]


@pytest.mark.parametrize("already_active", [False, True])
def test_adapter_source_change_invalidates_successful_receipt(rsi, monkeypatch, already_active):
    def measured_adapter(*args, **kwargs):
        elapsed = invoke(*args, **kwargs)
        return 0.1 if "kind" in args[2] else elapsed

    monkeypatch.setattr("kernel_envolve.rsi.trials.invoke", measured_adapter)
    identity = stage(rsi, proposal(rsi))
    result = validate(rsi, identity)
    assert result["accepted"], result
    if already_active:
        adopt(rsi, result["receipt"])
    adapter = Path(rsi.config["suites"]["demo"]["command"][1])
    adapter.write_text("raise SystemExit('Changed solver version')\n")
    with pytest.raises(ValueError, match="[Aa]dapter"):
        adopt(rsi, result["receipt"])
    if already_active:
        audit(rsi)
    assert rsi.active("demo") == "base"


def test_adapter_change_during_validation_rejects_candidate(rsi, monkeypatch):
    adapter = Path(rsi.config["suites"]["demo"]["command"][1])

    def changing_adapter(*args, **kwargs):
        elapsed = invoke(*args, **kwargs)
        if "kind" in args[2]:
            adapter.write_text(adapter.read_text() + "\n# Changed during the experiment\n")
        return elapsed

    monkeypatch.setattr("kernel_envolve.rsi.trials.invoke", changing_adapter)
    identity = stage(rsi, proposal(rsi))
    with pytest.raises(ValueError, match="Adapter changed during validation"):
        validate(rsi, identity)
    assert rsi.candidate(identity)["status"] == "rejected"
    assert rsi.active("demo") == "base"
    assert not rsi.db.execute("SELECT * FROM validations").fetchall()


def test_explicit_adapter_dependency_identity(tmp_path):
    from kernel_envolve.rsi.trials import adapter_identity

    dependency = tmp_path / "prompt.txt"
    dependency.write_text("initial prompt")
    suite = {"command": [sys.executable, "-m", "package"], "adapter_files": [str(dependency)]}
    before = adapter_identity(suite)
    dependency.write_text("modified prompt")
    assert before != adapter_identity(suite)
    dependency.unlink()
    with pytest.raises(ValueError, match="Adapter file"):
        adapter_identity(suite)
