"""Real CPU receipts test scoped reuse feedback, not GPU or learning efficacy."""

import ast
import json
from pathlib import Path

import pytest

from kernel_envolve.cli import evaluate
from kernel_envolve.rsi import artifact_lineage as lineage
from kernel_envolve.rsi import reuse_memory as module
from kernel_envolve.rsi.research import run
from kernel_envolve.rsi.reuse_memory import ReuseMemory
from kernel_envolve.storage import write_json

REFERENCE = (
    "import numpy as np\nimport time\n"
    "def make_inputs(case,seed): return (np.ones(case['n']),)\n"
    "def reference(x): return x+1\n"
    "def baseline(x):\n time.sleep(.006)\n return x+1\n"
)
INITIAL = "import time\ndef run(x):\n time.sleep(.003)\n return x+1\n"
GOOD = "def run(x): return x+1\n"
SCOPED = "def run(x): return x+1 if x.size==16 else x+2\n"
SLOWER = "import time\ndef run(x):\n time.sleep(.004)\n return x+1\n"


def make_task(root, n=16, initial=INITIAL):
    root.mkdir()
    (root / "task.py").write_text(REFERENCE)
    (root / "candidate.py").write_text(initial)
    write_json(
        root / "task.json",
        {
            "name": "reuse-fixture",
            "objective": "CPU engineering fixture",
            "device": "cpu",
            "cases": [{"id": "a", "args": {"n": n}, "weight": 1}],
            "warmup": 1,
            "full_trials": 3,
            "quick_trials": 1,
            "full_seeds": [0],
            "quick_seeds": [0],
        },
    )
    return root


def trial(task, output, source=None, snapshot=None, *, max_actions=4):
    packet = {
        "task": str(task),
        "kind": "generic",
        "editable_files": [str(task / "candidate.py")],
        "model": "deterministic-no-model",
        "seed": 0,
        "strategy": "",
        "seconds": 30,
    }
    calls = []

    def decide(request, timeout):
        calls.append(request)
        if source is not None and (
            not request["history"] or len(request["history"]) == 1 and snapshot
        ):
            return {
                "action": "propose",
                "source": source,
                "mode": "full",
                "hypothesis": "CPU fixture",
            }
        return {"action": "stop", "hypothesis": "CPU fixture stop"}

    trace = run(
        packet,
        output,
        decide,
        max_actions=max_actions,
        artifact_snapshot=snapshot,
        source_validator=ast.parse,
    )
    final, _ = evaluate(task, "full", timeout=10)
    return trace, final, calls


@pytest.fixture
def setup(tmp_path):
    archive = tmp_path / "archive"
    counter = 0

    def seed(source=GOOD):
        nonlocal counter
        counter += 1
        task = make_task(tmp_path / f"training-{counter}")
        out = tmp_path / f"seed-trace-{counter}"
        # A slower seed must still be the best full source on its formation task.
        if source == SLOWER:
            (task / "candidate.py").write_text(INITIAL.replace(".003", ".005"))
        _, final, _ = trial(task, out, source)
        return lineage.collect(archive, task, out, {}, confirmation_run=final)

    first = seed()
    memory = ReuseMemory.create(tmp_path / "memory", archive)
    entry = memory.register(first)
    scope = lineage.audit(archive, first)["scope"]
    yield memory, seed, scope, entry, first
    memory.close()


def test_reuse_uses_one_fixed_action_and_feedback_does_not_run_work(setup, tmp_path, monkeypatch):
    memory, _, scope, _, _ = setup
    task = make_task(tmp_path / "target")
    out = tmp_path / "target-trace"
    offer = memory.prepare(scope, task, {}, out)
    trace, final, calls = trial(task, out, snapshot=offer["snapshot"])
    assert trace["max_actions"] == 4 and not trace.get("adaptive_artifact_budget")
    assert trace["steps"][0]["decision_kind"] == "artifact_proposal"
    assert calls[0]["remaining_actions"] == 3
    assert len(calls) == 1  # Reuse itself made no callback; the fixture then stopped.
    from kernel_envolve.rsi import research

    monkeypatch.setattr(
        research, "evaluate", lambda *a, **kw: pytest.fail("No feedback evaluation")
    )
    report = memory.observe(offer["ticket"], confirmation_run=final)
    assert report["status"] == "helpful"
    assert report["formed_candidate"] is None
    assert report["reported_cost"]["model_tokens"] is None
    assert memory.observe(offer["ticket"], confirmation_run=final) == report
    (task / "candidate.py").write_text(INITIAL)
    again = memory.prepare(scope, task, {}, tmp_path / "again")
    assert again["ticket"] and again["snapshot"] == offer["snapshot"]
    assert not (task / ".kernel_envolve/best.json").exists()


def test_failed_reuse_suppressed_for_same_configuration_across_incumbents(setup, tmp_path):
    memory, seed, scope, first_entry, _ = setup
    memory.set_entry_enabled(first_entry, False)
    scoped_record = seed(SCOPED)
    scoped_entry = memory.register(scoped_record)
    task = make_task(tmp_path / "wide", n=32)
    out = tmp_path / "wide-trace"
    offer = memory.prepare(scope, task, {}, out)
    _, final, _ = trial(task, out, snapshot=offer["snapshot"])
    report = memory.observe(offer["ticket"], confirmation_run=final)
    assert report["status"] == "failed_on_target"
    assert (task / "candidate.py").read_text() == INITIAL
    assert memory.register(seed(SCOPED)) == scoped_entry  # A fresh receipt earns no reset.
    renamed = make_task(tmp_path / "renamed", n=32)
    cfg = json.loads((renamed / "task.json").read_text())
    cfg.update(name="another-name", objective="another-label")
    write_json(renamed / "task.json", cfg)
    assert memory.prepare(scope, renamed, {}, tmp_path / "renamed-trace")["ticket"] is None
    narrow = make_task(tmp_path / "narrow", n=16)
    assert memory.prepare(scope, narrow, {}, tmp_path / "narrow-trace")["ticket"]
    (task / "candidate.py").write_text(INITIAL + "# a distinct incumbent\n")
    assert memory.prepare(scope, task, {}, tmp_path / "changed-incumbent")["ticket"] is None
    # Withdrawals do not erase failure facts, but the next explicitly registered
    # source remains available on the failed target.
    memory.set_entry_enabled(first_entry, True)
    fallback = memory.prepare(scope, renamed, {}, tmp_path / "fallback")
    assert fallback["ticket"]
    assert memory._ticket(fallback["ticket"])[1]["entry"] == first_entry


def test_correct_but_slower_reuse_is_not_recommended_again(setup, tmp_path):
    memory, seed, scope, first_entry, _ = setup
    memory.set_entry_enabled(first_entry, False)
    memory.register(seed(SLOWER))
    task = make_task(tmp_path / "target")
    out = tmp_path / "target-trace"
    offer = memory.prepare(scope, task, {}, out)
    _, final, _ = trial(task, out, snapshot=offer["snapshot"])
    report = memory.observe(offer["ticket"], confirmation_run=final)
    assert report["status"] == "unhelpful_on_target"
    assert memory.prepare(scope, task, {}, tmp_path / "again")["ticket"] is None
    (task / "candidate.py").write_text(INITIAL.replace(".003", ".005"))
    assert memory.prepare(scope, task, {}, tmp_path / "weaker-incumbent")["ticket"]


def test_pending_and_abandoned_uses_never_gain_automatic_retries(setup, tmp_path):
    memory, _, scope, entry, _ = setup
    task = make_task(tmp_path / "target")
    offer = memory.prepare(scope, task, {}, tmp_path / "out")
    other = ReuseMemory(memory.root)
    try:
        assert other.prepare(scope, task, {}, tmp_path / "concurrent")["ticket"] is None
        report = memory.abandon(offer["ticket"], "User stopped before independent final")
        assert report["status"] == "incomplete" and report["record"] is None
        assert memory.abandon(offer["ticket"], "User stopped before independent final") == report
        memory.set_entry_enabled(entry, False)
        memory.set_entry_enabled(entry, True)
        assert other.prepare(scope, task, {}, tmp_path / "after")["ticket"] is None
    finally:
        other.close()


def test_disabled_or_exhausted_memory_performs_no_raw_audit(setup, tmp_path, monkeypatch):
    memory, _, scope, _, _ = setup
    task = make_task(tmp_path / "target")
    memory.set_enabled(False)
    monkeypatch.setattr(
        lineage, "audit", lambda *a: pytest.fail("Disabled means no archive traversal")
    )
    assert memory.prepare(scope, task, {}, tmp_path / "out")["reason"] == "disabled"


def test_incumbent_and_unrelated_scope_skip_work(setup, tmp_path, monkeypatch):
    memory, _, scope, _, _ = setup
    task = make_task(tmp_path / "target", initial=GOOD)
    no_offer = memory.prepare(scope, task, {}, tmp_path / "out")
    assert no_offer["ticket"] is None
    assert no_offer["skipped"][0]["reason"] == "already_incumbent"
    monkeypatch.setattr(lineage, "audit", lambda *a: pytest.fail("No unrelated raw audit"))
    assert memory.prepare("f" * 64, task, {}, tmp_path / "other")["ticket"] is None


def test_raw_evidence_change_cannot_hide_behind_snapshot(setup, tmp_path):
    memory, _, scope, _, record_id = setup
    record = lineage.audit(memory.archive, record_id)
    path = Path(record["provenance"]["artifacts"]) / "search-trace.json"
    trace = json.loads(path.read_text())
    trace["error"] = {"type": "InterruptedError", "message": "incomplete"}
    write_json(path, trace)
    task = make_task(tmp_path / "target")
    offer = memory.prepare(scope, task, {}, tmp_path / "out")
    assert offer["ticket"] is None and offer["skipped"][0]["reason"] == "invalid_evidence"
    assert memory.db.execute("SELECT count(*) FROM trials").fetchone()[0] == 0


@pytest.mark.parametrize("mutation", ["budget", "initial", "task", "missing_final"])
def test_feedback_rejects_unbound_or_incomplete_trials(setup, tmp_path, mutation):
    memory, _, scope, _, _ = setup
    task = make_task(tmp_path / "target")
    out = tmp_path / "out"
    offer = memory.prepare(scope, task, {}, out)
    if mutation == "initial":
        (task / "candidate.py").write_text(INITIAL + "# changed before execution\n")
    if mutation == "task":
        cfg = json.loads((task / "task.json").read_text())
        cfg["cases"][0]["args"]["n"] = 64
        write_json(task / "task.json", cfg)
    _, final, _ = trial(
        task, out, snapshot=offer["snapshot"], max_actions=2 if mutation == "budget" else 4
    )
    if mutation == "missing_final":
        final = task / ".kernel_envolve/runs/missing"
    with pytest.raises((ValueError, FileNotFoundError)):
        memory.observe(offer["ticket"], confirmation_run=final)
    assert memory._ticket(offer["ticket"])[0]["status"] == "pending"
    memory.abandon(offer["ticket"], "Invalid or incomplete evidence retained")


def test_fallback_is_recorded_as_unactivated_not_a_quality_failure(setup, tmp_path):
    memory, _, scope, _, _ = setup
    task = make_task(tmp_path / "target")
    out = tmp_path / "out"
    offer = memory.prepare(scope, task, {}, out)
    unavailable = offer["snapshot"] | {"path": str(tmp_path / "missing.json")}
    _, final, _ = trial(task, out, snapshot=unavailable)
    report = memory.observe(offer["ticket"], confirmation_run=final)
    assert report["status"] == "not_activated"
    assert report["reuse_observations"] == []


def test_lifetime_limits_and_future_trace_reservations(setup, tmp_path):
    original, _, scope, _, seed_id = setup
    small = ReuseMemory.create(tmp_path / "small", original.archive, max_entries=1, max_trials=1)
    try:
        small.register(seed_id)
        task = make_task(tmp_path / "target")
        out = tmp_path / "out"
        offer = small.prepare(scope, task, {}, out)
        small.abandon(offer["ticket"], "No work started")
        assert small.prepare(scope, task, {}, tmp_path / "again")["reason"] == "trial_limit"
        with pytest.raises(ValueError, match="2–8 fixed"):
            small.prepare(scope, task, {}, out, max_actions=1)
        out.mkdir()
        (out / "search-trace.json").write_text("{}")
        with pytest.raises(ValueError, match="future unused"):
            small.prepare(scope, task, {}, out)
    finally:
        small.close()


def test_memory_code_identity_does_not_change_historical_lineage(setup, tmp_path, monkeypatch):
    memory, _, scope, _, record = setup
    monkeypatch.setattr(module, "_implementation", lambda: "0" * 64)
    with pytest.raises(ValueError, match="code changed"):
        ReuseMemory(memory.root)
    assert lineage.audit(memory.archive, record)["kind"] == "seed"


@pytest.mark.parametrize("withdraw", ["global", "entry"])
def test_withdrawal_during_audit_prevents_new_offer(setup, tmp_path, monkeypatch, withdraw):
    memory, _, scope, entry, _ = setup
    task = make_task(tmp_path / "target")
    original = memory._entry

    def audited(row):
        value = original(row)
        if withdraw == "global":
            memory.set_enabled(False)
        else:
            memory.set_entry_enabled(entry, False)
        return value

    monkeypatch.setattr(memory, "_entry", audited)
    assert memory.prepare(scope, task, {}, tmp_path / "out")["ticket"] is None
    assert memory.db.execute("SELECT count(*) FROM trials").fetchone()[0] == 0


def test_same_evidence_directory_cannot_be_reserved_for_two_tasks(setup, tmp_path):
    memory, _, scope, _, _ = setup
    first = make_task(tmp_path / "first")
    second = make_task(tmp_path / "second", n=32)
    out = tmp_path / "out"
    assert memory.prepare(scope, first, {}, out)["ticket"]
    with pytest.raises(ValueError, match="already reserved"):
        memory.prepare(scope, second, {}, out)
    assert memory.db.execute("SELECT count(*) FROM trials").fetchone()[0] == 1


def test_repaired_descendant_is_returned_but_not_automatically_registered(setup, tmp_path):
    memory, seed, scope, first_entry, _ = setup
    memory.set_entry_enabled(first_entry, False)
    parent = seed(SCOPED)
    memory.register(parent)
    task = make_task(tmp_path / "target", n=32)
    out = tmp_path / "out"
    offer = memory.prepare(scope, task, {}, out)
    _, final, _ = trial(task, out, source=GOOD, snapshot=offer["snapshot"])
    report = memory.observe(offer["ticket"], confirmation_run=final)
    assert report["status"] == "failed_on_target"  # Do not credit the failed parent for repair.
    assert report["formed_candidate"]
    child = lineage.audit(memory.archive, report["formed_candidate"])
    assert child["kind"] == "candidate" and child["parent"] == parent
    assert memory.db.execute("SELECT count(*) FROM entries").fetchone()[0] == 2
    (task / "candidate.py").write_text(INITIAL)
    assert memory.prepare(scope, task, {}, tmp_path / "after")["ticket"] is None


def test_negative_feedback_avoids_repeated_raw_audits(setup, tmp_path, monkeypatch):
    memory, _, scope, _, _ = setup
    task = make_task(tmp_path / "target")
    offer = memory.prepare(scope, task, {}, tmp_path / "out")
    memory.abandon(offer["ticket"], "No complete result")
    monkeypatch.setattr(
        lineage, "audit", lambda *a: pytest.fail("No repeat scan of blocked source")
    )
    assert memory.prepare(scope, task, {}, tmp_path / "after")["ticket"] is None


def test_trace_change_during_collection_cannot_relabel_feedback(setup, tmp_path, monkeypatch):
    memory, _, scope, _, _ = setup
    task = make_task(tmp_path / "target")
    out = tmp_path / "out"
    offer = memory.prepare(scope, task, {}, out)
    _, final, _ = trial(task, out, snapshot=offer["snapshot"])
    original = lineage.collect

    def mutated(*args, **kwargs):
        identity = original(*args, **kwargs)
        trace = json.loads((out / "search-trace.json").read_text())
        trace["max_actions"] = 2
        write_json(out / "search-trace.json", trace)
        return identity

    monkeypatch.setattr(lineage, "collect", mutated)
    with pytest.raises(ValueError, match="trace changed"):
        memory.observe(offer["ticket"], confirmation_run=final)
    assert memory._ticket(offer["ticket"])[0]["status"] == "pending"


@pytest.mark.parametrize("state", ["pending", "incomplete"])
def test_reserved_evidence_directory_is_rejected_even_when_no_offer(setup, tmp_path, state):
    memory, _, scope, _, _ = setup
    task = make_task(tmp_path / "target")
    out = tmp_path / "out"
    offer = memory.prepare(scope, task, {}, out)
    if state == "incomplete":
        memory.abandon(offer["ticket"], "Stopped without creating a trace")
    with pytest.raises(ValueError, match="already reserved"):
        memory.prepare(scope, task, {}, out)


@pytest.mark.parametrize("destination", ["memory", "archive", "memory_child", "archive_child"])
def test_prepare_rejects_control_state_and_evidence_directory_overlap(setup, tmp_path, destination):
    memory, _, scope, _, _ = setup
    task = make_task(tmp_path / "target")
    root = memory.root if destination.startswith("memory") else memory.archive
    out = root / "new-trace" if destination.endswith("child") else root
    with pytest.raises(ValueError, match="overlap"):
        memory.prepare(scope, task, {}, out)
    assert memory.db.execute("SELECT count(*) FROM trials").fetchone()[0] == 0


def test_changed_code_cannot_reenable_existing_memory(setup, monkeypatch):
    memory, _, _, entry, _ = setup
    memory.set_enabled(False)
    memory.set_entry_enabled(entry, False)
    monkeypatch.setattr(module, "_implementation", lambda: "0" * 64)
    with pytest.raises(ValueError, match="code changed"):
        memory.set_enabled(True)
    with pytest.raises(ValueError, match="code changed"):
        memory.set_entry_enabled(entry, True)


def test_recovery_allows_inspection_disable_and_abandon_but_no_new_work(
    setup, tmp_path, monkeypatch
):
    memory, _, scope, entry, record = setup
    task = make_task(tmp_path / "target")
    offer = memory.prepare(scope, task, {}, tmp_path / "out")
    monkeypatch.setattr(module, "_implementation", lambda: "0" * 64)
    recovery = ReuseMemory(memory.root, recovery=True)
    try:
        status = recovery.status()
        assert not status["code_compatible"]
        assert status["pending"][0]["ticket"] == offer["ticket"]
        for action in [
            lambda: recovery.set_enabled(True),
            lambda: recovery.set_entry_enabled(entry, True),
            lambda: recovery.register(record),
            lambda: recovery.prepare(scope, task, {}, tmp_path / "new"),
            lambda: recovery.observe(offer["ticket"], confirmation_run=tmp_path / "unused"),
        ]:
            with pytest.raises(ValueError, match="code changed"):
                action()
        recovery.set_enabled(False)
        recovery.set_entry_enabled(entry, False)
        recovery.abandon(offer["ticket"], "Operator closed interrupted work after upgrade")
        status = recovery.status()
        assert not status["enabled"] and status["trial_counts"] == {"incomplete": 1}
    finally:
        recovery.close()


def test_readonly_status_is_bounded_and_cannot_modify_state(setup, tmp_path):
    import sqlite3

    memory, _, scope, _, _ = setup
    for n in [16, 32]:
        memory.prepare(scope, make_task(tmp_path / f"target-{n}", n=n), {}, tmp_path / f"out-{n}")
    before = list(memory.db.iterdump())
    readonly = ReuseMemory(memory.root, recovery=True, readonly=True)
    try:
        status = readonly.status(pending_limit=1)
        assert status["pending_truncated"] and len(status["pending"]) == 1
        assert status["trial_counts"] == {"pending": 2}
        with pytest.raises(sqlite3.OperationalError, match="readonly"):
            readonly.set_enabled(False)
    finally:
        readonly.close()
    assert list(memory.db.iterdump()) == before


def test_invalid_or_consumed_ticket_writes_no_feedback_files(setup, tmp_path):
    memory, _, scope, _, _ = setup
    with pytest.raises(ValueError, match="Unknown"):
        memory.abandon("unknown", "This request must not write an orphan")
    assert not (memory.root / "feedback").exists()
    offer = memory.prepare(scope, make_task(tmp_path / "target"), {}, tmp_path / "out")
    memory.abandon(offer["ticket"], "No final")
    before = set((memory.root / "feedback").iterdir())
    with pytest.raises(ValueError, match="already consumed"):
        memory.abandon(offer["ticket"], "Another reason cannot rewrite history")
    assert set((memory.root / "feedback").iterdir()) == before


def test_memory_creation_inside_archive_is_rejected_without_writes(setup):
    memory, *_ = setup
    nested = memory.archive / "nested-memory"
    with pytest.raises(ValueError, match="overlap"):
        ReuseMemory.create(nested, memory.archive)
    assert not nested.exists()
