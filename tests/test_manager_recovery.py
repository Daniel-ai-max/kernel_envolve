"""Real CPU subprocesses exercise diagnosis, unproductive retries and budget exhaustion."""

import json
import math
import os
import sys
import time
from unittest.mock import patch

import pytest
from test_manager import launch, replace_command

from kernel_envolve.manager.process import identity
from kernel_envolve.manager.recovery import begin, implementation
from kernel_envolve.manager.views import overview

pytest_plugins = ["test_manager"]


def run_round(store):
    key = store.schedule()[0]
    process = launch(store, key)
    out, err = process.communicate(timeout=15)
    assert process.returncode in (0, 2), (out, err)
    return key


def advance_retry(store):
    task = store.task("toy")
    next_at = task["next_at"]
    assert math.isfinite(next_at) and next_at > task["created"]
    # Process teardown can outlast the 0.1 s success cooldown on a busy runner.
    # Test admission just before the recorded deadline, not at a racing wall time.
    with patch("kernel_envolve.manager.store.time.time", return_value=next_at - 0.001):
        assert not store.schedule()
    # Fast-forward only the test's admission timestamp; actual subprocesses and
    # accumulated wall/GPU usage are never mocked or reset.
    store.db.execute("UPDATE tasks SET next_at=0 WHERE id='toy'")


def recovery(store):
    return overview(store)["tasks"][0]["recovery"]


def test_three_failures_deliver_evidence_to_real_repair_round(managed):
    store, _, work = managed
    (work / "candidate.py").write_text(
        "def run(x): raise RuntimeError('missing diagnostic dependency sentinel')\n"
    )
    script = work / "agent.py"
    script.write_text("""import json,os,pathlib,subprocess,sys
prompt=sys.stdin.read()
p=pathlib.Path('attempt');n=int(p.read_text())+1 if p.exists() else 1;p.write_text(str(n))
if n<=3:
 raise SystemExit(subprocess.run([sys.executable,'-c','import candidate;candidate.run(1)']).returncode)
assert 'missing diagnostic dependency sentinel' in prompt
assert 'diagnosis_requested' in prompt and 'stderr.log' in prompt
pathlib.Path('candidate.py').write_text('def run(x): return x + 1\\n')
subprocess.run([sys.executable,'-c','import candidate;assert candidate.run(1)==2'],check=True)
pathlib.Path(os.environ['KERNEL_ENVOLVE_REPORT']).write_text(json.dumps({'outcome':'repaired implementation after inspecting the dependency failure','next_action':'verify repaired implementation through evaluator'}))
""")
    replace_command(store, [sys.executable, str(script)])
    consumed = []
    for attempt in range(3):
        key = run_round(store)
        assert store.job(key)["status"] == "failed"
        assert store.task("toy")["desired"] == "active"
        assert recovery(store)["retry_after_seconds"] == 10 * 2**attempt
        consumed.append(store.task("toy")["wall_seconds"])
        advance_retry(store)
    repaired = run_round(store)
    assert store.job(repaired)["status"] == "succeeded"
    assert recovery(store)["observable_progress"]
    assert recovery(store)["implementation_changed"]
    assert recovery(store)["state"] == "checkpoint_received"
    assert store.task("toy")["failures"] == 0
    assert consumed[0] < consumed[1] < consumed[2] < store.task("toy")["wall_seconds"]
    task = store.task("toy")
    assert recovery(store)["retry_after_seconds"] == task["spec"]["cooldown_seconds"]
    # Assert admission before the recorded deadline, independent of teardown speed.
    with patch("kernel_envolve.manager.store.time.time", return_value=task["next_at"] - 0.001):
        assert not store.schedule()


def test_empty_bridge_like_rounds_back_off_without_claiming_model_activity(managed):
    store, _, work = managed
    script = work / "bridge.py"
    script.write_text("""import json,os,pathlib
pathlib.Path(os.environ['KERNEL_ENVOLVE_REPORT']).write_text(json.dumps({'outcome':'waiting for host','next_action':'receive external agent request'}))
""")
    replace_command(store, [sys.executable, str(script)])
    for attempt in range(1, 5):
        run_round(store)
        state = recovery(store)
        assert not state["observable_progress"]
        assert state["state"] == "diagnosis_requested"
        assert "external bridge" in state["executor"]
        assert store.task("toy")["failures"] == attempt
        assert state["retry_after_seconds"] == min(300, 5 * 2**attempt)
        if attempt != 4:
            advance_retry(store)


def test_same_gpu_command_result_and_reworded_report_do_not_reset_backoff(managed):
    store, _, work = managed
    script = work / "agent.py"
    script.write_text("""import json,os,pathlib,subprocess,sys
p=pathlib.Path('attempt');n=int(p.read_text())+1 if p.exists() else 1;p.write_text(str(n))
subprocess.run([sys.executable,'-m','kernel_envolve.cli','manage','gpu-run','--',sys.executable,'-c','print("diagnostic-result")'],check=True)
pathlib.Path(os.environ['KERNEL_ENVOLVE_REPORT']).write_text(json.dumps({'outcome':'same diagnostic, new wording '+str(n),'next_action':'inspect whether a distinct hypothesis is needed'}))
""")
    replace_command(store, [sys.executable, str(script)])
    run_round(store)
    assert recovery(store)["observable_progress"]
    advance_retry(store)
    run_round(store)
    assert not recovery(store)["observable_progress"]
    assert store.task("toy")["failures"] == 1
    assert len(store.rows("SELECT id FROM jobs WHERE kind='gpu' AND status='succeeded'")) == 2


def test_real_retry_exhausts_original_wall_budget(managed):
    store, _, work = managed
    limit = 0.5
    replace_command(
        store,
        [sys.executable, "-c", "import time;time.sleep(.3);raise SystemExit(9)"],
        wall_budget_seconds=limit,
    )
    first = run_round(store)
    before = store.task("toy")["wall_seconds"]
    rounds = [first]
    # Startup and teardown can consume the entire original budget on a busy
    # runner. Retry only when the actual recorded usage still permits admission.
    if before < limit:
        assert store.task("toy")["desired"] == "active"
        advance_retry(store)
        rounds.append(run_round(store))
        assert store.task("toy")["wall_seconds"] > before
    else:
        assert store.task("toy")["wall_seconds"] == before
    for key in rounds:
        job = store.job(key)
        if job["status"] == "cancelled":
            assert job["error"] == "wall budget exhausted"
        else:
            assert job["status"] == "failed"
            assert job["exit_code"] == 9
    task = store.task("toy")
    assert task["spec"]["wall_budget_seconds"] == limit
    assert task["wall_seconds"] >= limit
    assert task["desired"] == "budget"
    assert not store.schedule()
    assert len(store.rows("SELECT id FROM jobs WHERE kind='agent'")) == len(rounds)


def test_budget_is_recognized_even_during_backoff(managed):
    store, _, _ = managed
    replace_command(store, [sys.executable, "-c", "pass"], gpu_budget_seconds=1)
    store.db.execute("UPDATE tasks SET gpu_seconds=1,next_at=?", (time.time() + 300,))
    assert not store.schedule()
    assert store.task("toy")["desired"] == "budget"


@pytest.mark.parametrize("desired", ["paused", "budget", "superseded"])
def test_recovery_never_reactivates_terminal_or_user_paused_task(managed, desired):
    store, _, _ = managed
    key = store.schedule()[0]
    store.claim(key, os.getpid(), identity(os.getpid()))
    begin(store, store.task("toy"), store.job(key))
    store.db.execute("UPDATE tasks SET desired=?", (desired,))
    store.finish(key, "interrupted", "supervisor gone")
    assert store.task("toy")["desired"] == desired
    assert not store.schedule()


def test_notes_or_native_reference_changes_are_not_implementation_progress(managed):
    store, _, work = managed
    before = implementation(store.task("toy"))
    (work / "notes.md").write_text("new wording")
    assert implementation(store.task("toy")) == before
    spec = store.task("toy")["spec"] | {"kind": "native"}
    (work / "native.json").write_text(
        json.dumps(
            {
                "sources": {"candidate": {"core": str(work)}},
                "implementation_paths": {"core": ["candidate.py"]},
            }
        )
    )
    task = {"spec": spec}
    before = implementation(task)
    (work / "reference.py").write_text("untrusted mutation")
    assert implementation(task) == before


def test_exact_legacy_block_recovers_once_without_refunding_budget(managed):
    store, _, _ = managed
    from kernel_envolve.manager.recovery import LEGACY_BLOCK_REASON

    store.db.execute(
        "UPDATE tasks SET desired='blocked',reason=?,failures=3,wall_seconds=17,gpu_seconds=9",
        (LEGACY_BLOCK_REASON,),
    )
    assert (
        not store.schedule()
    )  # Migration schedules a diagnostic backoff, not an immediate replay.
    task = store.task("toy")
    assert task["desired"] == "active" and task["failures"] == 3
    assert task["wall_seconds"] == 17 and task["gpu_seconds"] == 9
    assert recovery(store)["state"] == "diagnosis_requested"
    next_at = task["next_at"]
    assert next_at > time.time()
    assert not store.schedule() and store.task("toy")["next_at"] == next_at
    advance_retry(store)
    assert len(store.schedule()) == 1
    assert len(store.rows("SELECT seq FROM events WHERE kind='legacy_recovery_requested'")) == 1


@pytest.mark.parametrize(
    "state,reason,superseded",
    [
        ("blocked", "external unavailable; human decision", False),
        ("paused", "Three consecutive failed rounds; inspect and resume", False),
        ("superseded", "Three consecutive failed rounds; inspect and resume", False),
        ("blocked", "Three consecutive failed rounds; inspect and resume", True),
    ],
)
def test_legacy_migration_never_revives_other_blocks_or_superseded_metadata(
    managed, state, reason, superseded
):
    store, _, _ = managed
    store.db.execute("UPDATE tasks SET desired=?,reason=?,failures=3", (state, reason))
    if superseded:
        store.db.execute(
            "INSERT INTO metadata VALUES('superseded:toy',?)",
            (json.dumps({"superseded_by": "toy_v2"}),),
        )
    assert not store.schedule()
    assert store.task("toy")["desired"] == state
    assert not store.rows("SELECT seq FROM events WHERE kind='legacy_recovery_requested'")


def test_legacy_block_with_spent_budget_stays_stopped(managed):
    store, _, _ = managed
    from kernel_envolve.manager.recovery import LEGACY_BLOCK_REASON

    replace_command(store, [sys.executable, "-c", "pass"], wall_budget_seconds=1)
    store.db.execute(
        "UPDATE tasks SET desired='blocked',reason=?,failures=3,wall_seconds=2",
        (LEGACY_BLOCK_REASON,),
    )
    assert not store.schedule()
    assert store.task("toy")["desired"] == "budget"
    assert store.task("toy")["wall_seconds"] == 2


def test_native_relative_implementation_paths_resolve_from_task_dir(managed, monkeypatch):
    store, _, work = managed
    directory = work / "task"
    directory.mkdir()
    (directory / "native.json").write_text(
        json.dumps(
            {
                "sources": {"candidate": {"core": ".."}},
                "implementation_paths": {"core": ["candidate.py"]},
            }
        )
    )
    task = {"spec": store.task("toy")["spec"] | {"kind": "native", "task_dir": str(directory)}}
    monkeypatch.chdir(directory.parent.parent)
    before = implementation(task)
    assert before is not None
    (work / "candidate.py").write_text("def run(x): return x + 2\n")
    assert implementation(task) != before


def test_non_source_files_count_towards_fingerprint_scan_bound(managed):
    store, _, work = managed
    folder = work / "implementation"
    folder.mkdir()
    for i in range(260):
        (folder / (str(i) + ".bin")).touch()
    (folder / "kernel.py").write_text("pass")
    (work / "native.json").write_text(
        json.dumps(
            {
                "sources": {"candidate": {"core": "."}},
                "implementation_paths": {"core": ["implementation"]},
            }
        )
    )
    assert implementation({"spec": store.task("toy")["spec"] | {"kind": "native"}}) is None


def test_finish_never_hashes_source_under_sqlite_write_transaction(managed, monkeypatch):
    store, _, _ = managed
    key = store.schedule()[0]
    from kernel_envolve.manager import recovery as module

    original = module.implementation

    def guarded(task):
        assert not store.db.in_transaction
        return original(task)

    monkeypatch.setattr(module, "implementation", guarded)
    store.finish(key, "failed", "missing provider")
    assert recovery(store)["state"] == "diagnosis_requested"


def test_real_gpu_budget_is_not_refunded_by_recovery(managed):
    store, _, work = managed
    script = work / "agent.py"
    script.write_text("""import subprocess,sys
subprocess.run([sys.executable,'-m','kernel_envolve.cli','manage','gpu-run','--',sys.executable,'-c','import time;time.sleep(1)'])
""")
    replace_command(store, [sys.executable, str(script)], gpu_budget_seconds=0.2)
    run_round(store)
    used = store.task("toy")["gpu_seconds"]
    assert used >= 0.2 and store.task("toy")["desired"] == "budget"
    assert not store.schedule()
    assert store.task("toy")["gpu_seconds"] == used


def test_repeated_failure_backoff_is_capped_but_never_removed(managed):
    store, _, _ = managed
    delays = []
    for _ in range(9):
        key = store.schedule()[0]
        store.finish(key, "failed", "provider remains unavailable")
        delays.append(recovery(store)["retry_after_seconds"])
        assert store.task("toy")["desired"] == "active"
        assert not store.schedule()
        store.db.execute("UPDATE tasks SET next_at=0 WHERE id='toy'")
    assert delays == [10, 20, 40, 80, 160, 300, 300, 300, 300]
