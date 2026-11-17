"""The overview separates research notes, execution facts and contract-specific best."""

import json

from test_manager_supersede import change_spec

from kernel_envolve.manager.views import detail, overview, snapshot

pytest_plugins = ["test_manager_supersede"]


def meta(store, key, value):
    store.db.execute("INSERT OR REPLACE INTO metadata VALUES(?,?)", (key, json.dumps(value)))


def record_run(store, task, path, at):
    data = {
        "run": path,
        "recorded_at": at,
        "mode": "quick",
        "correctness": "pass",
        "execution": "completed",
    }
    store.db.execute("INSERT INTO runs VALUES(?,?,?,?,NULL)", (path, task, at, json.dumps(data)))
    meta(store, "latest:" + task, data)
    return data


def gpu(store, task, started, ended, status="succeeded"):
    job = store.enqueue(store.task(task), "gpu", ["python", "diagnostic.py"])
    store.db.execute(
        "UPDATE jobs SET status=?,started=?,ended=? WHERE id=?", (status, started, ended, job)
    )
    return job


def test_history_is_traceable_without_transferring_best(replaced):
    store = replaced
    old_run = record_run(store, "old", "old-run", 10)
    old_run["baseline_speedup"] = 9
    store.db.execute("UPDATE runs SET data=? WHERE path='old-run'", (json.dumps(old_run),))
    meta(store, "best:old", {"run": "old-run"})
    gpu(store, "old", 11, 12)
    store.supersede("old", "new")
    change_spec(store, "third", wall_budget_seconds=60, gpu_budget_seconds=50)
    store.supersede("new", "third")
    task = detail(store, "third", now=20)
    assert task["best"] is None and task["latest_evaluation"] is None
    assert [row["task"] for row in task["lineage"]] == ["third", "new", "old"]
    assert task["history_counts"] == {"current_runs": 0, "previous_runs": 1, "gpu_jobs": 1}
    assert task["recent_evaluation"]["task"] == "old"
    assert task["recent_evaluation"]["inherited"]
    assert task["recent_execution"]["recorded_at"] == 12
    assert task["gpu_history"][0]["task"] == "old"
    assert detail(store, "old")["runs"][0]["path"] == "old-run"


def test_diagnostic_only_report_is_visible_but_not_a_validated_speedup(replaced):
    store = replaced
    report = {
        "phase": "维护服务",
        "hypothesis": "诊断调度",
        "outcome": "局部测试快 20%，未完整验收",
        "next_action": "检查公共 VJP",
    }
    store.db.execute("UPDATE tasks SET report=? WHERE id='new'", (json.dumps(report),))
    store.event("new", "report", report)
    job = gpu(store, "new", 10, 20)
    tasks = {task["id"]: task for task in overview(store, compact=True, now=30)["tasks"]}
    task = tasks["new"]
    assert task["report"]["outcome"] == report["outcome"]
    assert task["report"]["next_action"] == report["next_action"]
    assert task["research_record"]["outcome"] == report["outcome"]
    assert task["recent_execution"]["id"] == job
    assert task["recent_execution"]["recorded_at"] == 20
    assert task["report_as_of"] > 20  # Maintenance report is newer, but is not an experiment.
    assert task["best"] is None and task["history_counts"]["current_runs"] == 0


def test_idle_session_does_not_become_gpu_work_and_unstarted_jobs_are_not_counted(replaced):
    store = replaced
    store.db.execute("UPDATE tasks SET desired='active' WHERE id='new'")
    session = store.enqueue(store.task("new"), "agent", ["python"])
    store.db.execute(
        "UPDATE jobs SET status='running',started=10,heartbeat=29 WHERE id=?", (session,)
    )
    gpu(store, "new", None, 15, status="failed")
    task = detail(store, "new", now=30)
    assert task["phase"] == "agent_running"
    assert task["execution_state"] == "idle"
    assert task["history_counts"]["gpu_jobs"] == 0
    assert task["recent_execution"] is None
    assert task["gpu_history"] == []


def test_only_explicit_migration_links_history_and_report_fallback(replaced):
    store = replaced
    record_run(store, "old", "old-run", 10)
    store.db.execute(
        "UPDATE tasks SET report=? WHERE id='old'", (json.dumps({"outcome": "旧契约的结论"}),)
    )
    assert detail(store, "new")["history_counts"]["previous_runs"] == 0
    store.supersede("old", "new")
    task = detail(store, "new")
    assert task["research_record"]["task"] == "old"
    assert task["research_record"]["inherited"]
    # A damaged cycle cannot loop or count an identity twice.
    meta(store, "superseded:new", {"task": "new", "superseded_by": "old"})
    assert len(detail(store, "new")["lineage"]) == 2


def test_snapshot_uses_ledger_without_reading_evidence_or_source_files(replaced, monkeypatch):
    store = replaced
    record_run(store, "old", "old-run", 10)
    store.supersede("old", "new")

    def no_io(*args, **kwargs):
        raise AssertionError("Overview must not read receipts or source files")

    monkeypatch.setattr("pathlib.Path.read_bytes", no_io)
    monkeypatch.setattr("pathlib.Path.read_text", no_io)
    data = snapshot(store)
    task = next(t for t in data["tasks"] if t["id"] == "new")
    assert task["history_counts"]["previous_runs"] == 1


def test_empty_evidence_projection_does_not_break_overview(replaced):
    store = replaced
    meta(store, "latest:new", None)
    task = next(t for t in snapshot(store)["tasks"] if t["id"] == "new")
    assert task["latest_evaluation"] is None
    assert task["recent_evaluation"] is None
    assert task["recent_execution"] is None


def test_ambiguous_predecessor_fails_closed(replaced):
    store = replaced
    record_run(store, "old", "old-run", 10)
    store.supersede("old", "new")
    meta(store, "superseded:third", {"task": "third", "superseded_by": "new"})
    task = detail(store, "new")
    assert not task["history_complete"]
    assert [entry["task"] for entry in task["lineage"]] == ["new"]
    assert task["history_counts"]["previous_runs"] == 0
