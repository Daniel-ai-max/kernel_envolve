"""UI projections use evaluator facts, supervisor state and separately labeled agent notes."""

import json
import time
from pathlib import Path

PROGRESS_NOTICE_SECONDS = 300


def progress_time(store, task):
    """Activity is a changed research report or a GPU job, never a process heartbeat."""
    report = store.one("SELECT data FROM metadata WHERE key=?", (f"report_progress:{task['id']}",))
    if report:
        reported_at = json.loads(report["data"])
    else:
        legacy = store.one(
            "SELECT MAX(at) AS at FROM events WHERE task=? AND kind='report'", (task["id"],)
        )
        reported_at = legacy["at"] if legacy else None
    gpu = store.one(
        "SELECT MAX(COALESCE(ended,started,created)) AS at FROM jobs WHERE task=? AND kind='gpu'",
        (task["id"],),
    )
    first_session = store.one(
        "SELECT MIN(started) AS at FROM jobs WHERE task=? AND kind='agent'", (task["id"],)
    )
    # Waiting for the first session slot must not consume its startup grace period.
    return max(task["created"], first_session["at"] or 0, reported_at or 0, gpu["at"] or 0)


def best(store, task):
    projection = store.one("SELECT data FROM metadata WHERE key=?", (f"best:{task['id']}",))
    projection = json.loads(projection["data"]) if projection else None
    if not projection:
        return None
    row = store.one("SELECT * FROM runs WHERE path=? AND task=?", (projection["run"], task["id"]))
    return json.loads(row["data"]) if row and not row["error"] else None


def lineage_index(store):
    """Only explicit replacement receipts link contracts; names never imply history."""
    predecessors = {}
    for row in store.rows("SELECT key,data FROM metadata WHERE key LIKE 'superseded:%'"):
        receipt = json.loads(row["data"])
        old = row["key"][len("superseded:") :]
        new = receipt.get("superseded_by")
        if new and receipt.get("task", old) == old:
            predecessors[new] = old if new not in predecessors else None
    return predecessors


def research_summary(store, task, predecessors):
    """Read the indexed ledger only. Never scan receipts, source trees or report numbers."""
    lineage, seen, key = [], set(), task["id"]
    history_complete = True
    while key and key not in seen:
        seen.add(key)
        if not store.one("SELECT id FROM tasks WHERE id=?", (key,)):
            history_complete = False
            break
        counts = store.one("SELECT COUNT(*) AS count FROM runs WHERE task=?", (key,))
        gpu_count = store.one(
            "SELECT COUNT(*) AS count FROM jobs WHERE task=? AND kind='gpu' AND started IS NOT NULL",
            (key,),
        )
        lineage.append(
            {
                "task": key,
                "inherited": key != task["id"],
                "run_count": counts["count"],
                "gpu_job_count": gpu_count["count"],
            }
        )
        if key in predecessors and predecessors[key] is None:
            history_complete = False
        key = predecessors.get(key)
    if key in seen:
        history_complete = False
    keys = [row["task"] for row in lineage]
    placeholders = ",".join("?" for _ in keys)
    recent_gpu = store.one(
        f"SELECT id,task,status,started,ended,seconds,error FROM jobs WHERE task IN ({placeholders}) "
        "AND kind='gpu' AND started IS NOT NULL ORDER BY COALESCE(ended,started) DESC,id DESC LIMIT 1",
        keys,
    )
    if recent_gpu:
        recent_gpu["recorded_at"] = recent_gpu["ended"] or recent_gpu["started"]
        recent_gpu["inherited"] = recent_gpu["task"] != task["id"]
    evaluations, record = [], None
    for entry in lineage:
        key = entry["task"]
        latest = store.one("SELECT data FROM metadata WHERE key=?", (f"latest:{key}",))
        if latest:
            data = json.loads(latest["data"])
            if isinstance(data, dict) and data:
                evaluations.append({**data, "task": key, "inherited": entry["inherited"]})
        report = (
            task["report"]
            if key == task["id"]
            else json.loads(store.one("SELECT report FROM tasks WHERE id=?", (key,))["report"])
        )
        if record is None and any(report.get(k) for k in ("outcome", "next_action", "hypothesis")):
            at = store.one(
                "SELECT MAX(at) AS at FROM events WHERE task=? AND kind='report'", (key,)
            )
            record = {
                "task": key,
                "inherited": entry["inherited"],
                "recorded_at": at["at"],
                **{k: report.get(k, "") for k in ("phase", "hypothesis", "outcome", "next_action")},
            }
    recent_evaluation = max(
        evaluations, key=lambda value: value.get("recorded_at") or 0, default=None
    )
    # A report can change during maintenance. Its timestamp is never an experiment timestamp.
    executions = []
    if recent_gpu:
        executions.append({**recent_gpu, "kind": "gpu_job"})
    if recent_evaluation:
        executions.append({**recent_evaluation, "kind": "evaluation"})
    return {
        "lineage": lineage,
        "history_complete": history_complete,
        "research_record": record,
        "recent_evaluation": recent_evaluation,
        "recent_gpu_job": recent_gpu,
        "recent_execution": max(
            executions, key=lambda value: value.get("recorded_at") or 0, default=None
        ),
        "history_counts": {
            "current_runs": lineage[0]["run_count"],
            "previous_runs": sum(row["run_count"] for row in lineage[1:]),
            "gpu_jobs": sum(row["gpu_job_count"] for row in lineage),
        },
    }


def task_view(store, key, now=None, predecessors=None):
    now = time.time() if now is None else now
    task = store.task(key)
    recovery = store.one("SELECT data FROM metadata WHERE key=?", ("recovery:" + key,))
    task["recovery"] = json.loads(recovery["data"]) if recovery else None
    if task["recovery"]:
        task["recovery"].pop("execution_signatures", None)
    jobs = store.rows(
        "SELECT * FROM jobs WHERE task=? AND status IN ('queued','running') ORDER BY created",
        (key,),
    )
    task["jobs"] = jobs
    task.update(
        research_summary(
            store, task, lineage_index(store) if predecessors is None else predecessors
        )
    )
    task["best"] = best(store, task)
    evidence = store.one("SELECT data FROM metadata WHERE key=?", (f"evidence:{key}",))
    task["evidence_as_of"] = json.loads(evidence["data"])["observed_at"] if evidence else None
    task["evidence_stale"] = task["evidence_as_of"] is None or now - task["evidence_as_of"] > 15
    scan = store.one("SELECT data FROM metadata WHERE key='evidence_scan'")
    scan = json.loads(scan["data"]) if scan else {}
    task["evidence_scan_error"] = scan.get("error")
    task["evidence_scan_slow"] = bool(
        scan.get("running") and now - scan.get("updated_at", now) > 300
    )
    task["evidence_status"] = (
        "error"
        if task["evidence_error"] or scan.get("error")
        else "slow"
        if task["evidence_scan_slow"]
        else "verifying"
        if scan.get("running") and scan.get("task") == key
        else "queued"
        if scan.get("running")
        else "pending"
        if task["evidence_stale"]
        else "current"
    )
    latest = store.one("SELECT data FROM metadata WHERE key=?", (f"latest:{key}",))
    task["latest_evaluation"] = json.loads(latest["data"]) if latest else None
    report = store.one("SELECT MAX(at) AS at FROM events WHERE task=? AND kind='report'", (key,))
    task["report_as_of"] = report["at"]
    task["phase"] = task["desired"]
    if task["desired"] == "active":
        gpu = next((j for j in jobs if j["kind"] == "gpu" and j["status"] == "running"), None)
        waiting = next((j for j in jobs if j["kind"] == "gpu" and j["status"] == "queued"), None)
        if gpu:
            task["phase"] = "gpu_running"
        elif waiting:
            task["phase"] = "waiting_gpu"
        elif jobs:
            task["phase"] = "agent_running" if jobs[0]["status"] == "running" else "queued"
        else:
            if store.resource(task["spec"]["resource"])["problem"]:
                task["phase"] = "resource_blocked"
            elif task["next_at"] > now:
                task["phase"] = "backoff" if task["failures"] else "cooldown"
            else:
                task["phase"] = "waiting_session"
    task["execution_state"] = (
        "running"
        if any(j["kind"] == "gpu" and j["status"] == "running" for j in jobs)
        else "queued"
        if any(j["kind"] == "gpu" and j["status"] == "queued" for j in jobs)
        else "idle"
    )
    task["heartbeat"] = max((j["heartbeat"] or j["created"] for j in jobs), default=None)
    task["stale"] = bool(jobs and now - task["heartbeat"] > 10)
    task["progress_as_of"] = progress_time(store, task)
    task["progress_age_seconds"] = max(0, now - task["progress_as_of"])
    task["progress_stale"] = (
        task["phase"] == "agent_running" and task["progress_age_seconds"] >= PROGRESS_NOTICE_SECONDS
    )
    if task["progress_stale"]:
        task["phase"] = "awaiting_progress"
    task["last_promotion"] = store.one(
        "SELECT at,data FROM promotions WHERE task=? ORDER BY at DESC LIMIT 1", (key,)
    )
    if task["last_promotion"]:
        task["last_promotion"] = json.loads(task["last_promotion"]["data"]).get("promoted_at")
    task["run_count"] = store.db.execute(
        "SELECT COUNT(*) FROM runs WHERE task=?", (key,)
    ).fetchone()[0]
    return task


def overview(store, *, compact=False, now=None):
    now = time.time() if now is None else now
    coordinator = store.one("SELECT data FROM metadata WHERE key='coordinator'")
    coordinator = json.loads(coordinator["data"]) if coordinator else None
    predecessors = lineage_index(store)
    tasks = [
        task_view(store, r["id"], now, predecessors)
        for r in store.rows("SELECT id FROM tasks ORDER BY created,id")
    ]
    if compact:
        for task in tasks:
            task["report"] = {
                k: task["report"].get(k, "")
                for k in ("phase", "hypothesis", "outcome", "next_action")
            }
            if task["best"]:
                data = task["best"]
                task["best"] = {
                    k: data.get(k)
                    for k in ("run", "baseline_speedup", "incremental_speedup", "native_checks")
                }
                task["best"]["decision"] = {"coverage": data["decision"].get("coverage")}
    resources = []
    for row in store.rows("SELECT id FROM resources ORDER BY id"):
        resource = store.resource(row["id"])
        resource["jobs"] = store.rows(
            "SELECT * FROM jobs WHERE resource=? AND status IN ('queued','running') ORDER BY created,id",
            (row["id"],),
        )
        resource["tasks"] = [t["id"] for t in tasks if t["spec"]["resource"] == row["id"]]
        resource["gpu_seconds"] = store.db.execute(
            "SELECT COALESCE(SUM(seconds),0) FROM jobs WHERE resource=? AND kind='gpu'",
            (row["id"],),
        ).fetchone()[0]
        resources.append(resource)
    return {
        "coordinator": coordinator,
        "as_of": now,
        "tasks": tasks,
        "resources": resources,
        "measurement": "Speedup = 1 / (1 − median full-batch fixed-baseline latency reduction). No framework-wide speedup is inferred.",
    }


def detail(store, key, now=None):
    task = task_view(store, key, now)
    task["history"] = []
    for row in store.rows("SELECT * FROM promotions WHERE task=? ORDER BY at", (key,)):
        data = json.loads(row["data"])
        current = store.one("SELECT error FROM runs WHERE path=?", (row["run"],))
        data["evidence_error"] = current["error"] if current else "Evidence missing"
        if task["evidence_error"] and not task["best"]:
            data["evidence_error"] = data["evidence_error"] or "Promotion chain needs verification"
        data["gpu_seconds_at_observation"] = row["gpu_seconds"]
        task["history"].append(data)
    task["runs"] = [
        {**r, "data": json.loads(r["data"])}
        for r in store.rows(
            "SELECT * FROM runs WHERE task=? ORDER BY imported DESC LIMIT 100", (key,)
        )
    ]
    keys = [row["task"] for row in task["lineage"]]
    placeholders = ",".join("?" for _ in keys)
    task["gpu_history"] = store.rows(
        f"SELECT id,task,status,started,ended,seconds,error FROM jobs WHERE task IN ({placeholders}) "
        "AND kind='gpu' AND started IS NOT NULL ORDER BY COALESCE(ended,started) DESC LIMIT 50",
        keys,
    )
    task["events"] = [
        {**r, "data": json.loads(r["data"])}
        for r in store.rows("SELECT * FROM events WHERE task=? ORDER BY seq DESC LIMIT 100", (key,))
    ]
    task["rounds"] = store.rows(
        "SELECT * FROM jobs WHERE task=? ORDER BY created DESC LIMIT 100", (key,)
    )
    task["notes"] = ""
    for directory in ("task_dir", "workspace"):
        notes = Path(task["spec"][directory]) / "notes.md"
        if notes.is_file():
            with notes.open(errors="replace") as stream:
                task["notes"] = stream.read(32000)
            break
    return task


def snapshot(store, selected=None):
    """One WAL read snapshot, shared by the summary and the visible task detail."""
    with store.transaction(readonly=True):
        now = time.time()
        data = overview(store, compact=True, now=now)
        data["detail"] = detail(store, selected, now) if selected else None
        if data["detail"]:
            task = data["detail"]
            task.pop("rounds")
            task["history"] = [
                {k: h.get(k) for k in ("run", "promoted_at", "baseline_speedup", "evidence_error")}
                for h in task["history"]
            ]
            for index, run in enumerate(task["runs"]):
                record = run["data"]
                run["data"] = {
                    "mode": record.get("mode"),
                    "correctness": record.get("correctness"),
                    "decision": {
                        k: record.get("decision", {}).get(k) for k in ("performance", "reasons")
                    },
                    "source": record.get("source") if index == 0 else None,
                }
    return data
