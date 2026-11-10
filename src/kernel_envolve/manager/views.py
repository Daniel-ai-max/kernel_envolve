"""UI projections use evaluator facts, supervisor state and separately labeled agent notes."""

import json
import time
from pathlib import Path

from ..storage import digest, read_json
from .evidence import state_path


def best(store, task):
    state = state_path(task["spec"])
    try:
        pointer = read_json(state / "best.json")
        projection = store.one("SELECT data FROM metadata WHERE key=?", (f"best:{task['id']}",))
        projection = json.loads(projection["data"]) if projection else None
        if not projection or projection["pointer_hash"] != digest(pointer):
            return None
        path = (state / pointer["run"]).resolve()
        row = store.one("SELECT * FROM runs WHERE path=? AND task=?", (str(path), task["id"]))
        if row and not row["error"]:
            data = json.loads(row["data"])
            if (
                data["result_hash"] == pointer["result_hash"]
                and data["decision"]["eligible_for_best"]
            ):
                return data
    except (OSError, ValueError, KeyError, TypeError):
        pass
    return None


def task_view(store, key):
    task = store.task(key)
    jobs = store.rows(
        "SELECT * FROM jobs WHERE task=? AND status IN ('queued','running') ORDER BY created",
        (key,),
    )
    task["jobs"] = jobs
    task["best"] = best(store, task)
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
            elif task["next_at"] > time.time():
                task["phase"] = "backoff" if task["failures"] else "cooldown"
            else:
                task["phase"] = "waiting_session"
    task["heartbeat"] = max((j["heartbeat"] or j["created"] for j in jobs), default=None)
    task["stale"] = bool(jobs and time.time() - task["heartbeat"] > 10)
    task["last_promotion"] = store.one(
        "SELECT at,data FROM promotions WHERE task=? ORDER BY at DESC LIMIT 1", (key,)
    )
    if task["last_promotion"]:
        task["last_promotion"] = json.loads(task["last_promotion"]["data"]).get("promoted_at")
    task["run_count"] = store.db.execute(
        "SELECT COUNT(*) FROM runs WHERE task=?", (key,)
    ).fetchone()[0]
    return task


def overview(store):
    coordinator = store.one("SELECT data FROM metadata WHERE key='coordinator'")
    coordinator = json.loads(coordinator["data"]) if coordinator else None
    tasks = [
        task_view(store, r["id"]) for r in store.rows("SELECT id FROM tasks ORDER BY created,id")
    ]
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
        "as_of": time.time(),
        "tasks": tasks,
        "resources": resources,
        "measurement": "Speedup = 1 / (1 − median full-batch fixed-baseline latency reduction). No framework-wide speedup is inferred.",
    }


def detail(store, key):
    task = task_view(store, key)
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
    task["events"] = [
        {**r, "data": json.loads(r["data"])}
        for r in store.rows("SELECT * FROM events WHERE task=? ORDER BY seq DESC LIMIT 100", (key,))
    ]
    task["rounds"] = store.rows(
        "SELECT * FROM jobs WHERE task=? ORDER BY created DESC LIMIT 100", (key,)
    )
    notes = Path(task["spec"]["task_dir"]) / "notes.md"
    task["notes"] = notes.read_text(errors="replace")[:32000] if notes.is_file() else ""
    return task
