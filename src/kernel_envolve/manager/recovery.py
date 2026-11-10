"""Observable recovery attempts; a fresh process or rewritten report is not progress."""

import hashlib
import json
import os
from pathlib import Path

from ..storage import file_hash

LEGACY_BLOCK_REASON = "Three consecutive failed rounds; inspect and resume"


def migrate_legacy(store, task):
    """Resume only the old automatic three-failure block, with original accounting."""
    if (
        task["desired"] != "blocked"
        or task["reason"] != LEGACY_BLOCK_REASON
        or store.one("SELECT key FROM metadata WHERE key=?", ("superseded:" + task["id"],))
    ):
        return task
    import time

    budget = store.budget_reason(task)
    delay = min(300, 5 * 2 ** min(max(task["failures"], 1), 6))
    next_at = max(task["next_at"], time.time() + delay)
    desired = "budget" if budget else "active"
    reason = budget or "Legacy failure block migrated; agent diagnosis requested"
    value = {
        "state": "budget_exhausted" if budget else "diagnosis_requested",
        "reason": reason,
        "attempts": task["failures"],
        "observable_progress": False,
        "retry_after_seconds": delay,
        "executor": "Configured command; an external bridge still needs its agent host.",
    }
    store.db.execute(
        "UPDATE tasks SET desired=?,reason=?,next_at=? WHERE id=?",
        (desired, reason, next_at, task["id"]),
    )
    store.db.execute(
        "INSERT OR REPLACE INTO metadata VALUES(?,?)",
        ("recovery:" + task["id"], json.dumps(value)),
    )
    store.event(task["id"], "legacy_recovery_requested", value)
    return store.task(task["id"])


def implementation(task):
    """Fingerprint only declared implementation, never notes or report timestamps."""
    directory = Path(task["spec"]["task_dir"])
    if task["spec"]["kind"] == "generic":
        selected = [directory / "candidate.py"]
    else:
        try:
            cfg = json.loads((directory / "native.json").read_text())
            selected = [
                directory / os.path.expandvars(cfg["sources"]["candidate"][root]) / name
                for root, names in cfg.get("implementation_paths", {}).items()
                for name in names
            ]
        except (OSError, ValueError, KeyError, TypeError, AttributeError):
            return None
    files = {}
    total = visited = 0
    pending = list(selected)
    try:
        while pending:
            source = pending.pop()
            visited += 1
            if visited + len(pending) > 256:
                return None  # Count all entries, including non-source and cache files.
            if source.is_symlink():
                return None
            if not source.resolve().is_relative_to(task["spec"]["workspace"]):
                return None
            if source.is_dir():
                with os.scandir(source) as entries:
                    for entry in entries:
                        pending.append(Path(entry.path))
                        if visited + len(pending) > 256:
                            return None
                continue
            if not source.is_file() or source.suffix not in {
                ".py",
                ".cu",
                ".cuh",
                ".cpp",
                ".c",
                ".h",
                ".hpp",
                ".ptx",
            }:
                continue
            total += source.stat().st_size
            if total > 16 * 1024 * 1024:
                return None
            files[str(source)] = file_hash(source)
    except OSError:
        return None
    return files or None


def begin(store, task, job):
    value = {"implementation": implementation(task), "outcome": task["report"].get("outcome")}
    store.db.execute(
        "INSERT OR REPLACE INTO metadata VALUES(?,?)",
        ("round_input:" + job["id"], json.dumps(value)),
    )


def assess(store, task, job, status, current):
    previous = store.one("SELECT data FROM metadata WHERE key=?", ("round_input:" + job["id"],))
    before = json.loads(previous["data"]) if previous else {}
    old = store.one("SELECT data FROM metadata WHERE key=?", ("recovery:" + task["id"],))
    old = json.loads(old["data"]) if old else {}
    seen = set(old.get("execution_signatures", []))
    novel = []
    for child in store.rows(
        "SELECT * FROM jobs WHERE parent=? AND kind='gpu' AND child_pid IS NOT NULL "
        "AND status IN ('succeeded','failed')",
        (job["id"],),
    ):
        signature = hashlib.sha256(
            json.dumps(
                [child["command"], child["status"], child["exit_code"]],
                sort_keys=True,
            ).encode()
        ).hexdigest()
        if signature not in seen:
            novel.append(child["id"])
        seen.add(signature)
    changed = (
        before.get("implementation") is not None
        and current is not None
        and current != before["implementation"]
    )
    report = task["report"]
    outcome = report.get("outcome", "").strip()
    checkpoint = bool(
        outcome and outcome != before.get("outcome") and report.get("next_action", "").strip()
    )
    progress = status == "succeeded" and checkpoint and bool(novel or changed)
    return {
        "state": "checkpoint_received" if progress else "diagnosis_requested",
        "round": job["id"],
        "observable_progress": progress,
        "implementation_changed": changed,
        "new_execution_jobs": novel,
        "execution_signatures": sorted(seen)[-256:],
        "checkpoint_changed": checkpoint,
        "verification": "Agent checkpoint and execution activity; evaluator acceptance is separate.",
        "executor": "Configured command; it may be an external bridge, not a running model.",
    }


def context(store, task):
    row = store.one("SELECT data FROM metadata WHERE key=?", ("recovery:" + task["id"],))
    if not row:
        return {}
    data = json.loads(row["data"])
    data.pop("execution_signatures", None)
    jobs = store.rows(
        "SELECT id,kind,status,error,exit_code FROM jobs WHERE task=? "
        "AND status IN ('failed','interrupted') ORDER BY ended DESC LIMIT 3",
        (task["id"],),
    )
    for job in jobs:
        directory = store.root / "jobs" / job["id"]
        job["logs"] = str(directory)
        for name in ("stderr.log", "supervisor.log"):
            path = directory / name
            try:
                with path.open("rb") as stream:
                    stream.seek(max(0, path.stat().st_size - 3000))
                    job[name + "_tail"] = stream.read().decode(errors="replace")
            except OSError:
                pass
    return data | {"failures": task["failures"], "recent_failures": jobs}
