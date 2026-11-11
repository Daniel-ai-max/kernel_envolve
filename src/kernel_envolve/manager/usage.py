"""Allocation receipts include idle time; command timers cannot measure rented GPU time."""

import json
import math
import time

from .store import encode


def allocations(store):
    return [
        json.loads(row["data"])
        for row in store.rows("SELECT data FROM metadata WHERE key LIKE 'allocation:%'")
    ]


def import_usage(store, document):
    """Idempotently import scheduler observations, with immutable allocation attribution."""
    if set(document) != {"version", "allocations"} or document["version"] != 1:
        raise ValueError("Expected version 1 allocation receipts")
    if not isinstance(document["allocations"], list) or not document["allocations"]:
        raise ValueError("allocations must be a nonempty list")
    seen = set()
    with store.transaction():
        for entry in document["allocations"]:
            if set(entry) != {
                "id",
                "started_at",
                "observed_at",
                "ended",
                "gpus",
                "tasks",
                "source",
            }:
                raise ValueError("Invalid allocation receipt fields")
            for field in ("id", "source"):
                if not isinstance(entry[field], str) or not entry[field].strip():
                    raise ValueError(f"{field} must be nonempty text")
            if entry["id"] in seen:
                raise ValueError("Duplicate allocation ID")
            seen.add(entry["id"])
            for field in ("started_at", "observed_at", "gpus"):
                value = entry[field]
                if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
                    raise ValueError(f"{field} must be finite and positive")
            if entry["observed_at"] < entry["started_at"] or entry["observed_at"] > time.time():
                raise ValueError("Invalid allocation observation interval")
            if type(entry["ended"]) is not bool or type(entry["gpus"]) is not int:
                raise ValueError("ended must be boolean and gpus must be an integer")
            if not isinstance(entry["tasks"], dict) or not entry["tasks"]:
                raise ValueError("Explicit task attribution is required")
            for task, cards in entry["tasks"].items():
                if not store.one("SELECT id FROM tasks WHERE id=?", (task,)):
                    raise ValueError(f"Unknown task: {task}")
                if type(cards) not in (int, float) or not math.isfinite(cards) or cards <= 0:
                    raise ValueError("Task GPU shares must be finite and positive")
            if not math.isclose(sum(entry["tasks"].values()), entry["gpus"], abs_tol=1e-9):
                raise ValueError("Task GPU shares must sum to allocated GPUs")
            key = "allocation:" + entry["id"]
            previous = store.one("SELECT data FROM metadata WHERE key=?", (key,))
            if previous:
                old = json.loads(previous["data"])
                if old == entry:
                    continue
                if (
                    old["ended"]
                    or entry["observed_at"] < old["observed_at"]
                    or any(old[field] != entry[field] for field in ("started_at", "gpus", "tasks"))
                ):
                    raise ValueError("Cannot rewrite allocation attribution or decrease usage")
            store.db.execute("INSERT OR REPLACE INTO metadata VALUES(?,?)", (key, encode(entry)))
            store.event(None, "allocation_usage", entry)


def usage_view(entries, tasks=None):
    selected = [e for e in entries if tasks is None or tasks.intersection(e["tasks"])]
    if not selected:
        return None
    seconds = sum(
        (e["observed_at"] - e["started_at"])
        * sum(cards for task, cards in e["tasks"].items() if tasks is None or task in tasks)
        for e in selected
    )
    return {
        "gpu_hours": seconds / 3600,
        "observed_at": min(e["observed_at"] for e in selected if not e["ended"])
        if any(not e["ended"] for e in selected)
        else max(e["observed_at"] for e in selected),
        "ended": all(e["ended"] for e in selected),
        "sources": [e["source"] for e in selected],
    }
