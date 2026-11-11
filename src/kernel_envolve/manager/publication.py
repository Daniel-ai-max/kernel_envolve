"""Bind publication to verified source contents and expire changed GitHub heads."""

import datetime as dt
import json
import re
import subprocess
import time
from pathlib import Path

from ..storage import digest
from .publication_sources import bind_source
from .store import encode

REFRESH_SECONDS = 60
MAX_AGE_SECONDS = 240
FIELDS = "url,updatedAt,headRefOid,baseRefName,baseRefOid,headRepository,state"


def github(arguments):
    result = subprocess.run(
        ["gh", *arguments], check=True, capture_output=True, text=True, timeout=8
    )
    return json.loads(result.stdout)


def metadata(url):
    if not re.fullmatch(r"https://github\.com/[\w.-]+/[\w.-]+/pull/[1-9]\d*", url):
        raise ValueError("Expected a GitHub pull request URL")
    data = github(["pr", "view", url, "--json", FIELDS])
    updated = dt.datetime.fromisoformat(data["updatedAt"].replace("Z", "+00:00"))
    if (
        data["url"] != url
        or updated.tzinfo is None
        or any(not re.fullmatch(r"[a-f0-9]{40}", data[k]) for k in ("headRefOid", "baseRefOid"))
        or not re.fullmatch(r"[\w.-]+/[\w.-]+", data["headRepository"]["nameWithOwner"])
        or not isinstance(data["baseRefName"], str)
    ):
        raise ValueError("Invalid GitHub PR metadata")
    data["updated_at"] = updated.timestamp()
    return data


def publication_view(store, task, best):
    row = store.one("SELECT data FROM metadata WHERE key=?", ("publication:" + task,))
    value = json.loads(row["data"]) if row else None
    if not (
        best
        and value
        and value.get("schema") == 2
        and value["run"] == best["run"]
        and value.get("result_hash") == best.get("result_hash")
        and value.get("confirmation") == best.get("confirmation")
    ):
        return None
    fresh = (
        value.get("verification") == "verified"
        and time.time() - value.get("checked_at", 0) <= MAX_AGE_SECONDS
    )
    return value | {
        "current_head_verified": fresh,
        "commit_url": f"https://github.com/{value['head_repository']}/commit/{value['head']}",
    }


def sync_pr(store, task, run, url, *, source_root, base_branch):
    from .evidence import inspect_run
    from .views import best

    if not re.fullmatch(r"https://github\.com/[\w.-]+/[\w.-]+/pull/[1-9]\d*", url):
        raise ValueError("Expected a GitHub pull request URL")
    if not base_branch or not source_root:
        raise ValueError("Declare source_root and expected base_branch")
    current = best(store, store.task(task))
    if not current or current["run"] != run:
        raise ValueError("PR must be bound to the current verified best run")
    # Never leave an older head verified after a failed attempt to update its binding.
    with store.transaction():
        old = store.one("SELECT data FROM metadata WHERE key=?", ("publication:" + task,))
        if old:
            prior = json.loads(old["data"]) | {"verification": "checking"}
            store.db.execute(
                "UPDATE metadata SET data=? WHERE key=?", (encode(prior), "publication:" + task)
            )
    spec = store.task(task)["spec"]
    verified = inspect_run(Path(run), spec)
    if (
        verified["mode"] != "full"
        or not verified["decision"]["eligible_for_best"]
        or verified["result_hash"] != current.get("result_hash")
    ):
        raise ValueError("Publication requires the unchanged verified full best")
    data = metadata(url)
    if data["baseRefName"] != base_branch or data["state"] not in ("OPEN", "MERGED"):
        raise ValueError("PR target branch or publication state differs from the declared target")
    binding = bind_source(spec, run, source_root, data, github, verified)
    # Source reads/API calls do not hold the SQLite writer lock. Recheck both endpoints.
    after = metadata(url)
    if any(after[k] != data[k] for k in ("headRefOid", "baseRefName", "baseRefOid", "state")):
        raise ValueError("PR head or target changed during source verification")
    rechecked = inspect_run(Path(run), spec)
    if any(rechecked.get(k) != verified.get(k) for k in ("result_hash", "source", "confirmation")):
        raise ValueError("Evaluated evidence changed during publication verification")
    receipt = {
        "schema": 2,
        "verification": "verified",
        "run": run,
        "url": url,
        "updated_at": after["updated_at"],
        "head": data["headRefOid"],
        "base_branch": base_branch,
        "base_head": data["baseRefOid"],
        "observed_at": time.time(),
        "checked_at": time.time(),
        "result_hash": verified["result_hash"],
        "confirmation": verified.get("confirmation"),
        "evaluated_source_hash": digest(verified["source"]),
        **binding,
    }
    with store.transaction():
        current = best(store, store.task(task))
        if (
            not current
            or current["run"] != run
            or current.get("result_hash") != receipt["result_hash"]
            or current.get("confirmation") != receipt["confirmation"]
        ):
            raise ValueError("Verified best changed while fetching PR")
        store.db.execute(
            "INSERT OR REPLACE INTO metadata VALUES(?,?)", ("publication:" + task, encode(receipt))
        )
        store.event(task, "publication", receipt)


def refresh_publications(store):
    """At most one due network check per coordinator scan; page reads remain local."""
    due = []
    for row in store.rows("SELECT key,data FROM metadata WHERE key LIKE 'publication:%'"):
        value = json.loads(row["data"])
        if (
            value.get("schema") == 2
            and value.get("verification") in ("verified", "unavailable")
            and time.time() - value.get("checked_at", 0) >= REFRESH_SECONDS
        ):
            due.append((row, value))
    if not due:
        return
    row, value = min(due, key=lambda item: item[1].get("checked_at", 0))
    try:
        data = metadata(value["url"])
        same = (
            data["headRefOid"] == value["head"]
            and data["baseRefName"] == value["base_branch"]
            and data["baseRefOid"] == value["base_head"]
            and data["state"] in ("OPEN", "MERGED")
        )
        updates = {
            "verification": "verified" if same else "head_changed",
            "checked_at": time.time(),
            "updated_at": data["updated_at"] if same else value["updated_at"],
            "error": None,
        }
    except (OSError, ValueError, KeyError, TypeError, subprocess.SubprocessError) as exc:
        updates = {"verification": "unavailable", "checked_at": time.time(), "error": str(exc)}
    with store.transaction():
        latest = store.one("SELECT data FROM metadata WHERE key=?", (row["key"],))
        if latest and latest["data"] == row["data"]:
            store.db.execute(
                "UPDATE metadata SET data=? WHERE key=?", (encode(value | updates), row["key"])
            )
