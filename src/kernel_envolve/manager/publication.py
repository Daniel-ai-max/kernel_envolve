"""Bind a published PR to an explicitly selected, verified best run."""

import datetime as dt
import json
import re
import subprocess
import time

from .store import encode


def publication_view(store, task, best):
    row = store.one("SELECT data FROM metadata WHERE key=?", ("publication:" + task,))
    value = json.loads(row["data"]) if row else None
    return value if best and value and value["run"] == best["run"] else None


def sync_pr(store, task, run, url):
    from .views import best

    match = re.fullmatch(r"https://github\.com/([\w.-]+)/([\w.-]+)/pull/([1-9]\d*)", url)
    if not match:
        raise ValueError("Expected a GitHub pull request URL")
    current = best(store, store.task(task))
    if not current or current["run"] != run:
        raise ValueError("PR must be bound to the current verified best run")
    result = subprocess.run(
        ["gh", "pr", "view", url, "--json", "url,updatedAt,headRefOid"],
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )
    data = json.loads(result.stdout)
    updated = dt.datetime.fromisoformat(data["updatedAt"].replace("Z", "+00:00"))
    if (
        data["url"] != url
        or updated.tzinfo is None
        or not re.fullmatch(r"[a-f0-9]{40}", data["headRefOid"])
    ):
        raise ValueError("Invalid GitHub PR metadata")
    receipt = {
        "run": run,
        "url": url,
        "updated_at": updated.timestamp(),
        "head": data["headRefOid"],
        "observed_at": time.time(),
    }
    with store.transaction():
        current = best(store, store.task(task))
        if not current or current["run"] != run:
            raise ValueError("Verified best changed while fetching PR")
        store.db.execute(
            "INSERT OR REPLACE INTO metadata VALUES(?,?)",
            ("publication:" + task, encode(receipt)),
        )
        store.event(task, "publication", receipt)
