"""Read-only dashboard projection of explicit research/native delivery relationships.

Research never acquires an evaluator best. Source/evidence checks run in the
background; page reads retain verified historical receipts and indexed native facts.
"""

import json
import subprocess
import time

from kernel_envolve.evaluation.shared.storage import digest
from kernel_envolve.manager.store import encode

REFRESH_SECONDS = 30


def links(store):
    from kernel_envolve.manager.research_policy import identity

    values = {}
    for row in store.rows("SELECT data FROM metadata WHERE key LIKE 'delivery_link:%'"):
        value = json.loads(row["data"])
        values[value["task"]] = value
    # The pre-projection policy already stored an explicit current binding.
    for row in store.rows("SELECT data FROM metadata WHERE key LIKE 'research_policy:%'"):
        policy = json.loads(row["data"])
        if link := policy.get("delivery"):
            values.setdefault(
                link["task"],
                link | {"research_task": policy["task"], "research_identity": policy["identity"]},
            )
    result = {}
    for key, value in values.items():
        try:
            parent = store.task(value["research_task"])["spec"]
            child = store.task(key)["spec"]
            if (
                parent["kind"] == "research"
                and child["kind"] == "native"
                and child.get("observe_only")
                and identity(parent) == value["research_identity"]
                and identity(child) == value["identity"]
                and parent["baseline"]["revision"] == child["baseline"]["revision"]
            ):
                result[key] = value
        except (KeyError, ValueError):
            continue
    return result


def refresh(store, task_id=None):
    """Audit frozen native delivery, independent of the current research checkout.

    Stop policy deliberately has a stronger current-research-source condition.
    An unavailable refresh never erases a previously verified delivery receipt.
    """
    from pathlib import Path

    from kernel_envolve.evaluation.shared.storage import file_hash, read_json
    from kernel_envolve.manager.dashboard.views import best
    from kernel_envolve.manager.results.publication import publication_view

    for child_id, link in links(store).items():
        if task_id is not None and task_id not in (child_id, link["research_task"]):
            continue
        meta_key = "delivery_display:" + child_id
        row = store.one("SELECT data FROM metadata WHERE key=?", (meta_key,))
        previous = json.loads(row["data"]) if row else {}
        if task_id is None and time.time() - previous.get("audited_at", 0) < REFRESH_SECONDS:
            continue
        value = previous | {"audited_at": time.time()}
        try:
            child = store.task(child_id)
            current = best(store, child)
            publication = publication_view(store, child_id, current)
            if not current or not publication or not publication["current_head_verified"]:
                raise ValueError("No current verified native publication")
            # The existing evidence scanner already deep-audits changed native
            # evidence (including independent confirmation) before indexing it.
            # sync-pr also deep-audits before publishing. Do not hash the full
            # source trees again for dashboard narration/display.
            verified = current
            run = Path(current["run"])
            raw = read_json(run / "result.json")
            cfg = read_json(run / "config.json")
            native_receipt = read_json(run / "receipt.json")
            if (
                not current.get("fingerprint")
                or not current.get("contract_verified")
                or current["contract"] != child["spec"]["baseline"]["contract"]
                or digest(raw) != current["result_hash"]
                or native_receipt["result_hash"] != current["result_hash"]
                or file_hash(run / "config.json") != native_receipt["config_hash"]
            ):
                value["invalid_result_hash"] = current["result_hash"]
                raise ValueError("Native indexed result/config receipt changed")
            root = link["source_root"]
            parent = store.task(link["research_task"])["spec"]
            if (
                verified["mode"] != "full"
                or not verified["decision"]["eligible_for_best"]
                or verified["confirmation_status"] != "passed"
                or not verified["confirmation"]
                or set(cfg["required_scopes"]) != {"operator", "numerical", "integration"}
                or raw["identities"]["baseline"][root]["git_head"] != parent["baseline"]["revision"]
                or publication["source_root"] != root
                or publication["evaluated_source_hash"] != digest(verified["source"])
            ):
                value["invalid_result_hash"] = current["result_hash"]
                raise ValueError("Native full, confirmation, baseline and publication differ")
            receipt = {
                "link": link,
                "checked_at": time.time(),
                "run": verified["run"],
                "result_hash": verified["result_hash"],
                "source_hash": digest(verified["source"]),
                "confirmation": verified["confirmation"],
                "publication": {k: v for k, v in publication.items() if k != "checked_at"},
            }
            value.pop("invalid_result_hash", None)
            value.update(receipt=receipt, error=None)
            store.db.execute(
                "INSERT OR REPLACE INTO metadata VALUES(?,?)",
                (
                    "delivery_display_history:" + child_id + ":" + verified["result_hash"],
                    encode(receipt),
                ),
            )
        except (OSError, ValueError, KeyError, TypeError, subprocess.SubprocessError) as exc:
            value["error"] = type(exc).__name__
        store.db.execute("INSERT OR REPLACE INTO metadata VALUES(?,?)", (meta_key, encode(value)))


def retained(store, child_id, link):
    """Use the recorded verified result, never the research candidate or report."""
    from kernel_envolve.manager.results.publication import publication_view

    row = store.one("SELECT data FROM metadata WHERE key=?", ("delivery_display:" + child_id,))
    projection = json.loads(row["data"]) if row else {}
    receipt = projection.get("receipt")
    if (
        not receipt
        or receipt["link"] != link
        or projection.get("invalid_result_hash") == receipt["result_hash"]
    ):
        return None
    row = store.one(
        "SELECT data,error FROM runs WHERE path=? AND task=?", (receipt["run"], child_id)
    )
    current = json.loads(row["data"]) if row and not row["error"] else None
    if (
        not current
        or current.get("confirmation_status") != "passed"
        or not current.get("decision", {}).get("eligible_for_best")
        or any(current.get(k) != receipt.get(k) for k in ("run", "result_hash", "confirmation"))
        or digest(current.get("source")) != receipt["source_hash"]
    ):
        return None
    publication = publication_view(store, child_id, current)
    expected = receipt["publication"]
    if publication and all(
        publication.get(k) == expected.get(k)
        for k in (
            "url",
            "head",
            "base_head",
            "result_hash",
            "confirmation",
            "implementation_digest",
        )
    ):
        publication = {k: v for k, v in publication.items() if k != "checked_at"}
    else:
        # The historical evaluated commit remains auditable, but is not the
        # current verified PR implementation after a new/unverified head.
        publication = expected | {"current_head_verified": False}
    return {
        "task": child_id,
        "status": "verified",
        "checked_at": receipt["checked_at"],
        "best": {
            k: current.get(k)
            for k in (
                "run",
                "result_hash",
                "confirmation",
                "baseline_speedup",
                "incremental_speedup",
                "recorded_at",
                "mode",
                "execution",
                "correctness",
            )
        },
        "implementation_pr": publication,
    }


def view(store, task, now):
    from kernel_envolve.manager.research_policy import get

    key = task["id"]
    relationships = links(store)
    if key in relationships:
        return {"delivery_parent": relationships[key]["research_task"]}
    if task["spec"]["kind"] != "research":
        return {}
    owned = {child: link for child, link in relationships.items() if link["research_task"] == key}
    history = [{"task": child, "bound_at": link.get("bound_at")} for child, link in owned.items()]
    policy = get(store, key) or {}
    binding = policy.get("delivery") or {}
    candidates = [
        result for child, link in owned.items() if (result := retained(store, child, link))
    ]
    # Attempts may use different contracts/modes. Do not rank incomparable
    # speedups; retain the most recently verified published result.
    delivered = max(candidates, key=lambda x: x["best"].get("recorded_at", 0), default=None)
    if delivered:
        delivered["policy_phase"] = policy.get("phase")
    elif binding.get("task") in owned:
        delivered = {
            "task": binding["task"],
            "status": "pending",
            "best": None,
            "implementation_pr": None,
            "policy_phase": policy.get("phase"),
        }
    return {
        "delivery_attempts": history,
        "delivery": delivered,
        "current_delivery_task": binding.get("task"),
    }
