"""Opt-in host policy: two fixed research windows, independently verified delivery.

This is scheduling policy, never evaluator evidence. No evaluator is needed to
install it. The host deadline adapter owns allocation/remote-guard admission.
"""

import json
import math
import signal
import statistics
import subprocess
import tempfile
import time
from pathlib import Path

from ..storage import digest, read_json
from .process import exclusive
from .store import encode

WINDOW = 4 * 3600
TERMINAL = {"succeeded", "exhausted", "suspended"}


def get(store, key):
    row = store.one("SELECT data FROM metadata WHERE key=?", ("research_policy:" + key,))
    return json.loads(row["data"]) if row else None


def save(store, key, old, value):
    with store.transaction():
        if get(store, key) != old:
            raise ValueError("Research policy changed; retry from current state")
        store.db.execute(
            "INSERT OR REPLACE INTO metadata VALUES(?,?)",
            ("research_policy:" + key, encode(value)),
        )
        store.event(key, "research_policy", value)
    return value


def install(store, key, *, started_at, min_gain, deadline_command):
    """Install once on a paused host task; retain all old counters and timestamps."""
    from .config import positive, resource_devices

    positive(started_at, "started_at")
    if not started_at <= time.time() < started_at + WINDOW:
        raise ValueError("Install within the original first window; never reset an expired policy")
    if type(min_gain) not in (int, float) or not math.isfinite(min_gain) or not 0 < min_gain < 1:
        raise ValueError("min_gain must be a latency-reduction fraction strictly between 0 and 1")
    if (
        not isinstance(deadline_command, list)
        or not deadline_command
        or any(not isinstance(arg, str) or not arg or "\0" in arg for arg in deadline_command)
    ):
        raise ValueError("Declare a trusted host deadline adapter argv")
    task = store.task(key)
    spec = task["spec"]
    if spec["kind"] != "research" or not spec.get("host_command") or spec.get("observe_only"):
        raise ValueError("Policy requires an executable research task with a real model host")
    if len(resource_devices(store.resource(spec["resource"])["spec"])) != 1:
        raise ValueError("Bounded research policy requires one allocated device per lane")
    if task["desired"] != "paused" or active_jobs(store, key):
        raise ValueError("Pause and drain the research task before installing policy")
    if get(store, key) is not None:
        raise ValueError(
            "Policy is already installed; its clock and consumed budgets are immutable"
        )
    value = {
        "schema": "bounded-research-policy-v1",
        "task": key,
        "started_at": started_at,
        "initial_end": started_at + WINDOW,
        "maximum_end": started_at + 2 * WINDOW,
        "min_gain": min_gain,
        "deadline_command": deadline_command,
        "identity": identity(spec),
        "initial_wall_seconds": task["wall_seconds"],
        "initial_gpu_seconds": task["gpu_seconds"],
        "phase": "starting",
        "extension_used": False,
        "delivery": None,
    }
    return save(store, key, None, value)


def identity(spec):
    return {k: spec[k] for k in ("kind", "workspace", "task_dir", "baseline")}


def active_jobs(store, key):
    return (
        store.one("SELECT id FROM jobs WHERE task=? AND status IN ('queued','running')", (key,))
        is not None
    )


def source_link(store, key, delivery, root):
    from ..native import snapshot
    from ..native_config import load_config

    research = store.task(key)["spec"]
    native = store.task(delivery)["spec"]
    if (
        native["kind"] != "native"
        or native["baseline"]["revision"] != research["baseline"]["revision"]
    ):
        raise ValueError("Delivery must be a distinct native task with the same frozen baseline")
    cfg = load_config(Path(native["task_dir"]))
    if root not in cfg["sources"]["candidate"] or root not in cfg["sources"]["baseline"]:
        raise ValueError("Unknown delivery source root")
    candidate = Path(cfg["sources"]["candidate"][root]).resolve()
    if not candidate.is_relative_to(Path(native["workspace"]).resolve()):
        raise ValueError("Delivery candidate must belong to its dedicated native workspace")
    if "confirmation" not in cfg or set(cfg["required_scopes"]) != {
        "operator",
        "numerical",
        "integration",
    }:
        raise ValueError("Delivery requires full native scopes and independent confirmation")
    baseline = snapshot(cfg["sources"]["baseline"][root], None, cfg["artifacts"].get(root, []))
    current = snapshot(candidate, None, cfg["artifacts"].get(root, []))
    original = snapshot(research["workspace"], None, cfg["artifacts"].get(root, []))
    if (
        baseline["git_head"] != research["baseline"]["revision"]
        or current["files"] != original["files"]
    ):
        raise ValueError("Delivery baseline or candidate contents do not match research")
    return {"task": delivery, "source_root": root, "identity": identity(native)}


def bind_delivery(store, key, delivery, source_root):
    value = get(store, key)
    if not value or value["phase"] in TERMINAL:
        raise ValueError("An active installed research policy is required")
    binding = source_link(store, key, delivery, source_root)
    return save(store, key, value, value | {"delivery": binding})


def delivery_proof(store, value):
    """Re-audit full + confirmation + fresh source-bound PR; summaries never qualify."""
    from ..native import native_decision, snapshot
    from .evidence import inspect_run
    from .publication import publication_view
    from .views import best

    link = value["delivery"]
    if not link:
        return None
    task = store.task(link["task"])
    if identity(task["spec"]) != link["identity"]:
        raise ValueError("Delivery identity changed")
    current = best(store, task)
    publication = publication_view(store, task["id"], current)
    if not publication or not publication["current_head_verified"]:
        return None
    audit_deadline = time.monotonic() + 20

    def checkpoint():
        remaining = audit_deadline - time.monotonic()
        if remaining <= 0:
            raise ValueError("Delivery audit exceeded the bounded policy handoff check")
        return remaining

    verified = inspect_run(Path(current["run"]), task["spec"], checkpoint)
    if (
        verified["mode"] != "full"
        or not verified["decision"]["eligible_for_best"]
        or verified["confirmation_status"] != "passed"
        or not verified["confirmation"]
        or any(verified.get(k) != current.get(k) for k in ("result_hash", "confirmation"))
        or publication["source_root"] != link["source_root"]
        or publication["evaluated_source_hash"] != digest(verified["source"])
    ):
        return None
    cfg = read_json(Path(current["run"]) / "config.json")
    result = read_json(Path(current["run"]) / "result.json")
    root = link["source_root"]
    if "confirmation" not in cfg or set(cfg["required_scopes"]) != {
        "operator",
        "numerical",
        "integration",
    }:
        return None
    research = store.task(value["task"])["spec"]
    actual = snapshot(research["workspace"], None, cfg["artifacts"].get(root, []), checkpoint)
    if (
        result["identities"]["baseline"][root]["git_head"] != research["baseline"]["revision"]
        or verified["source"][root]["files"] != actual["files"]
    ):
        return None
    child = Path(verified["confirmation"]["run"])
    held = native_decision(read_json(child / "result.json"), read_json(child / "config.json"))
    for decision in (verified["decision"], held):
        gains = decision.get("fixed_baseline_gain_by_batch", [])
        if (
            not gains
            or not decision["eligible_for_best"]
            or statistics.median(gains) < value["min_gain"]
        ):
            return None
    return {
        "delivery_task": task["id"],
        "run": current["run"],
        "result_hash": verified["result_hash"],
        "confirmation": verified["confirmation"],
        "publication": {
            k: publication[k]
            for k in (
                "url",
                "head",
                "base_head",
                "result_hash",
                "confirmation",
                "implementation_digest",
            )
        },
        "source_hash": digest(verified["source"]),
    }


def invoke(command, request):
    """Bound host synchronization; remote work must be idempotent by transition_id."""
    with tempfile.TemporaryFile() as output, tempfile.TemporaryFile() as errors:
        process = subprocess.Popen(
            command, stdin=subprocess.PIPE, stdout=output, stderr=errors, start_new_session=True
        )
        try:
            process.communicate(encode(request).encode(), timeout=10)
            if process.returncode != 0 or output.tell() > 65536:
                raise ValueError("Host deadline adapter failed or exceeded output bound")
            output.seek(0)
            return json.load(output)
        finally:
            try:
                import os

                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait()


def deadline_request(value):
    extension = value["phase"] == "extending"
    end = value["maximum_end"] if extension else value["initial_end"]
    request = {
        "schema": "research-policy-deadline-v1",
        "task": value["task"],
        "started_at": value["started_at"],
        "initial_end": value["initial_end"],
        "maximum_end": value["maximum_end"],
        "deadline_at": end,
        "extension_used": extension,
        "wall_budget_seconds": value["initial_wall_seconds"] + (2 if extension else 1) * WINDOW,
        "gpu_budget_seconds": value["initial_gpu_seconds"] + (2 if extension else 1) * WINDOW,
    }
    request["transition_id"] = digest(request)
    return request


def admission(store, value, sync):
    request = deadline_request(value)
    end = request["deadline_at"]
    reply = sync(value["deadline_command"], request)
    for field in ("schema", "task", "transition_id", "maximum_end"):
        if reply.get(field) != request[field]:
            raise ValueError("Host admission does not match the immutable policy transition")
    if reply.get("status") == "pending":
        return None
    deadline = reply.get("deadline_at")
    used = reply.get("allocation_gpu_seconds_used")
    limit = reply.get("allocation_gpu_seconds_limit")
    allocation_end = reply.get("allocation_deadline_at")
    if any(
        type(x) not in (int, float) or not math.isfinite(x)
        for x in (deadline, used, limit, allocation_end)
    ):
        raise ValueError("Host admission requires finite allocation clocks and cumulative usage")
    now = time.time()
    if not (
        now < deadline <= end
        and deadline <= allocation_end <= value["maximum_end"]
        and 0 <= used <= limit <= 2 * WINDOW
        and used + allocation_end - now <= limit + 1
    ):
        raise ValueError("Host admission exceeds original research or allocation budget")
    prior = value.get("admission", {})
    if used < prior.get("allocation_gpu_seconds_used", 0):
        raise ValueError("Cumulative allocation usage cannot decrease")
    if reply.get("status") != "ready":
        raise ValueError("Host must explicitly acknowledge ready or pending")
    return request, reply


def tick(store, key, *, sync=invoke):
    """Called by the coordinator, not by model reports. Never resumes user controls."""
    with exclusive(store.root / "research-policy.lock") as handle:
        if handle is None:
            return get(store, key)
        value = get(store, key)
        if not value or value["phase"] in TERMINAL:
            return value
        task = store.task(key)
        if identity(task["spec"]) != value["identity"] or task["superseded_by"]:
            raise ValueError("Research policy identity changed")
        now = time.time()
        if now >= value["maximum_end"]:
            next_value = value | {"phase": "exhausted"}
            store.control(key, "pause", _policy=(value, next_value))
            return next_value
        if task["desired"] == "blocked":
            return save(store, key, value, value | {"phase": "suspended", "error": task["reason"]})
        if value["phase"] in ("initial", "extended"):
            if task["desired"] not in ("active", "budget"):
                return value  # Blocked execution needs inspection; never silently clear it.
            try:
                proof = delivery_proof(store, value)
            except (OSError, ValueError, KeyError, TypeError, subprocess.SubprocessError) as exc:
                # Invalid delivery is no qualifying outcome, not a reason to lose
                # the remaining research window or its authorized continuation.
                store.db.execute(
                    "INSERT OR REPLACE INTO metadata VALUES(?,?)",
                    ("research_policy_error:" + key, encode({"error": str(exc), "at": now})),
                )
                proof = None
            if proof:
                next_value = value | {"phase": "stopping", "delivery_proof": proof}
                store.control(key, "pause", _policy=(value, next_value))
                return next_value
            if value["phase"] == "extended" and now >= task["spec"]["deadline_at"]:
                next_value = value | {"phase": "exhausted"}
                store.control(key, "pause", _policy=(value, next_value))
                return next_value
            if value["phase"] == "initial" and now >= task["spec"]["deadline_at"]:
                next_value = value | {"phase": "extending", "extension_used": True}
                store.control(key, "pause", _policy=(value, next_value))
                return next_value
            return value
        if active_jobs(store, key):
            return value
        if value["phase"] == "stopping":
            try:
                proof = delivery_proof(store, value)
            except (OSError, ValueError, KeyError, TypeError, subprocess.SubprocessError):
                proof = None
            if proof != value["delivery_proof"]:
                # Source/evidence drift during drain must not be recorded as success.
                return save(
                    store,
                    key,
                    value,
                    value | {"phase": "suspended", "error": "Delivery changed during stop"},
                )
            return save(store, key, value, value | {"phase": "succeeded", "completed_at": now})
        if value["phase"] in ("starting", "extending"):
            request = deadline_request(value)
            if value.get("pending_transition") != request:
                value = save(store, key, value, value | {"pending_transition": request})
            admitted = admission(store, value, sync)
            if admitted is None:
                return value
            request, reply = admitted
            changes = {k: request[k] for k in ("wall_budget_seconds", "gpu_budget_seconds")}
            changes["deadline_at"] = reply["deadline_at"]
            next_value = value | {
                "admission": reply,
                "phase": "extended" if value["extension_used"] else "initial",
            }
            store.control(key, "configure", changes, _policy=(value, value))
            store.control(key, "resume", _policy=(value, next_value))
            return next_value
        return value


def refresh(store):
    for row in store.rows("SELECT key FROM metadata WHERE key LIKE 'research_policy:%'"):
        key = row["key"].split(":", 1)[1]
        try:
            tick(store, key)
        except (OSError, ValueError, KeyError, TypeError, subprocess.SubprocessError) as exc:
            # Adapter/evidence failures do not stop independent lanes or extend budgets.
            data = {"error": str(exc), "at": time.time()}
            store.db.execute(
                "INSERT OR REPLACE INTO metadata VALUES(?,?)",
                ("research_policy_error:" + key, encode(data)),
            )
