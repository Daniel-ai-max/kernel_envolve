"""Node-local release guard, independent of the desktop/model transport.

The deployment supplies a hard platform TTL and stops the allocation when the
release marker appears. Cleanup failure must never prevent that marker.
"""

import json
import math
import os
import subprocess
import time
from pathlib import Path

from kernel_envolve._io import durable_bytes, durable_json
from kernel_envolve.manager.execution.process import terminate


def finite_number(value):
    try:
        return type(value) in (int, float) and math.isfinite(value)
    except OverflowError:
        return False


def release_reason(*, now, deadline, lease_expires, interrupted=False, reserve_seconds=180):
    if not all(finite_number(x) for x in (now, deadline, lease_expires, reserve_seconds)):
        return "invalid_allocation_control"
    if reserve_seconds < 0:
        return "invalid_allocation_control"
    if now >= deadline - reserve_seconds:
        return "allocation_deadline"
    if interrupted:
        return "autonomous_workflow_interrupted"
    if now >= lease_expires:
        return "model_host_lease_expired"
    return None


def release(marker, receipt, *, reason, cleanup_commands=(), timeout=30):
    """Best-effort bounded cleanup; outer platform TTL remains the final backstop."""
    marker, receipt = Path(marker), Path(receipt)
    record = dict(reason=reason, triggered_at=time.time(), cleanup=[])
    # A malformed timeout must not bypass the release finally block or hang cleanup.
    valid_timeout = finite_number(timeout) and 0 <= timeout <= 300
    until = time.monotonic() + (timeout if valid_timeout else 0)
    if not valid_timeout:
        record["cleanup"].append(dict(error="invalid_cleanup_timeout"))
    try:
        for argv in cleanup_commands:
            remaining = until - time.monotonic()
            if remaining <= 0:
                record["cleanup"].append(dict(argv=argv, error="cleanup_budget_exhausted"))
                break
            try:
                with subprocess.Popen(
                    argv,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    start_new_session=True,
                ) as p:
                    try:
                        p.wait(timeout=remaining)
                        record["cleanup"].append(dict(argv=argv, returncode=p.returncode))
                    finally:
                        terminate(p, grace=0)
            except (OSError, subprocess.TimeoutExpired) as exc:
                record["cleanup"].append(dict(argv=argv, error=type(exc).__name__))
    finally:
        # Release precedes optional receipt write so full/broken evidence storage
        # cannot indefinitely keep idle GPUs allocated.
        durable_bytes(marker, (reason + "\n").encode())
        record["release_requested_at"] = time.time()
        durable_json(receipt, record)
    return record


def watch(config_path):
    config = json.loads(Path(config_path).read_text())
    marker = Path(config["release_marker"])
    while not marker.exists():
        if config.get("heartbeat_path"):
            durable_json(Path(config["heartbeat_path"]), {"pid": os.getpid(), "at": time.time()})
        try:
            lease = json.loads(Path(config["lease_path"]).read_text())
            expires = lease["expires_at"]
            if not finite_number(expires) or lease.get("state") not in {
                "starting",
                "running",
                "recovering",
                "interrupted",
            }:
                raise ValueError("Invalid host lease")
            interrupted = lease["state"] == "interrupted"
        except (OSError, ValueError, KeyError, TypeError, OverflowError):
            expires, interrupted = 0, False  # Missing/malformed lease fails closed.
        reason = release_reason(
            now=time.time(),
            deadline=config["deadline"],
            lease_expires=expires,
            interrupted=interrupted,
            reserve_seconds=config.get("reserve_seconds", 180),
        )
        if reason:
            timeout = config.get("cleanup_timeout", 30)
            if finite_number(timeout) and finite_number(config["deadline"]):
                timeout = min(timeout, max(0, config["deadline"] - time.time()))
            release(
                marker,
                config["receipt"],
                reason=reason,
                cleanup_commands=config.get("cleanup_commands", []),
                timeout=timeout,
            )
            return
        time.sleep(2)


if __name__ == "__main__":
    import sys

    watch(sys.argv[1])
