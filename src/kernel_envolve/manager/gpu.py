"""FIFO device admission with a host-wide inherited lock and parent-round supervision."""

import math
import os
import subprocess
import time

from .process import alive, exclusive, gpu_idle, identity, lock_path
from .store import Store
from .worker import supervise


class DurationAdmissionError(ValueError):
    """A declared command cannot fit the remaining execution budgets."""


def require_duration(store, task, parent, expected_seconds):
    """Admission uses live elapsed time; declarations never reserve or bill usage."""
    if expected_seconds is None:
        return
    now = time.time()
    remaining = {
        "parent round": task["spec"]["round_timeout_seconds"]
        - max(parent["seconds"], now - parent["started"])
    }
    for kind, job_kind in (("wall", "agent"), ("gpu", "gpu")):
        limit = task["spec"][kind + "_budget_seconds"]
        if limit is None:
            continue
        # Heartbeats account usage periodically; also include time since the last one.
        unreported = sum(
            max(0.0, now - job["started"] - job["seconds"])
            for job in store.rows(
                "SELECT started,seconds FROM jobs WHERE task=? AND kind=? AND status='running'",
                (task["id"], job_kind),
            )
        )
        remaining["cumulative " + kind + " budget"] = limit - task[kind + "_seconds"] - unreported
    for name, seconds in remaining.items():
        if expected_seconds > seconds:
            raise DurationAdmissionError(
                f"GPU duration admission rejected: declared {expected_seconds:g}s exceeds "
                f"{max(0, seconds):.3f}s remaining in {name}. "
                "Save a checkpoint before rescheduling; no budget was extended."
            )


def run(root, task_id, round_id, command, *, expected_seconds=None):
    if os.environ.get("KERNEL_ENVOLVE_GPU_JOB"):
        raise ValueError("Nested gpu-run is not supported; wrap the entire GPU command once")
    if not command:
        raise ValueError("A GPU command is required")
    if expected_seconds is not None and (
        type(expected_seconds) not in (int, float)
        or not math.isfinite(expected_seconds)
        or expected_seconds <= 0
    ):
        raise ValueError("expected-seconds must be finite and positive")
    store = Store(root)
    key = None
    try:
        with store.transaction():
            task = store.task(task_id)
            parent = store.job(round_id)
            if (
                parent["task"] != task_id
                or parent["kind"] != "agent"
                or parent["status"] != "running"
            ):
                raise ValueError("GPU work requires the current running agent round")
            require_duration(store, task, parent, expected_seconds)
            key = store.enqueue(task, "gpu", command, round_id)
            if expected_seconds is not None:
                store.event(
                    task_id, "gpu_duration_declared", {"job": key, "seconds": expected_seconds}
                )
            store.db.execute(
                "UPDATE jobs SET pid=?,identity=?,heartbeat=? WHERE id=?",
                (os.getpid(), identity(os.getpid()), time.time(), key),
            )
        device = store.resource(parent["resource"])["spec"]["device"]
        while True:
            task = store.task(task_id)
            parent = store.job(round_id)
            if (
                store.job(key)["status"] != "queued"
                or parent["cancel_requested"]
                or task["desired"] != "active"
                or store.budget_reason(task)
                or parent["status"] != "running"
                or not alive(parent["pid"], parent["identity"])
            ):
                store.finish(key, "cancelled", "Parent ended, task paused, or budget exhausted")
                return 2
            with exclusive(lock_path("gpu", device)) as handle:
                if handle is not None:
                    try:
                        gpu_idle(device)
                    except (OSError, ValueError, subprocess.SubprocessError) as exc:
                        store.db.execute("UPDATE jobs SET error=? WHERE id=?", (str(exc), key))
                    else:
                        if store.claim(key, os.getpid(), identity(os.getpid())):
                            # Waiting in the device queue consumes parent/wall time.
                            with store.transaction():
                                require_duration(
                                    store,
                                    store.task(task_id),
                                    store.job(round_id),
                                    expected_seconds,
                                )
                            result = supervise(store, key, handle)
                            directory = store.root / "jobs" / key
                            # Forward bounded tails; full output remains in durable job logs.
                            for name in ("stdout.log", "stderr.log"):
                                if (directory / name).exists():
                                    with (directory / name).open("rb") as log:
                                        log.seek(max(0, (directory / name).stat().st_size - 65536))
                                        print(log.read().decode(errors="replace"), end="")
                            return result
            time.sleep(0.2)
    except BaseException as exc:
        if key:
            reason = str(exc) if isinstance(exc, ValueError) else "GPU waiter interrupted"
            store.finish(key, "cancelled", reason)
        raise
    finally:
        store.close()
