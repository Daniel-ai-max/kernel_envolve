"""FIFO device admission with a host-wide inherited lock and parent-round supervision."""

import os
import subprocess
import time

from .process import alive, exclusive, gpu_idle, identity, lock_path
from .store import Store
from .worker import supervise


def run(root, task_id, round_id, command):
    if os.environ.get("KERNEL_ENVOLVE_GPU_JOB"):
        raise ValueError("Nested gpu-run is not supported; wrap the entire GPU command once")
    if not command:
        raise ValueError("A GPU command is required")
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
            key = store.enqueue(task, "gpu", command, round_id)
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
    except BaseException:
        if key:
            store.finish(key, "cancelled", "GPU waiter interrupted")
        raise
    finally:
        store.close()
