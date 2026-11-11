"""Reconcile persisted intents with actual processes, then admit new rounds."""

import json
import os
import signal
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor

from .config import resource_devices
from .process import (
    alive,
    exclusive,
    gpu_group_idle,
    gpu_locks,
    identity,
    lock_path,
    reap_registered,
    stop_recorded,
)
from .store import Store, encode


def reconcile(store):
    for job in store.rows(
        "SELECT * FROM jobs WHERE status IN ('queued','running') ORDER BY created"
    ):
        # A transient process-probe error escapes: it must not cause replacement work.
        if alive(job["pid"], job["identity"]):
            continue
        if job["status"] == "queued" and job["pid"] is None and time.time() - job["created"] < 30:
            continue
        task = store.task(job["task"])
        devices = store.job_devices(job)
        guard = (
            exclusive(lock_path("task", task["spec"]["workspace"]))
            if job["kind"] == "agent"
            else gpu_locks(devices)
        )
        stop_recorded(job["child_pid"], job["child_identity"])
        reap_registered(store, job["id"])
        with guard as handle:
            if handle is None:
                # A surviving child inherited the lock. Do not declare the job terminal.
                store.db.execute(
                    "UPDATE jobs SET error=? WHERE id=?",
                    ("Supervisor absent; owned lock still held; inspect descendants", job["id"]),
                )
                continue
            if job["kind"] == "gpu" and job["status"] == "running":
                try:
                    gpu_group_idle(devices)
                except (OSError, ValueError, subprocess.SubprocessError) as exc:
                    store.db.execute(
                        "UPDATE resources SET problem=? WHERE id=?",
                        (f"Orphan GPU work; inspect before clearing: {exc}", job["resource"]),
                    )
            if job["started"]:
                # Conservative charge for the unobserved interval, never free budget on a crash.
                store.heartbeat(job["id"], max(job["seconds"], time.time() - job["started"]))
            store.finish(
                job["id"], "interrupted", "Supervisor exited; recovered without replaying the job"
            )


def clear_resource(store, key):
    resource = store.resource(key)
    devices = resource_devices(resource["spec"])
    with gpu_locks(devices) as handle:
        if handle is None:
            raise ValueError("Device lock is still held")
        gpu_group_idle(devices)
        with store.transaction():
            if store.one(
                "SELECT id FROM jobs WHERE resource=? AND kind='gpu' AND status='running'", (key,)
            ):
                raise ValueError("Reconcile the active GPU job first")
            store.db.execute("UPDATE resources SET problem=NULL WHERE id=?", (key,))
            store.event(None, "resource_cleared", {"resource": key})


def resource_sessions(store, key, sessions):
    if type(sessions) is not int or not 1 <= sessions <= 2:
        raise ValueError("sessions must be 1 or 2")
    with store.transaction():
        resource = store.resource(key)
        resource["spec"]["sessions"] = sessions
        store.db.execute(
            "UPDATE resources SET spec=? WHERE id=?", (json.dumps(resource["spec"]), key)
        )
        store.event(None, "resource_sessions", {"resource": key, "sessions": sessions})


def run(store, once=False):
    with exclusive(store.root / "coordinator.lock") as handle:
        if handle is None:
            raise ValueError("A coordinator already owns this state directory")
        stopped = threading.Event()
        stop_requested = False

        def on_signal(signum, frame):
            # A signal handler must not acquire Event locks held by this thread.
            nonlocal stop_requested
            stop_requested = True

        handlers = {s: signal.signal(s, on_signal) for s in (signal.SIGINT, signal.SIGTERM)}
        children = []
        pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="evidence")
        future, last_scan = None, 0
        evidence_error = None
        completed_once = False

        def scan():
            from .evidence import refresh

            local = Store(store.root)
            try:
                refresh(local, stop=stopped)
            finally:
                local.close()

        try:
            while not stop_requested:
                health = {
                    "pid": os.getpid(),
                    "identity": identity(os.getpid()),
                    "heartbeat": time.time(),
                }
                try:
                    reconcile(store)
                except (OSError, ValueError, subprocess.SubprocessError) as exc:
                    health["error"] = f"Process recovery unavailable; dispatch held: {exc}"
                    store.db.execute(
                        "INSERT OR REPLACE INTO metadata VALUES('coordinator',?)", (encode(health),)
                    )
                    if once:
                        raise
                    time.sleep(1)
                    continue
                if stop_requested:
                    break
                if future is not None and future.done():
                    try:
                        future.result()
                        evidence_error = None
                    except Exception as exc:
                        evidence_error = str(exc)
                    future = None
                if future is None and time.monotonic() - last_scan >= 5:
                    future = pool.submit(scan)
                    last_scan = time.monotonic()
                health["evidence_error"] = evidence_error
                store.db.execute(
                    "INSERT OR REPLACE INTO metadata VALUES('coordinator',?)", (encode(health),)
                )
                if stop_requested:
                    break
                for key in store.schedule():
                    directory = store.root / "jobs" / key
                    directory.mkdir(parents=True, exist_ok=True)
                    try:
                        with (directory / "supervisor.log").open("ab") as log:
                            children.append(
                                subprocess.Popen(
                                    [
                                        sys.executable,
                                        "-m",
                                        "kernel_envolve.manager.worker",
                                        str(store.root),
                                        key,
                                    ],
                                    stdin=subprocess.DEVNULL,
                                    stdout=log,
                                    stderr=log,
                                    start_new_session=True,
                                    env=os.environ.copy(),
                                )
                            )
                    except OSError as exc:
                        store.finish(key, "failed", str(exc))
                children = [p for p in children if p.poll() is None]
                if once:
                    completed_once = True
                    break
                time.sleep(1)
        finally:
            # Keep both locks owned until the evidence thread reaches a safe boundary.
            cancel_scan = stop_requested or not completed_once
            if cancel_scan:
                stopped.set()
            pool.shutdown(wait=True, cancel_futures=cancel_scan)
            for sig, handler in handlers.items():
                signal.signal(sig, handler)
        # Round supervisors intentionally survive a coordinator restart. Pause tasks to stop them.
