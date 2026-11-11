"""A foreground host's one-request-at-a-time bridge; never a model runner."""

import contextlib
import hashlib
import json
import os
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from .config import identifier
from .process import exclusive
from .store import encode


def atomic(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, delete=False) as stream:
        temporary = Path(stream.name)
        stream.write(encode(value))
        stream.flush()
        os.fsync(stream.fileno())
    try:
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def open_host(store, task_id, owner, claim_id):
    identifier(owner)
    identifier(claim_id)
    key = f"host-claim:{task_id}:{claim_id}"
    with store.transaction():
        old = store.one("SELECT data FROM metadata WHERE key=?", (key,))
        if old:
            value = json.loads(old["data"])
            if value["owner"] != owner:
                raise ValueError("Claim belongs to another host")
            return value
        task = store.task(task_id)
        if not task["spec"].get("host_driven") or task["desired"] != "active":
            raise ValueError("An active host_driven task is required")
        if store.budget_reason(task) or task["next_at"] > time.time():
            raise ValueError("Budget exhausted or retry delay has not elapsed")
        if store.one(
            "SELECT id FROM jobs WHERE task=? AND status IN ('queued','running')", (task_id,)
        ):
            raise ValueError("Continue the existing round instead of opening another")
        ready = store.one("SELECT data FROM metadata WHERE key=?", ("host-ready:" + task_id,))
        if ready and json.loads(ready["data"])["expires_at"] > time.time():
            raise ValueError("An unconsumed host claim already exists")
        value = {"claim": claim_id, "owner": owner, "expires_at": time.time() + 120}
        store.db.execute("INSERT INTO metadata VALUES(?,?)", (key, encode(value)))
        store.db.execute(
            "INSERT OR REPLACE INTO metadata VALUES(?,?)", ("host-ready:" + task_id, encode(value))
        )
        store.event(task_id, "host_ready", value)
        return value


def active(store, task, round_id):
    job = store.job(round_id)
    return (
        job["task"] == task
        and job["kind"] == "agent"
        and job["status"] == "running"
        and not job["cancel_requested"]
        and store.task(task)["desired"] == "active"
        and not store.budget_reason(store.task(task))
    )


def folder(root, task, round_id):
    return Path(root).resolve() / identifier(task) / identifier(round_id)


def read(path):
    return json.loads(path.read_text())


def result(directory, request_id):
    path = directory / "results" / (request_id + ".json")
    if not path.exists():
        return None
    raw = path.read_bytes()
    value = json.loads(raw)
    value["receipt_sha256"] = hashlib.sha256(raw).hexdigest()
    for name in ("stdout", "stderr"):
        log = directory / "logs" / f"{request_id}.{name}.log"
        if log.exists():
            with log.open("rb") as stream:
                stream.seek(max(0, log.stat().st_size - 65536))
                value[name] = stream.read().decode(errors="replace")
    return value


def request(store, root, task, round_id, owner, request_id, value=None, wait_seconds=0):
    identifier(request_id)
    if not 0 <= wait_seconds <= 60:
        raise ValueError("wait_seconds must be between 0 and 60")
    directory = folder(root, task, round_id)
    context = read(directory / "context.json")
    if context["owner"] != owner:
        raise ValueError("Request owner differs from the round host")
    target = directory / "requests" / (request_id + ".json")
    if value is not None:
        encoded = dict(value, id=request_id, round=round_id)
        with exclusive(directory / ".host.lock") as lock:
            if lock is None:
                raise ValueError("Another host operation is in progress")
            if target.exists():
                if read(target) != encoded:
                    raise ValueError("Request ID already has different content")
            else:
                if (
                    not active(store, task, round_id)
                    or read(directory / "context.json")["status"] != "ready"
                ):
                    raise ValueError("Round is no longer active")
                for previous in (directory / "requests").glob("*.json"):
                    if not (directory / "consumed" / previous.name).exists():
                        raise ValueError("Consume the previous result before submitting new work")
                kind = encoded.get("kind")
                if kind not in ("cpu", "gpu", "finish"):
                    raise ValueError("kind must be cpu, gpu or finish")
                if kind in ("cpu", "gpu"):
                    argv = encoded.get("argv")
                    if (
                        not isinstance(argv, list)
                        or not argv
                        or any(not isinstance(a, str) or not a or "\0" in a for a in argv)
                    ):
                        raise ValueError("argv must be a nonempty list of nonempty strings")
                    seconds = encoded.get("expected_seconds")
                    if type(seconds) not in (float, int) or not 0 < seconds <= 86400:
                        raise ValueError("expected_seconds must be in (0,86400]")
                atomic(target, encoded)
    elif not target.exists():
        raise ValueError("Unknown request ID; never infer a submission")
    deadline = time.monotonic() + wait_seconds
    while True:
        receipt = result(directory, request_id)
        if receipt:
            return receipt
        if (
            not active(store, task, round_id)
            or read(directory / "context.json")["status"] != "ready"
        ):
            return {
                "id": request_id,
                "status": "interrupted_without_receipt"
                if (directory / "claims" / target.name).exists()
                else "not_executed",
                "round": round_id,
            }
        atomic(directory / "host.json", {"owner": owner, "at": time.time()})
        if time.monotonic() >= deadline:
            return {"id": request_id, "status": "pending", "round": round_id}
        time.sleep(0.1)


def consume(store, root, task, round_id, owner, request_id, sha256, outcome, next_action):
    directory = folder(root, task, round_id)
    identifier(request_id)
    if read(directory / "context.json")["owner"] != owner or any(
        not isinstance(text, str) or not text.strip() or len(text) > 8000
        for text in (outcome, next_action)
    ):
        raise ValueError("Matching host, outcome and concrete next_action are required")
    value = result(directory, request_id)
    if value is None or value["receipt_sha256"] != sha256:
        raise ValueError("Read and match the completed result receipt before consuming it")
    record = {
        "request": request_id,
        "receipt_sha256": sha256,
        "owner": owner,
        "outcome": outcome,
        "next_action": next_action,
    }
    with exclusive(directory / ".host.lock") as lock:
        if lock is None:
            raise ValueError("Another host operation is in progress")
        target = directory / "consumed" / (request_id + ".json")
        if target.exists():
            if read(target) != record:
                raise ValueError("Result already consumed with a different analysis")
        else:
            if active(store, task, round_id):
                try:
                    store.report(task, round_id, {"outcome": outcome, "next_action": next_action})
                except ValueError:
                    if active(store, task, round_id):
                        raise
            atomic(target, record)
            store.event(task, "host_consumed_result", record | {"round": round_id})
        atomic(directory / "host.json", {"owner": owner, "at": time.time()})
    return record


def run_bridge(store, root, idle_seconds=120):
    if not 1 <= idle_seconds <= 300:
        raise ValueError("idle_seconds must be in [1,300]")
    task, round_id = os.environ["KERNEL_ENVOLVE_TASK"], os.environ["KERNEL_ENVOLVE_ROUND"]
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "" or os.environ.get("KERNEL_ENVOLVE_GPU_JOB"):
        raise ValueError("Bridge must run as the agent command outside gpu-run")
    claim = store.one("SELECT data FROM metadata WHERE key=?", ("host-round:" + round_id,))
    if not claim or not active(store, task, round_id):
        raise ValueError("The foreground host must open this round first")
    owner = json.loads(claim["data"])["owner"]
    directory = folder(root, task, round_id)
    if any(
        directory.is_relative_to(json.loads(row["spec"])["workspace"])
        for row in store.rows("SELECT spec FROM tasks")
    ):
        raise ValueError("Bridge artifacts must remain outside task workspaces")
    directory.mkdir(parents=True, exist_ok=False)
    for child in ("requests", "results", "claims", "consumed", "logs"):
        (directory / child).mkdir()
    (directory / "prompt.txt").write_text(sys.stdin.read())
    context = {"task": task, "round": round_id, "owner": owner, "status": "ready"}
    atomic(directory / "context.json", context)
    atomic(directory / "host.json", {"owner": owner, "at": time.time()})
    stopped = False
    child = None

    def stop(signum, frame):
        nonlocal stopped
        stopped = True
        if child is not None:
            child.terminate()

    signals = {s: signal.signal(s, stop) for s in (signal.SIGTERM, signal.SIGINT)}
    try:
        while not stopped and active(store, task, round_id):
            pending = [
                p
                for p in sorted((directory / "requests").glob("*.json"))
                if not (directory / "claims" / p.name).exists()
            ]
            if not pending:
                if time.time() - read(directory / "host.json")["at"] > idle_seconds:
                    store.event(task, "host_disconnected", {"round": round_id})
                    return 2
                time.sleep(0.1)
                continue
            path = pending[0]
            value = read(path)
            atomic(directory / "claims" / path.name, {"started_at": time.time()})
            receipt = {"id": path.stem, "round": round_id, "started_at": time.time()}
            try:
                if value["kind"] == "finish":
                    if (
                        not isinstance(value.get("report"), dict)
                        or not str(value["report"].get("next_action", "")).strip()
                    ):
                        raise ValueError("Finish requires a concrete next_action checkpoint")
                    store.report(task, round_id, value["report"])
                    atomic(Path(os.environ["KERNEL_ENVOLVE_REPORT"]), value["report"])
                    receipt.update(status="succeeded", returncode=0, finished_at=time.time())
                    atomic(directory / "results" / path.name, receipt)
                    return 0
                argv = value["argv"]
                if value["kind"] == "gpu":
                    release = os.environ.get("KERNEL_ENVOLVE_RELEASE")
                    cli = (
                        [str(Path(release) / ".entry/kernel-envolve")]
                        if release
                        else [sys.executable, "-m", "kernel_envolve.cli"]
                    )
                    argv = [
                        *cli,
                        "manage",
                        "gpu-run",
                        "--expected-seconds",
                        str(value["expected_seconds"]),
                        "--",
                        *argv,
                    ]
                with (
                    (directory / "logs" / f"{path.stem}.stdout.log").open("wb") as out,
                    (directory / "logs" / f"{path.stem}.stderr.log").open("wb") as err,
                ):
                    child = subprocess.Popen(argv, stdin=subprocess.DEVNULL, stdout=out, stderr=err)
                    start = time.monotonic()
                    while child.poll() is None:
                        if (
                            stopped
                            or not active(store, task, round_id)
                            or (
                                value["kind"] == "cpu"
                                and time.monotonic() - start > value["expected_seconds"]
                            )
                        ):
                            child.terminate()
                            stopped = True
                            break
                        time.sleep(0.1)
                    try:
                        code = child.wait(timeout=3)
                    except subprocess.TimeoutExpired:
                        child.kill()
                        code = child.wait()
                child = None
                receipt.update(
                    status="succeeded" if code == 0 and not stopped else "failed",
                    returncode=code,
                    finished_at=time.time(),
                )
                atomic(directory / "results" / path.name, receipt)
            except (OSError, ValueError, KeyError, TypeError) as exc:
                receipt.update(
                    status="failed", error=str(exc), returncode=None, finished_at=time.time()
                )
                atomic(directory / "results" / path.name, receipt)
        return 2
    finally:
        if child is not None and child.poll() is None:
            child.terminate()
            with contextlib.suppress(subprocess.TimeoutExpired):
                child.wait(timeout=3)
            if child.poll() is None:
                child.kill()
                child.wait()
        atomic(
            directory / "context.json", context | {"status": "closed", "finished_at": time.time()}
        )
        for sig, handler in signals.items():
            signal.signal(sig, handler)
