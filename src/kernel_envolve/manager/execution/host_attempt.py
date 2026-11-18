"""One durable external-model attempt, not a retry loop or scheduler.

Use under the broker's exclusive owner lock. A separate supervisor records exit
and cleanup even if the broker dies. Unknown process ownership fails closed.
"""

import hashlib
import json
import math
import os
import selectors
import signal
import subprocess
import sys
import time
from pathlib import Path

from kernel_envolve._io import durable_json
from kernel_envolve.manager.execution.host_model import terminal_failure
from kernel_envolve.manager.execution.process import (
    exclusive,
    group_alive,
    identity,
    stop_recorded,
    terminate,
)


def read_events(path):
    """Keep partial final lines in the raw trace, but never parse them as events."""
    if not Path(path).exists():
        return []
    result = []
    for line in Path(path).read_bytes().splitlines(keepends=True):
        if not line.endswith(b"\n"):
            continue
        try:
            value = json.loads(line)
            if isinstance(value, dict):
                result.append(value)
        except ValueError:
            pass
    return result


class HostAttempt:
    def __init__(self, directory):
        self.directory = Path(directory)
        self.process = None
        self.lease = None

    @classmethod
    def start(cls, directory, *, argv, prompt, cwd, deadline, metadata, env=None, idle_seconds=300):
        if not math.isfinite(deadline) or deadline <= time.time():
            raise ValueError("Original model deadline expired")
        if (
            type(idle_seconds) not in (int, float)
            or not math.isfinite(idle_seconds)
            or idle_seconds <= 0
        ):
            raise ValueError("Model silence must have a finite positive limit")
        self = cls(directory)
        self.directory.mkdir(parents=True, exist_ok=False)
        (self.directory / "prompt.txt").write_text(prompt)
        durable_json(
            self.directory / "intent.json",
            {
                "argv": argv,
                "cwd": str(cwd),
                "deadline": deadline,
                "idle_seconds": idle_seconds,
                "created_at": time.time(),
                "metadata": metadata,
                "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
            },
        )
        with exclusive(self.directory / "lifetime.lock") as lock:
            if lock is None:
                raise ValueError("Model attempt already owned")
            reader, self.lease = os.pipe()
            try:
                self.process = subprocess.Popen(
                    [sys.executable, "-m", __name__, str(self.directory), str(reader)],
                    pass_fds=(reader, lock.fileno()),
                    start_new_session=True,
                    stdin=subprocess.DEVNULL,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    env=env,
                )
                durable_json(
                    self.directory / "supervisor.json",
                    {
                        "pid": self.process.pid,
                        "identity": identity(self.process.pid, include_zombie=True),
                    },
                )
                # Do not launch the model before its supervisor registration is durable.
                os.write(self.lease, b"1")
            except BaseException:
                self.stop()
                raise
            finally:
                os.close(reader)
        return self

    def stop(self):
        if self.lease is not None:
            os.close(self.lease)
            self.lease = None
        if self.process is not None:
            try:
                self.process.wait(timeout=8)
            except subprocess.TimeoutExpired:
                terminate(self.process)

    def outcome(self):
        # The lock outlives the broker and the exit receipt. Never resume until
        # the supervisor has finished cleanup and closed its inherited descriptor.
        with exclusive(self.directory / "lifetime.lock") as lock:
            if lock is None:
                return None
            path = self.directory / "exit.json"
            if path.exists():
                return json.loads(path.read_text())
            process = self.directory / "process.json"
            if process.exists():
                recorded = json.loads(process.read_text())
                stop_recorded(recorded["pid"], recorded["identity"])
            intent = json.loads((self.directory / "intent.json").read_text())
            return {
                "returncode": None,
                "cleanup_confirmed": False,
                "reason": "supervisor_receipt_missing",
                "finished_at": intent["created_at"],
            }


def model_result(directory):
    """Collect or replay one immutable model result, including after broker restart."""
    directory = Path(directory)
    outcome = HostAttempt(directory).outcome()
    if outcome is None:
        return None
    saved = directory / "result.json"
    if saved.exists():
        return json.loads(saved.read_text())
    intent = json.loads((directory / "intent.json").read_text())
    metadata = intent["metadata"]
    trace = directory / "trace.jsonl"
    raw = trace.read_bytes() if trace.exists() else b""
    events = read_events(trace)
    sessions = {e.get("thread_id") for e in events if e.get("type") == "thread.started"}
    session = metadata.get("session")
    if session:
        sessions.add(session)
    sessions.discard(None)
    session = next(iter(sessions)) if len(sessions) == 1 else None
    code = outcome["returncode"]
    last = next(
        (e for e in reversed(events) if e.get("type") in {"turn.failed", "turn.completed"}), {}
    )
    status = (
        "succeeded"
        if (
            code == 0
            and last.get("type") == "turn.completed"
            and outcome["reason"] in {"exit", "owner_lost"}
        )
        else "failed"
    )
    exit_code = code if type(code) is int and code >= 0 else 128 - code if type(code) is int else 78
    if status == "failed" and exit_code == 0:
        exit_code = 2
    failure = terminal_failure(events, code)
    active_tools = set()
    for event in events:
        item = event.get("item", {})
        if isinstance(item, dict) and item.get("type") == "command_execution":
            if event.get("type") == "item.started":
                active_tools.add(item.get("id"))
            elif event.get("type") == "item.completed":
                active_tools.discard(item.get("id"))
    if active_tools:
        # A killed CLI cannot confirm cleanup of tools that created their own
        # sessions. Original bridge IDs must still be reconciled; never claim a
        # clean resumable boundary from the model leader's exit alone.
        failure = {
            "category": "host_tool_cleanup_unconfirmed",
            "exit_code": 78,
            "retryable": False,
            "process_exit_code": code,
        }
    if not outcome["cleanup_confirmed"] or not session:
        failure = {
            "category": "host_cleanup_unconfirmed"
            if not outcome["cleanup_confirmed"]
            else "host_session_unavailable",
            "exit_code": 78,
            "retryable": False,
            "process_exit_code": code,
        }
    if failure:
        status, exit_code = "failed", failure["exit_code"]
    if outcome["reason"] == "deadline":
        status, exit_code = "cancelled", 124
    messages = [
        e["item"]["text"]
        for e in events
        if e.get("type") == "item.completed" and e.get("item", {}).get("type") == "agent_message"
    ]
    stderr = directory / "stderr.log"
    value = dict(
        schema_version=1,
        id=directory.name,
        status=status,
        exit_code=exit_code,
        process_exit_code=code,
        process_exited=outcome["cleanup_confirmed"],
        termination_reason=outcome["reason"],
        stdout="\n".join(messages)[-16000:],
        stderr=stderr.read_text(errors="replace")[-8000:] if stderr.exists() else "",
        started_at=intent["created_at"],
        finished_at=outcome["finished_at"],
        session_id=session,
        trace_path=str(trace),
        trace_sha256=hashlib.sha256(raw).hexdigest(),
        usage=[e.get("usage") for e in events if e.get("type") == "turn.completed"],
    )
    if failure:
        value["model_failure"] = failure
        if failure["category"] == "host_tool_cleanup_unconfirmed":
            value["process_exited"] = False
    durable_json(saved, value)
    return value


def supervise(directory, lease):
    directory = Path(directory)
    intent = json.loads((directory / "intent.json").read_text())
    deadline = intent["deadline"]
    monotonic_end = time.monotonic() + max(0, deadline - time.time())
    last_output = time.monotonic()
    stopped = False

    def stop(*_):
        nonlocal stopped
        stopped = True

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    selector = selectors.DefaultSelector()
    selector.register(lease, selectors.EVENT_READ, "lease")
    child = None
    code = None
    reason = "launch_not_started"
    cleanup = True
    pending = b""
    try:
        # Parent death before the handshake cannot launch an unregistered model.
        if not selector.select(max(0, deadline - time.time())) or os.read(lease, 1) != b"1":
            return
        with (
            (directory / "prompt.txt").open("rb") as prompt,
            (directory / "trace.jsonl").open("wb") as trace,
            (directory / "timed-trace.jsonl").open("w") as timed,
            (directory / "stderr.log").open("wb") as err,
        ):
            if time.time() >= deadline:
                reason = "deadline"
                return
            child = subprocess.Popen(
                intent["argv"],
                cwd=intent["cwd"],
                stdin=prompt,
                stdout=subprocess.PIPE,
                stderr=err,
                start_new_session=True,
            )
            durable_json(
                directory / "process.json",
                {
                    "pid": child.pid,
                    "identity": identity(child.pid, include_zombie=True),
                    "started_at": time.time(),
                },
            )
            selector.register(child.stdout, selectors.EVENT_READ, "output")
            reason = "exit"
            while True:
                if stopped:
                    if reason != "owner_lost":
                        reason = "owner_stopped"
                    break
                if time.time() >= deadline or time.monotonic() >= monotonic_end:
                    reason = "deadline"
                    break
                if time.monotonic() - last_output >= intent["idle_seconds"]:
                    reason = "model_stalled"
                    break
                for key, _ in selector.select(0.05):
                    if key.data == "lease":
                        if not os.read(lease, 1):
                            reason = "owner_lost"
                            stopped = True
                        continue
                    chunk = os.read(child.stdout.fileno(), 65536)
                    if not chunk:
                        selector.unregister(child.stdout)
                        continue
                    trace.write(chunk)
                    trace.flush()
                    last_output = time.monotonic()
                    pending += chunk
                    while b"\n" in pending:
                        line, pending = pending.split(b"\n", 1)
                        try:
                            event = json.loads(line)
                        except ValueError:
                            event = {"unparsed": line.decode(errors="replace")}
                        timed.write(json.dumps({"observed_at": time.time(), "event": event}) + "\n")
                        timed.flush()
                if child.poll() is not None:
                    # Exit is not enough: descendants may still hold stdout open.
                    terminate(child, grace=0.2)
                    for key in list(selector.get_map().values()):
                        if key.data == "output":
                            os.set_blocking(child.stdout.fileno(), False)
                            while chunk := child.stdout.read(65536):
                                trace.write(chunk)
                            selector.unregister(child.stdout)
                    break
            # Preserve owner-loss reason even when child exit races with the lease EOF.
            if stopped and reason == "exit":
                reason = "owner_lost"
    except (OSError, ValueError, subprocess.SubprocessError):
        reason = "supervisor_error"
        cleanup = False
    finally:
        try:
            if child is not None:
                terminate(child, grace=0.2)
                code = child.wait(timeout=2)
                until = time.monotonic() + 2
                while group_alive(child.pid) and time.monotonic() < until:
                    time.sleep(0.05)
                cleanup = cleanup and not group_alive(child.pid)
        except (OSError, ValueError, subprocess.SubprocessError):
            cleanup = False
        selector.close()
        os.close(lease)
        durable_json(
            directory / "exit.json",
            {
                "returncode": code,
                "cleanup_confirmed": cleanup,
                "reason": reason,
                "finished_at": time.time(),
            },
        )


if __name__ == "__main__":
    supervise(sys.argv[1], int(sys.argv[2]))
