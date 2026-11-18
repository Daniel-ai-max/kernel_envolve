"""Durable Codex host adapter; retry decisions remain with HostRecovery and the broker."""

import json
import os
from pathlib import Path

from kernel_envolve._io import durable_json
from kernel_envolve.manager.config import identifier
from kernel_envolve.manager.execution.host_attempt import HostAttempt, model_result, read_events
from kernel_envolve.manager.execution.host_model import codex_command


class ModelHost:
    """One attempt at a time, under the deployment's exclusive broker lock.

    restore() inspects the original attempt; it never starts a process. The broker
    must reconcile outstanding bridge IDs before admitting the next attempt.
    """

    def __init__(
        self, root, trial, codex="codex", writable=(), model_options=None, idle_seconds=300
    ):
        self.root, self.trial = Path(root), Path(trial)
        self.root.mkdir(parents=True, exist_ok=True)
        self.codex, self.writable = codex, writable
        self.idle_seconds = idle_seconds
        self.model_options = dict(model_options or {})
        if not self.model_options.get("model"):
            raise ValueError("Explicit research model required")
        self.process = None
        self.request = None
        self.directory = None

    def _session(self, observed=None):
        path = self.root / "session.json"
        saved = json.loads(path.read_text())["session_id"] if path.exists() else None
        if observed and saved and observed != saved:
            raise ValueError("Model session identity changed")
        if observed and not saved:
            durable_json(path, {"session_id": observed})
        return observed or saved

    def start(self, request, prompt, deadline):
        if self.process:
            raise ValueError("Model already owns an attempt")
        identifier(request["id"])
        options = dict(self.model_options)
        key_file = options.pop("key_file", None)
        session = self._session()
        argv = codex_command(codex=self.codex, writable=self.writable, session=session, **options)
        env = os.environ.copy()
        if options.get("provider"):
            key_name = options["provider"]["env_key"]
            if key_file:
                path = Path(key_file)
                if path.stat().st_mode & 0o077:
                    raise ValueError("Provider key file must be private")
                env[key_name] = path.read_text().strip()
            if not env.get(key_name):
                raise ValueError("Provider credential is unavailable")
        self.request = request
        self.directory = self.root / "turns" / request["id"]
        self.process = HostAttempt.start(
            self.directory,
            argv=argv,
            prompt=prompt,
            cwd=self.trial,
            env=env,
            deadline=deadline,
            metadata={"request": request, "session": session, "model_selection": options},
            idle_seconds=self.idle_seconds,
        )

    def restore(self, attempt_id):
        if self.process:
            raise ValueError("Model already owns an attempt")
        identifier(attempt_id)
        directory = self.root / "turns" / attempt_id
        intent = json.loads((directory / "intent.json").read_text())
        metadata = intent["metadata"]
        if metadata["model_selection"]["model"] != self.model_options["model"]:
            raise ValueError("Cannot restore a different research model")
        if metadata["request"]["id"] != attempt_id:
            raise ValueError("Attempt identity changed")
        if metadata.get("session"):
            self._session(metadata["session"])
        self.directory, self.request = directory, metadata["request"]
        self.process = HostAttempt(directory)
        self.snapshot()
        return intent

    def snapshot(self):
        if not self.process:
            return None
        events = read_events(self.directory / "trace.jsonl")
        sessions = {e.get("thread_id") for e in events if e.get("type") == "thread.started"}
        if len(sessions) > 1 or any(not isinstance(s, str) or not s for s in sessions):
            raise ValueError("Invalid model session evidence")
        session = self._session(next(iter(sessions)) if sessions else None)
        record = self.directory / "process.json"
        value = json.loads(record.read_text()) if record.exists() else {}
        return {"id": self.request["id"], "session_id": session, "pid": value.get("pid")}

    def terminate(self):
        if self.process:
            self.process.stop()

    def collect(self):
        if not self.process:
            return None
        result = model_result(self.directory)
        if result is None:
            self.snapshot()
            return None
        if result["session_id"]:
            self._session(result["session_id"])
        self.process.stop()
        self.process = None
        return result
