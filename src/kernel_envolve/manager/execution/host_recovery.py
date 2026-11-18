"""Durable, bounded same-session recovery for a deployment's existing model host.

This is an admission ledger, not a second scheduler. The host owns its process
lock, persists raw traces, reconciles bridge requests, then resumes the model.
"""

import json
import math
from pathlib import Path

from kernel_envolve._io import durable_json
from kernel_envolve.manager.execution.host_model import terminal_failure


def transient_failure(events):
    terminal = next(
        (e for e in reversed(events) if e.get("type") in ("turn.failed", "turn.completed")),
        None,
    )
    if not terminal or terminal["type"] != "turn.failed":
        return False
    error = terminal.get("error")
    if not isinstance(error, dict):
        return False
    message = str(error.get("message", "")).lower()
    return error.get("code") in {"rate_limit_exceeded", "network_error", "timeout"} or any(
        text in message
        for text in ("stream disconnected", "network timeout", "error sending request", "http 429")
    )


class HostRecovery:
    """Use under the adapter's exclusive owner lock; identities never reset on restart."""

    def __init__(self, path, *, model, session, round_id, deadline, max_retries=2):
        if not model or not session or not round_id:
            raise ValueError("Recovery requires original model, session and round identities")
        if not math.isfinite(deadline) or type(max_retries) is not int or max_retries < 0:
            raise ValueError("Recovery must be bounded")
        self.path = Path(path)
        identity = dict(
            model=model, session=session, round=round_id, deadline=deadline, max_retries=max_retries
        )
        if self.path.exists():
            self.data = json.loads(self.path.read_text())
            if self.data["identity"] != identity:
                raise ValueError("Original recovery identity/budget cannot change")
        else:
            self.data = dict(identity=identity, retries=0, state="running", attempts=[])
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.save()

    def save(self):
        durable_json(self.path, self.data)

    def failed(self, events, returncode, *, trace_sha256, now):
        if not math.isfinite(now) or not trace_sha256:
            raise ValueError("Failure requires an observation time and raw trace digest")
        if self.data["state"] != "running":
            raise ValueError("An unresolved failure cannot be replaced")
        fatal = terminal_failure(events, returncode)
        terminal = next(
            (e for e in reversed(events) if e.get("type") in ("turn.failed", "turn.completed")),
            None,
        )
        # Unknown provider failures also get the same small recovery allowance;
        # they are not relabelled as network errors. Missing/malformed terminal
        # evidence and explicit credential/quota failures remain fail-closed.
        retryable = bool(
            not fatal
            and terminal
            and terminal["type"] == "turn.failed"
            and isinstance(terminal.get("error"), dict)
        )
        self.data["attempts"].append(
            dict(trace_sha256=trace_sha256, at=now, returncode=returncode, retryable=retryable)
        )
        self.data["state"] = "recovering" if retryable else "interrupted"
        self.data["reason"] = (
            fatal["category"]
            if fatal
            else "model_transport_failure"
            if transient_failure(events)
            else "model_execution_failure"
        )
        self.data["retry_at"] = now + min(30, 5 * 2 ** self.data["retries"])
        self.save()

    def admit(self, *, now, active, reconciled, session, model, reserve_seconds=90):
        identity = self.data["identity"]
        if session != identity["session"] or model != identity["model"]:
            raise ValueError("Recovery cannot change the research model or session")
        if not all(math.isfinite(x) for x in (now, reserve_seconds)) or reserve_seconds < 0:
            raise ValueError("Invalid cleanup reserve")
        if self.data["state"] == "interrupted":
            return "interrupt"
        if not active or now >= identity["deadline"] - reserve_seconds:
            self.data.update(state="interrupted", reason="stopped_or_budget_exhausted")
        elif self.data["state"] != "recovering":
            raise ValueError("No failed attempt to resume")
        elif self.data["retries"] >= identity["max_retries"]:
            self.data.update(state="interrupted", reason="same_model_recovery_exhausted")
        elif now < self.data["retry_at"] or not reconciled:
            return "wait"
        else:
            self.data["retries"] += 1
            self.data["state"] = "running"
            self.save()  # Crash after admission consumes the attempt, never silently replays it.
            return "resume"
        self.save()
        return "interrupt"
