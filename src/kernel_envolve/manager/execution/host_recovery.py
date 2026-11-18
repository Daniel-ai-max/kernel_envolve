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

    def __init__(
        self, path, *, model, session, round_id, deadline, max_retries=2, recovery_seconds=180
    ):
        if not model or not session or not round_id:
            raise ValueError("Recovery requires original model, session and round identities")
        if not math.isfinite(deadline) or type(max_retries) is not int or max_retries < 0:
            raise ValueError("Recovery must be bounded")
        if not math.isfinite(recovery_seconds) or recovery_seconds <= 0:
            raise ValueError("Recovery wait must be bounded")
        self.path = Path(path)
        identity = dict(
            model=model,
            session=session,
            round=round_id,
            deadline=deadline,
            max_retries=max_retries,
            recovery_seconds=recovery_seconds,
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

    def progress(self, events, *, now, trace_sha256):
        """End an outage only on model output/tool execution, never broker heartbeats.

        The total round retry count stays spent. A later distinct outage gets its
        own bounded window rather than inheriting a stale, already-expired window.
        """
        if not math.isfinite(now) or not trace_sha256:
            raise ValueError("Progress requires a timestamp and raw trace digest")
        if self.data["state"] != "running" or "recovery_started_at" not in self.data:
            return False
        meaningful = any(
            e.get("type") == "item.completed"
            and isinstance(e.get("item"), dict)
            and (
                (e["item"].get("type") == "agent_message" and e["item"].get("text", "").strip())
                or (
                    e["item"].get("type") == "command_execution"
                    and e["item"].get("status") == "completed"
                    and type(e["item"].get("exit_code")) is int
                )
            )
            for e in events
        )
        if not meaningful:
            return False
        self.data.setdefault("recoveries", []).append(dict(at=now, trace_sha256=trace_sha256))
        del self.data["recovery_started_at"]
        self.save()
        return True

    def failed(self, events, returncode, *, trace_sha256, now, process_exited=False):
        if not math.isfinite(now) or not trace_sha256:
            raise ValueError("Failure requires an observation time and raw trace digest")
        if self.data["state"] != "running":
            raise ValueError("An unresolved failure cannot be replaced")
        fatal = terminal_failure(events, returncode)
        terminal = next(
            (e for e in reversed(events) if e.get("type") in ("turn.failed", "turn.completed")),
            None,
        )
        # A killed process cannot emit turn.failed. Only an independent supervisor's
        # exit + cleanup receipt may authorize that path, never silence/heartbeat loss.
        retryable = bool(
            not fatal
            and returncode != 78
            and (
                (
                    terminal
                    and terminal["type"] == "turn.failed"
                    and isinstance(terminal.get("error"), dict)
                )
                or (
                    process_exited is True
                    and type(returncode) is int
                    and (not terminal or terminal["type"] != "turn.completed" or returncode != 0)
                )
            )
        )
        self.data["attempts"].append(
            dict(
                trace_sha256=trace_sha256,
                at=now,
                returncode=returncode,
                retryable=retryable,
                process_exited=process_exited,
            )
        )
        self.data["state"] = "recovering" if retryable else "interrupted"
        self.data["reason"] = (
            fatal["category"]
            if fatal
            else "model_transport_failure"
            if transient_failure(events)
            else "model_process_exited"
            if process_exited and not terminal
            else "model_execution_failure"
        )
        self.data["retry_at"] = now + min(30, 5 * 2 ** self.data["retries"])
        self.data.setdefault("recovery_started_at", now)
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
        elif now >= self.data["recovery_started_at"] + identity["recovery_seconds"]:
            self.data.update(state="interrupted", reason="recovery_window_exhausted")
        elif self.data["retries"] >= identity["max_retries"]:
            self.data.update(state="interrupted", reason="same_model_recovery_exhausted")
        elif now < self.data["retry_at"] or not reconciled:
            return "wait"
        else:
            self.data["retries"] += 1
            self.data["state"] = "running"
            self.data["admitted_at"] = now
            self.save()  # Crash after admission consumes the attempt, never silently replays it.
            return "resume"
        self.save()
        return "interrupt"
