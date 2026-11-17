"""Verified, complete research episodes for optional background retrieval experiments.

This is a data archive, not a policy publisher or an executable tool library. It
does not infer causal repair edges, new-task performance, or permission to execute
saved code. No production foreground path imports this module.
"""

import json
import math
import os
import re
import stat
import statistics
import tempfile
from pathlib import Path

from ..manager.evidence import inspect_run
from ..storage import config, contract, digest
from .model import profile, read, text
from .research import source_id

MAX_EPISODE_BYTES = 384 * 1024
MAX_SOURCE_BYTES = 24000
MAX_VIEW_BYTES = 32768


def _source(path):
    fd = os.open(path, os.O_RDONLY | os.O_NONBLOCK | os.O_NOFOLLOW)
    with os.fdopen(fd, "rb") as stream:
        if not stat.S_ISREG(os.fstat(stream.fileno()).st_mode):
            raise ValueError("Episode source must be a regular file")
        raw = stream.read(MAX_SOURCE_BYTES + 1)
    if len(raw) > MAX_SOURCE_BYTES:
        raise ValueError("Episode source exceeds limit; incomplete code is not archived")
    return raw.decode("utf-8")


def _identity(value):
    if not isinstance(value, str) or not re.fullmatch(r"[a-f0-9]{64}", value):
        raise ValueError("Invalid episode/source identity")
    return value


def _encoded(value):
    raw = json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False).encode()
    if len(raw) > MAX_EPISODE_BYTES:
        raise ValueError("Complete episode exceeds archive limit")
    return raw


def _reported_seconds(value):
    if value is not None and (
        type(value) not in (int, float) or not math.isfinite(value) or value < 0
    ):
        raise ValueError("Invalid reported trace duration")
    return value


def build(task, artifacts, task_profile, *, origin_task=None, confirmation_run=None):
    """Recompute all measured outcomes; trace prose/claimed scores are not evidence.

    origin_task is an operator-supplied path mapping for an offline archive import.
    It maps one original task directory to one local task directory, never arbitrary
    paths. Production callers omit it. Missing/partial artifacts fail closed.
    """
    task, artifacts = Path(task).resolve(), Path(artifacts).resolve()
    # Keep the producer's spelling for archive paths (Linux /tmp becomes
    # /private/tmp under macOS resolve). inspect_run separately canonicalizes it.
    origin = Path(origin_task) if origin_task is not None else task
    if not origin.is_absolute() or ".." in origin.parts:
        raise ValueError("Origin task must be an absolute, normalized archive path")
    cfg = config(task)
    if cfg["candidate_files"] != ["candidate.py"]:
        raise ValueError("Episodes currently require a single generic candidate.py")
    spec = {
        "kind": "generic",
        "task_dir": str(origin.resolve()),
        "baseline": {"contract": contract(task, cfg)},
    }
    trace = read(artifacts / "search-trace.json", MAX_EPISODE_BYTES)
    if trace.get("error"):
        raise ValueError("Incomplete research trace; retain raw failure logs outside this index")
    steps = trace["steps"]
    if not isinstance(steps, list) or len(steps) > 8:
        raise ValueError("Episode exceeds eight research actions")
    observations, sources, actions = [], {}, []

    def measured(observed):
        if observed.get("status") != "evaluated":
            raise ValueError("Only actually evaluated observations can enter the archive")
        relative = Path(observed["run"]).relative_to(origin / ".kernel_envolve/runs")
        if len(relative.parts) != 1 or not re.fullmatch(r"[A-Za-z0-9_-]+", relative.name):
            raise ValueError("Episode run must be directly inside the task run archive")
        run = task / ".kernel_envolve/runs" / relative
        if not run.resolve().is_relative_to(task) or run.is_symlink():
            raise ValueError("Episode run escapes the imported task")
        verified = inspect_run(run, spec)
        identity = _identity(observed["candidate"])
        source = _source(run / "source/candidate.py")
        if (
            source_id(source) != identity
            or source != _source(artifacts / f"candidate-{identity}.py")
            or observed["mode"] != verified["mode"]
        ):
            raise ValueError("Episode source/evaluation identity mismatch")
        sources[identity] = source
        passed = verified["execution"] == "completed" and verified["correctness"] == "pass"
        gains = verified["decision"].get("fixed_baseline_gain_by_batch", [])
        quality = (
            max(0.0, statistics.median(gains))
            if passed and gains and verified["decision"]["eligible_for_best"]
            else 0.0
            if passed
            else -1.0
        )
        observations.append(
            {
                "source": identity,
                "run": relative.name,
                "result_hash": verified["result_hash"],
                "mode": verified["mode"],
                "execution": verified["execution"],
                "correctness": verified["correctness"],
                "quality": quality,
                "decision": verified["decision"],
                "failed_checks": verified["failed_checks"],
                "evaluator_hash": verified["evaluator_hash"],
                "environment": verified["environment"],
            }
        )
        return len(observations) - 1

    measured(trace["initial_observation"])
    if observations[0]["source"] != trace["initial"]:
        raise ValueError("Episode initial source mismatch")
    for step in steps:
        text(step["hypothesis"], "untrusted hypothesis", 1200)
        action = {"action": step["action"], "untrusted_hypothesis": step["hypothesis"]}
        if "decision_error" in step:
            if (
                step["decision_error"] != {"type": "TimeoutError", "tokens": None}
                or step.get("stop_origin") != "controller"
                or step["action"] != "stop"
            ):
                raise ValueError("Invalid reported decision timeout")
            action["reported_decision_error"] = step["decision_error"]
            action["reported_stop_origin"] = "controller"
        if "decision_kind" in step:
            if step["decision_kind"] not in {"adapter", "artifact_proposal"}:
                raise ValueError("Unsupported reported decision kind")
            action["reported_decision_kind"] = step["decision_kind"]
            if step.get("artifact_episode_id") is not None:
                action["reported_artifact_episode_id"] = _identity(step["artifact_episode_id"])
        if step["action"] not in {"propose", "evaluate", "stop"}:
            raise ValueError("Unsupported episode action")
        observed = step.get("observation")
        if step["action"] == "stop" and observed:
            raise ValueError("Stop action cannot claim a measurement")
        if observed and observed.get("status") == "evaluated":
            action["observation"] = measured(observed)
        elif observed and observed.get("status") == "duplicate_skipped":
            if not any(
                row["source"] == observed["candidate"] and row["mode"] == observed["mode"]
                for row in observations
            ):
                raise ValueError("Duplicate claim has no prior measurement")
            action["measurement_status"] = "duplicate_skipped"
        elif observed or step["action"] != "stop":
            raise ValueError("Incomplete episode action; missing outcome is not a success")
        actions.append(action)
    full = [row for row in observations if row["mode"] == "full"]
    if not full or trace["selected"] != max(full, key=lambda row: row["quality"])["source"]:
        raise ValueError("Episode selection is not the best observed full candidate")
    confirmation = None
    if confirmation_run is not None:
        if Path(confirmation_run).name in {row["run"] for row in observations}:
            raise ValueError("Final confirmation must be a separate evaluation")
        confirmation = measured(
            {
                "status": "evaluated",
                "run": str(confirmation_run),
                "candidate": trace["selected"],
                "mode": "full",
            }
        )
    value = {
        "version": 1,
        "profile": profile(task_profile),
        "origin_task": str(origin),
        "contract": spec["baseline"]["contract"],
        "task_source": _source(task / "task.py"),
        "task_config": cfg,
        "sources": sources,
        "observations": observations,
        "actions": actions,
        "selected": trace["selected"],
        "stop_reason": trace.get("stop_reason"),
        "final_confirmation": confirmation,
        "reported_cost": {
            "trace_seconds": _reported_seconds(trace.get("seconds")),
            "model_seconds_by_action": [
                _reported_seconds(step.get("model_seconds")) for step in steps
            ],
            "model_tokens": None,
            "authority": "Trace-reported durations, not independently verified API usage. Null means unknown, never zero. External all-attempt accounting remains required.",
        },
        "scope": "Historical measurements only; sequence does not prove causal repair. Saved code is untrusted data, never validated on a new task by this archive.",
    }
    _encoded(value)
    return value


def save(directory, episode):
    """Publish a content-addressed data file without overwriting an existing file."""
    raw = _encoded(episode)
    identity = digest(episode)
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=directory, delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temporary, directory / f"{identity}.json")
        except FileExistsError:
            load(directory, identity)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return identity


def load(directory, identity):
    """Integrity check only; callers must pin approved IDs and audit source receipts."""
    value = read(Path(directory) / f"{_identity(identity)}.json", MAX_EPISODE_BYTES)
    if digest(value) != identity:
        raise ValueError("Episode content changed")
    return value


def view(directory, identity):
    """Bounded model-facing facts with complete code, including losing attempts.

    No advice or causal explanation is generated here. All observations and case
    timing rows remain represented. Oversized views are rejected, never silently
    shortened to a partial program. The complete archive retains noise statistics.
    """
    episode = load(directory, identity)
    observations = []
    for row in episode["observations"]:
        decision = row["decision"]
        observations.append(
            {
                **{
                    k: row[k]
                    for k in (
                        "source",
                        "mode",
                        "execution",
                        "correctness",
                        "quality",
                        "failed_checks",
                    )
                },
                "eligible_for_best": decision["eligible_for_best"],
                "performance": decision.get("performance"),
                "cases": [
                    {
                        key: case[key]
                        for key in (
                            "case",
                            "batch",
                            "candidate_us",
                            "comparator_us",
                            "baseline_us",
                            "speedup",
                        )
                        if key in case
                    }
                    for case in decision.get("cases", [])
                ],
            }
        )
    value = {
        "episode_id": identity,
        **{
            key: episode[key]
            for key in (
                "profile",
                "task_source",
                "task_config",
                "sources",
                "actions",
                "selected",
                "final_confirmation",
                "reported_cost",
                "scope",
            )
        },
        "observations": observations,
        "omitted": "Detailed noise statistics and environment remain in the pinned archive; no sources or observed attempts are omitted.",
    }
    if len(json.dumps(value, allow_nan=False).encode()) > MAX_VIEW_BYTES:
        raise ValueError("Complete episode view exceeds model input limit")
    return value
