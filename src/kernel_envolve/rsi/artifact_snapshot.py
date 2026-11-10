"""Bounded, operator-pinned artifact views, built from independently verified episodes.

Materialization is background work, not admission or publication. A caller must
explicitly pin an approved identity. No production Manager path reads these files.
The optional research path uses its existing initial full result for compatibility;
historical evidence never substitutes for a new full or a separate final full.
"""

import json
from pathlib import Path

from ..storage import digest
from .episodes import _identity, build, save
from .model import keys, profile, read
from .research import source_id
from .reuse import runtime_key

MAX_SNAPSHOT_BYTES = 32768
COMPATIBILITY = (
    "reference_files",
    "candidate_files",
    "device",
    "rtol",
    "atol",
    "equal_nan",
    "mutable_inputs",
    "deterministic",
)


def materialize(task, artifacts, task_profile, directory, *, confirmation_run):
    """Verify original receipts and save one immutable unexecuted candidate view.

    This does not update an active pointer, rank episodes, or certify future gain.
    A failed independent final cannot create a view, even if search once passed.
    """
    task, directory = Path(task).resolve(), Path(directory).resolve()
    if directory.is_relative_to(task):
        raise ValueError("Artifact snapshots must be outside the candidate workspace")
    if confirmation_run is None:
        raise ValueError("Artifact snapshot requires independent final confirmation")
    episode = build(task, artifacts, task_profile, confirmation_run=confirmation_run)
    final = episode["observations"][episode["final_confirmation"]]
    if (
        final["mode"] != "full"
        or final["execution"] != "completed"
        or final["correctness"] != "pass"
        or not final["decision"]["eligible_for_best"]
        or final["source"] != episode["selected"]
    ):
        raise ValueError("Artifact final confirmation did not qualify")
    cfg = episode["task_config"]
    if cfg["reference_files"] != ["task.py"] or cfg["candidate_files"] != ["candidate.py"]:
        raise ValueError("Artifact snapshots require single-file reference and candidate")
    value = {
        "schema": 1,
        "profile": episode["profile"],
        "compatibility": {k: cfg[k] for k in COMPATIBILITY},
        "runtime": runtime_key(final["environment"]),
        "evaluator_hash": final["evaluator_hash"],
        "origin_contract": episode["contract"],
        "confirmation_result_hash": final["result_hash"],
        "proposal": {
            "episode_id": digest(episode),
            "source_hash": episode["selected"],
            "task_source_hash": source_id(episode["task_source"]),
            "source": episode["sources"][episode["selected"]],
        },
    }
    if len(json.dumps(value["proposal"], allow_nan=False).encode()) > MAX_SNAPSHOT_BYTES:
        raise ValueError("Artifact proposal exceeds research transport bound")
    if (
        len(json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False).encode())
        > MAX_SNAPSHOT_BYTES
    ):
        raise ValueError("Complete artifact snapshot exceeds foreground bound")
    # Keep complete success/failure observations behind the bounded view. An
    # interrupted write may leave an orphan data archive, never an active pointer.
    episode_id = save(directory / "episodes", episode)
    if episode_id != value["proposal"]["episode_id"]:
        raise ValueError("Artifact episode identity changed during materialization")
    identity = save(directory, value)
    return {"path": str(directory / f"{identity}.json"), "identity": identity}


def proposal(path, identity, task_profile, cfg, initial_result):
    """One bounded file read; return (proposal, reason), falling back on mismatch.

    initial_result is the trusted current evaluator result already obtained by the
    caller, not model prose or an archived environment. A pinned identity provides
    integrity, not publisher trust or revocation; approval belongs to the caller.
    """
    try:
        _identity(identity)
        value = read(path, MAX_SNAPSHOT_BYTES)
        keys(
            value,
            {
                "schema",
                "profile",
                "compatibility",
                "runtime",
                "evaluator_hash",
                "origin_contract",
                "confirmation_result_hash",
                "proposal",
            },
        )
        if type(value["schema"]) is not int or value["schema"] != 1 or digest(value) != identity:
            return None, "snapshot_identity_mismatch"
        facts, required = profile(task_profile), profile(value["profile"])
        if any(not set(required[k]) <= set(facts[k]) for k in facts):
            return None, "profile_mismatch"
        if value["compatibility"] != {k: cfg[k] for k in COMPATIBILITY}:
            return None, "task_compatibility_mismatch"
        if (
            initial_result["mode"] != "full"
            or initial_result["execution"] != "completed"
            or initial_result["correctness"] != "pass"
            or initial_result["signal_only"] is not False
            or initial_result["check_only"] is not False
            or initial_result["selection"] != {}
        ):
            return None, "current_initial_full_unavailable"
        if value["evaluator_hash"] != initial_result["identity"]["evaluator_hash"]:
            return None, "evaluator_mismatch"
        runtime = runtime_key(initial_result["environment"])
        if value["runtime"] != runtime or str(cfg["device"]).startswith("cuda") != (
            runtime["gpu"] is not None
        ):
            return None, "runtime_mismatch"
        offer = value["proposal"]
        keys(offer, {"episode_id", "source_hash", "task_source_hash", "source"})
        for field in ("episode_id", "source_hash", "task_source_hash"):
            _identity(offer[field])
        for field in ("evaluator_hash", "origin_contract", "confirmation_result_hash"):
            _identity(value[field])
        if offer["task_source_hash"] != initial_result["identity"]["files"]["task.py"]:
            return None, "reference_mismatch"
        source = offer["source"]
        if (
            not isinstance(source, str)
            or not source.strip()
            or len(source.encode()) > 24000
            or source_id(source) != offer["source_hash"]
            or len(json.dumps(offer, allow_nan=False).encode()) > MAX_SNAPSHOT_BYTES
        ):
            return None, "source_mismatch"
        return offer, "compatible_proposal"
    except (OSError, ValueError, KeyError, TypeError, OverflowError, RecursionError):
        return None, "snapshot_unavailable"
