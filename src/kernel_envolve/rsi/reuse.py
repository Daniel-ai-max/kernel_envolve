"""Optional exact-source artifact selection for operator-configured experiments.

Approved content IDs and the runtime/profile are supplied by the operator. This
helper neither executes sources nor certifies a new task. The research loop must
spend an ordinary full-evaluation action before a source can replace its incumbent.
No production Manager path imports this module.
"""

from pathlib import Path

from ..storage import config
from .episodes import _source, load
from .model import profile
from .research import source_id


def runtime_key(environment):
    """Compare declared compute settings, excluding physical UUIDs and paths."""
    if (
        not isinstance(environment, dict)
        or not isinstance(environment.get("packages"), dict)
        or not environment["packages"]
        or not isinstance(environment.get("runtime_settings"), dict)
        or not environment["runtime_settings"]
    ):
        raise ValueError("Artifact reuse requires observed package/runtime information")
    gpu = environment.get("gpu")
    if gpu is not None and not isinstance(gpu, dict):
        raise ValueError("Invalid observed GPU environment")
    if gpu is not None and not all(gpu.get(key) for key in ("name", "capability", "cuda_runtime")):
        raise ValueError("Incomplete observed GPU environment")
    return {
        "packages": environment["packages"],
        "gpu": None
        if gpu is None
        else {key: gpu.get(key) for key in ("name", "capability", "cuda_runtime")},
        "runtime_settings": {
            key: value
            for key, value in environment.get("runtime_settings", {}).items()
            if key != "cuda"  # Logical device/stream numbers are not hardware identities.
        },
    }


def select(directory, approved_ids, task, task_profile, environment):
    """Return one unexecuted proposal or None; no fuzzy match or score ranking.

    Only a complete single-file reference/input-generator match is supported.
    Input cases, measurement settings and seeds may differ, so even an exact
    source match does not imply the historical implementation will pass or win.
    IDs authenticate contents, not publisher trust; callers must pin an approved
    manifest and selector in adapter_files and audit their original receipts.
    """
    if (
        not isinstance(approved_ids, (list, tuple))
        or not 1 <= len(approved_ids) <= 32
        or any(not isinstance(identity, str) for identity in approved_ids)
        or len(set(approved_ids)) != len(approved_ids)
    ):
        raise ValueError("Artifact selection requires 1–32 unique approved episode IDs")
    task = Path(task).resolve()
    cfg = config(task)
    if cfg["reference_files"] != ["task.py"] or cfg["candidate_files"] != ["candidate.py"]:
        return None
    source = _source(task / "task.py")
    facts = profile(task_profile)
    runtime = runtime_key(environment)
    if str(cfg["device"]).startswith("cuda") != (runtime["gpu"] is not None):
        raise ValueError("Observed runtime does not match the task device")
    for identity in sorted(approved_ids):
        episode = load(directory, identity)
        if episode["task_source"] != source:
            continue
        old = episode["task_config"]
        if any(
            old.get(key) != cfg[key]
            for key in (
                "reference_files",
                "candidate_files",
                "device",
                "rtol",
                "atol",
                "equal_nan",
                "mutable_inputs",
                "deterministic",
            )
        ):
            continue
        if not set(episode["profile"]["tags"]) <= set(facts["tags"]) or not set(
            episode["profile"]["capabilities"]
        ) <= set(facts["capabilities"]):
            continue
        index = episode["final_confirmation"]
        if type(index) is not int or not 0 <= index < len(episode["observations"]):
            continue
        confirmation = episode["observations"][index]
        if (
            confirmation["source"] != episode["selected"]
            or confirmation["mode"] != "full"
            or confirmation["execution"] != "completed"
            or confirmation["correctness"] != "pass"
            or not confirmation["decision"]["eligible_for_best"]
            or runtime_key(confirmation["environment"]) != runtime
        ):
            continue
        candidate = episode["sources"][episode["selected"]]
        if source_id(candidate) != episode["selected"]:
            raise ValueError("Historical artifact source identity mismatch")
        return {
            "episode_id": identity,
            "source_hash": episode["selected"],
            "task_source_hash": source_id(source),
            "source": candidate,
        }
    return None
