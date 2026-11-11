"""Background formation of auditable artifact descendants, without adoption.

The caller supplies an original completed research trial, never an agent's claimed
score or hand-picked source. Every collection rebuilds the complete episode. A
descendant needs an actual parent snapshot activation and an independently eligible
final for a changed, agent-proposed implementation. Failure observations remain.

This module does not execute code, call models, scan production histories, rank
unrelated tasks, publish an active version, or certify improvement. Candidate
snapshots are inputs to a separately authorized validation experiment only.
"""

from pathlib import Path

from ..storage import digest
from .artifact_snapshot import COMPATIBILITY, _value
from .episodes import MAX_EPISODE_BYTES, _identity, build, load, save
from .model import keys, profile, read
from .research import source_id
from .reuse import runtime_key

MAX_ANCESTORS = 32


def _scope(episode):
    initial = episode["observations"][0]
    return digest(
        {
            "task_source_hash": source_id(episode["task_source"]),
            "compatibility": {k: episode["task_config"][k] for k in COMPATIBILITY},
            "profile": episode["profile"],
            "runtime": runtime_key(initial["environment"]),
            "evaluator_hash": initial["evaluator_hash"],
        }
    )


def _linked(provenance, episode, parent_view):
    """Match the actual activated file, proposal, initial receipt and first action."""
    artifacts = Path(provenance["artifacts"])
    trace = read(artifacts / "search-trace.json", MAX_EPISODE_BYTES)
    activation = trace.get("artifact_snapshot", {})
    offer = parent_view["proposal"]
    initial = episode["observations"][0]
    origin = Path(episode["origin_task"])
    # The second metadata read must not replace the action whose source/run was
    # actually verified by build(). Files can change between these two reads.
    if not episode["actions"]:
        return False
    first = episode["actions"][0]
    if (
        first.get("reported_decision_kind") != "artifact_proposal"
        or first.get("reported_artifact_episode_id") != offer["episode_id"]
        or first["action"] != "propose"
    ):
        return False
    used = (
        episode["observations"][first["observation"]]
        if "observation" in first
        else initial
        if first.get("measurement_status") == "duplicate_skipped"
        else None
    )
    if used is None or used["source"] != offer["source_hash"]:
        return False
    if (
        activation.get("identity") != digest(parent_view)
        or activation.get("reason") != "compatible_proposal"
        or activation.get("initial_result_hash") != initial["result_hash"]
        or activation.get("initial_run") != str(origin / ".kernel_envolve/runs" / initial["run"])
        or trace.get("artifact_proposal_hash") != digest(offer)
        or not trace["steps"]
    ):
        return False
    step = trace["steps"][0]
    if (
        step.get("index") != 0
        or step.get("decision_kind") != "artifact_proposal"
        or step.get("artifact_episode_id") != offer["episode_id"]
        or step.get("action") != "propose"
        or step.get("observation", {}).get("candidate") != offer["source_hash"]
    ):
        return False
    return read(artifacts / "artifact-proposal.json", 32768) == offer


def _form(provenance, parent_id, parent_record, parent_view):
    from .trials import validator_identity

    keys(provenance, {"task", "artifacts", "profile", "confirmation_run", "origin_task"})
    episode = build(
        provenance["task"],
        provenance["artifacts"],
        provenance["profile"],
        confirmation_run=provenance["confirmation_run"],
        origin_task=provenance["origin_task"],
    )
    if episode["final_confirmation"] is None:
        raise ValueError("Artifact lineage requires an independent final, including failed finals")
    scope = _scope(episode)
    final = episode["observations"][episode["final_confirmation"]]
    linked = parent_id is not None and _linked(provenance, episode, parent_view)
    reasons = []
    initial = episode["observations"][0]
    try:
        context_matches = final["evaluator_hash"] == initial["evaluator_hash"] and runtime_key(
            final["environment"]
        ) == runtime_key(initial["environment"])
    except (TypeError, ValueError):
        context_matches = False
    if not context_matches:
        reasons.append("independent_final_context_changed")
    if parent_id is not None:
        if scope != parent_record["scope"]:
            reasons.append("parent_scope_mismatch")
        if not linked:
            reasons.append("parent_not_activated")
        if episode["selected"] == parent_view["proposal"]["source_hash"]:
            reasons.append("unchanged_implementation")
        if not any(
            action.get("reported_decision_kind") == "adapter"
            and action["action"] == "propose"
            and "observation" in action
            and episode["observations"][action["observation"]]["source"] == episode["selected"]
            for action in episode["actions"]
        ):
            reasons.append("selected_source_not_proposed_by_child_agent")
    value = None
    if (
        final["execution"] != "completed"
        or final["correctness"] != "pass"
        or not final["decision"]["eligible_for_best"]
    ):
        reasons.append("independent_final_not_eligible")
    if not reasons:
        # An oversized/incompatible view is not a partially valid candidate. The
        # complete verified episode remains useful background evidence.
        try:
            value = _value(episode)
        except (KeyError, TypeError, ValueError):
            reasons.append("unsupported_bounded_view")
    parent_observations = []
    if linked:
        for index, observed in enumerate(episode["observations"]):
            if observed["source"] == parent_view["proposal"]["source_hash"]:
                parent_observations.append(
                    {
                        "observation": index,
                        "contract": episode["contract"],
                        "correctness": observed["correctness"],
                        "eligible_for_best": observed["decision"]["eligible_for_best"],
                        "result_hash": observed["result_hash"],
                    }
                )
    record = {
        "schema": 1,
        "validator_hash": validator_identity(),
        "kind": "observation" if value is None else "seed" if parent_id is None else "candidate",
        "depth": 1 if parent_record is None else parent_record["depth"] + 1,
        "parent": parent_id,
        "parent_activated": linked,
        "scope": scope,
        "episode": digest(episode),
        "snapshot": digest(value) if value is not None else None,
        "proposal_key": digest({"parent": parent_id, "scope": scope, "source": episode["selected"]})
        if value is not None
        else None,
        "reasons": reasons,
        "parent_observations": parent_observations,
        "provenance": provenance,
        "authority": "Verified artifact formation only; no admission, publication or gain claim. Parent failures apply to their observed contract, never all future tasks.",
    }
    return record, episode, value


def audit(directory, identity):
    """Rebuild the entire bounded ancestry from its original research receipts.

    Missing or changed raw evidence invalidates this audit, even if the materialized
    files' content hashes still match. Hashes alone are not publisher trust.
    """
    directory = Path(directory).resolve()
    from .trials import validator_identity

    def verify(current, seen):
        _identity(current)
        if current in seen or len(seen) >= MAX_ANCESTORS:
            raise ValueError("Artifact ancestry is cyclic or exceeds its audit bound")
        record = load(directory / "records", current)
        if record.get("validator_hash") != validator_identity():
            raise ValueError("Artifact formation code changed; audit with its frozen collector")
        parent_record = parent_view = None
        if record["parent"] is not None:
            parent_record, parent_view = verify(record["parent"], seen | {current})
            if parent_view is None:
                raise ValueError("An observation cannot be used as an artifact parent")
        rebuilt, episode, value = _form(
            record["provenance"], record["parent"], parent_record, parent_view
        )
        if rebuilt != record or load(directory / "episodes", record["episode"]) != episode:
            raise ValueError("Artifact lineage or original episode evidence changed")
        if value is not None and load(directory / "views", record["snapshot"]) != value:
            raise ValueError("Artifact snapshot changed")
        return record, value

    return verify(identity, set())[0]


def collect(
    directory, task, artifacts, task_profile, *, confirmation_run, parent=None, origin_task=None
):
    """Automatically form a record from one completed trial; no manual source choice.

    Repeating the same collection is idempotent. Multiple successful records of
    the same parent/source share proposal_key, so a future admission ledger must
    not grant fresh validation chances for changed receipt IDs or explanations.
    """
    directory, task, artifacts = (Path(p).resolve() for p in (directory, task, artifacts))
    if directory.is_relative_to(task) or directory.is_relative_to(artifacts):
        raise ValueError("Artifact lineage archive must be outside research workspaces")
    if confirmation_run is None:
        raise ValueError("Artifact lineage requires independent final confirmation")
    parent_record = parent_view = None
    if parent is not None:
        parent_record = audit(directory, parent)
        if parent_record["depth"] >= MAX_ANCESTORS:
            raise ValueError("Artifact ancestry exceeds its audit bound")
        if parent_record["snapshot"] is None:
            raise ValueError("An observation cannot be used as an artifact parent")
        parent_view = load(directory / "views", parent_record["snapshot"])
    provenance = {
        "task": str(task),
        "artifacts": str(artifacts),
        "profile": profile(task_profile),
        "confirmation_run": str(confirmation_run),
        "origin_task": str(origin_task) if origin_task is not None else None,
    }
    record, episode, value = _form(provenance, parent, parent_record, parent_view)
    # Content-addressed files are immutable. An interrupted collection may leave
    # orphan evidence, never an activated version or a partial mutable manifest.
    save(directory / "episodes", episode)
    if value is not None:
        save(directory / "views", value)
    return save(directory / "records", record)


def validation_snapshot(directory, identity):
    """Explicit input to an authorized experiment, NOT proof of adoption."""
    record = audit(directory, identity)
    if record["snapshot"] is None:
        raise ValueError("This observation did not form an artifact candidate")
    return {
        "path": str(Path(directory).resolve() / "views" / f"{record['snapshot']}.json"),
        "identity": record["snapshot"],
        "profile": record["provenance"]["profile"],
    }
