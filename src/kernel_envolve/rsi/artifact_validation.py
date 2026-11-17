"""Offline admission checks over original trial and model transport receipts.

The publisher/controller, its timing manifest and complete transport inventory are
trusted. They must be collected outside the solver's write authority after job
release. Hashes detect drift, not a malicious publisher or an incomplete provider
billing export. This module executes neither candidates nor model requests.
"""

import json
import statistics
from pathlib import Path

from ..storage import config, digest
from . import artifact_lineage as lineage
from .episodes import MAX_EPISODE_BYTES, build, load
from .model import number, read
from .research import source_id
from .reuse import runtime_key
from .trials import files, verdict, verify_template

ARMS = ("control", "seed", "parent", "candidate")


def task_key(cfg, source):
    """Exclude display-only differences from learning/holdout separation."""
    return digest(
        {
            "config": {k: v for k, v in cfg.items() if k not in {"name", "objective"}},
            "source": source,
        }
    )


def trial_order(suite):
    """Reverse paired blocks so no arm always receives the earlier GPU state."""
    return [
        (r, t, arm)
        for r in range(suite["repeats"])
        for t in range(len(suite["tasks"]))
        for arm in (ARMS if (r + t) % 2 == 0 else tuple(reversed(ARMS)))
    ]


def check_suite(suite, slots):
    """Keep original protections; extra practical criteria may only tighten them."""
    if type(suite["repeats"]) is not int or not 2 <= suite["repeats"] <= 16:
        raise ValueError("Admission requires 2–16 matched repeats")
    if type(suite["max_actions"]) is not int or not 1 <= suite["max_actions"] <= 8:
        raise ValueError("Invalid action limit")
    if not 1 <= len(suite["tasks"]) <= 64:
        raise ValueError("Invalid validation task count")
    for field, lower, upper in (
        ("max_quality_regression", -1, 0.02),
        ("max_cost_regression", -1, 0.05),
        ("min_quality_gain", 0, 1),
        ("min_cost_gain", 0, 1),
        ("min_relative_quality_gain", 0, 1),
        ("min_wall_saving", 0, 1),
        ("max_token_regression", -1, 0.05),
    ):
        value = number(suite[field], field, lower, upper)
        if value < 0:
            raise ValueError("Negative admission threshold")
    if suite["min_quality_gain"] < 0.01 or suite["min_cost_gain"] < 0.10:
        raise ValueError("Admission cannot weaken original benefit gates")
    if suite["min_relative_quality_gain"] < 0.10 or suite["min_wall_saving"] < 0.10:
        raise ValueError("Admission requires at least ten percent practical benefit")
    number(suite["max_seconds"], "validation seconds")
    if type(suite["max_tokens"]) is not int or suite["max_tokens"] <= 0:
        raise ValueError("Invalid token reservation")
    seen, coverage = set(), set()
    for task in suite["tasks"]:
        verify_template(task)
        if task["kind"] != "generic" or task["editable"] != ["task/candidate.py"]:
            raise ValueError("Artifact admission currently supports single generic candidates")
        if task["scope"] not in slots or task["partition"] not in {"retention", "fresh"}:
            raise ValueError("Unknown scope or validation partition")
        root = Path(task["path"]) / "task"
        key = task_key(config(root), (root / "task.py").read_text())
        if key in seen:
            raise ValueError("Duplicate validation template")
        seen.add(key)
        coverage.add((task["scope"], task["partition"]))
    if coverage != {(s, p) for s in slots for p in ("retention", "fresh")}:
        raise ValueError("Every catalog slot needs retention and fresh validation")


def _request(call, trace, episode, suite, repeat, seen_responses):
    request = call["request"]
    if request["model"] != suite["model"] or request.get("reasoning") != suite["reasoning"]:
        raise ValueError("Model configuration changed")
    if {k: v for k, v in request.items() if k not in {"model", "reasoning", "input"}} != suite[
        "request_options"
    ]:
        raise ValueError("Model transport options changed")
    prefix, encoded = request["input"].split("\n", 1)
    if digest(prefix) != suite["prompt_prefix_hash"]:
        raise ValueError("Model prompt changed")
    packet = json.loads(encoded)
    stage = suite["max_actions"] - packet["remaining_actions"]
    if type(stage) is not int or not 0 <= stage < len(trace["steps"]):
        raise ValueError("Model call stage is outside the actual search")
    step = trace["steps"][stage]
    if step["decision_kind"] != "adapter" or step["index"] != stage:
        raise ValueError("Model call replaces an artifact action")
    if (
        packet["model"] != suite["model"]
        or packet["repeat_identity"] != repeat
        or packet["strategy"] != ""
        or "historical_experience" in packet
        or packet["task_config"] != episode["task_config"]
        or packet["task_source"] != episode["task_source"]
    ):
        raise ValueError("Model request is not bound to this matched trial")
    history = packet["history"]
    if packet.get("history_representation") == "candidate-observation-refs-v1":
        history = json.loads(json.dumps(history))
        for action in history:
            ref = action.pop("observation_ref", None)
            if ref is None:
                continue
            observation = {
                "candidate": ref["candidate"],
                "mode": ref["mode"],
                "status": ref["status"],
            }
            if ref["status"] == "evaluated":
                matches = [
                    o
                    for c in packet["candidates"]
                    if c["id"] == ref["candidate"]
                    for o in c["observations"]
                    if o["run"] == ref["run"] and o["mode"] == ref["mode"]
                ]
                if len(matches) != 1:
                    raise ValueError("Ambiguous compact observation")
                observation.update(matches[0])
            action["observation"] = observation
    prior = trace["steps"][:stage]
    if history != prior:
        raise ValueError("Transport history differs from verified search")
    observations = [trace["initial_observation"]] + [
        s["observation"] for s in prior if s.get("observation", {}).get("status") == "evaluated"
    ]
    if (
        packet["selected"]
        != max((o for o in observations if o["mode"] == "full"), key=lambda o: o["quality"])[
            "candidate"
        ]
    ):
        raise ValueError("Model was shown a different incumbent")
    expected_sources = {o["candidate"] for o in observations}
    expected_sources.update(
        s["observation"]["candidate"] for s in prior if s.get("action") == "propose"
    )
    if {c["id"] for c in packet["candidates"]} != expected_sources:
        raise ValueError("Model candidate inventory changed")
    for candidate in packet["candidates"]:
        if source_id(candidate["source"]) != candidate["id"] or candidate["observations"] != [
            {k: v for k, v in o.items() if k not in {"candidate", "status"}}
            for o in observations
            if o["candidate"] == candidate["id"]
        ]:
            raise ValueError("Model source/observation binding changed")
    response = call.get("response") or {}
    usage = response.get("usage")
    if call["status"] != "completed" or response.get("status") != "completed" or not usage:
        raise ValueError("Unknown or incomplete model usage forbids admission")
    response_id = response["id"]
    if not isinstance(response_id, str) or not response_id or response_id in seen_responses:
        raise ValueError("Missing or reused transport response")
    seen_responses.add(response_id)
    for field in ("input_tokens", "output_tokens", "total_tokens"):
        if type(usage[field]) is not int or usage[field] < 0:
            raise ValueError("Invalid model usage")
    if usage["total_tokens"] != usage["input_tokens"] + usage["output_tokens"]:
        raise ValueError("Model usage totals disagree")
    content = "".join(
        c["text"]
        for o in response["output"]
        if o.get("type") == "message"
        for c in o["content"]
        if c.get("type") == "output_text"
    ).strip()
    if content.startswith("```"):
        content = "\n".join(content.splitlines()[1:-1])
    decision = json.loads(content)
    if any(decision.get(k) != step.get(k) for k in ("action", "hypothesis")):
        raise ValueError("Transport response differs from executed decision")
    if decision["action"] != "stop":
        observed = step["observation"]
        proposed = (
            source_id(decision["source"])
            if decision["action"] == "propose"
            else decision["candidate"]
        )
        if proposed != observed["candidate"] or decision["mode"] != observed["mode"]:
            raise ValueError("Executed source differs from transport response")
    return stage, usage["total_tokens"]


def verify(archive, catalogs, suite, evidence):
    """Rebuild outcomes; durations are explicitly trusted-controller measurements.

    evidence has trials, calls (the complete transport inventory) and a released
    job receipt reference for the external controller's audit. Resource release
    and provider-wide budgets remain that controller's responsibility.
    """
    archive = Path(archive)
    check_suite(suite, catalogs["candidate"])
    expected = trial_order(suite)
    trials = evidence["trials"]
    if [(t["repeat"], t["task"], t["arm"]) for t in trials] != expected:
        raise ValueError("Incomplete or duplicate four-arm validation")
    release = read(evidence["release_receipt"])
    if (
        release.get("phase") not in {"Completed", "Failed", "Stopped", "Deleted"}
        or release.get("pods") != []
        or release.get("live_read_status", {}).get("pods") != "available"
    ):
        raise ValueError("Controller must collect evidence after resource release")
    calls = [read(p, 2 * 1024 * 1024) for p in evidence["calls"]]
    if len(set(evidence["calls"])) != len(calls):
        raise ValueError("Duplicate model receipt path")
    metrics, fingerprints, run_ids, response_ids, call_count = {}, [], set(), set(), 0
    records = {
        identity: lineage.audit(archive, identity)
        for slots in catalogs.values()
        for identity in slots.values()
    }
    for item in trials:
        r, t, arm = (item[k] for k in ("repeat", "task", "arm"))
        task = suite["tasks"][t]
        provenance = item["provenance"]
        workspace = Path(provenance["task"]).parent
        after = files(workspace)
        if set(after) != set(task["files"]) or any(
            after[k] != v for k, v in task["files"].items() if k not in task["editable"]
        ):
            raise ValueError("Validation workspace changed protected files")
        episode = build(**provenance)
        if episode["final_confirmation"] is None:
            raise ValueError("Independent final evaluation is required")
        if episode["profile"] != task["profile"] or lineage._scope(episode) != task["scope"]:
            raise ValueError("Validation scope changed")
        if episode["observations"][0]["source"] != task["files"]["task/candidate.py"]:
            raise ValueError("Validation initial candidate changed")
        final = episode["observations"][episode["final_confirmation"]]
        run_key = (episode["origin_task"], final["run"])
        if run_key in run_ids:
            raise ValueError("Final evaluation reused across trials")
        run_ids.add(run_key)
        initial = episode["observations"][0]
        if final["evaluator_hash"] != initial["evaluator_hash"] or runtime_key(
            final["environment"]
        ) != runtime_key(initial["environment"]):
            raise ValueError("Final evaluation context changed")
        trace = read(Path(provenance["artifacts"]) / "search-trace.json", MAX_EPISODE_BYTES)
        observed = [trace["initial_observation"]] + [
            s["observation"]
            for s in trace["steps"]
            if s.get("observation", {}).get("status") == "evaluated"
        ]
        for claimed, actual in zip(observed, episode["observations"][:-1], strict=True):
            if any(claimed[k] != actual[k] for k in ("quality", "correctness", "mode")):
                raise ValueError("Trace outcomes differ from original evaluator receipts")
        if (
            trace["max_actions"] != suite["max_actions"]
            or len(trace["steps"]) > suite["max_actions"]
            or trace.get("adaptive_artifact_budget")
        ):
            raise ValueError("Validation action allowance changed")
        if arm == "control":
            if (
                trace.get("artifact_snapshot")
                or trace.get("artifact_proposal_hash")
                or any(a.get("reported_decision_kind") != "adapter" for a in episode["actions"])
            ):
                raise ValueError("No-memory control received an artifact")
        else:
            record = records[catalogs[arm][task["scope"]]]
            view = load(archive / "views", record["snapshot"])
            if not lineage._linked(provenance, episode, view):
                raise ValueError("Assigned artifact was not actually activated")
        trial_id = f"{r}-{t}-{arm}"
        matched = [c for c in calls if c["trial_id"] == trial_id]
        stages, tokens = [], 0
        for call in matched:
            stage, total = _request(call, trace, episode, suite, r, response_ids)
            stages.append(stage)
            tokens += total
        if sorted(stages) != [
            s["index"] for s in trace["steps"] if s["decision_kind"] == "adapter"
        ]:
            raise ValueError("Missing, duplicate or extra model requests")
        call_count += len(matched)
        seconds = number(item["seconds"], "controller trial seconds")
        solver_seconds = number(item["solver_seconds"], "controller solver seconds", 0, seconds)
        metrics[r, t, arm] = {
            "passed": final["execution"] == "completed" and final["correctness"] == "pass",
            "quality": final["quality"],
            "seconds": seconds,
            "solver_seconds": solver_seconds,
            "tokens": tokens,
        }
        fingerprints.append(digest(episode))
    if call_count != len(calls):
        raise ValueError("Unassigned model attempts in transport inventory")
    decisions = {}
    for arm in ARMS[:-1]:
        pairs = [
            {
                "repeat": r,
                "task": t,
                "control": metrics[r, t, arm],
                "candidate": metrics[r, t, "candidate"],
            }
            for r in range(suite["repeats"])
            for t in range(len(suite["tasks"]))
        ]
        decisions[arm] = verdict(pairs, suite)
    reasons = [f"{arm}: {reason}" for arm, d in decisions.items() for reason in d["reasons"]]
    practical = {}
    for arm, decision in decisions.items():
        practical[arm] = []
        for totals in decision["repeats"]:
            old, new = totals["control_quality"], totals["candidate_quality"]
            relative_gain = new / old - 1 if old > 0 else None
            saving = 1 - totals["candidate_seconds"] / totals["control_seconds"]
            passed = (
                relative_gain is not None and relative_gain >= suite["min_relative_quality_gain"]
            ) or (new >= old and saving >= suite["min_wall_saving"])
            practical[arm].append(
                {
                    "repeat": totals["repeat"],
                    "quality_relative_gain": relative_gain,
                    "wall_saving": saving,
                    "benefit_floor_met": passed,
                }
            )
            if not passed:
                reasons.append(f"{arm}: No practical benefit in repeat {totals['repeat']}")
    for r in range(suite["repeats"]):
        for arm in ARMS[:-1]:

            def total(role):
                return sum(metrics[r, t, role]["tokens"] for t in range(len(suite["tasks"])))

            if total("candidate") > total(arm) * (1 + suite["max_token_regression"]):
                reasons.append(f"{arm}: token cost regression")
        for partition in ("retention", "fresh"):
            indices = [t for t, task in enumerate(suite["tasks"]) if task["partition"] == partition]
            if (
                statistics.mean(metrics[r, t, "candidate"]["quality"] for t in indices)
                < statistics.mean(metrics[r, t, "parent"]["quality"] for t in indices) - 1e-12
            ):
                reasons.append(f"parent: {partition} aggregate quality regression")
    seconds = sum(m["seconds"] for m in metrics.values())
    tokens = sum(m["tokens"] for m in metrics.values())
    if seconds > suite["max_seconds"] or tokens > suite["max_tokens"]:
        reasons.append("Validation reservation exceeded")
    return {
        "accepted": not reasons,
        "reasons": sorted(set(reasons)),
        "comparisons": decisions,
        "practical": practical,
        "seconds": seconds,
        "tokens": tokens,
        "episodes": fingerprints,
        "transport_hashes": [digest(c) for c in calls],
        "release_hash": digest(release),
        "scope": "Declared four-arm validation only; not campaign ROI or statistical superiority",
    }
