"""Offline verification of every collected receipt; never reruns model or GPU code."""

import argparse
import collections
import hashlib
import json
import sqlite3
import statistics
from pathlib import Path

from kernel_envolve.manager.evidence import inspect_run
from kernel_envolve.rsi.model import context
from kernel_envolve.rsi.trials import verdict
from kernel_envolve.storage import digest, evaluator_hash

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("pilot_directory", type=Path)
ROOT = parser.parse_args().pilot_directory.resolve()
E = ROOT / "evidence"
r = json.loads((E / "pilot-result.json").read_text())
m = json.loads((ROOT / "manifest.json").read_text())
assert r["completed"] and len(r["trials"]) == 16
assert r["active"] == "base"


def local(path):
    return E / Path(path).relative_to("/tmp/kernel-rsi-pilot")


verified = {}
for path in sorted(E.rglob("receipt.json")):
    run = path.parent
    if not (run / "result.json").exists():
        continue
    raw = json.loads((run / "result.json").read_text())
    spec = {
        "kind": "generic",
        "task_dir": str(Path(raw["task_root"]).resolve()),
        "baseline": {"contract": raw["identity"]["contract"]},
    }
    value = inspect_run(run, spec)
    assert value["evaluator_hash"] == evaluator_hash()
    verified[str(run)] = value
trace_observations = 0
summaries = []
for item in r["trials"]:
    result = item["result"]
    trace = item["trace"]
    counts = collections.Counter()
    modes = collections.Counter()
    bad = 0
    repeats = 0
    if trace:
        observations = [trace["initial_observation"]] + [
            s["observation"] for s in trace["steps"] if "observation" in s
        ]
        directory = E / "comparison" / f"{item['repeat']}-{item['task']}-{item['arm']}"
        for ob in observations:
            if ob["status"] == "duplicate_skipped":
                repeats += 1
                continue
            value = verified[str(local(ob["run"]))]
            source = (local(ob["run"]) / "source/candidate.py").read_bytes()
            assert hashlib.sha256(source).hexdigest() == ob["candidate"]
            assert source == (directory / f"candidate-{ob['candidate']}.py").read_bytes()
            assert ob["correctness"] == value["correctness"] and ob["mode"] == value["mode"]
            modes[ob["mode"]] += 1
            bad += value["correctness"] != "pass"
            trace_observations += 1
        for step in trace["steps"]:
            counts[step.get("action", "missing")] += 1
    entry = {k: item[k] for k in ["repeat", "task", "arm", "error"]}
    entry.update(
        actions=dict(counts),
        search_evaluations=dict(modes),
        wrong_candidate_evaluations=bad,
        identical_evaluations_skipped=repeats,
        stop_reason=trace.get("stop_reason") if trace else None,
        model_seconds=sum(s.get("model_seconds", 0) for s in trace["steps"]) if trace else None,
    )
    if result:
        value = verified[str(local(result["run"]))]
        assert value["result_hash"] == result["result_hash"]
        passed = value["execution"] == "completed" and value["correctness"] == "pass"
        gains = value["decision"].get("fixed_baseline_gain_by_batch", [])
        quality = (
            max(0, statistics.median(gains))
            if passed and gains and value["decision"]["eligible_for_best"]
            else 0
            if passed
            else -1
        )
        assert passed == result["passed"] and quality == result["quality"]
        entry.update(
            passed=passed,
            quality=quality,
            eligible_for_best=value["decision"]["eligible_for_best"],
            baseline_speedup=value["baseline_speedup"],
            fixed_baseline_gain_by_batch=gains,
            solver_seconds=result["solver_seconds"],
            trial_seconds=result["seconds"],
            result_hash=result["result_hash"],
        )
    summaries.append(entry)
suite = json.loads((E / "rsi-state/config.json").read_text())["suites"]["b300-study"]
for arm in ["fixed", "raw", "learned"]:
    pairs = []
    for repeat in range(2):
        for task in range(2):
            subset = {
                x["arm"]: x["result"]
                for x in r["trials"]
                if x["repeat"] == repeat and x["task"] == task
            }
            pairs.append(
                {
                    "repeat": repeat,
                    "task": task,
                    "control": subset["control"],
                    "candidate": subset[arm],
                }
            )
    assert verdict(pairs, suite) == r["gate_comparisons"][arm]
for arm, rules in r["strategies"].items():
    assert digest(rules) == r["strategy_hashes"][arm]
aggregates = {}
for arm in ["control", "fixed", "raw", "learned"]:
    rows = [x for x in summaries if x["arm"] == arm]
    aggregates[arm] = {
        "trials": len(rows),
        "errors": sum(x["error"] is not None for x in rows),
        "passed": sum(x.get("passed", False) for x in rows),
        "promotable": sum(x.get("eligible_for_best", False) for x in rows),
        "solver_seconds": sum(x.get("solver_seconds", 0) for x in rows),
        "trial_seconds": sum(x.get("trial_seconds", 0) for x in rows),
        "model_seconds": sum(x.get("model_seconds", 0) or 0 for x in rows),
        "wrong_candidate_evaluations": sum(x["wrong_candidate_evaluations"] for x in rows),
        "identical_evaluations_skipped": sum(x["identical_evaluations_skipped"] for x in rows),
        "search_evaluations": dict(
            sum((collections.Counter(x["search_evaluations"]) for x in rows), collections.Counter())
        ),
        "actions": dict(
            sum((collections.Counter(x["actions"]) for x in rows), collections.Counter())
        ),
    }
for arm, values in aggregates.items():
    rows = [x for x in summaries if x["arm"] == arm]
    values["completed_trial_seconds"] = values.pop("trial_seconds")
    values["completed_solver_seconds"] = values.pop("solver_seconds")
    values["failed_trial_seconds"] = sum(x["error"]["seconds"] for x in rows if x["error"])
    values["all_attempts_trial_seconds"] = (
        values["completed_trial_seconds"] + values["failed_trial_seconds"]
    )
    values["cost_comparison_complete"] = not values["errors"]
calls = []
for f in sorted((ROOT / "model-calls").glob("*.json")):
    d = json.loads(f.read_text())
    usage = (d.get("response") or {}).get("usage")
    arm = "learning"
    if "-solve-" in f.stem:
        packet = json.loads(d["request"]["input"].split("\n", 1)[1])
        matching = [
            name for name, rules in r["strategies"].items() if context(rules) == packet["strategy"]
        ]
        assert len(matching) == 1
        arm = matching[0]
    calls.append(
        {
            "arm": arm,
            "call": f.stem,
            "status": d["status"],
            "seconds": d.get("seconds"),
            "usage": usage,
        }
    )
usage = {
    key: sum((c["usage"] or {}).get(key, 0) for c in calls)
    for key in ["input_tokens", "output_tokens", "total_tokens", "cost"]
}
for arm, values in aggregates.items():
    selected = [c for c in calls if c["arm"] == arm]
    values["model_attempts"] = len(selected)
    values["missing_usage_calls"] = sum(c["usage"] is None for c in selected)
    values["model_attempt_seconds"] = sum(c["seconds"] for c in selected)
    values["known_tokens"] = sum((c["usage"] or {}).get("total_tokens", 0) for c in selected)
allocations = []
for name in [
    "rsi-batch-pilot-20261110",
    "rsi-batch-pilot-20261110-b",
    "rsi-batch-pilot-20261110-c",
    "rsi-multistep-pilot-20261110",
    ROOT.name,
]:
    p = ROOT.parent / name
    start = json.loads((p / "create-attempt.json").read_text())
    end = json.loads((p / "released.json").read_text())
    assert end["pods"] == [] and end["live_read_status"]["pods"] == "available"
    allocations.append(
        {
            "directory": name,
            "job": end["id"],
            "phase": end["phase"],
            "create_at": start["at"],
            "release_at": end["at"],
            "conservative_gpu_seconds": end["at"] - start["at"],
            "live_pods_empty": True,
        }
    )
con = sqlite3.connect(f"file:{E}/rsi-state/rsi.sqlite?mode=ro", uri=True)
try:
    assert (
        json.loads(con.execute("SELECT value FROM meta WHERE key='enabled'").fetchone()[0]) is False
    )
    assert con.execute("SELECT COUNT(*) FROM history").fetchone()[0] == 0
finally:
    con.close()
summary = {
    "date": "2026-11-10",
    "source_commit": m["source_commit"],
    "job": allocations[-1]["job"],
    "model": r["model"],
    "environment": r["environment"],
    "completed": r["completed"],
    "scope": r["scope"],
    "active_after": r["active"],
    "disabled_after": True,
    "strategies": r["strategies"],
    "strategy_hashes": r["strategy_hashes"],
    "training_reuse": r["training_reuse"],
    "trials": summaries,
    "aggregates": aggregates,
    "gate_comparisons": r["gate_comparisons"],
    "ledger": r["ledger"],
    "driver_seconds": r["seconds"],
    "model_calls": calls,
    "known_usage_totals": usage,
    "missing_usage_calls": sum(c["usage"] is None for c in calls),
    "cost_unit": "Provider field; currency not independently verified",
    "allocations_under_90_minute_approval": allocations,
    "conservative_gpu_seconds_total": sum(a["conservative_gpu_seconds"] for a in allocations),
    "separate_earlier_overrun_seconds": 6511.071,
    "verification": {
        "raw_receipts_recomputed": len(verified),
        "trace_source_bindings_checked": trace_observations,
        "gates_recomputed": True,
        "policy_hashes_checked": True,
        "disabled_and_no_adoption_checked": True,
    },
    "sha256": {
        name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
        for name in [
            "evidence.tar.gz",
            "manifest.json",
            "protocol.json",
            "approved-budget.json",
            "live-policy-verification.json",
        ]
    },
    "limitations": [
        "Two development templates and two repeats; not independent final audit or statistical superiority",
        "All arms share the same bounded research tools; improvements from tools are not RSI gains",
        "Fixed/raw baselines are not reproductions of ACE or GEPA",
        "Learning trajectories reused after correcting report attribution; historical hardware UUID is retained",
        "No supported paired model random seed",
        "More context and provider latency can confound wall time",
        "Token totals include policy learning; prior development/learning calls remain separately recorded",
        "No production policy activation",
        "Deadline configured and verified live, not expiry-tested",
    ],
}
(ROOT / "summary.json").write_text(json.dumps(summary, indent=2))
print(
    json.dumps(
        {
            "aggregates": aggregates,
            "gates": {k: v["accepted"] for k, v in r["gate_comparisons"].items()},
            "verification": summary["verification"],
            "usage": usage,
            "gpu_seconds": summary["conservative_gpu_seconds_total"],
        },
        indent=2,
    )
)
