"""Decisions are recomputable from complete raw evidence, never candidate claims."""

import math
import statistics as stats


def plan(cfg, mode):
    cases = [c for c in cfg["cases"] if mode == "full" or c.get("quick", True)]
    return cases, cfg[f"{mode}_seeds"], cfg[f"{mode}_trials"], 2 if mode == "full" else 1


def score(result, cfg):
    decision = {"eligible_for_best": False, "performance": "not_measured", "reasons": []}
    if result["execution"] != "completed" or result["correctness"] != "pass":
        decision["reasons"] = ["Evaluation did not complete with correctness pass"]
        return decision
    mode = result["mode"]
    cases, seeds, trials, batches = plan(cfg, mode)
    roles = ["candidate"] if result["signal_only"] else ["candidate", "comparator"]
    expected_checks = {
        (c["id"], seed, kind, role)
        for c in cases
        for seed in seeds
        for kind in ("fresh", "reuse")
        for role in roles
    }
    checks = [(r["case"], r["seed"], r["kind"], r["role"]) for r in result["checks"]]
    if len(checks) != len(expected_checks) or set(checks) != expected_checks:
        raise ValueError("Incomplete or duplicate correctness coverage")
    if any(r["status"] != "pass" for r in result["checks"]):
        raise ValueError("Correctness status contradicts individual checks")
    expected = {
        (c["id"], b, t, seeds[t % len(seeds)], role, repeat)
        for c in cases
        for b in range(batches)
        for t in range(trials)
        for role in roles
        for repeat in range(2)
    }
    records = result["samples"]
    keys = [(r["case"], r["batch"], r["trial"], r["seed"], r["role"], r["repeat"]) for r in records]
    if len(keys) != len(expected) or set(keys) != expected:
        raise ValueError("Incomplete or duplicate timing coverage")
    for r in records:
        if not isinstance(r["us"], (float, int)) or not math.isfinite(r["us"]) or r["us"] <= 0:
            raise ValueError("Timing must be finite and positive")
    summaries = []
    gains, guards = [], []
    for b in range(batches):
        totals = {role: 0.0 for role in roles}
        uncertainties = {role: 0.0 for role in roles}
        for c in cases:
            row = {"case": c["id"], "batch": b}
            for role in roles:
                values = [
                    r["us"]
                    for r in records
                    if r["case"] == c["id"] and r["batch"] == b and r["role"] == role
                ]
                median = stats.median(values)
                mad = stats.median(abs(v - median) for v in values)
                row[role + "_us"] = median
                row[role + "_mad_us"] = mad
                totals[role] += c["weight"] * median
                uncertainties[role] += c["weight"] * mad
            if "max_us" in c and row["candidate_us"] > c["max_us"]:
                decision["reasons"].append(f"{c['id']}: exceeds absolute latency cap in batch {b}")
            if not result["signal_only"]:
                row["speedup"] = row["comparator_us"] / row["candidate_us"]
                if row["candidate_us"] > row["comparator_us"] * (1 + cfg["max_regression"]):
                    decision["reasons"].append(
                        f"{c['id']}: exceeds regression allowance in batch {b}"
                    )
            summaries.append(row)
        if any(not math.isfinite(v) or v <= 0 for v in totals.values()):
            raise ValueError("Weighted latency must be finite and positive")
        if not all(math.isfinite(v) for v in uncertainties.values()):
            raise ValueError("Weighted noise must be finite")
        if not result["signal_only"]:
            gains.append(1 - totals["candidate"] / totals["comparator"])
            guards.append(
                cfg["noise_multiplier"] * sum(uncertainties.values()) / totals["comparator"]
            )
    decision["cases"] = summaries
    if result["signal_only"]:
        decision["reasons"].append("Candidate-only signal has no measured comparator")
        return decision
    decision["gain_by_batch"] = gains
    decision["noise_guard_by_batch"] = guards
    good = all(g > max(cfg["min_gain"], n) for g, n in zip(gains, guards))
    bad = all(g < -max(cfg["min_gain"], n) for g, n in zip(gains, guards))
    decision["performance"] = "faster" if good else "slower" if bad else "inconclusive"
    if not good:
        decision["reasons"].append(
            "Gain does not exceed the configured threshold and MAD noise guard in every batch"
        )
    if mode != "full":
        decision["reasons"].append("Quick results cannot update best")
    decision["eligible_for_best"] = mode == "full" and good and not decision["reasons"]
    return decision
