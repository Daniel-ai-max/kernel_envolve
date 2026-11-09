"""Decisions are recomputable from complete raw evidence, never candidate claims."""

import math
import statistics as stats

from .protocol import order, plan, post_seed, roles, timing_seed


def score(result, cfg):
    decision = {"eligible_for_best": False, "performance": "not_measured", "reasons": []}
    if result["execution"] != "completed" or result["correctness"] != "pass":
        decision["reasons"] = ["Evaluation did not complete with correctness pass"]
        return decision
    mode = result["mode"]
    cases, seeds, trials, batches = plan(cfg, mode, result.get("selection"))
    implementations = roles(result)
    expected_checks = {
        (c["id"], seed, kind, role)
        for c in cases
        for seed in seeds
        for kind in ("fresh", "reuse")
        for role in implementations
    }
    if not result.get("check_only"):
        expected_checks |= {
            (c["id"], post_seed(c["id"], seeds), "post", role)
            for c in cases
            for role in implementations
        }
        expected_checks |= {
            (c["id"], timing_seed(c["id"], batches - 1, trials - 1, 1, seeds), "timed_output", role)
            for c in cases
            for role in implementations
        }
    checks = [(r["case"], r["seed"], r["kind"], r["role"]) for r in result["checks"]]
    if len(checks) != len(expected_checks) or set(checks) != expected_checks:
        raise ValueError("Incomplete or duplicate correctness coverage")
    if any(r["status"] != "pass" for r in result["checks"]):
        raise ValueError("Correctness status contradicts individual checks")
    if result.get("check_only"):
        if mode != "quick" or result["samples"]:
            raise ValueError("Check-only diagnostics cannot supply full/timing evidence")
        decision["reasons"] = ["Correctness-only diagnostic cannot update best"]
        return decision
    expected = {
        (c["id"], b, t, timing_seed(c["id"], b, t, repeat, seeds), role, repeat)
        for c in cases
        for b in range(batches)
        for t in range(trials)
        for role in implementations
        for repeat in range(2)
    }
    records = result["samples"]
    keys = [(r["case"], r["batch"], r["trial"], r["seed"], r["role"], r["repeat"]) for r in records]
    if len(keys) != len(expected) or set(keys) != expected:
        raise ValueError("Incomplete or duplicate timing coverage")
    ordered_keys = []
    for b in range(batches):
        for c in cases if b == 0 else reversed(cases):
            for t in range(trials):
                repeats = dict.fromkeys(implementations, 0)
                for role in order(implementations, b, t):
                    repeat = repeats[role]
                    ordered_keys.append(
                        (c["id"], b, t, timing_seed(c["id"], b, t, repeat, seeds), role, repeat)
                    )
                    repeats[role] += 1
    if keys != ordered_keys:
        raise ValueError("Timing order does not match the paired schedule")
    for r in records:
        if not isinstance(r["us"], (float, int)) or not math.isfinite(r["us"]) or r["us"] <= 0:
            raise ValueError("Timing must be finite and positive")
    summaries = []
    gains, guards, baseline_gains, baseline_guards = [], [], [], []
    for b in range(batches):
        totals = {role: 0.0 for role in implementations}
        uncertainties = {role: 0.0 for role in implementations}
        for c in cases:
            row = {"case": c["id"], "batch": b}
            for role in implementations:
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
                if mode == "full":
                    original = row.get("baseline_us", row["comparator_us"])
                    if row["candidate_us"] > original * (1 + cfg["max_baseline_regression"]):
                        decision["reasons"].append(
                            f"{c['id']}: exceeds fixed baseline budget in batch {b}"
                        )
            summaries.append(row)
        if any(not math.isfinite(v) or v <= 0 for v in totals.values()):
            raise ValueError("Weighted latency must be finite and positive")
        if not all(math.isfinite(v) for v in uncertainties.values()):
            raise ValueError("Weighted noise must be finite")
        if not result["signal_only"]:
            gains.append(1 - totals["candidate"] / totals["comparator"])
            guards.append(
                cfg["noise_multiplier"]
                * (uncertainties["candidate"] + uncertainties["comparator"])
                / totals["comparator"]
            )
            if mode == "full":
                baseline_gains.append(
                    1 - totals["candidate"] / totals.get("baseline", totals["comparator"])
                )
                baseline_guards.append(
                    cfg["noise_multiplier"]
                    * (
                        uncertainties["candidate"]
                        + uncertainties.get("baseline", uncertainties["comparator"])
                    )
                    / totals.get("baseline", totals["comparator"])
                )
    decision["cases"] = summaries
    if result["signal_only"]:
        decision["reasons"].append("Candidate-only signal has no measured comparator")
        return decision
    decision["gain_by_batch"] = gains
    decision["noise_guard_by_batch"] = guards
    if baseline_gains:
        decision["fixed_baseline_gain_by_batch"] = baseline_gains
        decision["fixed_baseline_noise_guard_by_batch"] = baseline_guards
        if not all(g > max(cfg["min_gain"], n) for g, n in zip(baseline_gains, baseline_guards)):
            decision["reasons"].append(
                "Meaningful gain over fixed baseline is not confirmed in every batch"
            )
    good = all(g > max(cfg["min_gain"], n) for g, n in zip(gains, guards))
    bad = all(g < -max(cfg["min_gain"], n) for g, n in zip(gains, guards))
    decision["performance"] = "faster" if good else "slower" if bad else "inconclusive"
    if not good:
        decision["reasons"].append(
            "Gain does not exceed the configured threshold and MAD noise guard in every batch"
        )
    if mode != "full":
        decision["reasons"].append("Quick results cannot update best")
    if result.get("same_candidate_as_best"):
        decision["reasons"].append(
            "Candidate source is identical to best; no implementation change"
        )
    decision["eligible_for_best"] = mode == "full" and good and not decision["reasons"]
    return decision
