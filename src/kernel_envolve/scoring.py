"""Decisions are recomputable from complete raw evidence, never candidate claims."""

import math
import statistics as stats

from .protocol import (
    measured_cases,
    objective_cases,
    order,
    performance_role,
    plan,
    post_seed,
    roles,
    timing_seed,
)


def ratio(numerator, denominator):
    value = numerator / denominator
    if not math.isfinite(value):
        raise ValueError("Derived metric is not finite")
    return value


def distribution(values):
    if not all(math.isfinite(v) for v in values):
        raise ValueError("Derived timing distribution is not finite")
    median = stats.median(values)
    mad = stats.median(abs(v - median) for v in values)
    # A one-trial quick probe cannot estimate spread; it never grants promotion.
    if len(values) > 1:
        q25, _, q75 = stats.quantiles(values, n=4, method="inclusive")
        iqr = q75 - q25
    else:
        iqr = 0.0
    if not all(math.isfinite(v) for v in (median, mad, iqr)):
        raise ValueError("Derived timing distribution is not finite")
    return median, mad, iqr, max(mad, iqr / 2)


def score(result, cfg):
    decision = {"eligible_for_best": False, "performance": "not_measured", "reasons": []}
    if result["execution"] != "completed" or result["correctness"] != "pass":
        decision["reasons"] = ["Evaluation did not complete with correctness pass"]
        return decision
    mode = result["mode"]
    cases, seeds, trials, batches = plan(cfg, mode, result.get("selection"))
    timed_cases = [] if result.get("check_only") else measured_cases(cases)
    decision["coverage"] = {
        "correctness_cases": [case["id"] for case in cases],
        "timed_cases": [case["id"] for case in timed_cases],
    }
    implementations = roles(result)
    expected_checks = {
        (c["id"], seed, kind, role)
        for c in cases
        for seed in seeds
        for kind in (
            ("fresh", "repeat", "reuse") if cfg.get("deterministic") else ("fresh", "reuse")
        )
        for role in implementations
    }
    if not result.get("check_only"):
        expected_checks |= {
            (c["id"], post_seed(c["id"], seeds), "post", role)
            for c in timed_cases
            for role in implementations
        }
        expected_checks |= {
            (c["id"], timing_seed(c["id"], batches - 1, trials - 1, 1, seeds), "timed_output", role)
            for c in timed_cases
            for role in implementations
        }
    checks = [(r["case"], r["seed"], r["kind"], r["role"]) for r in result["checks"]]
    if len(checks) != len(expected_checks) or set(checks) != expected_checks:
        raise ValueError("Incomplete or duplicate correctness coverage")
    if any(r["status"] != "pass" for r in result["checks"]):
        raise ValueError("Correctness status contradicts individual checks")
    if result.get("check_only"):
        if result["samples"]:
            raise ValueError("Check-only diagnostics cannot supply timing evidence")
        decision["reasons"] = ["Correctness-only diagnostic cannot update best"]
        return decision
    if not timed_cases:
        raise ValueError("Performance evaluation requires measured cases")
    objectives = objective_cases(timed_cases)
    if not objectives:
        raise ValueError("Performance evaluation requires measured objective cases")
    if any(performance_role(c) == "guard" for c in timed_cases):
        decision["coverage"]["objective_cases"] = [c["id"] for c in objectives]
        decision["coverage"]["guard_cases"] = [
            c["id"] for c in timed_cases if performance_role(c) == "guard"
        ]
    expected = {
        (c["id"], b, t, timing_seed(c["id"], b, t, repeat, seeds), role, repeat)
        for c in timed_cases
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
        for c in timed_cases if b == 0 else reversed(timed_cases):
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
    grouped = {}
    for r in records:
        if type(r["us"]) not in (float, int) or not math.isfinite(r["us"]) or r["us"] <= 0:
            raise ValueError("Timing must be finite and positive")
        key = (r["case"], r["batch"], r["role"])
        if key not in grouped:
            grouped[key] = [[] for _ in range(trials)]
        grouped[key][r["trial"]].append(r["us"])
    summaries = []
    gains, guards, baseline_gains, baseline_guards = [], [], [], []
    paired_gains, baseline_paired_gains = [], []
    for b in range(batches):
        totals = {role: 0.0 for role in implementations}
        contrasts = {role: 0.0 for role in implementations if role != "candidate"}
        uncertainties = dict.fromkeys(contrasts, 0.0)
        for c in timed_cases:
            row = {"case": c["id"], "batch": b}
            objective = performance_role(c) == "objective"
            if "performance_role" in c:
                row["performance_role"] = c["performance_role"]
            trial_medians = {}
            for role in implementations:
                by_trial = grouped[(c["id"], b, role)]
                values = [v for pair in by_trial for v in pair]
                median, mad, iqr, _ = distribution(values)
                row[role + "_us"] = median
                row[role + "_mad_us"] = mad
                row[role + "_iqr_us"] = iqr
                if objective:
                    totals[role] += c["weight"] * median
                if contrasts:
                    trial_medians[role] = [stats.median(pair) for pair in by_trial]
            for role in contrasts:
                differences = [
                    candidate - control
                    for candidate, control in zip(trial_medians["candidate"], trial_medians[role])
                ]
                delta, mad, iqr, spread = distribution(differences)
                row[f"paired_{role}_delta_us"] = delta
                row[f"paired_{role}_mad_us"] = mad
                row[f"paired_{role}_iqr_us"] = iqr
                if objective:
                    contrasts[role] += c["weight"] * delta
                    uncertainties[role] += c["weight"] * spread
            if "max_us" in c and row["candidate_us"] > c["max_us"]:
                decision["reasons"].append(f"{c['id']}: exceeds absolute latency cap in batch {b}")
            if not result["signal_only"]:
                row["speedup"] = ratio(row["comparator_us"], row["candidate_us"])
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
        if not all(math.isfinite(v) for v in contrasts.values()):
            raise ValueError("Weighted paired contrast must be finite")
        if not result["signal_only"]:
            gains.append(1 - ratio(totals["candidate"], totals["comparator"]))
            paired_gains.append(ratio(-contrasts["comparator"], totals["comparator"]))
            guards.append(
                ratio(
                    cfg["noise_multiplier"] * uncertainties["comparator"],
                    totals["comparator"],
                )
            )
            if mode == "full":
                baseline_role = "baseline" if "baseline" in totals else "comparator"
                baseline_gains.append(1 - ratio(totals["candidate"], totals[baseline_role]))
                baseline_paired_gains.append(
                    ratio(-contrasts[baseline_role], totals[baseline_role])
                )
                baseline_guards.append(
                    ratio(
                        cfg["noise_multiplier"] * uncertainties[baseline_role],
                        totals[baseline_role],
                    )
                )
    decision["cases"] = summaries
    if result["signal_only"]:
        decision["reasons"].append("Candidate-only signal has no measured comparator")
        return decision
    decision["gain_by_batch"] = gains
    decision["paired_gain_by_batch"] = paired_gains
    decision["noise_guard_by_batch"] = guards
    if baseline_gains:
        decision["fixed_baseline_gain_by_batch"] = baseline_gains
        decision["fixed_baseline_paired_gain_by_batch"] = baseline_paired_gains
        decision["fixed_baseline_noise_guard_by_batch"] = baseline_guards
        if not all(
            g > cfg["min_gain"] and p > max(cfg["min_gain"], n)
            for g, p, n in zip(baseline_gains, baseline_paired_gains, baseline_guards)
        ):
            decision["reasons"].append(
                "Meaningful gain over fixed baseline is not confirmed in every batch"
            )
    good = all(
        g > cfg["min_gain"] and p > max(cfg["min_gain"], n)
        for g, p, n in zip(gains, paired_gains, guards)
    )
    bad = all(
        g < -cfg["min_gain"] and p < -max(cfg["min_gain"], n)
        for g, p, n in zip(gains, paired_gains, guards)
    )
    decision["performance"] = "faster" if good else "slower" if bad else "inconclusive"
    if not good:
        decision["reasons"].append(
            "Aggregate gain and paired gain do not both clear their thresholds in every batch"
        )
    if mode != "full":
        decision["reasons"].append("Quick results cannot update best")
    if result.get("same_candidate_as_best"):
        decision["reasons"].append(
            "Candidate source is identical to best; no implementation change"
        )
    decision["eligible_for_best"] = mode == "full" and good and not decision["reasons"]
    return decision
