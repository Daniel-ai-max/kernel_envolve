"""Optional action allocation from a current full result, never a promotion gate.

The caller supplies the evaluator's verified, untruncated result. A half-budget
margin is an engineering heuristic, not a confidence bound or future guarantee.
No historical score, model assessment, I/O, or extra measurement is used here.
"""

import math

from ..protocol import measured_cases, objective_cases

POLICY = "artifact-margin-v1"


def artifact_action_budget(result, cfg):
    """Shorten four ordinary actions to two only with complete current margins.

    Unsupported, missing or ambiguous evidence retains four actions. Both objective
    and guard cases count. Absolute latency caps are not modeled by this heuristic,
    so their presence also retains four. All original evaluator gates still apply.
    """
    decision = {"policy": POLICY, "actions": 4, "reason": "incomplete_result"}

    def keep(reason):
        return {**decision, "reason": reason}

    def finite(value):
        return type(value) in (int, float) and math.isfinite(value)

    try:
        if (
            result["mode"] != "full"
            or result["execution"] != "completed"
            or result["correctness"] != "pass"
            or result.get("signal_only") is not False
            or result.get("check_only") is not False
            or result.get("selection") != {}
            or result["decision"]["eligible_for_best"] is not True
        ):
            return keep("not_eligible_full")
        # Import lazily: reuse also exposes the research-loop proposal interface.
        from .reuse import runtime_key

        runtime = runtime_key(result["environment"])
        if str(cfg["device"]).startswith("cuda") != (runtime["gpu"] is not None):
            return keep("runtime_device_mismatch")
        limits = [cfg["max_baseline_regression"], cfg["max_regression"]]
        if not all(finite(v) and 0 <= v < 1 for v in limits):
            return keep("invalid_limits")
        if not isinstance(cfg["cases"], list) or len(cfg["cases"]) > 256:
            return keep("unsupported_coverage")
        cases = measured_cases(cfg["cases"])
        if not 1 <= len(cases) <= 128 or not objective_cases(cases):
            return keep("unsupported_coverage")
        if any("max_us" in case for case in cases):
            return keep("absolute_case_limit")
        ids = [case["id"] for case in cases]
        if any(not isinstance(i, str) for i in ids) or len(set(ids)) != len(ids):
            return keep("ambiguous_coverage")
        expected = {(identity, batch) for identity in ids for batch in range(2)}
        rows = result["decision"]["cases"]
        if not isinstance(rows, list) or len(rows) != len(expected):
            return keep("incomplete_coverage")
        actual = set()
        baseline_slowdowns, comparator_slowdowns = [], []
        kind = result["comparator"]["kind"]
        if kind not in {"baseline", "best"}:
            return keep("unknown_comparator")
        for row in rows:
            if not isinstance(row["case"], str) or type(row["batch"]) is not int:
                return keep("ambiguous_coverage")
            actual.add((row["case"], row["batch"]))
            candidate, comparator = row["candidate_us"], row["comparator_us"]
            # A best-so-far comparator is never a substitute for the fixed baseline.
            baseline = comparator if kind == "baseline" else row.get("baseline_us")
            if kind == "baseline" and "baseline_us" in row and row["baseline_us"] != comparator:
                return keep("ambiguous_baseline")
            if not all(finite(v) and v > 0 for v in (candidate, comparator, baseline)):
                return keep("invalid_timings")
            a, b = candidate / baseline - 1, candidate / comparator - 1
            if not math.isfinite(a) or not math.isfinite(b):
                return keep("invalid_timings")
            baseline_slowdowns.append(a)
            comparator_slowdowns.append(b)
        if actual != expected:
            return keep("ambiguous_coverage")
        baseline_max, comparator_max = max(baseline_slowdowns), max(comparator_slowdowns)
        short = baseline_max <= limits[0] * 0.5 and comparator_max <= limits[1] * 0.5
        return {
            "policy": POLICY,
            "actions": 2 if short else 4,
            "reason": "within_margin" if short else "near_performance_limit",
            "case_batch_count": len(rows),
            "max_baseline_slowdown": baseline_max,
            "max_comparator_slowdown": comparator_max,
            "baseline_margin": limits[0] * 0.5,
            "comparator_margin": limits[1] * 0.5,
        }
    except (KeyError, TypeError, ValueError, OverflowError):
        return decision
