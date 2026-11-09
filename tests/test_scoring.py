import copy

import pytest

from kernel_envolve.protocol import order, post_seed, roles, timing_seed
from kernel_envolve.scoring import score


def evidence(mode="full", signal_only=False, compare_best=False):
    cfg = {
        "cases": [{"id": "a", "weight": 1}, {"id": "b", "weight": 2}],
        "quick_seeds": [0],
        "full_seeds": [0, 1],
        "quick_trials": 3,
        "full_trials": 3,
        "min_gain": 0.02,
        "noise_multiplier": 2,
        "max_regression": 0.05,
        "max_baseline_regression": 0.05,
    }
    result = {
        "mode": mode,
        "comparator": {"kind": "best" if compare_best else "baseline"},
        "signal_only": signal_only,
        "execution": "completed",
        "correctness": "pass",
        "checks": [],
        "samples": [],
    }
    implementations = roles(result)
    seeds = cfg[f"{mode}_seeds"]
    for case in cfg["cases"]:
        for seed in seeds:
            for role in implementations:
                for kind in ("fresh", "reuse"):
                    result["checks"].append(
                        {
                            "case": case["id"],
                            "seed": seed,
                            "role": role,
                            "kind": kind,
                            "status": "pass",
                        }
                    )
        for role in implementations:
            result["checks"].append(
                {
                    "case": case["id"],
                    "seed": post_seed(case["id"], seeds),
                    "role": role,
                    "kind": "post",
                    "status": "pass",
                }
            )
            result["checks"].append(
                {
                    "case": case["id"],
                    "seed": timing_seed(case["id"], (2 if mode == "full" else 1) - 1, 2, 1, seeds),
                    "role": role,
                    "kind": "timed_output",
                    "status": "pass",
                }
            )
    for b in range(2 if mode == "full" else 1):
        for case in cfg["cases"] if b == 0 else reversed(cfg["cases"]):
            for t in range(3):
                repeats = dict.fromkeys(implementations, 0)
                for role in order(implementations, b, t):
                    repeat = repeats[role]
                    result["samples"].append(
                        {
                            "case": case["id"],
                            "batch": b,
                            "trial": t,
                            "seed": timing_seed(case["id"], b, t, repeat, seeds),
                            "role": role,
                            "repeat": repeat,
                            "us": 8 if role == "candidate" else 10,
                        }
                    )
                    repeats[role] += 1
    return result, cfg


def test_two_batches_required_for_promotion():
    result, cfg = evidence()
    assert score(result, cfg)["eligible_for_best"]
    for row in result["samples"]:
        if row["batch"] == 1 and row["role"] == "candidate":
            row["us"] = 10
    assert not score(result, cfg)["eligible_for_best"]


@pytest.mark.parametrize("mode,signal", [("quick", False), ("quick", True)])
def test_quick_never_promotes(mode, signal):
    result, cfg = evidence(mode, signal)
    decision = score(result, cfg)
    assert not decision["eligible_for_best"]
    if signal:
        assert "gain_by_batch" not in decision
        assert decision["performance"] == "not_measured"


@pytest.mark.parametrize("field", ["checks", "samples"])
def test_missing_and_duplicate_coverage(field):
    result, cfg = evidence()
    result[field].pop()
    with pytest.raises(ValueError, match="coverage"):
        score(result, cfg)
    result[field].append(copy.deepcopy(result[field][0]))
    with pytest.raises(ValueError, match="coverage"):
        score(result, cfg)


@pytest.mark.parametrize("value", [0, -1, float("inf"), float("nan")])
def test_invalid_time(value):
    result, cfg = evidence()
    result["samples"][0]["us"] = value
    with pytest.raises(ValueError, match="finite and positive"):
        score(result, cfg)


def test_seed_is_part_of_coverage_identity():
    result, cfg = evidence()
    result["samples"][0]["seed"] = 999
    with pytest.raises(ValueError, match="coverage"):
        score(result, cfg)


def test_weighted_latency_not_mean_speedup():
    result, cfg = evidence()
    cfg["max_regression"] = 0.9
    for row in result["samples"]:
        if row["case"] == "a":
            row["us"] = 1 if row["role"] == "candidate" else 10
        else:
            row["us"] = 120 if row["role"] == "candidate" else 100
    # Mean speedup > 5x, but total weighted latency grows 210 -> 241.
    assert score(result, cfg)["performance"] == "slower"


def test_noise_rejects_apparent_gain():
    result, cfg = evidence()
    for row in result["samples"]:
        if row["role"] == "candidate":
            row["us"] = [5, 8, 11][row["trial"]]
    assert score(result, cfg)["performance"] == "inconclusive"


def test_case_caps_and_regression():
    result, cfg = evidence()
    cfg["cases"][0]["max_us"] = 7
    assert not score(result, cfg)["eligible_for_best"]
    del cfg["cases"][0]["max_us"]
    cfg["cases"][0]["weight"] = 0.001
    for row in result["samples"]:
        if row["case"] == "a" and row["role"] == "candidate":
            row["us"] = 11
    decision = score(result, cfg)
    assert decision["performance"] == "faster"
    assert not decision["eligible_for_best"]


def test_numeric_failure_is_completed_but_ineligible():
    result, cfg = evidence()
    result["correctness"] = "fail"
    assert not score(result, cfg)["eligible_for_best"]


def test_cumulative_regression_cannot_hide_behind_better_aggregate():
    result, cfg = evidence(compare_best=True)
    for r in result["samples"]:
        # a: 100 -> prior best 104 -> candidate 108; this step fits 5%, total does not.
        # b: 100 -> 90 -> 70; this dominates the aggregate, but cannot excuse a.
        values = {
            "a": {"baseline": 100, "comparator": 104, "candidate": 108},
            "b": {"baseline": 100, "comparator": 90, "candidate": 70},
        }
        r["us"] = values[r["case"]][r["role"]]
    decision = score(result, cfg)
    assert decision["performance"] == "faster"
    assert not decision["eligible_for_best"]
    assert any("fixed baseline budget" in r for r in decision["reasons"])


def test_improving_degraded_best_is_not_original_baseline_win():
    result, cfg = evidence(compare_best=True)
    cfg["max_baseline_regression"] = 0.5
    for r in result["samples"]:
        r["us"] = {"baseline": 10, "comparator": 20, "candidate": 15}[r["role"]]
    decision = score(result, cfg)
    assert decision["performance"] == "faster"
    assert not decision["eligible_for_best"]
    assert any("Meaningful gain over fixed baseline" in r for r in decision["reasons"])


def test_three_role_measurement_can_qualify():
    result, cfg = evidence(compare_best=True)
    assert score(result, cfg)["eligible_for_best"]
    result["samples"] = [r for r in result["samples"] if r["role"] != "baseline"]
    with pytest.raises(ValueError, match="coverage"):
        score(result, cfg)


def test_wrong_temporal_order_rejected_even_with_complete_coverage():
    result, cfg = evidence()
    result["samples"].sort(key=lambda r: r["role"])
    with pytest.raises(ValueError, match="paired schedule"):
        score(result, cfg)


def test_post_timing_check_is_required():
    result, cfg = evidence()
    result["checks"] = [r for r in result["checks"] if r["kind"] != "post"]
    with pytest.raises(ValueError, match="coverage"):
        score(result, cfg)


def test_identical_source_cannot_be_promoted_by_noise():
    result, cfg = evidence(compare_best=True)
    result["same_candidate_as_best"] = True
    assert not score(result, cfg)["eligible_for_best"]


def test_fixed_baseline_noise_cannot_be_used_to_confirm_original_gain():
    result, cfg = evidence(compare_best=True)
    for r in result["samples"]:
        if r["role"] == "baseline":
            r["us"] = [5, 10, 15][r["trial"]]
    assert not score(result, cfg)["eligible_for_best"]
