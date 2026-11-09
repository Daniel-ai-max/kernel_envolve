import copy

import pytest

from kernel_envolve.scoring import score


def evidence(mode="full", signal_only=False):
    cfg = {
        "cases": [{"id": "a", "weight": 1}, {"id": "b", "weight": 2}],
        "quick_seeds": [0],
        "full_seeds": [0, 1],
        "quick_trials": 3,
        "full_trials": 3,
        "min_gain": 0.02,
        "noise_multiplier": 2,
        "max_regression": 0.05,
    }
    result = {
        "mode": mode,
        "signal_only": signal_only,
        "execution": "completed",
        "correctness": "pass",
        "checks": [],
        "samples": [],
    }
    roles = ["candidate"] if signal_only else ["candidate", "comparator"]
    for case in cfg["cases"]:
        for seed in cfg[f"{mode}_seeds"]:
            for role in roles:
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
        for b in range(2 if mode == "full" else 1):
            for t in range(3):
                for role in roles:
                    for repeat in range(2):
                        result["samples"].append(
                            {
                                "case": case["id"],
                                "batch": b,
                                "trial": t,
                                "seed": cfg[f"{mode}_seeds"][t % len(cfg[f"{mode}_seeds"])],
                                "role": role,
                                "repeat": repeat,
                                "us": 8 if role == "candidate" else 10,
                            }
                        )
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
