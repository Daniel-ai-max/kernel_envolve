"""Allocation safety boundaries on explicit result data; no efficacy simulation."""

import copy

import pytest

from kernel_envolve.rsi.allocation import artifact_action_budget


@pytest.fixture
def current():
    cfg = {
        "device": "cpu",
        "max_regression": 0.05,
        "max_baseline_regression": 0.05,
        "cases": [
            {"id": "objective", "weight": 1},
            {"id": "guard", "performance_role": "guard"},
            {"id": "correctness-only", "measure": False},
        ],
    }
    result = {
        "mode": "full",
        "execution": "completed",
        "correctness": "pass",
        "signal_only": False,
        "check_only": False,
        "selection": {},
        "environment": {"packages": {"numpy": "fixture"}, "runtime_settings": {"threads": 1}},
        "comparator": {"kind": "baseline"},
        "decision": {
            "eligible_for_best": True,
            "cases": [
                {"case": case, "batch": batch, "candidate_us": 80.0, "comparator_us": 100.0}
                for case in ("objective", "guard")
                for batch in range(2)
            ],
        },
    }
    return result, cfg


def test_complete_current_margins_shorten_without_changing_inputs(current):
    result, cfg = current
    original = copy.deepcopy(current)
    decision = artifact_action_budget(result, cfg)
    assert decision["actions"] == 2
    assert decision["case_batch_count"] == 4
    assert decision["baseline_margin"] == 0.025
    assert current == original


def test_late_guard_case_near_limit_retains_four(current):
    result, cfg = current
    result["decision"]["cases"][-1]["candidate_us"] = 104
    decision = artifact_action_budget(result, cfg)
    assert decision["actions"] == 4
    assert decision["reason"] == "near_performance_limit"
    # The evaluator's eligibility flag and thresholds are not rewritten.
    assert result["decision"]["eligible_for_best"] is True
    assert cfg["max_baseline_regression"] == 0.05


def test_best_comparator_is_not_a_fixed_baseline(current):
    result, cfg = current
    result["comparator"]["kind"] = "best"
    assert artifact_action_budget(result, cfg)["actions"] == 4
    for row in result["decision"]["cases"]:
        row["baseline_us"] = 78  # Candidate looks faster than comparator, slower than baseline.
    assert artifact_action_budget(result, cfg)["actions"] == 4
    for row in result["decision"]["cases"]:
        row["baseline_us"] = 120
    assert artifact_action_budget(result, cfg)["actions"] == 2
    for row in result["decision"]["cases"]:
        row["comparator_us"] = 77  # Safely beats baseline but approaches incumbent's limit.
    assert artifact_action_budget(result, cfg)["actions"] == 4


def test_all_cases_not_a_truncated_model_view_are_required(current):
    result, cfg = current
    cfg["cases"] = [{"id": str(i), "weight": 1} for i in range(12)]
    result["decision"]["cases"] = [
        {"case": str(i), "batch": batch, "candidate_us": 80, "comparator_us": 100}
        for batch in range(2)
        for i in range(12)
    ]
    result["decision"]["cases"][-1]["candidate_us"] = 104
    assert artifact_action_budget(result, cfg)["actions"] == 4
    result["decision"]["cases"] = result["decision"]["cases"][:16]
    assert artifact_action_budget(result, cfg)["reason"] == "incomplete_coverage"


def test_ambiguous_baseline_and_overlarge_coverage_do_not_shorten(current):
    result, cfg = current
    result["decision"]["cases"][0]["baseline_us"] = 200
    assert artifact_action_budget(result, cfg)["reason"] == "ambiguous_baseline"
    cfg["cases"] *= 100
    assert artifact_action_budget(result, cfg)["reason"] == "unsupported_coverage"


@pytest.mark.parametrize(
    "corruption",
    [
        "missing_batch",
        "duplicate_batch",
        "boolean_batch",
        "unknown_case",
        "nan",
        "infinity",
        "boolean_time",
        "zero_time",
        "overflow",
        "not_full",
        "not_eligible",
        "not_correct",
        "signal_only",
        "check_only",
        "selected_cases",
        "runtime_missing",
        "runtime_device",
        "unknown_comparator",
        "absolute_cap",
        "invalid_limit",
    ],
)
def test_incomplete_or_unsupported_data_never_shortens(current, corruption):
    result, cfg = current
    rows = result["decision"]["cases"]
    if corruption == "missing_batch":
        rows.pop()
    elif corruption == "duplicate_batch":
        rows[-1] = copy.deepcopy(rows[-2])
    elif corruption == "boolean_batch":
        rows[-1]["batch"] = True
    elif corruption == "unknown_case":
        rows[-1]["case"] = "not-a-task-case"
    elif corruption in {"nan", "infinity", "boolean_time", "zero_time"}:
        rows[-1]["candidate_us"] = {
            "nan": float("nan"),
            "infinity": float("inf"),
            "boolean_time": True,
            "zero_time": 0,
        }[corruption]
    elif corruption == "overflow":
        rows[-1].update(candidate_us=1e308, comparator_us=1e-308)
    elif corruption == "not_full":
        result["mode"] = "quick"
    elif corruption == "not_eligible":
        result["decision"]["eligible_for_best"] = False
    elif corruption == "not_correct":
        result["correctness"] = "fail"
    elif corruption in {"signal_only", "check_only"}:
        result[corruption] = True
    elif corruption == "selected_cases":
        result["selection"] = {"cases": ["objective"]}
    elif corruption == "runtime_missing":
        del result["environment"]
    elif corruption == "runtime_device":
        cfg["device"] = "cuda:0"
    elif corruption == "unknown_comparator":
        result["comparator"]["kind"] = "historical"
    elif corruption == "absolute_cap":
        cfg["cases"][1]["max_us"] = 100
    elif corruption == "invalid_limit":
        cfg["max_baseline_regression"] = True
    assert artifact_action_budget(result, cfg)["actions"] == 4
