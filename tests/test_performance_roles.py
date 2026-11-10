"""Guard measurements cannot create a win or disappear from acceptance evidence."""

import copy
import json

import pytest
from test_scoring import evidence

from kernel_envolve.cli import evaluate, promote
from kernel_envolve.native import evaluate_native, promote_native
from kernel_envolve.native_config import load_config
from kernel_envolve.scoring import score
from kernel_envolve.storage import config, contract

pytest_plugins = ["test_native"]


def guarded_evidence(**kwargs):
    result, cfg = evidence(**kwargs)
    cfg["cases"][1] = {"id": "b", "performance_role": "guard"}
    return result, cfg


def test_slow_public_guard_does_not_dilute_target_gain():
    result, cfg = guarded_evidence(compare_best=True)
    for row in result["samples"]:
        if row["case"] == "b":
            row["us"] = 100000  # A long unchanged API must not dilute an 8 vs 10 us kernel.
    decision = score(result, cfg)
    assert decision["eligible_for_best"]
    assert decision["gain_by_batch"] == pytest.approx([0.2, 0.2])
    assert decision["fixed_baseline_gain_by_batch"] == pytest.approx([0.2, 0.2])
    assert decision["coverage"]["objective_cases"] == ["a"]
    assert decision["coverage"]["guard_cases"] == ["b"]
    assert decision["coverage"]["timed_cases"] == ["a", "b"]
    assert all(
        row["performance_role"] == "guard" for row in decision["cases"] if row["case"] == "b"
    )


@pytest.mark.parametrize("target", [10, 12])
def test_guard_speedup_cannot_hide_unchanged_or_slower_objective(target):
    result, cfg = guarded_evidence()
    for row in result["samples"]:
        if row["case"] == "b":
            row["us"] = 1 if row["role"] == "candidate" else 100000
        elif row["role"] == "candidate":
            row["us"] = target
    decision = score(result, cfg)
    assert not decision["eligible_for_best"]
    assert decision["performance"] == ("inconclusive" if target == 10 else "slower")
    assert decision["gain_by_batch"] == pytest.approx([1 - target / 10] * 2)


def test_guard_noise_is_reported_without_entering_objective_uncertainty():
    result, cfg = guarded_evidence()
    for row in result["samples"]:
        if row["case"] == "b":
            row["us"] = [50, 100, 150][row["trial"]] if row["role"] == "candidate" else 100
    decision = score(result, cfg)
    assert decision["eligible_for_best"]
    assert decision["noise_guard_by_batch"] == [0, 0]
    assert all(
        row["paired_comparator_mad_us"] > 0 for row in decision["cases"] if row["case"] == "b"
    )


def test_guard_speedup_cannot_hide_noisy_objective():
    result, cfg = guarded_evidence()
    for row in result["samples"]:
        if row["case"] == "a" and row["role"] == "candidate":
            row["us"] = [1, 8, 15][row["trial"]]
        if row["case"] == "b":
            row["us"] = 1 if row["role"] == "candidate" else 100000
    assert not score(result, cfg)["eligible_for_best"]


@pytest.mark.parametrize("limit", ["regression", "fixed_baseline", "max_us"])
def test_guard_limits_block_an_objective_win(limit):
    result, cfg = guarded_evidence(compare_best=True)
    values = {"candidate": 11, "comparator": 10, "baseline": 11}
    expected = "regression allowance"
    if limit == "fixed_baseline":
        values = {"candidate": 11, "comparator": 11, "baseline": 10}
        expected = "fixed baseline budget"
    if limit == "max_us":
        values = {"candidate": 11, "comparator": 11, "baseline": 11}
        cfg["cases"][1]["max_us"] = 10
        expected = "absolute latency cap"
    for row in result["samples"]:
        if row["case"] == "b":
            row["us"] = values[row["role"]]
    decision = score(result, cfg)
    assert decision["performance"] == "faster"
    assert not decision["eligible_for_best"]
    assert any("b:" in reason and expected in reason for reason in decision["reasons"])


@pytest.mark.parametrize("field", ["checks", "samples"])
def test_guard_coverage_cannot_be_removed(field):
    result, cfg = guarded_evidence()
    result[field] = [row for row in result[field] if row["case"] != "b"]
    with pytest.raises(ValueError, match="coverage"):
        score(result, cfg)


@pytest.mark.parametrize("value", [0, -1, float("nan"), float("inf")])
def test_invalid_guard_timing_is_rejected(value):
    result, cfg = guarded_evidence()
    next(row for row in result["samples"] if row["case"] == "b")["us"] = value
    with pytest.raises(ValueError, match="finite and positive"):
        score(result, cfg)


def test_explicit_objective_matches_legacy_scoring():
    result, cfg = evidence(compare_best=True)
    legacy = score(result, cfg)
    for case in cfg["cases"]:
        case["performance_role"] = "objective"
    explicit = score(result, cfg)
    for row in explicit["cases"]:
        assert row.pop("performance_role") == "objective"
    assert explicit == legacy


def test_guard_correctness_failure_cannot_be_ignored():
    result, cfg = guarded_evidence()
    next(row for row in result["checks"] if row["case"] == "b")["status"] = "fail"
    with pytest.raises(ValueError, match="contradicts"):
        score(result, cfg)
    result["correctness"] = "fail"
    assert not score(result, cfg)["eligible_for_best"]


@pytest.mark.parametrize("mode,signal", [("quick", False), ("quick", True)])
def test_guarded_diagnostics_never_promote(mode, signal):
    result, cfg = guarded_evidence(mode=mode, signal_only=signal)
    assert not score(result, cfg)["eligible_for_best"]


def load_cases(task, kind, cases):
    if kind == "generic":
        path, loader = task / "task.json", config
        raw = json.loads(path.read_text())
    else:
        path, loader = task / "native.json", load_config
        (task / "adapter.py").write_text("# Configuration-only fixture\n")
        raw = {
            "name": "configuration",
            "objective": "target latency",
            "device": "cpu",
            "sources": {"candidate": {"core": str(task)}, "baseline": {"core": str(task)}},
            "imports": {"fixture": "core"},
        }
    raw["cases"] = cases
    path.write_text(json.dumps(raw))
    return loader(task)


@pytest.mark.parametrize("kind", ["generic", "native"])
def test_guard_config_requires_no_weight_and_preserves_legacy_defaults(task, kind):
    cases = [
        {"id": "target", "args": {}, "weight": 2},
        {"id": "api", "args": {}, "performance_role": "guard", "max_us": 100},
    ]
    loaded = load_cases(task, kind, cases)
    assert loaded["cases"] == cases
    assert "performance_role" not in loaded["cases"][0]


@pytest.mark.parametrize("kind", ["generic", "native"])
@pytest.mark.parametrize(
    "change",
    [
        {"performance_role": "unknown"},
        {"performance_role": None},
        {"performance_role": "guard", "weight": 1},
        {"performance_role": "guard", "weight": 0},
        {"performance_role": "guard", "measure": False},
        {"performance_role": "objective", "measure": False},
    ],
)
def test_invalid_performance_role_contract_is_rejected(task, kind, change):
    cases = [{"id": "target", "args": {}, "weight": 1}, {"id": "other", "args": {}, **change}]
    with pytest.raises(ValueError):
        load_cases(task, kind, cases)


@pytest.mark.parametrize("kind", ["generic", "native"])
@pytest.mark.parametrize("only_full_objective", [False, True])
def test_each_mode_requires_a_measured_objective(task, kind, only_full_objective):
    cases = [{"id": "api", "args": {}, "performance_role": "guard"}]
    if only_full_objective:
        cases.append({"id": "target", "args": {}, "weight": 1, "quick": False})
    with pytest.raises(ValueError, match="objective"):
        load_cases(task, kind, cases)


def test_generic_full_keeps_guard_evidence_and_freezes_role(task):
    path = task / "task.json"
    raw = json.loads(path.read_text())
    guard = copy.deepcopy(raw["cases"][0])
    guard.update(id="api", performance_role="guard")
    guard.pop("weight")
    raw["cases"].append(guard)
    path.write_text(json.dumps(raw))
    frozen = contract(task, config(task))
    run, result = evaluate(task, "full")
    assert result["decision"]["eligible_for_best"], result
    assert any(row["case"] == "api" for row in result["checks"])
    assert any(row["case"] == "api" for row in result["samples"])
    promote(run)
    raw["cases"][-1].update(performance_role="objective", weight=1)
    path.write_text(json.dumps(raw))
    assert contract(task, config(task)) != frozen
    with pytest.raises(ValueError, match="Live task contract changed"):
        promote(run)


def test_guard_only_selection_is_correctness_diagnostic(task):
    path = task / "task.json"
    raw = json.loads(path.read_text())
    raw["cases"][1]["performance_role"] = "guard"
    raw["cases"][1].pop("weight")
    path.write_text(json.dumps(raw))
    name = raw["cases"][1]["id"]
    with pytest.raises(ValueError, match="No objective cases"):
        evaluate(task, cases=[name])
    _, result = evaluate(task, cases=[name], check_only=True)
    assert result["correctness"] == "pass"
    assert not result["samples"]
    assert not result["decision"]["eligible_for_best"]


def test_native_full_preserves_guard_gradients_and_all_native_checks(native_task):
    task, _ = native_task
    path = task / "native.json"
    raw = json.loads(path.read_text())
    raw["cases"].append({"id": "api", "args": {}, "performance_role": "guard"})
    path.write_text(json.dumps(raw))
    run, result = evaluate_native(task, "full", timeout=90)
    assert result["decision"]["eligible_for_best"], result.get("error") or result["decision"]
    assert len(result["native_checks"]) == 3
    assert any(row["case"] == "api" for row in result["checks"])
    assert any(row["case"] == "api" for row in result["samples"])
    promote_native(run)
