"""Broad correctness coverage stays mandatory without repeated timing of every case."""

import copy
import json

import pytest

from kernel_envolve.cli import evaluate, promote
from kernel_envolve.scoring import score
from kernel_envolve.storage import config


@pytest.fixture
def broad_task(task):
    cfg = json.loads((task / "task.json").read_text())
    cfg["cases"] += [
        {
            "id": f"guard-{n}",
            "args": {"n": n, "stride": 3},
            "measure": False,
            "quick": False,
        }
        for n in (255, 256, 257)
    ]
    (task / "task.json").write_text(json.dumps(cfg))
    return task


def test_broad_correctness_is_gated_but_not_timed(broad_task):
    run, result = evaluate(broad_task, "full")
    assert result["correctness"] == "pass", result
    assert result["decision"]["eligible_for_best"], result
    coverage = result["decision"]["coverage"]
    assert coverage["correctness_cases"] == [
        "contiguous",
        "strided",
        "guard-255",
        "guard-256",
        "guard-257",
    ]
    assert coverage["timed_cases"] == ["contiguous", "strided"]
    assert len(result["samples"]) == 48
    assert len(result["checks"]) == 68
    assert {r["case"] for r in result["decision"]["cases"]} == {"contiguous", "strided"}
    promote(run)
    # The third implementation (fixed baseline) must pass the broader cases too.
    _, next_result = evaluate(broad_task, "full")
    assert next_result["correctness"] == "pass", next_result
    assert len(next_result["samples"]) == 72
    assert len(next_result["checks"]) == 102
    assert {r["role"] for r in next_result["checks"] if r["case"].startswith("guard-")} == {
        "candidate",
        "comparator",
        "baseline",
    }


def test_size_specific_bug_blocks_full_despite_passing_quick(broad_task):
    (broad_task / "candidate.py").write_text(
        "def run(x,y):\n return x-y if x.size == 257 else x+y\n"
    )
    _, quick = evaluate(broad_task)
    assert quick["correctness"] == "pass"
    run, full = evaluate(broad_task, "full")
    assert full["correctness"] == "fail"
    assert full["checks"][-1]["case"] == "guard-257"
    assert not full["samples"]
    with pytest.raises(ValueError, match="Not eligible"):
        promote(run)


def test_guard_cannot_be_removed_from_evidence_or_given_timing(broad_task):
    _, result = evaluate(broad_task, "full")
    cfg = config(broad_task)
    missing = copy.deepcopy(result)
    missing["checks"] = [r for r in missing["checks"] if r["case"] != "guard-256"]
    with pytest.raises(ValueError, match="correctness coverage"):
        score(missing, cfg)
    extra = copy.deepcopy(result)
    extra["samples"].append({**extra["samples"][0], "case": "guard-256"})
    with pytest.raises(ValueError, match="timing coverage"):
        score(extra, cfg)


def test_unmeasured_case_can_be_diagnosed_without_a_performance_claim(broad_task):
    for signal in (False, True):
        with pytest.raises(ValueError, match="--check-only"):
            evaluate(broad_task, cases=["guard-257"], signal_only=signal)
    _, result = evaluate(broad_task, cases=["guard-257"], check_only=True)
    assert result["correctness"] == "pass"
    assert result["decision"]["coverage"] == {"correctness_cases": ["guard-257"], "timed_cases": []}
    assert result["decision"]["performance"] == "not_measured"
    assert not result["decision"]["eligible_for_best"]


def test_changing_coverage_cannot_reuse_an_existing_promotion(broad_task):
    run, result = evaluate(broad_task, "full")
    assert result["decision"]["eligible_for_best"], result
    cfg = json.loads((broad_task / "task.json").read_text())
    cfg["cases"] = cfg["cases"][:2]
    (broad_task / "task.json").write_text(json.dumps(cfg))
    with pytest.raises(ValueError, match="Live task contract changed"):
        promote(run)


@pytest.mark.parametrize(
    "change, message",
    [
        ({"measure": "false"}, "boolean"),
        ({"measure": 0}, "boolean"),
        ({"weight": 1}, "Unmeasured cases"),
        ({"max_us": 100}, "Unmeasured cases"),
    ],
)
def test_misleading_measurement_config_is_rejected(broad_task, change, message):
    cfg = json.loads((broad_task / "task.json").read_text())
    cfg["cases"][-1].update(change)
    (broad_task / "task.json").write_text(json.dumps(cfg))
    with pytest.raises(ValueError, match=message):
        config(broad_task)


def test_task_needs_a_measured_quick_case(broad_task):
    cfg = json.loads((broad_task / "task.json").read_text())
    for case in cfg["cases"]:
        case["measure"] = False
        case.pop("weight", None)
    (broad_task / "task.json").write_text(json.dumps(cfg))
    with pytest.raises(ValueError, match="measured quick case"):
        config(broad_task)


def test_row_padding_and_independent_input_strides_are_covered(task):
    cfg = json.loads((task / "task.json").read_text())
    cfg["cases"] = [
        {"id": "packed", "args": {"pad": 0}, "weight": 1},
        {"id": "padded", "args": {"pad": 7}, "measure": False, "quick": False},
    ]
    (task / "task.json").write_text(json.dumps(cfg))
    (task / "task.py").write_text(
        "import numpy as np\n"
        "def make_inputs(case,seed):\n"
        " g=np.random.default_rng(seed)\n"
        " x=g.normal(size=(3,33+case['pad'])).astype(np.float32)[:,:33]\n"
        " y=g.normal(size=(3,99)).astype(np.float32)[:,::3]\n"
        " return x,y\n"
        "def reference(x,y): return x+y\n"
        "baseline=reference\n"
    )
    # Model the frozen GPU kernels' assumption that row stride is width * column stride.
    (task / "candidate.py").write_text(
        "import numpy as np\n"
        "def run(x,y):\n"
        " packed=np.lib.stride_tricks.as_strided(x,shape=x.shape,"
        "strides=(x.shape[1]*x.strides[1],x.strides[1]))\n"
        " return packed+y\n"
    )
    _, quick = evaluate(task)
    assert quick["correctness"] == "pass"
    _, full = evaluate(task, "full")
    assert full["correctness"] == "fail"
    assert full["checks"][-1]["case"] == "padded"
    assert not full["samples"]
    (task / "candidate.py").write_text("def run(x,y): return x+y\n")
    _, repaired = evaluate(task, "full")
    assert repaired["correctness"] == "pass", repaired
