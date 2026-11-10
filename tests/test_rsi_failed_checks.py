"""Preserve verified failure conditions without inventing diagnostic coverage."""

import json
from pathlib import Path

import pytest

from kernel_envolve.cli import evaluate
from kernel_envolve.manager.evidence import failed_checks, inspect_run
from kernel_envolve.rsi.research import observation
from kernel_envolve.storage import config, contract, write_json


def test_failed_check_bounds_do_not_imply_unlisted_cases_passed():
    args = {"dtype": "bfloat16", "stride": 3, "shape": [513, 2049]}
    cfg = {"cases": [{"id": "wide", "args": args}, {"id": "narrow", "args": {}}]}
    result = {
        "checks": [
            {"case": "wide", "status": "fail", "role": "candidate", "kind": "reuse", "seed": 11}
        ]
    }
    value = failed_checks(result, cfg)
    assert value["checks"][0]["args"] == args
    assert value["checks"][0]["kind"] == "reuse"
    assert value["count"] == 1 and not value["truncated"]
    assert "unlisted cases are unknown" in value["scope"]
    assert all(row["case"] != "narrow" for row in value["checks"])

    cfg["cases"][0]["args"] = {"large": "字" * 512}
    result["checks"] *= 20
    bounded = failed_checks(result, cfg)
    assert bounded["count"] == 20 and bounded["truncated"]
    assert len(bounded["checks"]) == 8
    assert all(row["args"] is None and row["args_truncated"] for row in bounded["checks"])
    assert len(json.dumps(bounded, ensure_ascii=False).encode()) < 8192


def test_no_checks_is_not_claimed_complete_coverage():
    value = failed_checks({"execution": "error"}, {"cases": []})
    assert value["checks"] == [] and value["count"] == 0
    assert "may stop early" in value["scope"]


def test_verified_cpu_failure_preserves_frozen_case_and_detects_tampering(tmp_path):
    task = tmp_path / "task"
    task.mkdir()
    (task / "task.py").write_text(
        "import numpy as np\n"
        "def make_inputs(case,seed): return (np.ones(case['n']),)\n"
        "def reference(x): return x+1\n"
        "def baseline(x): return x+1\n"
    )
    (task / "candidate.py").write_text("def run(x): return x+2\n")
    write_json(
        task / "task.json",
        {
            "name": "failed-case-test",
            "objective": "Synthetic CPU provenance test",
            "device": "cpu",
            "cases": [{"id": "narrow", "args": {"n": 7}, "weight": 1}],
            "full_trials": 3,
            "quick_trials": 1,
            "full_seeds": [11],
            "quick_seeds": [11],
            "warmup": 1,
        },
    )
    spec = {
        "kind": "generic",
        "task_dir": str(task),
        "baseline": {"contract": contract(task, config(task))},
    }
    run, result = evaluate(task, "full")
    assert result["correctness"] == "fail"
    projected = inspect_run(run, spec)["failed_checks"]
    assert projected["checks"][0]["args"] == {"n": 7}
    assert observation(run, result)["failed_checks"] == projected
    # Editing live task metadata cannot change the frozen receipt's conditions.
    live = json.loads((task / "task.json").read_text())
    live["cases"][0]["args"]["n"] = 99
    write_json(task / "task.json", live)
    assert inspect_run(run, spec)["failed_checks"] == projected
    frozen = Path(run) / "source/task.json"
    frozen.write_text(frozen.read_text().replace('"n": 7', '"n": 99'))
    with pytest.raises(ValueError, match="Source changed"):
        inspect_run(run, spec)
