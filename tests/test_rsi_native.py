"""Native source freezing and a real autograd/gate trial, without a GPU or LLM."""

import sys

from kernel_envolve.rsi.trials import freeze_template, trial, verify_template

pytest_plugins = ["test_native"]


def test_native_trial_preserves_baseline_and_runs_original_gates(native_task, tmp_path):
    task, candidate = native_task
    original = (candidate / "toyop.py").read_bytes()
    frozen = freeze_template(
        {"path": str(task), "kind": "native", "editable": ["candidate/core/toyop.py"]},
        tmp_path / "frozen",
    )
    frozen["kind"] = "native"
    directory = tmp_path / "trial"
    directory.mkdir()
    result = trial(
        directory,
        frozen,
        {
            "command": [sys.executable, "-c", "pass"],
            "model": "fixture",
            "solver_seconds": 5,
            "evaluation_seconds": 90,
        },
        [],
        0,
        100,
        "cpu",
    )
    assert result["passed"]
    assert result["quality"] > 0
    verify_template(frozen)
    assert (candidate / "toyop.py").read_bytes() == original
