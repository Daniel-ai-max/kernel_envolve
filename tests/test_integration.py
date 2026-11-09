import json
import shutil
import sys
from pathlib import Path

import pytest

from kernel_envolve.cli import evaluate, promote
from kernel_envolve.storage import read_best, read_json


@pytest.fixture
def task(tmp_path):
    source = Path(__file__).parents[1] / "examples" / "vector_add"
    for p in source.glob("*.py"):
        shutil.copy(p, tmp_path / p.name)
    cfg = json.loads((source / "task.json").read_text())
    cfg.update(full_trials=3, quick_trials=1, warmup=1)
    (tmp_path / "task.json").write_text(json.dumps(cfg))
    return tmp_path


def set_candidate(task, code):
    (task / "candidate.py").write_text(code)


def test_real_full_and_snapshot_best(task):
    run, result = evaluate(task, "full")
    assert result["execution"] == "completed", result
    assert result["correctness"] == "pass", result
    assert result["decision"]["eligible_for_best"], result
    pointer = promote(run)
    set_candidate(task, "raise RuntimeError('mutable working copy must not be imported by best')")
    _, best_run, _ = read_best(task)
    assert best_run == run
    assert pointer["run"] == str(run.relative_to(task / ".kernel_envolve"))
    # A new candidate is measured against the saved implementation, not the edited working copy.
    set_candidate(task, "import numpy as np\ndef run(x,y): return np.add(x,y)\n")
    _, second = evaluate(task, "quick")
    assert second["correctness"] == "pass", second
    assert second["comparator"]["kind"] == "best"


@pytest.mark.parametrize(
    "code,kind",
    [
        ("def run(x,y): return x-y\n", "fail"),
        ("def run(x,y):\n x += y\n return x\n", "fail"),
        ("def run(x,y): raise RuntimeError('bad launch')\n", "error"),
    ],
)
def test_failures(task, code, kind):
    set_candidate(task, code)
    run, result = evaluate(task)
    assert result["execution"] == ("error" if kind == "error" else "completed")
    if kind == "fail":
        assert result["correctness"] == "fail"
    assert not result["samples"]
    with pytest.raises(ValueError, match="Not eligible"):
        promote(run)


def test_same_object_cache_rejected(task):
    set_candidate(
        task,
        "cache={}\ndef run(x,y):\n key=id(x)\n if key not in cache: cache[key]=x+y\n return cache[key]\n",
    )
    _, result = evaluate(task)
    assert result["correctness"] == "fail"
    assert result["checks"][-1]["kind"] == "reuse"


def test_timeout_is_structured_error(task):
    set_candidate(task, "import time\ndef run(x,y):\n time.sleep(30)\n return x+y\n")
    _, result = evaluate(task, timeout=1)
    assert result["execution"] == "error"
    assert "timeout" in result["error"]["message"]


def test_quick_and_signal_do_not_promote(task):
    run, result = evaluate(task)
    assert result["correctness"] == "pass"
    with pytest.raises(ValueError, match="Quick"):
        promote(run)
    _, signal = evaluate(task, signal_only=True)
    assert "gain_by_batch" not in signal["decision"]
    with pytest.raises(ValueError, match="quick mode"):
        evaluate(task, mode="full", signal_only=True)


def test_stale_comparator_cannot_promote(task):
    first, _ = evaluate(task, "full")
    second, _ = evaluate(task, "full")
    promote(first)
    with pytest.raises(ValueError, match="Best changed"):
        promote(second)


def test_source_result_and_contract_edits_rejected(task):
    run, _ = evaluate(task, "full")
    result_path = run / "result.json"
    original = result_path.read_text()
    changed = json.loads(original)
    changed["decision"]["eligible_for_best"] = False
    result_path.write_text(json.dumps(changed))
    with pytest.raises(ValueError, match="Result changed"):
        promote(run)
    result_path.write_text(original)
    snap = run / "source" / "candidate.py"
    old_source = snap.read_text()
    snap.write_text(old_source + "\n# edited\n")
    with pytest.raises(ValueError, match="Source changed"):
        promote(run)
    snap.write_text(old_source)
    (task / "task.py").write_text((task / "task.py").read_text() + "\n# changed oracle\n")
    with pytest.raises(ValueError, match="Live task contract changed"):
        promote(run)


def test_relative_helpers_are_frozen(task):
    cfg_path = task / "task.json"
    cfg = read_json(cfg_path)
    cfg["candidate_files"] = ["candidate.py", "helper.py"]
    cfg_path.write_text(json.dumps(cfg))
    (task / "helper.py").write_text("def add(x,y): return x+y\n")
    set_candidate(task, "from .helper import add\ndef run(x,y): return add(x,y)\n")
    run, result = evaluate(task, "full")
    assert result["correctness"] == "pass"
    promote(run)
    (task / "helper.py").write_text("raise RuntimeError('working helper')\n")
    set_candidate(task, "def run(x,y): return x+y\n")
    _, result = evaluate(task)
    assert result["correctness"] == "pass", result


def test_declared_inplace_inputs_are_reset(task):
    cfg = read_json(task / "task.json")
    cfg["mutable_inputs"] = [0]
    (task / "task.json").write_text(json.dumps(cfg))
    with (task / "task.py").open("a") as f:
        f.write("\ndef reference(x,y):\n x += y\n return x\nbaseline = reference\n")
    set_candidate(task, "def run(x,y):\n x += y\n return x\n")
    _, result = evaluate(task, "full")
    assert result["correctness"] == "pass", result


def test_declared_inplace_wrong_side_effect_rejected(task):
    cfg = read_json(task / "task.json")
    cfg["mutable_inputs"] = [0]
    (task / "task.json").write_text(json.dumps(cfg))
    with (task / "task.py").open("a") as f:
        f.write("\ndef reference(x,y):\n x += y\n return x\nbaseline = reference\n")
    set_candidate(task, "def run(x,y):\n out=x+y\n x.fill(0)\n return out\n")
    _, result = evaluate(task)
    assert result["correctness"] == "fail"
    assert "mutated input" in result["error"]["message"]


def test_baseline_is_also_verified(task):
    with (task / "task.py").open("a") as f:
        f.write("\ndef baseline(x,y): return x-y\n")
    _, result = evaluate(task)
    assert result["correctness"] == "fail"
    assert result["checks"][-1]["role"] == "comparator"


def test_task_lock_prevents_overlap(task):
    from kernel_envolve.cli import lock

    state = task / ".kernel_envolve"
    state.mkdir()
    with lock(state / "evaluate.lock"):
        with pytest.raises(ValueError, match="Another operation"):
            evaluate(task)


def test_cli_exit_status(task):
    import subprocess

    set_candidate(task, "def run(x,y): return x-y\n")
    proc = subprocess.run(
        [sys.executable, "-m", "kernel_envolve.cli", "evaluate", "--task", str(task)],
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 1
    assert json.loads(proc.stdout)["correctness"] == "fail"
