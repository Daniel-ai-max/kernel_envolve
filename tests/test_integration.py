import json
import sys

import pytest

from kernel_envolve.cli import evaluate, promote
from kernel_envolve.storage import read_best, read_json


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


def test_targeted_check_only_keeps_contract_and_covers_excluded_case(task):
    original = (task / "task.json").read_bytes()
    run, result = evaluate(task, cases=["strided"], seeds=[17], check_only=True)
    assert result["execution"] == "completed", result
    assert result["correctness"] == "pass", result
    assert {r["case"] for r in result["checks"]} == {"strided"}
    assert {r["seed"] for r in result["checks"]} == {17}
    assert not result["samples"]
    assert (task / "task.json").read_bytes() == original
    with pytest.raises(ValueError, match="diagnostic"):
        promote(run)


@pytest.mark.parametrize("kwargs", [{"cases": ["strided"]}, {"seeds": [17]}, {"check_only": True}])
def test_full_cannot_silently_reduce_coverage(task, kwargs):
    with pytest.raises(ValueError, match="quick mode"):
        evaluate(task, mode="full", **kwargs)


def test_timing_values_vary_and_match_across_roles(task):
    _, result = evaluate(task, "full")
    rows = [r for r in result["samples"] if r["case"] == "contiguous"]
    for role in ("candidate", "comparator"):
        seeds = [r["seed"] for r in rows if r["role"] == role]
        assert len(seeds) == len(set(seeds))
        assert not set(seeds) & {0, 1, 2}
    candidate = {
        (r["batch"], r["trial"], r["repeat"]): r["seed"] for r in rows if r["role"] == "candidate"
    }
    comparator = {
        (r["batch"], r["trial"], r["repeat"]): r["seed"] for r in rows if r["role"] == "comparator"
    }
    assert candidate == comparator


def test_best_full_includes_fixed_baseline_but_quick_does_not(task):
    run, _ = evaluate(task, "full")
    promote(run)
    _, full = evaluate(task, "full")
    assert full["execution"] == "completed", full
    assert full["same_candidate_as_best"]
    assert not full["decision"]["eligible_for_best"]
    assert {s["role"] for s in full["samples"]} == {"candidate", "comparator", "baseline"}
    assert any(c["role"] == "baseline" and c["kind"] == "post" for c in full["checks"])
    _, quick = evaluate(task)
    assert {s["role"] for s in quick["samples"]} == {"candidate", "comparator"}
    _, signal = evaluate(task, signal_only=True)
    assert signal["execution"] == "completed", signal


def test_post_check_detects_failure_after_initial_correctness(task):
    # quick: two correctness calls, one warmup, two timed calls, then post check.
    set_candidate(
        task, "count=0\ndef run(x,y):\n global count\n count+=1\n return x+y if count<6 else x-y\n"
    )
    _, result = evaluate(task)
    assert result["execution"] == "completed"
    assert result["correctness"] == "fail"
    assert result["checks"][-1]["kind"] == "post"
    assert result["samples"]


def test_actual_timed_output_is_checked_without_candidate_rerun(task):
    # Initial and post checks would pass. Only the outputs actually timed are wrong.
    set_candidate(
        task,
        "count=0\ndef run(x,y):\n global count\n count+=1\n return x-y if 4<=count<=5 else x+y\n",
    )
    _, result = evaluate(task)
    assert result["correctness"] == "fail", result
    assert result["checks"][-1]["kind"] == "timed_output"


def test_nondeterministic_factory_is_configuration_error(task):
    with (task / "task.py").open("a") as f:
        f.write(
            "\ndef make_inputs(case,seed): return tuple(np.random.randn(10).astype(np.float32) for _ in range(2))\n"
        )
    _, result = evaluate(task)
    assert result["execution"] == "error"
    assert "factory is not deterministic" in result["error"]["message"]


def test_task_validator_checks_output_alias(task):
    with (task / "task.py").open("a") as f:
        f.write(
            "\ndef reference(x,y): return x\nbaseline=reference\ndef validate(inputs,output):\n assert output is inputs[0], 'output must alias input 0'\n"
        )
    set_candidate(task, "def run(x,y): return x.copy()\n")
    _, result = evaluate(task, check_only=True)
    assert result["correctness"] == "fail"
    assert "alias input" in result["error"]["message"]
    set_candidate(task, "def run(x,y): return x\n")
    _, result = evaluate(task, check_only=True)
    assert result["correctness"] == "pass"


def test_validator_cannot_silently_return_false(task):
    with (task / "task.py").open("a") as f:
        f.write("\ndef validate(inputs,output): return False\n")
    _, result = evaluate(task, check_only=True)
    assert result["execution"] == "error"
    assert "must return None" in result["error"]["message"]


def test_public_entry_uses_verified_snapshot_in_fresh_process(task):
    import subprocess

    run, _ = evaluate(task, "full")
    promote(run)
    set_candidate(task, "raise RuntimeError('unverified working copy')\n")
    proc = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys,numpy as np; from kernel_envolve import load_best; "
            "f=load_best(sys.argv[1]); a=np.arange(5,dtype=np.float32); "
            "np.testing.assert_array_equal(f(a,a),a+a)",
            str(task),
        ],
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stderr
    (run / "source" / "candidate.py").write_text("def run(x,y): return x-y\n")
    from kernel_envolve import load_best

    with pytest.raises(ValueError, match="Source changed"):
        load_best(task)
