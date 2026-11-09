import json
import os
import py_compile
import shutil
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from test_scoring import evidence

from kernel_envolve.arrays import Mismatch, assert_close
from kernel_envolve.cli import audit_result, evaluate
from kernel_envolve.scoring import score
from kernel_envolve.storage import config, load_module, verify_source
from kernel_envolve.worker import Clock


def settings(task, **kwargs):
    path = task / "task.json"
    path.write_text(json.dumps(json.loads(path.read_text()) | kwargs))


def shared_reference(task):
    with (task / "task.py").open("a") as f:
        f.write(
            "\ndef reference(x,y):\n"
            " if not hasattr(np, '_ke_buf') or np._ke_buf.shape != x.shape: np._ke_buf=np.empty_like(x)\n"
            " np.add(x,y,out=np._ke_buf)\n return np._ke_buf\nbaseline=reference\n"
        )


def test_oracle_output_cannot_be_overwritten_by_candidate(task):
    shared_reference(task)
    (task / "candidate.py").write_text(
        "import numpy as np\ndef run(x,y):\n np._ke_buf.fill(0)\n return np._ke_buf\n"
    )
    _, result = evaluate(task, check_only=True)
    assert result["correctness"] == "fail", result


def test_actual_timed_output_cannot_be_overwritten_by_later_oracle(task):
    shared_reference(task)
    (task / "candidate.py").write_text(
        "import numpy as np\ncount=0\ndef run(x,y):\n global count\n count+=1\n"
        " if 4<=count<=5: np._ke_buf.fill(0)\n else: np.add(x,y,out=np._ke_buf)\n return np._ke_buf\n"
    )
    _, result = evaluate(task)
    assert result["correctness"] == "fail", result
    assert result["checks"][-1]["kind"] == "timed_output"


def test_cached_factory_inputs_are_rejected(task):
    with (task / "task.py").open("a") as f:
        f.write(
            "\nCACHED=(np.ones(8,dtype=np.float32),np.ones(8,dtype=np.float32))\ndef make_inputs(case,seed): return CACHED\n"
        )
    _, result = evaluate(task, check_only=True)
    assert result["execution"] == "error", result
    assert "independent storage" in result["error"]["message"]


def test_factory_shape_drift_cannot_include_compilation_in_timing(task):
    with (task / "task.py").open("a") as f:
        f.write(
            "\ndef make_inputs(case,seed): return tuple(np.ones(8 if seed<100 else 9,dtype=np.float32) for _ in range(2))\n"
        )
    _, result = evaluate(task)
    assert result["execution"] == "error", result
    assert "contract changed within a case" in result["error"]["message"]


def test_optional_exact_value_determinism(task):
    settings(task, atol=1.0, rtol=0)
    (task / "candidate.py").write_text(
        "count=0\ndef run(x,y):\n global count\n count+=1\n return x+y+(0.1 if count%2 else 0)\n"
    )
    _, loose = evaluate(task, check_only=True)
    assert loose["correctness"] == "pass", loose
    settings(task, deterministic=True)
    _, strict = evaluate(task, check_only=True)
    assert strict["correctness"] == "fail", strict
    assert strict["checks"][-1]["kind"] == "repeat"
    assert "determinism" in strict["error"]["message"]


def test_numpy_global_error_policy_cannot_leak_between_roles(task):
    (task / "candidate.py").write_text(
        "import numpy as np\nnp.seterr(all='ignore')\ndef run(x,y): return x+y\n"
    )
    _, result = evaluate(task)
    assert result["execution"] == "error", result
    assert "Persistent runtime setting" in result["error"]["message"]


@pytest.mark.parametrize("at_import", [True, False])
def test_torch_default_dtype_cannot_leak_between_roles(task, at_import):
    pytest.importorskip("torch")
    with (task / "task.py").open("a") as f:
        f.write("\nimport torch\n")
    code = "import torch\n"
    code += (
        "torch.set_default_dtype(torch.float64)\ndef run(x,y): return x+y\n"
        if at_import
        else "def run(x,y):\n torch.set_default_dtype(torch.float64)\n return x+y\n"
    )
    (task / "candidate.py").write_text(code)
    _, result = evaluate(task, check_only=True)
    assert result["execution"] == "error", result
    assert "Persistent runtime setting" in result["error"]["message"]


def test_torch_expanded_views_and_empty_storage_are_valid(task):
    pytest.importorskip("torch")
    (task / "task.py").write_text(
        "import torch\ndef make_inputs(case,seed):\n"
        " g=torch.Generator().manual_seed(seed)\n"
        " return tuple(torch.randn(8,generator=g).expand(3,8) for _ in range(2))\n"
        "def reference(x,y): return x+y\nbaseline=reference\n"
    )
    (task / "candidate.py").write_text("def run(x,y): return x+y\n")
    _, result = evaluate(task)
    assert result["execution"] == "completed" and result["correctness"] == "pass", result
    source = (
        (task / "task.py")
        .read_text()
        .replace("torch.randn(8,generator=g).expand(3,8)", "torch.empty(0)")
    )
    (task / "task.py").write_text(source)
    _, result = evaluate(task, check_only=True)
    assert result["correctness"] == "pass", result


def test_float16_tolerance_math_must_not_overflow_to_infinity():
    a = np.array([65504], dtype=np.float16)
    b = -a
    with pytest.raises(Mismatch):
        assert_close(a, b, {"rtol": 1.5, "atol": 0, "equal_nan": False})


def test_torch_complex128_conjugate_view_compares_correctly():
    torch = pytest.importorskip("torch")
    a = torch.tensor([1 + 2j], dtype=torch.complex128).conj()
    b = torch.tensor([1 - 2j], dtype=torch.complex128)
    assert_close(a, b, {"rtol": 0, "atol": 0, "equal_nan": False})


def test_bimodal_timing_cannot_hide_behind_zero_mad():
    result, cfg = evidence()
    for row in result["samples"]:
        if row["role"] == "candidate":
            row["us"] = 100 if row["trial"] == 2 else 8
    decision = score(result, cfg)
    assert all(r["candidate_mad_us"] == 0 for r in decision["cases"])
    assert decision["performance"] == "inconclusive"
    assert not decision["eligible_for_best"]


def test_derived_ratio_overflow_rejected_with_finite_raw_samples():
    result, cfg = evidence()
    for row in result["samples"]:
        row["us"] = 1e-308 if row["role"] == "candidate" else 1e308
    with pytest.raises(ValueError, match="not finite"):
        score(result, cfg)


def test_repeat_checks_cannot_be_omitted_when_required():
    result, cfg = evidence()
    cfg["deterministic"] = True
    with pytest.raises(ValueError, match="coverage"):
        score(result, cfg)


def test_supervisor_does_not_trust_eligibility_or_wrong_identity():
    result, cfg = evidence(mode="quick")
    result.update(schema=3, decision={"eligible_for_best": True})
    request = {"mode": "quick", "signal_only": False}
    assert not audit_result(result, request, cfg)["eligible_for_best"]
    result["mode"] = "full"
    with pytest.raises(ValueError, match="does not match request"):
        audit_result(result, request, cfg)


@pytest.mark.parametrize("bad", [[], {"unexpected": float("nan")}])
def test_malformed_worker_envelope_is_rejected(bad):
    with pytest.raises(ValueError):
        audit_result(bad, {}, {})


def test_boolean_is_not_a_timing_measurement():
    result, cfg = evidence()
    result["samples"][0]["us"] = True
    with pytest.raises(ValueError, match="finite and positive"):
        score(result, cfg)


def test_case_quick_flag_is_not_a_truthy_string(task):
    path = task / "task.json"
    cfg = json.loads(path.read_text())
    cfg["cases"][0]["quick"] = "false"
    path.write_text(json.dumps(cfg))
    with pytest.raises(ValueError, match="boolean"):
        config(task)


@pytest.mark.parametrize(
    "field,value",
    [
        ("equal_nan", "false"),
        ("deterministic", "true"),
        ("device", "cudaBOGUS"),
        ("timer", "unknown"),
        ("rtol", True),
    ],
)
def test_configuration_cannot_silently_change_semantics(task, field, value):
    settings(task, **{field: value})
    with pytest.raises(ValueError):
        config(task)


def test_snapshot_import_ignores_timestamp_valid_stale_bytecode(tmp_path):
    (tmp_path / "candidate.py").write_text("from .helper import value\n")
    helper = tmp_path / "helper.py"
    helper.write_text("value=2\n")
    stamp = helper.stat()
    py_compile.compile(str(helper), doraise=True)
    helper.write_text("value=1\n")
    os.utime(helper, ns=(stamp.st_atime_ns, stamp.st_mtime_ns))
    assert load_module(tmp_path, "candidate").value == 1


def test_added_source_file_invalidates_manifest(task):
    run, result = evaluate(task, check_only=True)
    (run / "source" / "unlisted.py").write_text("value=0\n")
    with pytest.raises(ValueError, match="inventory changed"):
        verify_source(run, result["identity"])


def test_wall_clock_synchronizes_before_and_after_device_work():
    # Control-flow unit test; does not simulate GPU latency or establish CUDA hardware validation.
    events = []
    clock = Clock("cpu", timer="wall")
    clock.torch = SimpleNamespace(cuda=SimpleNamespace(synchronize=lambda: events.append("sync")))

    def work():
        events.append("work")
        return 7

    elapsed, output = clock.measure(work, ())
    assert events == ["sync", "work", "sync"]
    assert elapsed > 0 and output == 7


def test_golden_gate_accepts_improved_accuracy_and_rejects_bias_within_allclose(task):
    example = Path(__file__).parents[1] / "examples" / "reduction"
    for name in ("task.json", "task.py", "candidate.py"):
        shutil.copy(example / name, task / name)
    (task / "candidate.py").write_text(
        "import math,numpy as np\ndef run(x): return np.asarray(math.fsum(float(v) for v in x),dtype=np.float32)\n"
    )
    _, good = evaluate(task, check_only=True)
    assert good["correctness"] == "pass", good
    cfg = config(task)
    module = load_module(task, "task")
    inputs = module.make_inputs(cfg["cases"][0]["args"], 0)
    expected = module.reference(*inputs)
    assert_close(
        np.asarray(expected + np.float32(0.0005)), expected, cfg
    )  # allclose alone accepts it
    (task / "candidate.py").write_text(
        "import numpy as np\ndef run(x): return np.asarray(np.sum(x,dtype=np.float32)+np.float32(0.0005))\n"
    )
    _, bad = evaluate(task, check_only=True)
    assert bad["correctness"] == "fail", bad
    assert "golden error" in bad["error"]["message"]


def test_python_optimization_cannot_disable_task_assertions(task):
    process = subprocess.run(
        [sys.executable, "-O", "-m", "kernel_envolve.cli", "evaluate", "--task", str(task)],
        capture_output=True,
        text=True,
    )
    assert process.returncode == 2
    assert "without -O" in process.stderr


def test_snapshot_loader_keeps_assertions_even_in_optimized_python(tmp_path):
    (tmp_path / "helper.py").write_text("assert False, 'snapshot assertion remains active'\n")
    process = subprocess.run(
        [
            sys.executable,
            "-O",
            "-c",
            "import sys; from pathlib import Path; from kernel_envolve.storage import load_module; load_module(Path(sys.argv[1]), 'helper')",
            str(tmp_path),
        ],
        capture_output=True,
        text=True,
    )
    assert process.returncode != 0
    assert "snapshot assertion remains active" in process.stderr


@pytest.mark.parametrize("check_only", [True, False])
def test_cost_breakdown_includes_oracle_work_without_changing_coverage(task, check_only):
    with (task / "task.py").open("a") as f:
        f.write(
            "\nimport time\noriginal_reference=reference\n"
            "def reference(*args):\n time.sleep(0.01)\n return original_reference(*args)\n"
        )
    _, result = evaluate(task, check_only=check_only)
    assert result["execution"] == "completed" and result["correctness"] == "pass", result
    phases = result["phase_wall_seconds"]
    assert set(phases) == {"setup", "correctness", "measurement", "finalize"}
    assert all(value >= 0 for value in phases.values())
    assert phases["correctness"] >= 0.04  # Four oracle calls, each deliberately sleeps 10ms.
    assert sum(phases.values()) <= result["worker_process_wall_seconds"]
    assert result["worker_process_wall_seconds"] <= result["worker_wall_seconds"]
    assert len(result["checks"]) == (4 if check_only else 8)
    assert len(result["samples"]) == (0 if check_only else 4)
