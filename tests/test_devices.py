import json
import os
import shutil
from pathlib import Path

import pytest

from kernel_envolve.cli import evaluate


def test_torch_cpu_end_to_end(tmp_path):
    pytest.importorskip("torch")
    (tmp_path / "task.json").write_text(
        json.dumps(
            {
                "name": "torch-cpu",
                "objective": "test strided inputs",
                "cases": [{"id": "strided", "args": {}, "weight": 1}],
                "warmup": 1,
                "quick_trials": 1,
            }
        )
    )
    (tmp_path / "task.py").write_text(
        "import torch\n"
        "def make_inputs(case,seed):\n"
        " g=torch.Generator().manual_seed(seed)\n"
        " return tuple(torch.randn(100,generator=g)[::2] for _ in range(2))\n"
        "def reference(x,y): return x+y\n"
        "baseline=reference\n"
    )
    (tmp_path / "candidate.py").write_text("def run(x,y): return x+y\n")
    _, result = evaluate(tmp_path, timeout=60)
    assert result["execution"] == "completed", result
    assert result["correctness"] == "pass", result
    assert result["environment"]["timer"] == "perf_counter_ns"


@pytest.mark.skipif(
    os.environ.get("KERNEL_ENVOLVE_TEST_CUDA") != "1", reason="explicit GPU opt-in required"
)
@pytest.mark.parametrize("timer", ["auto", "wall"])
def test_triton_cuda_end_to_end(tmp_path, timer):
    torch = pytest.importorskip("torch")
    pytest.importorskip("triton")
    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    example = Path(__file__).parents[1] / "examples" / "cuda_vector_add"
    for file in example.glob("*.py"):
        shutil.copy(file, tmp_path / file.name)
    cfg = json.loads((example / "task.json").read_text())
    cfg.update(full_trials=3, warmup=1, timer=timer)
    (tmp_path / "task.json").write_text(json.dumps(cfg))
    _, result = evaluate(tmp_path, "full", timeout=600)
    assert result["execution"] == "completed", result
    assert result["correctness"] == "pass", result
    assert result["environment"]["timer"] == (
        "cuda_events_current_stream" if timer == "auto" else "synchronized_wall"
    )
    assert all(s["us"] > 0 for s in result["samples"])


def cuda_task(tmp_path):
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    (tmp_path / "task.json").write_text(
        json.dumps(
            {
                "name": "cuda-state-probes",
                "objective": "Synthetic CUDA evaluator verification, not kernel performance",
                "device": "cuda:0",
                "cases": [{"id": "tail", "args": {}, "weight": 1}],
                "warmup": 1,
                "full_trials": 3,
            }
        )
    )
    (tmp_path / "task.py").write_text(
        "import torch\n"
        "def make_inputs(case,seed):\n"
        " g=torch.Generator(device='cuda').manual_seed(seed)\n"
        " return tuple(torch.randn(257,device='cuda',generator=g) for _ in range(2))\n"
        "def reference(x,y): return x+y\n"
        "def baseline(x,y):\n"
        " result=x+y\n torch.cuda._sleep(1000000)\n return result\n"
    )
    return torch


@pytest.mark.skipif(
    os.environ.get("KERNEL_ENVOLVE_TEST_CUDA") != "1", reason="explicit GPU opt-in required"
)
@pytest.mark.parametrize(
    "source,execution,correctness",
    [
        ("def run(x,y): return x+y+1\n", "completed", "fail"),
        ("def run(x,y):\n x.add_(1)\n return x+y\n", "completed", "fail"),
        (
            "cached=None\ndef run(x,y):\n global cached\n"
            " if cached is None: cached=x+y\n return cached\n",
            "completed",
            "fail",
        ),
        (
            "import torch\n"
            "torch.backends.cuda.matmul.allow_tf32=not torch.backends.cuda.matmul.allow_tf32\n"
            "def run(x,y): return x+y\n",
            "error",
            "not_run",
        ),
    ],
    ids=["wrong-value", "input-mutation", "stale-output", "precision-policy-leak"],
)
def test_cuda_rejects_invalid_candidates(tmp_path, source, execution, correctness):
    cuda_task(tmp_path)
    (tmp_path / "candidate.py").write_text(source)
    _, result = evaluate(tmp_path, check_only=True, timeout=120)
    assert result["execution"] == execution, result
    assert result["correctness"] == correctness, result
    assert not result["decision"]["eligible_for_best"]


@pytest.mark.skipif(
    os.environ.get("KERNEL_ENVOLVE_TEST_CUDA") != "1", reason="explicit GPU opt-in required"
)
def test_cuda_promotion_and_public_entry(tmp_path):
    from kernel_envolve import load_best
    from kernel_envolve.cli import promote

    torch = cuda_task(tmp_path)
    (tmp_path / "candidate.py").write_text("def run(x,y): return x+y\n")
    run, result = evaluate(tmp_path, mode="full", timeout=120)
    assert result["execution"] == "completed" and result["correctness"] == "pass", result
    # Baseline deliberately sleeps to exercise promotion without depending on a real speed win.
    promote(run)
    verified = load_best(tmp_path)
    x, y = torch.randn(257, device="cuda"), torch.randn(257, device="cuda")
    torch.testing.assert_close(verified(x, y), x + y)
    _, repeat = evaluate(tmp_path, mode="full", timeout=120)
    assert repeat["execution"] == "completed" and repeat["correctness"] == "pass", repeat
    assert repeat["same_candidate_as_best"] and not repeat["decision"]["eligible_for_best"]
