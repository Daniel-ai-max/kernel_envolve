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
def test_triton_cuda_end_to_end(tmp_path):
    torch = pytest.importorskip("torch")
    pytest.importorskip("triton")
    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    example = Path(__file__).parents[1] / "examples" / "cuda_vector_add"
    for file in example.glob("*.py"):
        shutil.copy(file, tmp_path / file.name)
    cfg = json.loads((example / "task.json").read_text())
    cfg.update(full_trials=3, warmup=1)
    (tmp_path / "task.json").write_text(json.dumps(cfg))
    _, result = evaluate(tmp_path, "full", timeout=600)
    assert result["execution"] == "completed", result
    assert result["correctness"] == "pass", result
    assert result["environment"]["timer"] == "cuda_events_current_stream"
    assert all(s["us"] > 0 for s in result["samples"])
