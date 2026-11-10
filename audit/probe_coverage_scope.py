"""Reproduce coverage/sampling behavior locally; this is not a GPU speed benchmark."""

import argparse
import hashlib
import json
import tempfile
from pathlib import Path

from kernel_envolve.cli import evaluate
from kernel_envolve.storage import evaluator_hash

TASK = """import numpy as np
def make_inputs(case, seed):
    rng = np.random.default_rng(seed)
    n, stride = case["n"], case["stride"]
    return tuple(rng.normal(size=n*stride).astype(case["dtype"])[::stride] for _ in range(2))
def reference(x, y):
    return x + y
def baseline(x, y):
    return np.array([a+b for a,b in zip(x,y)], dtype=x.dtype)
"""
GOOD = "def run(x,y):\n return x+y\n"
BUG = "def run(x,y):\n return x-y if x.size == 1025 else x+y\n"


def probe():
    sizes = [1, 31, 32, 33, 255, 256, 257, 1023, 1024, 1025, 4095, 4097]
    cases = [
        {"id": f"n{n}-s{stride}-{dtype}", "args": {"n": n, "stride": stride, "dtype": dtype}}
        for n in sizes
        for stride in (1, 3)
        for dtype in ("float32", "float64")
    ]
    results = {}
    raw = {}
    with tempfile.TemporaryDirectory(prefix="ke-coverage-") as directory:
        for scope in ("all_timed", "representative_timing", "size_specific_bug"):
            root = Path(directory) / scope
            root.mkdir()
            cfg = {
                "name": scope,
                "objective": "Equal weights over the declared performance cases; coverage probe only.",
                "full_trials": 3,
                "warmup": 1,
                "cases": [],
            }
            for case in cases:
                timed = scope == "all_timed" or case["args"]["n"] in (1, 257, 4097)
                cfg["cases"].append(
                    case | ({"weight": 1} if timed else {"measure": False, "quick": False})
                )
            source = BUG if scope == "size_specific_bug" else GOOD
            (root / "task.json").write_text(json.dumps(cfg))
            (root / "task.py").write_text(TASK)
            (root / "candidate.py").write_text(source)
            _, result = evaluate(root, "full")
            raw[scope] = {"config": cfg, "result": result}
            results[scope] = {
                "execution": result["execution"],
                "correctness": result["correctness"],
                "initial_checks": sum(r["kind"] in ("fresh", "reuse") for r in result["checks"]),
                "timing_checks": sum(
                    r["kind"] in ("timed_output", "post") for r in result["checks"]
                ),
                "timing_samples": len(result["samples"]),
                "coverage": result["decision"].get("coverage"),
                "failed_case": next(
                    (r["case"] for r in result["checks"] if r["status"] == "fail"), None
                ),
                "worker_seconds": result["worker_process_wall_seconds"],
                "source_sha256": hashlib.sha256(source.encode()).hexdigest(),
            }
    a, b, c = (results[k] for k in ("all_timed", "representative_timing", "size_specific_bug"))
    assert a["correctness"] == b["correctness"] == "pass"
    assert a["initial_checks"] == b["initial_checks"] == 576
    assert a["timing_samples"] == 1152 and b["timing_samples"] == 288
    assert c["correctness"] == "fail" and c["timing_samples"] == 0
    assert c["failed_case"].startswith("n1025-")
    return {
        "evaluator_hash": evaluator_hash(),
        "scope": "CPU evaluator control-flow validation, not kernel generality or GPU wall-time speedup.",
        "case_count": len(cases),
        "results": results,
    }, raw


if __name__ == "__main__":
    import gzip

    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    summary, raw = probe()
    args.output.write_text(json.dumps(summary, indent=2) + "\n")
    args.output.with_suffix(".raw.json.gz").write_bytes(
        gzip.compress(json.dumps(raw).encode(), mtime=0)
    )
    print(json.dumps(summary, indent=2))
