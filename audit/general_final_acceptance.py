"""Independent broad-domain acceptance; no runtime evaluator or development verdicts.

Use a frozen single-file submission and a prespecified task directory/seed file.
Run in a fresh externally time-limited process. This reports measured performance,
not best-promotion eligibility or a statistical claim about agent effectiveness.
"""

import argparse
import hashlib
import importlib.util
import json
import statistics
import time
import traceback
from pathlib import Path

import numpy as np
import torch

# Deliberately separate from the development validator's constants and Torch oracle.
TOLERANCES = {
    "row_sum": {"float32": (1e-5, 2e-4), "float16": (0.003, 0.003), "bfloat16": (0.025, 0.025)},
    "matmul": {"float32": (1e-4, 5e-4), "float16": (0.003, 0.003), "bfloat16": (0.025, 0.025)},
    "fused_relu": {"float32": (1e-5, 1e-5), "float16": (0.003, 0.003), "bfloat16": (0.025, 0.025)},
}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(path):
    spec = importlib.util.spec_from_file_location("acceptance_source", path)
    module = importlib.util.module_from_spec(spec)
    # Source bytes, not timestamp-dependent pyc, determine executed code.
    exec(compile(path.read_bytes(), str(path), "exec"), module.__dict__)
    return module


def golden(name, inputs):
    arrays = [x.detach().double().cpu().numpy() for x in inputs]
    if name == "row_sum":
        expected = np.sum(arrays[0], axis=1, dtype=np.float64)
    elif name == "matmul":
        expected = np.matmul(arrays[0], arrays[1])
    else:
        x, scale, bias, residual = arrays
        expected = np.maximum(x * scale + bias, 0) + residual
    # The interface returns its input dtype, including BF16 (unsupported by NumPy).
    return torch.from_numpy(expected).to(inputs[0].dtype).double().numpy()


def row_sum(x):
    return x.sum(dim=1)


def matmul(a, b):
    return a @ b


def fused_relu(x, scale, bias, residual):
    return torch.relu(x * scale + bias) + residual


def settings():
    values = {
        "matmul_tf32": torch.backends.cuda.matmul.allow_tf32,
        "precision": torch.get_float32_matmul_precision(),
        "default_dtype": str(torch.get_default_dtype()),
        "fp16_reduction": torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction,
        "bf16_reduction": torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction,
        "threads": torch.get_num_threads(),
        "grad": torch.is_grad_enabled(),
        "deterministic": torch.are_deterministic_algorithms_enabled(),
        "cudnn_tf32": torch.backends.cudnn.allow_tf32,
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "numpy_errors": np.geterr(),
    }
    if torch.cuda.is_initialized():
        values.update(
            cuda_device=torch.cuda.current_device(),
            cuda_stream=torch.cuda.current_stream().cuda_stream,
        )
    return values


def synchronize(device):
    if device.startswith("cuda"):
        torch.cuda.synchronize(device)


def check_call(name, function, inputs, expected, device):
    policy = settings()
    # Include the whole factory-owned storage so writes to padding are observable.
    states = [
        (x._base, x._base.clone(), x.shape, x.stride(), x.data_ptr(), x.dtype, x.device)
        for x in inputs
    ]
    synchronize(device)
    start = time.perf_counter_ns()
    output = function(*inputs)
    synchronize(device)
    elapsed = (time.perf_counter_ns() - start) / 1000
    assert settings() == policy, "Changed precision settings"
    assert isinstance(output, torch.Tensor), "Output must be a tensor"
    assert output.shape == expected.shape, "Wrong output shape"
    assert output.dtype == inputs[0].dtype, "Wrong output dtype"
    assert output.device == inputs[0].device, "Wrong output device"
    actual = output.detach().double().cpu().numpy().copy()
    assert np.isfinite(actual).all(), "Nonfinite output"
    dtype = str(inputs[0].dtype).split(".")[-1]
    rtol, atol = TOLERANCES[name][dtype]
    np.testing.assert_allclose(actual, expected, rtol=rtol, atol=atol, equal_nan=False)
    for x, (base, before, shape, stride, pointer, dtype, dev) in zip(inputs, states, strict=True):
        assert x._base is base and (x.shape, x.stride(), x.data_ptr(), x.dtype, x.device) == (
            shape,
            stride,
            pointer,
            dtype,
            dev,
        ), "Input layout changed"
        assert torch.equal(torch.isnan(base), torch.isnan(before)), "Input padding changed"
        assert torch.equal(base.nan_to_num(), before.nan_to_num()), "Input values changed"
    assert np.isfinite(elapsed) and elapsed > 0, "Invalid timing"
    return elapsed


def accept(folder, candidate_path, seeds, *, check_only=False, trials=10):
    cfg = json.loads((folder / "task.json").read_text())
    name = cfg["name"].removeprefix("general-")
    if name not in TOLERANCES:
        raise ValueError("Unknown task")
    assert trials >= 1 and seeds and len(set(seeds)) == len(seeds)
    assert all(type(s) is int and 0 <= s < 2**63 - 1000003 for s in seeds)
    device = cfg["device"]
    paths = [candidate_path, folder / "task.json", folder / "workload.py", Path(__file__)]
    sources = {str(p): digest(p) for p in paths}
    report = {
        "name": name,
        "device": device,
        "sources": sources,
        "seeds": seeds,
        "contract": cfg,
        "scope": "independent numerical and state acceptance; synchronized eager wall timing",
        "torch": torch.__version__,
        "numpy": np.__version__,
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(device) if device.startswith("cuda") else None,
        "correct": False,
        "checks": [],
        "samples": [],
        "trials": trials,
    }
    context = {}
    started = time.perf_counter()
    try:
        factory = load(folder / "workload.py").make_inputs
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.set_float32_matmul_precision("highest")
        policy = settings()
        report["runtime_policy"] = policy
        candidate = load(candidate_path).run
        assert policy == settings(), "Import changed precision settings"
        functions = {
            "candidate": candidate,
            "baseline": {"row_sum": row_sum, "matmul": matmul, "fused_relu": fused_relu}[name],
        }
        for case in cfg["cases"]:
            for seed in seeds:
                for role, fn in functions.items():
                    inputs = factory(name, case["args"], seed)
                    for kind, current_seed in (("fresh", seed), ("reuse", seed + 1000003)):
                        context = dict(case=case["id"], role=role, seed=current_seed, kind=kind)
                        if kind == "reuse":
                            for x, replacement in zip(
                                inputs, factory(name, case["args"], current_seed), strict=True
                            ):
                                x.copy_(replacement)
                        check_call(name, fn, inputs, golden(name, inputs), device)
                        report["checks"].append(context)
        if not check_only:
            timed = [c for c in cfg["cases"] if c.get("measure", True)]
            for batch in range(2):
                for case in timed if batch == 0 else reversed(timed):
                    for role, fn in functions.items():
                        for warmup in range(3):
                            context = dict(case=case["id"], role=role, kind="warmup", batch=batch)
                            inputs = factory(name, case["args"], seeds[0] + warmup)
                            check_call(name, fn, inputs, golden(name, inputs), device)
                    for trial in range(trials):
                        key = json.dumps([case["id"], batch, trial, seeds]).encode()
                        seed = int.from_bytes(hashlib.sha256(key).digest()[:8], "big") % (2**63 - 2)
                        expected = [
                            golden(name, factory(name, case["args"], seed + r)) for r in range(2)
                        ]
                        order = ["candidate", "baseline", "baseline", "candidate"]
                        if (batch + trial) % 2:
                            order = ["baseline" if r == "candidate" else "candidate" for r in order]
                        repeats = {r: 0 for r in functions}
                        for role in order:
                            repeat = repeats[role]
                            context = dict(
                                case=case["id"],
                                role=role,
                                kind="timed",
                                batch=batch,
                                trial=trial,
                                repeat=repeat,
                                seed=seed + repeat,
                            )
                            inputs = factory(name, case["args"], seed + repeat)
                            us = check_call(name, functions[role], inputs, expected[repeat], device)
                            report["samples"].append(context | {"us": us})
                            repeats[role] += 1
            rows, gains = [], []
            for batch in range(2):
                totals = {r: 0.0 for r in functions}
                for case in timed:
                    row = {"case": case["id"], "batch": batch, "dtype": case["args"]["dtype"]}
                    for role in functions:
                        values = [
                            r["us"]
                            for r in report["samples"]
                            if (r["case"], r["batch"], r["role"]) == (case["id"], batch, role)
                        ]
                        row[role + "_us"] = statistics.median(values)
                        totals[role] += case["weight"] * row[role + "_us"]
                    row["regression"] = row["candidate_us"] / row["baseline_us"] - 1
                    rows.append(row)
                gains.append(1 - totals["candidate"] / totals["baseline"])
            report.update(case_results=rows, gain_by_batch=gains)
        assert all(digest(Path(p)) == value for p, value in sources.items()), "Source changed"
        report["correct"] = True
    except Exception as exc:
        report["error"] = {
            "type": type(exc).__name__,
            "message": str(exc),
            "context": context,
            "traceback": traceback.format_exc(),
        }
    report["wall_seconds"] = time.perf_counter() - started
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-directory", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--seeds", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--trials", type=int, default=10)
    args = parser.parse_args()
    if not __debug__:
        raise ValueError("Assertions must be enabled")
    values = json.loads(args.seeds.read_text())
    if len(values) < 3:
        raise ValueError("Use at least three prespecified seeds")
    result = accept(
        args.task_directory, args.candidate, values, check_only=args.check_only, trials=args.trials
    )
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(
        json.dumps(
            {
                "correct": result["correct"],
                "checks": len(result["checks"]),
                "samples": len(result["samples"]),
                "error": result.get("error"),
            }
        )
    )
    raise SystemExit(0 if result["correct"] else 1)
