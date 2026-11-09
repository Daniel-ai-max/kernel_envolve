"""Independent final acceptance for the three pilot workloads, outside agent search.

This audit tool deliberately imports no kernel_envolve evaluator/scoring code.
It checks single-file frozen candidates against NumPy FP64 arithmetic, checks
input state and same-storage value replacement, and reports synchronized eager
wall timings. It does not select candidates or consume development verdicts.
Run each invocation in a fresh process with an external timeout.
"""

import argparse
import hashlib
import json
import statistics
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent / "search_tasks"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_source(path):
    namespace = {"__name__": "frozen_pilot_candidate", "__file__": str(path)}
    exec(compile(path.read_bytes(), str(path), "exec"), namespace)
    return namespace


def golden(name, inputs):
    arrays = [x.detach().cpu().numpy().astype(np.float64) for x in inputs]
    if name == "row_sum":
        value = arrays[0].sum(axis=1, dtype=np.float64)
    elif name == "matmul":
        value = arrays[0] @ arrays[1]
    else:
        x, scale, bias, residual = arrays
        value = np.maximum(x * scale + bias, 0) + residual
    return value.astype(np.float32)


def sync(device):
    if device.startswith("cuda"):
        torch.cuda.synchronize(device)


def check_call(fn, inputs, expected, cfg, measure=False):
    # Snapshot before the call, including storage/layout and numerical policy.
    snapshots = [x.detach().clone() for x in inputs]
    layouts = [(x.shape, x.stride(), x.data_ptr(), x.dtype, x.device) for x in inputs]
    settings = (torch.backends.cuda.matmul.allow_tf32, torch.get_float32_matmul_precision())
    sync(cfg["device"])
    start = time.perf_counter_ns()
    output = fn(*inputs)
    sync(cfg["device"])
    elapsed = (time.perf_counter_ns() - start) / 1000
    assert settings == (
        torch.backends.cuda.matmul.allow_tf32,
        torch.get_float32_matmul_precision(),
    ), "Candidate changed shared precision settings"
    assert isinstance(output, torch.Tensor), "Output must be a tensor"
    assert output.dtype == torch.float32, "Wrong output dtype"
    assert output.device == inputs[0].device, "Wrong output device"
    value = output.detach().cpu().numpy().copy()
    assert value.shape == expected.shape, "Wrong output shape"
    assert np.isfinite(value).all(), "Nonfinite output"
    assert np.allclose(value, expected, rtol=cfg["rtol"], atol=cfg["atol"]), "Wrong output values"
    for x, before, layout in zip(inputs, snapshots, layouts):
        assert (x.shape, x.stride(), x.data_ptr(), x.dtype, x.device) == layout, (
            "Input layout changed"
        )
        assert torch.equal(x, before), "Input values changed"
    assert elapsed > 0 and np.isfinite(elapsed), "Invalid timing"
    return elapsed if measure else None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=["row_sum", "matmul", "fused_relu"], required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--seeds", type=Path, required=True)
    parser.add_argument("--device", choices=["cpu", "cuda:0"], required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    if not __debug__:
        raise ValueError("Assertions must be enabled")
    seeds = json.loads(args.seeds.read_text())
    if not isinstance(seeds, list) or len(seeds) < 3 or any(type(s) is not int for s in seeds):
        raise ValueError("Supply at least three prespecified integer seeds")
    if len(set(seeds)) != len(seeds) or any(s < 0 or s >= 2**63 - 1000003 for s in seeds):
        raise ValueError("Seeds must be unique and in range")
    folder = ROOT / args.task
    cfg = json.loads((folder / "task.json").read_text())
    cfg["device"] = args.device
    for case in cfg["cases"]:
        case["args"]["device"] = args.device
    sources = {str(p): sha(p) for p in [args.candidate, folder / "task.py", folder / "task.json"]}
    report = {
        "task": args.task,
        "device": args.device,
        "scope": "Independent frozen single-file submission acceptance; no development verdicts",
        "timer": "synchronized_eager_wall",
        "sources": sources,
        "verifier_sha256": sha(Path(__file__)),
        "seeds_sha256": sha(args.seeds),
        "seeds": seeds,
        "torch": torch.__version__,
        "numpy": np.__version__,
        "cuda_runtime": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(args.device) if args.device.startswith("cuda") else None,
        "checks": [],
        "samples": [],
        "correct": False,
    }
    try:
        trusted = load_source(folder / "task.py")
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.set_float32_matmul_precision("highest")
        candidate = load_source(args.candidate)["run"]
        assert not torch.backends.cuda.matmul.allow_tf32, "Import changed precision settings"
        functions = {"candidate": candidate, "baseline": trusted["baseline"]}
        for case in cfg["cases"]:
            for seed in seeds:
                for role, fn in functions.items():
                    inputs = trusted["make_inputs"](case["args"], seed)
                    expected = golden(args.task, inputs)
                    check_call(fn, inputs, expected, cfg)
                    report["checks"].append([case["id"], seed, role, "fresh"])
                    replacements = trusted["make_inputs"](case["args"], seed + 1000003)
                    expected = golden(args.task, replacements)
                    for x, replacement in zip(inputs, replacements):
                        x.copy_(replacement)
                    check_call(fn, inputs, expected, cfg)
                    report["checks"].append([case["id"], seed, role, "reuse"])
        if not args.check_only:
            for batch in range(2):
                cases = cfg["cases"] if batch == 0 else list(reversed(cfg["cases"]))
                for case in cases:
                    for role, fn in functions.items():
                        for _ in range(3):
                            inputs = trusted["make_inputs"](case["args"], seeds[0])
                            check_call(fn, inputs, golden(args.task, inputs), cfg)
                    for trial in range(10):
                        seed = (seeds[trial % len(seeds)] + 7919 * (1 + batch * 10 + trial)) % 2**63
                        order = ["candidate", "baseline", "baseline", "candidate"]
                        if (batch + trial) % 2:
                            order = ["baseline" if x == "candidate" else "candidate" for x in order]
                        repeats = {"candidate": 0, "baseline": 0}
                        for role in order:
                            repeat = repeats[role]
                            inputs = trusted["make_inputs"](case["args"], seed + repeat)
                            expected = golden(args.task, inputs)
                            us = check_call(functions[role], inputs, expected, cfg, measure=True)
                            report["samples"].append(
                                dict(
                                    case=case["id"],
                                    batch=batch,
                                    trial=trial,
                                    role=role,
                                    repeat=repeat,
                                    seed=seed + repeat,
                                    us=us,
                                )
                            )
                            repeats[role] += 1
            summaries = []
            gains = []
            for batch in range(2):
                totals = {"candidate": 0.0, "baseline": 0.0}
                for case in cfg["cases"]:
                    row = {"case": case["id"], "batch": batch}
                    for role in functions:
                        values = [
                            r["us"]
                            for r in report["samples"]
                            if (r["case"], r["batch"], r["role"]) == (case["id"], batch, role)
                        ]
                        row[role + "_us"] = statistics.median(values)
                        totals[role] += case["weight"] * row[role + "_us"]
                    row["regression_over_5_percent"] = (
                        row["candidate_us"] > 1.05 * row["baseline_us"]
                    )
                    summaries.append(row)
                gains.append(1 - totals["candidate"] / totals["baseline"])
            report.update(case_results=summaries, gain_by_batch=gains)
        assert all(sha(Path(p)) == digest for p, digest in sources.items()), (
            "Source changed during run"
        )
        report["correct"] = True
    except Exception as exc:
        report["error"] = {"type": type(exc).__name__, "message": str(exc)}
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(
        json.dumps(
            {
                "correct": report["correct"],
                "checks": len(report["checks"]),
                "samples": len(report["samples"]),
                "error": report.get("error"),
            }
        )
    )
    raise SystemExit(0 if report["correct"] else 1)


if __name__ == "__main__":
    main()
