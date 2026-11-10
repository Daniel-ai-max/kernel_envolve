"""Broad-workload positive/negative controls; no agent or inference calls.

Creates disposable development and public-sample tasks. CPU is an audit option;
actual operator iteration uses the target GPU directly. Preserve this public
sample as probe evidence; it must not later be called a hidden acceptance set.
"""

import argparse
import gzip
import hashlib
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

from build_general_tasks import BASELINES, NAMES, materialize

from kernel_envolve.cli import evaluate
from kernel_envolve.storage import evaluator_hash

ROOT = Path(__file__).resolve().parent


def independent(task, candidate, output, seeds, measure=False):
    command = [
        sys.executable,
        str(ROOT / "general_final_acceptance.py"),
        "--task-directory",
        str(task),
        "--candidate",
        str(candidate),
        "--seeds",
        str(seeds),
        "--output",
        str(output),
    ]
    if not measure:
        command.append("--check-only")
    proc = subprocess.Popen(
        command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, start_new_session=True
    )
    try:
        stdout, stderr = proc.communicate(timeout=180)
    except subprocess.TimeoutExpired:
        os.killpg(proc.pid, signal.SIGKILL)
        stdout, stderr = proc.communicate(timeout=10)
        raise RuntimeError(
            f"Independent verifier timed out: {stdout[-1000:]} {stderr[-1000:]}"
        ) from None
    if not output.exists():
        raise RuntimeError(f"Verifier did not produce evidence: {stdout[-2000:]} {stderr[-2000:]}")
    result = json.loads(output.read_text())
    assert proc.returncode == (0 if result["correct"] else 1), (proc.returncode, stderr)
    return result


def run(directory, device, measure):
    directory.mkdir(parents=True, exist_ok=False)
    for scope, seed in (("development", None), ("sample", 20261110)):
        materialize(directory / scope, device, seed)
    seeds = directory / "seeds.json"
    seeds.write_text("[1907101,1907107,1907117]\n")
    report = {
        "scope": "Public controls for a new broad contract; not agent search or hidden final tests",
        "device": device,
        "model_calls": 0,
        "evaluator_hash": evaluator_hash(),
        "records": [],
    }
    started = time.perf_counter()

    def save():
        report["wall_seconds"] = time.perf_counter() - started
        data = json.dumps(report, allow_nan=False).encode()
        (directory / "raw.json.gz").write_bytes(gzip.compress(data))
        (directory / "summary.json").write_text(
            json.dumps(
                {k: v for k, v in report.items() if k != "records"}
                | {
                    "records": [
                        {k: v for k, v in r.items() if k not in ("development", "independent")}
                        for r in report["records"]
                    ]
                },
                indent=2,
            )
            + "\n"
        )

    try:
        for name in NAMES:
            for scope in ("development", "sample"):
                task = directory / scope / name
                cfg = json.loads((task / "task.json").read_text())
                _, dev = evaluate(
                    task,
                    cases=[c["id"] for c in cfg["cases"]],
                    seeds=[0, 1, 2],
                    check_only=True,
                    timeout=180,
                )
                final = independent(
                    task,
                    task / "candidate.py",
                    task / "final.json",
                    seeds,
                    measure and scope == "sample",
                )
                row = {
                    "task": name,
                    "probe": "correct-" + scope,
                    "dev_pass": dev.get("correctness") == "pass",
                    "independent_pass": final["correct"],
                    "dev_checks": len(dev["checks"]),
                    "final_checks": len(final["checks"]),
                    "final_samples": len(final["samples"]),
                    "development": dev,
                    "independent": final,
                }
                report["records"].append(row)
                save()
                assert row["dev_pass"] and row["independent_pass"], (
                    name,
                    scope,
                    dev.get("error"),
                    final.get("error"),
                )
                print(
                    json.dumps(
                        {k: v for k, v in row.items() if k not in ("development", "independent")}
                    ),
                    flush=True,
                )
            task = directory / "development" / name
            cfg = json.loads((task / "task.json").read_text())
            original = BASELINES[name].replace("def run(", "def original(")
            faults = {
                "wrong-values": "    return original(*inputs) + 1\n",
                "precision-bias": "    return original(*inputs) + 0.01\n",
                "input-mutation": "    out=original(*inputs)\n    inputs[0].add_(1)\n    return out\n",
                "padding-write": "    out=original(*inputs)\n    inputs[0]._base[-1]=0\n    return out\n",
                "wrong-dtype": "    return original(*inputs).float()\n",
                "precision-policy": "    import torch\n    policy=torch.backends.cuda.matmul\n    policy.allow_bf16_reduced_precision_reduction=not policy.allow_bf16_reduced_precision_reduction\n    return original(*inputs)\n",
                "packed-row-assumption": "    import torch\n    x=inputs[0]\n    x=torch.as_strided(x,x.shape,(x.shape[1]*x.stride(1),x.stride(1)),x.storage_offset())\n    return original(x,*inputs[1:])\n",
                "empty-reduction-assumption": "    import torch\n    if inputs[0].shape[1]==0: return torch.empty((0,),dtype=inputs[0].dtype,device=inputs[0].device)\n    return original(*inputs)\n",
            }
            for fault, body in faults.items():
                cid = (
                    "float16-1"
                    if fault == "wrong-dtype"
                    else "float32-6"
                    if fault == "empty-reduction-assumption"
                    else "float32-1"
                )
                source = original + "\ndef run(*inputs):\n" + body
                (task / "candidate.py").write_text(source)
                _, dev = evaluate(task, cases=[cid], seeds=[0], check_only=True, timeout=90)
                control = directory / "controls" / name / fault
                control.mkdir(parents=True)
                for file in ("task.py", "workload.py", "candidate.py"):
                    (control / file).write_bytes((task / file).read_bytes())
                (control / "task.json").write_text(
                    json.dumps(cfg | {"cases": [c for c in cfg["cases"] if c["id"] == cid]})
                )
                final = independent(
                    control, control / "candidate.py", control / "final.json", seeds
                )
                row = {
                    "task": name,
                    "probe": fault,
                    "dev_rejected": (
                        dev.get("execution") == "error"
                        if fault == "precision-policy"
                        else dev.get("correctness") == "fail"
                    ),
                    "independent_rejected": not final["correct"],
                    "development": dev,
                    "independent": final,
                }
                report["records"].append(row)
                save()
                assert row["dev_rejected"] and row["independent_rejected"], (
                    name,
                    fault,
                    dev.get("error"),
                    final.get("error"),
                )
                print(
                    json.dumps(
                        {k: v for k, v in row.items() if k not in ("development", "independent")}
                    ),
                    flush=True,
                )
            (task / "candidate.py").write_text(BASELINES[name])
        report["completed"] = True
    finally:
        report["source_sha256"] = {
            name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
            for name in (
                "general_workloads.py",
                "build_general_tasks.py",
                "general_final_acceptance.py",
                "probe_general_tasks.py",
            )
        }
        save()
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--device", choices=["cpu", "cuda:0"], required=True)
    parser.add_argument("--measure", action="store_true")
    args = parser.parse_args()
    run(args.directory, args.device, args.measure)
