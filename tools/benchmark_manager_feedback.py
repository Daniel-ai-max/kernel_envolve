"""Measure CPU-only command feedback overhead; never a GPU/search-quality benchmark.

Run with the source tree under test on PYTHONPATH and --output outside the checkout.
Direct and managed arms execute identical commands and compare their actual output.
"""

import argparse
import hashlib
import json
import math
import os
import platform
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from kernel_envolve.manager import worker
from kernel_envolve.manager.config import validate
from kernel_envolve.manager.process import identity
from kernel_envolve.manager.service import reconcile
from kernel_envolve.manager.store import Store
from kernel_envolve.storage import config, contract

WORKLOAD = """import hashlib,json,sys,time
count=int(sys.argv[1])
delay=float(sys.argv[2])
result=[]
for i in range(count):
    result.append(hashlib.sha256(str(i).encode()).hexdigest())
    time.sleep(delay)
print(json.dumps(result))
"""


def run(args):
    records = []
    env = os.environ | {"KERNEL_ENVOLVE_AUTO_UPDATE": "0", "CUDA_VISIBLE_DEVICES": ""}
    for key in (
        "KERNEL_ENVOLVE_GPU_JOB",
        "KERNEL_ENVOLVE_RELEASE",
        "KERNEL_ENVOLVE_UPDATE_SESSION",
    ):
        env.pop(key, None)
    with tempfile.TemporaryDirectory(prefix="ke-feedback-") as temporary:
        root = Path(temporary).resolve()
        work = root / "work"
        work.mkdir()
        (work / "task.json").write_text(
            json.dumps(
                {
                    "name": "feedback",
                    "device": "cpu",
                    "objective": "command overhead only",
                    "cases": [{"id": "one", "args": {"n": 1}, "weight": 1}],
                }
            )
        )
        (work / "candidate.py").write_text("def run(x): return x\n")
        (work / "task.py").write_text("def reference(x): return x\n")
        deployment = validate(
            {
                "version": 1,
                "resources": [{"id": "cpu", "device": "cpu", "sessions": 1}],
                "tasks": [
                    {
                        "id": "feedback",
                        "name": "CPU feedback benchmark",
                        "workspace": str(work),
                        "task_dir": str(work),
                        "kind": "generic",
                        "resource": "cpu",
                        "command": [sys.executable, "-c", "pass"],
                        "round_timeout_seconds": 3600,
                        "baseline": {
                            "repository": "local CPU diagnostic",
                            "revision": args.label,
                            "entrypoint": "synthetic command",
                            "hardware": "CPU",
                            "introduced_on": None,
                            "introduction_url": None,
                            "frozen_on": "2026-11-11",
                            "contract": contract(work, config(work)),
                        },
                    }
                ],
            },
            root,
        )
        store = Store(root / "state")
        try:
            store.register(deployment)
            with store.transaction():
                store.db.executemany(
                    "INSERT INTO jobs(id,task,kind,resource,status,command,created,ended) "
                    "VALUES(?,'feedback','gpu','cpu','succeeded','[]',1,2)",
                    ((f"history-{i}",) for i in range(args.history_jobs)),
                )
            parent = store.schedule()[0]
            assert store.claim(parent, os.getpid(), identity(os.getpid()))
            env |= {
                "KERNEL_ENVOLVE_STATE": str(store.root),
                "KERNEL_ENVOLVE_TASK": "feedback",
                "KERNEL_ENVOLVE_ROUND": parent,
            }
            wrapper = [
                sys.executable,
                "-m",
                "kernel_envolve.cli",
                "manage",
                "gpu-run",
                "--expected-seconds",
                "30",
                "--",
            ]
            for count in args.batch_sizes:
                command = [sys.executable, "-c", WORKLOAD, str(count), str(args.delay)]
                expected = [hashlib.sha256(str(i).encode()).hexdigest() for i in range(count)]
                # Include the initial pair; label it separately instead of discarding it.
                for trial in range(args.pairs):
                    for arm in ("direct", "managed") if trial % 2 == 0 else ("managed", "direct"):
                        started = time.perf_counter()
                        result = subprocess.run(
                            (wrapper if arm == "managed" else []) + command,
                            cwd=work,
                            env=env,
                            capture_output=True,
                            text=True,
                            timeout=30,
                        )
                        elapsed = time.perf_counter() - started
                        if result.returncode or json.loads(result.stdout) != expected:
                            raise RuntimeError(
                                f"Invalid {arm} result: {result.stderr} {result.stdout}"
                            )
                        records.append(
                            {
                                "batch_size": count,
                                "trial": trial,
                                "arm": arm,
                                "seconds": elapsed,
                                "output_verified": True,
                            }
                        )
            jobs = store.rows(
                "SELECT status,seconds,exit_code FROM jobs WHERE kind='gpu' AND parent=?", (parent,)
            )
            if len(jobs) != len(args.batch_sizes) * args.pairs or any(
                job["status"] != "succeeded" or job["exit_code"] != 0 for job in jobs
            ):
                raise RuntimeError("Managed jobs did not all finish successfully")
            if store.rows("SELECT * FROM runs") or list(work.glob(".kernel*")):
                raise RuntimeError("Local commands unexpectedly created evaluator runs")
            store.finish(parent, "succeeded")
        finally:
            # subprocess.run reaps a timed-out wrapper; reconcile its recorded child
            # before deleting this benchmark's private state directory.
            try:
                reconcile(store)
            finally:
                store.close()
    summaries = []
    for count in args.batch_sizes:
        values = {
            arm: [r["seconds"] for r in records if r["batch_size"] == count and r["arm"] == arm]
            for arm in ("direct", "managed")
        }
        deltas = [b - a for a, b in zip(values["direct"], values["managed"], strict=True)]
        summaries.append(
            {
                "batch_size": count,
                "direct_median_seconds": statistics.median(values["direct"]),
                "managed_median_seconds": statistics.median(values["managed"]),
                "paired_overhead_median_seconds": statistics.median(deltas),
                "overhead_per_item_seconds": statistics.median(deltas) / count,
            }
        )
    return {
        "label": args.label,
        "platform": platform.platform(),
        "python": sys.version,
        "workload_sha256": hashlib.sha256(WORKLOAD.encode()).hexdigest(),
        "supervisor_sha256": hashlib.sha256(Path(worker.__file__).read_bytes()).hexdigest(),
        "synthetic_history_jobs": args.history_jobs,
        "delay_per_item_seconds": args.delay,
        "pairs": args.pairs,
        "gpu_used": False,
        "scope": "CPU command feedback; no model search or GPU evaluation",
        "records": records,
        "jobs": jobs,
        "summary": summaries,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--pairs", type=int, default=12)
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 8])
    parser.add_argument("--delay", type=float, default=0.02)
    parser.add_argument("--history-jobs", type=int, default=0)
    args = parser.parse_args()
    if (
        args.pairs < 2
        or not math.isfinite(args.delay)
        or args.delay < 0
        or args.history_jobs < 0
        or any(n < 1 for n in args.batch_sizes)
        or len(set(args.batch_sizes)) != len(args.batch_sizes)
    ):
        parser.error(
            "Require two pairs, unique positive batch sizes, finite nonnegative delay/history"
        )
    result = run(args)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result["summary"], indent=2))
