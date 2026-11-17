"""Paired CPU engineering overhead; no model, GPU, or kernel quality benefit claim."""

import argparse
import json
import os
import platform
import statistics
import sys
import time
from pathlib import Path

from kernel_envolve.manager.config import validate
from kernel_envolve.manager.evidence import inspect_run
from kernel_envolve.manager.gpu import run as gpu_run
from kernel_envolve.manager.process import identity
from kernel_envolve.manager.store import Store
from kernel_envolve.rsi.managed import evaluate
from kernel_envolve.storage import config, contract, file_hash, write_json


def main(root):
    root = root.resolve()
    root.mkdir(parents=True, exist_ok=False)
    task = root / "task"
    task.mkdir()
    write_json(
        task / "task.json",
        {
            "name": "managed-overhead",
            "objective": "CPU admission overhead only",
            "device": "cpu",
            "cases": [{"id": "a", "args": {}, "weight": 1}],
            "warmup": 1,
            "full_trials": 3,
            "full_seeds": [1],
            "quick_seeds": [1],
        },
    )
    (task / "task.py").write_text(
        "import numpy as np\ndef make_inputs(case,seed): return (np.ones(16),)\n"
        "def reference(x): return x+1\ndef baseline(x): return x+1\n"
    )
    (task / "candidate.py").write_text("def run(x): return x+1\n")
    spec = {
        "id": "overhead",
        "name": "CPU overhead only",
        "workspace": str(task),
        "task_dir": str(task),
        "kind": "generic",
        "resource": "cpu",
        "command": [sys.executable, "-c", "pass"],
        "round_timeout_seconds": 600,
        "wall_budget_seconds": 600,
        "gpu_budget_seconds": 300,
        "baseline": {
            "repository": "local-engineering-fixture",
            "revision": "1",
            "entrypoint": "task.py:baseline",
            "introduced_on": None,
            "introduction_url": None,
            "frozen_on": "2026-11-11",
            "hardware": "CPU",
            "contract": contract(task, config(task)),
        },
    }
    deployment = validate(
        {"version": 1, "resources": [{"id": "cpu", "device": "cpu"}], "tasks": [spec]}, root
    )
    store = Store(root / "manager")
    store.register(deployment)
    parent = store.schedule()[0]
    assert store.claim(parent, os.getpid(), identity(os.getpid()))
    os.environ.update(
        KERNEL_ENVOLVE_STATE=str(store.root),
        KERNEL_ENVOLVE_TASK="overhead",
        KERNEL_ENVOLVE_ROUND=parent,
        KERNEL_ENVOLVE_AUTO_UPDATE="0",
    )
    rows = []
    try:
        for pair in range(-1, 12):  # Explicit warm-up pair is recorded, excluded from estimates.
            arms = ["ordinary-manager", "managed-research"]
            if pair % 2:
                arms.reverse()
            for arm in arms:
                directory = root / f"{pair}-{arm}"
                began = time.perf_counter()
                if arm == "managed-research":
                    path, _ = evaluate(task, "full", 5, directory)
                else:
                    directory.mkdir()
                    script = (
                        "import sys;from pathlib import Path;from kernel_envolve.cli import evaluate;"
                        "path,result=evaluate(sys.argv[1],'full',timeout=5);"
                        "Path(sys.argv[2]).write_text(str(path))"
                    )
                    assert (
                        gpu_run(
                            store.root,
                            "overhead",
                            parent,
                            [sys.executable, "-c", script, str(task), str(directory / "run.txt")],
                            expected_seconds=5,
                        )
                        == 0
                    )
                    path = Path((directory / "run.txt").read_text())
                elapsed = time.perf_counter() - began
                verified = inspect_run(path, spec)
                assert verified["correctness"] == "pass"
                rows.append(
                    {
                        "pair": pair,
                        "arm": arm,
                        "seconds": elapsed,
                        "run": str(path),
                        "result_hash": verified["result_hash"],
                    }
                )
                write_json(root / "progress.json", rows)
        store.finish(parent, "succeeded")
        assert not store.rows(
            "SELECT id FROM jobs WHERE kind='gpu' AND status IN ('queued','running')"
        )
    finally:
        store.close()
    pairs = []
    for pair in range(12):
        values = {row["arm"]: row["seconds"] for row in rows if row["pair"] == pair}
        pairs.append(values["managed-research"] - values["ordinary-manager"])
    result = {
        "scope": "CPU/no-model engineering measurement. Existing Manager admission versus admitted research including extra receipt verification. Not cloud GPU usage or an RSI efficacy/production latency result.",
        "platform": platform.platform(),
        "python": platform.python_version(),
        "helper_sha256": file_hash(Path(__file__)),
        "managed_source_sha256": file_hash(Path(sys.modules[evaluate.__module__].__file__)),
        "pairs": 12,
        "warmup_pairs_recorded": 1,
        "median_extra_seconds": statistics.median(pairs),
        "min_extra_seconds": min(pairs),
        "max_extra_seconds": max(pairs),
        "arm_medians": {
            arm: statistics.median(
                row["seconds"] for row in rows if row["pair"] >= 0 and row["arm"] == arm
            )
            for arm in ["ordinary-manager", "managed-research"]
        },
        "rows": rows,
    }
    write_json(root / "summary.json", result)
    print(json.dumps({k: v for k, v in result.items() if k != "rows"}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    main(parser.parse_args().root)
