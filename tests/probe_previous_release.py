"""Run identical fault probes against a selected evaluator via PYTHONPATH.

This emits observations, not pass/fail expectations, so it also runs on older releases.
Run with the development environment: python tests/probe_previous_release.py
"""

import json
import os
import py_compile
import tempfile
from pathlib import Path

import numpy as np

from kernel_envolve import __version__
from kernel_envolve.arrays import assert_close
from kernel_envolve.cli import evaluate
from kernel_envolve.protocol import order, post_seed, timing_seed
from kernel_envolve.scoring import score
from kernel_envolve.storage import bundle, config, load_module, verify_source


def run():
    observations = {}
    with tempfile.TemporaryDirectory() as temp:
        root = Path(temp).resolve()
        task = root / "task"
        task.mkdir()
        cfg = {
            "name": "fault-probe",
            "objective": "synthetic evaluator regression probes",
            "cases": [{"id": "a", "weight": 1}],
            "quick_seeds": [0],
            "full_seeds": [0],
            "warmup": 1,
            "quick_trials": 1,
            "full_trials": 3,
        }
        (task / "task.json").write_text(json.dumps(cfg))
        shared = (
            "import numpy as np\n"
            "def make_inputs(case,seed):\n r=np.random.default_rng(seed)\n return (r.random(8).astype(np.float32),)\n"
            "def reference(x):\n"
            " if not hasattr(np,'_ke_probe'): np._ke_probe=np.empty_like(x)\n"
            " np.copyto(np._ke_probe,x)\n return np._ke_probe\nbaseline=reference\n"
        )
        (task / "task.py").write_text(shared)
        (task / "candidate.py").write_text(
            "import numpy as np\ndef run(x):\n np._ke_probe.fill(0)\n return np._ke_probe\n"
        )
        _, r = evaluate(task, check_only=True)
        observations["shared_oracle_buffer"] = (r["execution"], r["correctness"])

        (task / "candidate.py").write_text(
            "import numpy as np\nn=0\ndef run(x):\n global n\n n+=1\n"
            " if 4<=n<=5: np._ke_probe.fill(0)\n else: np.copyto(np._ke_probe,x)\n return np._ke_probe\n"
        )
        _, r = evaluate(task)
        observations["shared_timed_output"] = (r["execution"], r["correctness"])

        (task / "task.py").write_text(
            "import numpy as np\ndef make_inputs(case,seed): return (np.ones(8,dtype=np.float32),)\ndef reference(x): return x+1\nbaseline=reference\n"
        )
        (task / "candidate.py").write_text(
            "import numpy as np\nnp.seterr(all='ignore')\ndef run(x): return x+1\n"
        )
        _, r = evaluate(task, check_only=True)
        observations["global_numeric_policy_leak"] = (r["execution"], r["correctness"])

        (task / "candidate.py").write_text("def run(x): return x+1\n")
        saved = root / "saved"
        saved.mkdir()
        identity = bundle(task, saved / "source", config(task))
        (saved / "source" / "unlisted.py").write_text("x=1\n")
        try:
            verify_source(saved, identity)
            observations["unlisted_snapshot_source"] = "accepted"
        except ValueError:
            observations["unlisted_snapshot_source"] = "rejected"

        helper = root / "helper.py"
        helper.write_text("value=2\n")
        stamp = helper.stat()
        py_compile.compile(str(helper), doraise=True)
        helper.write_text("value=1\n")
        os.utime(helper, ns=(stamp.st_atime_ns, stamp.st_mtime_ns))
        observations["stale_bytecode_value"] = load_module(root, "helper").value

        effective = config(task)
        result = {
            "mode": "full",
            "signal_only": False,
            "comparator": {"kind": "baseline"},
            "execution": "completed",
            "correctness": "pass",
            "checks": [],
            "samples": [],
        }
        for role in ("candidate", "comparator"):
            for kind, seed in [
                ("fresh", 0),
                ("reuse", 0),
                ("post", post_seed("a", [0])),
                ("timed_output", timing_seed("a", 1, 2, 1, [0])),
            ]:
                result["checks"].append(
                    {"case": "a", "seed": seed, "kind": kind, "role": role, "status": "pass"}
                )
        for batch in range(2):
            for trial in range(3):
                repeats = dict.fromkeys(("candidate", "comparator"), 0)
                for role in order(list(repeats), batch, trial):
                    repeat = repeats[role]
                    result["samples"].append(
                        {
                            "case": "a",
                            "batch": batch,
                            "trial": trial,
                            "repeat": repeat,
                            "seed": timing_seed("a", batch, trial, repeat, [0]),
                            "role": role,
                            "us": (100 if trial == 2 else 8) if role == "candidate" else 10,
                        }
                    )
                    repeats[role] += 1
        decision = score(result, effective)
        observations["zero_mad_multimodal_eligible"] = decision["eligible_for_best"]
        try:
            assert_close(
                np.array([65504], dtype=np.float16),
                np.array([-65504], dtype=np.float16),
                {"rtol": 1.5, "atol": 0, "equal_nan": False},
            )
            observations["float16_tolerance_overflow"] = "accepted"
        except AssertionError:
            observations["float16_tolerance_overflow"] = "rejected"
    return {"version": __version__, "observations": observations}


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
