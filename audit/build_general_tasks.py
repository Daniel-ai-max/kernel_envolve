"""Materialize broad tasks in ordinary directories; no service, model call or allocation.

The public probe seed is not a secret final-test seed. A future experiment must
freeze a separate validation sample outside the agent workspace before searching.
"""

import argparse
import hashlib
import json
import random
from pathlib import Path

NAMES = ("row_sum", "matmul", "fused_relu")
DTYPES = ("float32", "float16", "bfloat16")
LAYOUTS = (
    {"column": 1, "pad": 0, "offset": 0},
    {"column": 1, "pad": 7, "offset": 5},
    {"column": 3, "pad": 11, "offset": 1},
    {"column": 2, "pad": 5, "offset": 3, "transpose": True},
)
BASELINES = {
    "row_sum": "def run(x):\n    return x.sum(dim=1)\n",
    "matmul": "def run(a,b):\n    return a @ b\n",
    "fused_relu": "import torch\ndef run(x,scale,bias,residual):\n    return torch.relu(x*scale+bias)+residual\n",
}


def public_cases(name, device):
    # Every dtype gets the same dimensional coverage, with independently varied layouts.
    shapes = (
        [(1, 1, 1), (31, 65, 127), (128, 128, 128), (3, 257, 1023), (257, 7, 129), (0, 5, 0)]
        if name == "matmul"
        else [(1, 1), (17, 257), (32, 1024), (3, 8193), (7, 4095), (0, 0)]
    )
    if name == "row_sum":
        shapes[4] = (7, 32769)
    shapes += (
        [(5, 7, 0), (512, 512, 1024)]
        if name == "matmul"
        else [(3, 0), (128, 8192) if name == "row_sum" else (512, 1024)]
    )
    result = []
    for di, dtype in enumerate(DTYPES):
        for i, shape in enumerate(shapes):
            keys = ("m", "n", "k") if name == "matmul" else ("rows", "columns")
            count = 2 if name == "matmul" else 4 if name == "fused_relu" else 1
            args = dict(zip(keys, shape, strict=True))
            args.update(
                device=device,
                dtype=dtype,
                layouts=[dict(LAYOUTS[(i + j + di) % len(LAYOUTS)]) for j in range(count)],
                pattern="zeros" if i == 0 else "cancellation" if i == 4 else "normal",
            )
            case = {"id": f"{dtype}-{i}", "args": args, "quick": i == 1}
            case.update({"weight": 1} if i in (1, 2, 3, 7) else {"measure": False})
            result.append(case)
    return result


def sample_cases(name, device, seed):
    """A fixed-size stratified same-domain sample, independent of candidate output."""
    rng = random.Random(seed)
    result = []
    for di, dtype in enumerate(DTYPES):
        for i in range(4):
            if name == "matmul":
                # Cover small, skinny, larger and K-boundary regions in each dtype.
                shape = [
                    (rng.randint(2, 30), rng.randint(2, 30), rng.randint(2, 62)),
                    (rng.randint(1, 8), rng.randint(129, 512), rng.randint(65, 257)),
                    (rng.randint(129, 512), rng.randint(33, 256), rng.randint(33, 256)),
                    (rng.randint(9, 64), rng.randint(9, 64), rng.choice([255, 256, 257, 1024])),
                ][i]
                args = dict(zip(("m", "n", "k"), shape, strict=True))
                count = 2
            else:
                width_max = 32769 if name == "row_sum" else 8193
                shape = [
                    (rng.randint(1, 8), rng.randint(2, 254)),
                    (rng.randint(9, 31), rng.choice([255, 256, 258, 1023, 1025])),
                    (rng.randint(33, 128), rng.randint(1026, 4094)),
                    (rng.randint(1, 16), rng.randint(4096, width_max)),
                ][i]
                args = dict(zip(("rows", "columns"), shape, strict=True))
                count = 4 if name == "fused_relu" else 1
            layouts = []
            for j in range(count):
                layout = dict(LAYOUTS[(i + j + di + 1) % len(LAYOUTS)])
                layout.update(pad=rng.randint(1, 17), offset=rng.randint(0, 9))
                layouts.append(layout)
            args.update(
                device=device,
                dtype=dtype,
                layouts=layouts,
                pattern="cancellation" if i == 3 else "normal",
            )
            case = {"id": f"sample-{dtype}-{i}", "args": args, "quick": i == 0}
            case.update({"weight": 1} if i in (0, 2) else {"measure": False})
            result.append(case)
    return result


def materialize(root, device, sample_seed=None):
    root.mkdir(parents=True, exist_ok=False)
    helper = Path(__file__).with_name("general_workloads.py").read_bytes()
    manifest = {"device": device, "sample_seed": sample_seed, "files": {}}
    for name in NAMES:
        folder = root / name
        folder.mkdir()
        cases = (
            public_cases(name, device)
            if sample_seed is None
            else sample_cases(name, device, sample_seed)
        )
        cfg = {
            "name": "general-" + name,
            "objective": "Equal-weight total eager wall latency on representative cases; all declared cases must be correct. Independent dtype and layout coverage.",
            "device": device,
            "timer": "wall",
            "rtol": 0.025,
            "atol": 0.025,
            "reference_files": ["task.py", "workload.py"],
            "min_gain": 0.02,
            "max_regression": 0.05,
            "max_baseline_regression": 0.05,
            "cases": cases,
        }
        task = f"""from . import workload
import torch
torch.backends.cuda.matmul.allow_tf32 = False
torch.set_float32_matmul_precision("highest")
def make_inputs(case,seed): return workload.make_inputs({name!r},case,seed)
def reference(*inputs): return workload.reference({name!r},*inputs)
{BASELINES[name].replace("def run(", "def baseline(")}
def validate(inputs,output): return workload.validate({name!r},inputs,output)
"""
        for filename, content in (
            ("task.json", json.dumps(cfg, indent=2).encode()),
            ("task.py", task.encode()),
            ("workload.py", helper),
            ("candidate.py", BASELINES[name].encode()),
        ):
            (folder / filename).write_bytes(content)
            manifest["files"][name + "/" + filename] = hashlib.sha256(content).hexdigest()
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--device", choices=["cpu", "cuda:0"], required=True)
    parser.add_argument("--sample-seed", type=int)
    args = parser.parse_args()
    materialize(args.directory, args.device, args.sample_seed)
