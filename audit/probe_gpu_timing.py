import base64
import gzip
import hashlib
import inspect
import json
import pathlib
import statistics
import subprocess
import time

import torch
import triton
import triton.testing

from kernel_envolve.storage import evaluator_hash, load_module
from kernel_envolve.worker import Clock

root = pathlib.Path.cwd().resolve()
module = load_module(root / "examples/cuda_vector_add", "candidate")
report = {
    "kind": "timer-scope-calibration",
    "evaluator_hash": evaluator_hash(),
    "device": torch.cuda.get_device_name(0),
    "torch": torch.__version__,
    "triton": triton.__version__,
    "cuda": torch.version.cuda,
    "scenarios": [],
    "methods": {},
}
report["nvidia_smi"] = subprocess.check_output(
    [
        "nvidia-smi",
        "--query-gpu=name,uuid,driver_version,pstate,clocks.sm,clocks.mem,power.limit",
        "--format=csv,noheader",
    ],
    text=True,
)
for name in ["do_bench", "do_bench_cudagraph"]:
    source = inspect.getsource(getattr(triton.testing, name))
    report["methods"][name] = {
        "source": source,
        "sha256": hashlib.sha256(source.encode()).hexdigest(),
    }
config = json.loads((root / "examples/cuda_vector_add/task.json").read_text())
g = torch.Generator(device="cuda").manual_seed(96351)
for case in config["cases"]:
    n = case["args"]["n"]
    stride = case["args"]["stride"]
    x, y = tuple(torch.randn(n * stride, device="cuda", generator=g)[::stride] for _ in range(2))

    def delayed_torch(x, y):
        time.sleep(0.0002)
        return x + y

    functions = {"torch": lambda x, y: x + y, "triton": module.run}
    if case["id"] == "tiny-tail":
        functions["torch_host_delay_200us"] = delayed_torch
    # Repeat and rotate order; results from different scopes remain separate.
    for round in range(3):
        names = list(functions)
        names = names[round % len(names) :] + names[: round % len(names)]
        for name in names:
            fn = functions[name]
            torch.testing.assert_close(fn(x, y), x + y)
            for _ in range(10):
                fn(x, y)
            torch.cuda.synchronize()
            row = {"case": case["id"], "role": name, "round": round, "fixed_input": True}
            for timer in ["cuda_event", "wall"]:
                clock = Clock("cuda:0", timer)
                samples = []
                for _ in range(30):
                    elapsed, out = clock.measure(fn, (x, y))
                    samples.append(elapsed)
                row[timer] = {"median_us": statistics.median(samples), "samples_us": samples}
                torch.testing.assert_close(out, x + y)
            row["triton_do_bench_us"] = (
                triton.testing.do_bench(
                    lambda: fn(x, y), warmup=10, rep=50, quantiles=[0.5, 0.25, 0.75]
                )[0]
                * 1000
            )
            row["triton_cudagraph_us"] = (
                triton.testing.do_bench_cudagraph(lambda: fn(x, y), rep=50) * 1000
            )
            report["scenarios"].append(row)
            print(
                "KE_TIMER "
                + json.dumps(
                    {k: v for k, v in row.items() if k not in ["cuda_event", "wall"]}
                    | {
                        "cuda_event_us": row["cuda_event"]["median_us"],
                        "wall_us": row["wall"]["median_us"],
                    }
                ),
                flush=True,
            )
report["completed"] = True
(root / "timer-probe.json").write_text(json.dumps(report, indent=2))
print(
    "KE_REPORT_GZ_B64 " + base64.b64encode(gzip.compress(json.dumps(report).encode())).decode(),
    flush=True,
)
print("KE_VALIDATION_DONE", flush=True)
