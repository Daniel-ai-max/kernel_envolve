"""Validate paired signal and noise together against the pinned 0.3.1 scorer.

python audit/probe_paired_policy.py --output result.json

Synthetic, single-shape timing sensitivity; neither GPU measurements nor search
success rates. Reuses nine explored noise families with a fresh seed and adds
four prespecified families. Both scorers consume identical 80-observation records.
"""

import argparse
import copy
import hashlib
import importlib.util
import json
import math
import statistics
import subprocess
import time
import types
from pathlib import Path

import numpy as np

from kernel_envolve.protocol import post_seed, timing_seed
from kernel_envolve.scoring import score
from kernel_envolve.storage import evaluator_hash

BASELINE = "243059c875a0a6beb6e2023b0f792cc054055d36"
SEED = 20261113
REPETITIONS = 2000
CONDITIONS = [
    "quiet_iid",
    "noisy_iid",
    "shared_quartet_clock",
    "smooth_clock_cycle",
    "heavy_tail",
    "rare_pause",
    "ar1_clock",
    "single_step",
    "alternating_clock",
    "gamma_iid",
    "block_pause",
    "two_steps",
    "exponential_iid",
]


def trace(name, index, repeat):
    rng = np.random.default_rng(np.random.SeedSequence([SEED, index, repeat]))
    if name in {"quiet_iid", "noisy_iid"}:
        log = rng.normal(0, 0.005 if name == "quiet_iid" else 0.10, 80)
    elif name == "shared_quartet_clock":
        log = rng.normal(0, 0.002, 80) + np.repeat(rng.normal(0, 0.15, 20), 4)
    elif name == "smooth_clock_cycle":
        log = rng.normal(0, 0.002, 80)
        log += 0.15 * np.cos(np.arange(80) * 2 * math.pi / 80 + rng.uniform(0, 2 * math.pi))
    elif name == "heavy_tail":
        log = 0.03 * rng.standard_t(3, 80)
    elif name == "rare_pause":
        log = rng.normal(0, 0.002, 80) + math.log(3) * (rng.random(80) < 0.03)
    elif name == "ar1_clock":
        log = np.zeros(80)
        log[0] = rng.normal(0, 0.1)
        for k in range(1, 80):
            log[k] = 0.9 * log[k - 1] + rng.normal(0, 0.1 * math.sqrt(1 - 0.9**2))
    elif name == "single_step":
        log = rng.normal(0, 0.002, 80)
        log[rng.integers(1, 80) :] += math.log(1.3)
    elif name == "alternating_clock":
        log = rng.normal(0, 0.002, 80) + 0.15 * ((np.arange(80) % 2) * 2 - 1)
    elif name == "gamma_iid":
        return rng.gamma(100, 0.01, 80).tolist()
    elif name == "block_pause":
        log = rng.normal(0, 0.002, 80)
        start = rng.integers(0, 75)
        log[start : start + 6] += math.log(3)
    elif name == "two_steps":
        log = rng.normal(0, 0.002, 80)
        for offset in [0, 40]:
            log[offset + rng.integers(1, 40) : offset + 40] += math.log(1.3)
    elif name == "exponential_iid":
        return (0.9 + rng.exponential(0.1, 80)).tolist()
    else:
        raise ValueError(name)
    return np.exp(log).tolist()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location(
        "fixtures", root / "audit/probe_acceptance_policy.py"
    )
    fixtures = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fixtures)
    base, cfg = fixtures.template()
    old_source = subprocess.check_output(
        ["git", "show", f"{BASELINE}:src/kernel_envolve/scoring.py"],
        cwd=root,
        text=True,
    )
    old_protocol = subprocess.check_output(
        ["git", "show", f"{BASELINE}:src/kernel_envolve/protocol.py"],
        cwd=root,
    )
    if old_protocol != (root / "src/kernel_envolve/protocol.py").read_bytes():
        raise ValueError("This replay requires the unchanged sampling protocol")
    old = types.ModuleType("kernel_envolve._score_031")
    old.__package__ = "kernel_envolve"
    exec(compile(old_source, "<pinned-v031-score>", "exec"), old.__dict__)
    results = []
    for index, name in enumerate(CONDITIONS):
        for gain in [0.0, 0.05, 0.20]:
            bits = {"previous": [], "current": []}
            examples = []
            for repeat in range(REPETITIONS):
                clock = trace(name, index, repeat)
                for sample, value in zip(base["samples"], clock):
                    sample["us"] = float(
                        100 * value * (1 - gain if sample["role"] == "candidate" else 1)
                    )
                previous, current = old.score(base, cfg), score(base, cfg)
                bits["previous"].append(previous["eligible_for_best"])
                bits["current"].append(current["eligible_for_best"])
                if repeat == 0 or (
                    gain == 0 and current["eligible_for_best"] and len(examples) < 2
                ):
                    examples.append(
                        {
                            "repeat": repeat,
                            "evidence": copy.deepcopy(base),
                            "previous": previous,
                            "current": current,
                        }
                    )
            row = {
                "condition": name,
                "true_gain": gain,
                "repetitions": REPETITIONS,
                "policies": {
                    k: {
                        "accepted": sum(v),
                        "rate": sum(v) / REPETITIONS,
                        "wilson_95": fixtures.wilson(sum(v), REPETITIONS),
                        "outcome_bits": "".join("1" if x else "0" for x in v),
                    }
                    for k, v in bits.items()
                },
                "examples": examples,
            }
            results.append(row)
            print(name, gain, {k: v["accepted"] for k, v in row["policies"].items()}, flush=True)

    # Compare score-only CPU cost at one and sixteen cases; never kernel timing.
    costs = []
    for count in [1, 16]:
        evidence, settings = copy.deepcopy(base), copy.deepcopy(cfg)
        settings["cases"] = [{"id": str(i), "weight": 1} for i in range(count)]
        evidence["checks"], evidence["samples"] = [], []
        for case in settings["cases"]:
            for check in base["checks"]:
                item = dict(check, case=case["id"])
                if item["kind"] == "post":
                    item["seed"] = post_seed(case["id"], cfg["full_seeds"])
                elif item["kind"] == "timed_output":
                    item["seed"] = timing_seed(case["id"], 1, 9, 1, cfg["full_seeds"])
                evidence["checks"].append(item)
        for batch in [0, 1]:
            cases = settings["cases"] if batch == 0 else list(reversed(settings["cases"]))
            for case in cases:
                for sample in base["samples"]:
                    if sample["batch"] == batch:
                        evidence["samples"].append(
                            dict(
                                sample,
                                case=case["id"],
                                seed=timing_seed(
                                    case["id"],
                                    batch,
                                    sample["trial"],
                                    sample["repeat"],
                                    cfg["full_seeds"],
                                ),
                            )
                        )
        timings = {"previous": [], "current": []}
        for repeat in range(5):
            arms = [("previous", old.score), ("current", score)]
            for name, fn in arms if repeat % 2 == 0 else reversed(arms):
                start = time.perf_counter()
                for _ in range(30):
                    fn(evidence, settings)
                timings[name].append((time.perf_counter() - start) / 30)
        costs.append(
            {
                "cases": count,
                "calls_per_block": 30,
                "seconds_per_call": timings,
                "median_seconds": {k: statistics.median(v) for k, v in timings.items()},
            }
        )
    args.output.write_text(
        json.dumps(
            {
                "scope": "Synthetic policy sensitivity and score-only CPU cost; no GPU or model-search rates",
                "baseline_commit": BASELINE,
                "baseline_scoring_sha256": hashlib.sha256(old_source.encode()).hexdigest(),
                "evaluator_hash": evaluator_hash(),
                "probe_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "seed": SEED,
                "numpy": np.__version__,
                "local_config": cfg,
                "results": results,
                "scoring_cost": costs,
            },
            indent=2,
            allow_nan=False,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
