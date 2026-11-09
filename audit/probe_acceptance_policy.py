"""Synthetic sensitivity study of native acceptance functions; no GPU/search claims.

python audit/probe_acceptance_policy.py --reference /path/to/pinned/atrex --output result.json

One shape, 80 scalar timing observations per policy, known multiplicative effect.
Atrex uses four native ABBA runs of 20 observations summarized by their medians;
kernel_envolve uses its native two-batch interleaved schedule (10 trials per batch).
Both consume the same latent clock trace but assign implementation roles according
to their own schedule. This models policies, not an actual clock-locked GPU runtime.
All conditions, effects, seeds and repetitions below are fixed before observation.
"""

import argparse
import copy
import hashlib
import json
import math
import platform
import statistics
import subprocess
import sys
from pathlib import Path

import numpy as np

from kernel_envolve.protocol import order, post_seed, timing_seed
from kernel_envolve.scoring import score
from kernel_envolve.storage import evaluator_hash

COMMIT = "bab3218768baea30ba56a87851705d60d64ebbf3"
VERIFIER_HASH = "0e87928fe50a05ce5f128d9435d6f2c1bf56f5f8545d8981cf05114bdc03bd9d"
REPETITIONS = 2000
SEED = 20261110
GAINS = [0.0, 0.05, 0.20]
CONDITIONS = {
    "quiet_iid": {"independent_log_sigma": 0.005},
    "noisy_iid": {"independent_log_sigma": 0.10},
    "shared_quartet_clock": {"independent_log_sigma": 0.002, "quartet_log_sigma": 0.15},
    "smooth_clock_cycle": {"independent_log_sigma": 0.002, "cycle_log_amplitude": 0.15},
}


def template():
    cfg = {
        "cases": [{"id": "one", "weight": 1}],
        "quick_seeds": [0],
        "full_seeds": [0, 1, 2],
        "quick_trials": 3,
        "full_trials": 10,
        "min_gain": 0.01,
        "noise_multiplier": 2.0,
        "max_regression": 0.05,
        "max_baseline_regression": 0.05,
    }
    result = {
        "mode": "full",
        "comparator": {"kind": "baseline"},
        "signal_only": False,
        "execution": "completed",
        "correctness": "pass",
        "checks": [],
        "samples": [],
    }
    for role in ["candidate", "comparator"]:
        for seed in cfg["full_seeds"]:
            for kind in ["fresh", "reuse"]:
                result["checks"].append(
                    {
                        "case": "one",
                        "seed": seed,
                        "kind": kind,
                        "role": role,
                        "status": "pass",
                    }
                )
        for kind, seed in [
            ("post", post_seed("one", cfg["full_seeds"])),
            ("timed_output", timing_seed("one", 1, 9, 1, cfg["full_seeds"])),
        ]:
            result["checks"].append(
                {
                    "case": "one",
                    "seed": seed,
                    "kind": kind,
                    "role": role,
                    "status": "pass",
                }
            )
    for batch in range(2):
        for trial in range(10):
            repeats = {"candidate": 0, "comparator": 0}
            for role in order(["candidate", "comparator"], batch, trial):
                repeat = repeats[role]
                result["samples"].append(
                    {
                        "case": "one",
                        "batch": batch,
                        "trial": trial,
                        "role": role,
                        "seed": timing_seed("one", batch, trial, repeat, cfg["full_seeds"]),
                        "repeat": repeat,
                        "us": 100.0,
                    }
                )
                repeats[role] += 1
    return result, cfg


def latent_trace(condition, condition_index, repeat):
    # Same trace for all three gains; independent seed per condition/repetition.
    rng = np.random.Generator(
        np.random.PCG64(np.random.SeedSequence([SEED, condition_index, repeat]))
    )
    log_clock = rng.normal(0, condition["independent_log_sigma"], 80)
    if "quartet_log_sigma" in condition:
        log_clock += np.repeat(rng.normal(0, condition["quartet_log_sigma"], 20), 4)
    if "cycle_log_amplitude" in condition:
        phase = rng.uniform(0, 2 * math.pi)
        log_clock += condition["cycle_log_amplitude"] * np.cos(
            np.arange(80) * 2 * math.pi / 80 + phase
        )
    return np.exp(log_clock).tolist()


def wilson(successes, total):
    z = 1.959963984540054
    p = successes / total
    denom = 1 + z * z / total
    center = (p + z * z / (2 * total)) / denom
    radius = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denom
    return [max(0.0, center - radius), min(1.0, center + radius)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.reference.resolve()
    commit = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True)
    if commit.strip() != COMMIT:
        raise ValueError("Reference commit differs")
    if subprocess.check_output(["git", "-C", str(root), "status", "--porcelain"], text=True):
        raise ValueError("Use a clean upstream checkout")
    verifier = root / "src/aka/light/acceptance/verifier.py"
    if hashlib.sha256(verifier.read_bytes()).hexdigest() != VERIFIER_HASH:
        raise ValueError("Reference verifier content differs")
    sys.path.insert(0, str(root / "src"))
    from aka.legacy.evaluation.abba import verification_schedule
    from aka.light.acceptance.verifier import score_verification_payload
    from aka.light.evaluation.policy import BENCH_ITERS

    assert BENCH_ITERS == 20
    schedule = verification_schedule(2)
    base, cfg = template()
    assert len(base["samples"]) == len(schedule) * BENCH_ITERS == 80
    results = []
    for condition_index, (name, condition) in enumerate(CONDITIONS.items()):
        for gain in GAINS:
            bits = {p: [] for p in ["kernel_envolve", "atrex_threshold_1pct", "atrex_default_0pct"]}
            examples = []
            for repeat in range(REPETITIONS):
                clock = latent_trace(condition, condition_index, repeat)
                evidence = copy.deepcopy(base)
                for sample, factor in zip(evidence["samples"], clock):
                    sample["us"] = 100 * factor * (1 - gain if sample["role"] == "candidate" else 1)
                local = score(evidence, cfg)
                bits["kernel_envolve"].append(local["eligible_for_best"])
                payload = {"runs": []}
                for index, step in enumerate(schedule):
                    factor = 1 - gain if step["revision"] == "candidate" else 1
                    values = [100 * v * factor for v in clock[index * 20 : (index + 1) * 20]]
                    # One shape: its median is also the per-run shape GeoMean.
                    payload["runs"].append(
                        {
                            **step,
                            "exit_code": 0,
                            "result": {
                                "all_pass": True,
                                "latency_us_geomean": statistics.median(values),
                            },
                        }
                    )
                native = {}
                for policy, threshold in [
                    ("atrex_threshold_1pct", 1.0),
                    ("atrex_default_0pct", 0.0),
                ]:
                    verdict = score_verification_payload(
                        payload,
                        schedule=schedule,
                        repeats=2,
                        min_improvement_pct=threshold,
                    )
                    if verdict.gate not in {"PASS", "FAIL"}:
                        raise AssertionError(verdict)
                    bits[policy].append(verdict.gate == "PASS")
                    native[policy] = verdict.as_dict()
                different = len({v[-1] for v in bits.values()}) > 1
                if repeat == 0 or (different and len(examples) < 2):
                    examples.append(
                        {
                            "repeat": repeat,
                            "latent_clock": clock,
                            "local_evidence": evidence,
                            "local_decision": local,
                            "native_payload": payload,
                            "native_decisions": native,
                        }
                    )
            row = {
                "condition": name,
                "true_gain": gain,
                "repetitions": REPETITIONS,
                "metric": "false_accept_rate" if gain == 0 else "true_gain_accept_rate",
                "policies": {
                    policy: {
                        "accepted": sum(values),
                        "rate": sum(values) / REPETITIONS,
                        "wilson_95": wilson(sum(values), REPETITIONS),
                        "outcome_bits": "".join("1" if v else "0" for v in values),
                    }
                    for policy, values in bits.items()
                },
                "examples": examples,
            }
            results.append(row)
            print(
                name, gain, {k: round(v["rate"], 4) for k, v in row["policies"].items()}, flush=True
            )
    loaded = {}
    for name, module in sys.modules.items():
        if name.startswith("aka") and getattr(module, "__file__", None):
            path = Path(module.__file__).resolve()
            loaded[str(path.relative_to(root))] = hashlib.sha256(path.read_bytes()).hexdigest()
    report = {
        "scope": (
            "Synthetic timing sensitivity, native score functions only; correctness assumed. "
            "Not real GPU error rates or model search. No claim of calibrated statistical "
            "confidence from either gate. Native run medians constructed by this probe; "
            "no native runner, clock locking, correctness or remote recovery executed."
        ),
        "reference_commit": COMMIT,
        "reference_loaded_sources_sha256": loaded,
        "evaluator_hash": evaluator_hash(),
        "probe_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "seed": SEED,
        "conditions": CONDITIONS,
        "true_gains": GAINS,
        "repetitions": REPETITIONS,
        "local_config": cfg,
        "native_schedule": schedule,
        "observations_per_policy": 80,
        "interval_scope": "Wilson intervals quantify Monte Carlo sampling only, not deployment validity",
        "results": results,
    }
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
