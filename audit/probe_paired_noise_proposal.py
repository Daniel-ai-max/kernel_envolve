"""Reproduce the rejected paired-dispersion-only proposal on nine noise families.

Run after installing kernel_envolve and NumPy:
  python audit/probe_paired_noise_proposal.py --output proposal.json

Only noise estimation is changed in this exploratory single-shape predicate.
It is deliberately not a general replacement for score() or promote().
"""

import argparse
import hashlib
import importlib.util
import json
import math
import statistics as st
from pathlib import Path

import numpy as np

parser = argparse.ArgumentParser(
    description="Rejected paired-noise-only proposal: synthetic sensitivity check"
)
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()
root = Path(__file__).resolve().parents[1]
s = importlib.util.spec_from_file_location("probe", root / "audit/probe_acceptance_policy.py")
p = importlib.util.module_from_spec(s)
s.loader.exec_module(p)
base, cfg = p.template()


def paired(evidence, decision):
    guards = []
    for b in range(2):
        delta = []
        for trial in range(10):
            c = [
                x["us"]
                for x in evidence["samples"]
                if x["batch"] == b and x["trial"] == trial and x["role"] == "candidate"
            ]
            r = [
                x["us"]
                for x in evidence["samples"]
                if x["batch"] == b and x["trial"] == trial and x["role"] == "comparator"
            ]
            delta.append(st.median(c) - st.median(r))
        med = st.median(delta)
        mad = st.median(abs(x - med) for x in delta)
        q1, _, q3 = st.quantiles(delta, n=4, method="inclusive")
        denom = decision["cases"][b]["comparator_us"]
        guards.append(2 * max(mad, (q3 - q1) / 2) / denom)
    return all(g > max(0.01, n) for g, n in zip(decision["gain_by_batch"], guards))


conditions = list(p.CONDITIONS) + [
    "heavy_tail",
    "rare_pause",
    "ar1_clock",
    "single_step",
    "alternating_clock",
]
rows = []
for ci, name in enumerate(conditions):
    for gain in [0.0, 0.05, 0.2]:
        counts = [0, 0]
        for rep in range(2000):
            rng = np.random.default_rng(np.random.SeedSequence([20261111, ci, rep]))
            if name in p.CONDITIONS:
                # Separate seed from the first comparison; equivalent noise families.
                log = rng.normal(0, p.CONDITIONS[name]["independent_log_sigma"], 80)
                if name == "shared_quartet_clock":
                    log += np.repeat(rng.normal(0, 0.15, 20), 4)
                if name == "smooth_clock_cycle":
                    log += 0.15 * np.cos(
                        np.arange(80) * 2 * math.pi / 80 + rng.uniform(0, 2 * math.pi)
                    )
            elif name == "heavy_tail":
                log = 0.03 * rng.standard_t(3, 80)
            elif name == "rare_pause":
                log = rng.normal(0, 0.002, 80) + np.log(3.0) * (rng.random(80) < 0.03)
            elif name == "ar1_clock":
                log = np.zeros(80)
                log[0] = rng.normal(0, 0.1)
                for k in range(1, 80):
                    log[k] = 0.9 * log[k - 1] + rng.normal(0, 0.1 * math.sqrt(1 - 0.9**2))
            elif name == "single_step":
                log = rng.normal(0, 0.002, 80)
                log[rng.integers(1, 80) :] += math.log(1.3)
            else:
                log = rng.normal(0, 0.002, 80) + 0.15 * ((np.arange(80) % 2) * 2 - 1)
            values = np.exp(log)
            for x, v in zip(base["samples"], values):
                x["us"] = float(100 * v * (1 - gain if x["role"] == "candidate" else 1))
            d = p.score(base, cfg)
            counts[0] += d["eligible_for_best"]
            counts[1] += paired(base, d)
        row = {
            "condition": name,
            "gain": gain,
            "repetitions": 2000,
            "current_accepted": counts[0],
            "paired_accepted": counts[1],
        }
        rows.append(row)
        print(row, flush=True)
args.output.write_text(
    json.dumps(
        {
            "scope": "Exploratory single-shape simulation of a rejected proposal; not a production scoring function or real GPU rates",
            "seed": 20261111,
            "evaluator_hash": p.evaluator_hash(),
            "probe_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "rows": rows,
        },
        indent=2,
    )
)
