"""Reconcile closed owned job receipts and model usage, without reading open trials.

Usage: python audit/rsi_campaign_cost.py WORKSPACE OUTPUT
Conservative allocation time is create reservation through confirmed empty Pods;
this is not active GPU compute or a billing estimate. Missing usage stays unknown.
"""

import hashlib
import json
import sys
from pathlib import Path

CLOSED = (
    "rsi-batch-pilot-20261110",
    "rsi-batch-pilot-20261110-b",
    "rsi-batch-pilot-20261110-c",
    "rsi-multistep-pilot-20261110",
    "rsi-multistep-pilot-20261110-b",
    "rsi-routing-pilot-20261110",
    "rsi-value-study-20261110/train-batch",
    "rsi-value-study-20261110/dev-round1-transport2",
    "rsi-value-study-20261110/dev-round2",
    "rsi-episode-study-20261110/development",
    "rsi-artifact-reuse-study-20261111/development",
    "rsi-linked-reuse-study-20261111/development",
    "rsi-compute-allocation-study-20261111/development",
    "rsi-managed-gpu-validation-20261111",
    "rsi-adaptive-budget-study-20261111/development",
    "rsi-generational-study-20261111/learning1",
    "rsi-seed-revalidation-20261111/run",
    "rsi-revalidated-evolution-study-20261111/learning1",
    "rsi-revalidated-evolution-study-20261111/admission1",
)


def read(path):
    return json.loads(path.read_text())


def hash_file(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def reconcile(workspace):
    entries, identities = [], set()
    running_total = 0.0
    for name in CLOSED:
        root = workspace / name
        release = read(root / "released.json")
        created = read(root / "create-attempt.json")
        approval = read(root / "approved-budget.json")
        assert release["id"] not in identities
        identities.add(release["id"])
        assert release["phase"] in {"Completed", "Stopped", "Failed"}
        assert release["pods"] == [] and release["live_read_status"]["pods"] == "available"
        assert release["at"] >= created["at"]
        if "prior_confirmed_gpu_seconds" in approval:
            assert abs(approval["prior_confirmed_gpu_seconds"] - running_total) < 1e-6
        seconds = release["at"] - created["at"]
        running_total += seconds
        calls = []
        for path in sorted((root / "model-calls").glob("*.json")):
            value = read(path)
            usage = (value.get("response") or {}).get("usage")
            tokens = None if usage is None else usage.get("total_tokens")
            assert tokens is None or type(tokens) is int and tokens >= 0
            calls.append(
                {
                    "file": path.name,
                    "sha256": hash_file(path),
                    "status": value.get("status", "unrecorded"),
                    "known_total_tokens": tokens,
                }
            )
        entries.append(
            {
                "folder": name,
                "job": release["id"],
                "created_at": created["at"],
                "released_at": release["at"],
                "conservative_gpu_seconds": seconds,
                "cumulative_gpu_seconds": running_total,
                "receipt_sha256": {
                    n: hash_file(root / n)
                    for n in ("released.json", "create-attempt.json", "approved-budget.json")
                },
                "model_attempts": len(calls),
                "known_tokens": sum(c["known_total_tokens"] or 0 for c in calls),
                "unknown_usage_attempts": sum(c["known_total_tokens"] is None for c in calls),
                "calls": calls,
            }
        )
    incident_path = workspace / "kernel_envolve-rsi/audit/rsi_pilot_incident_20261110.json"
    incident = read(incident_path)
    return {
        "scope": "All 19 closed campaign jobs, including failed and user-stopped studies; GPU use ended.",
        "entries": entries,
        "released_gpu_seconds_under_extended_ledger": running_total,
        "incident_separately_recorded_gpu_seconds": incident["conservative_resource_seconds"],
        "combined_historical_gpu_seconds": running_total
        + incident["conservative_resource_seconds"],
        "known_model_tokens_closed_jobs": sum(e["known_tokens"] for e in entries),
        "model_attempts_closed_jobs": sum(e["model_attempts"] for e in entries),
        "unknown_usage_attempts_closed_jobs": sum(e["unknown_usage_attempts"] for e in entries),
        "incident_model_attempts_separately_recorded": incident["model_calls"],
        "incident_model_usage": None,
        "incident_receipt_sha256": hash_file(incident_path),
        "production_roi_proven": False,
        "limitations": [
            "Missing usage is unknown, never a zero-cost call. Known tokens are a lower bound.",
            "Model pricing, monetary cost, human/agent development labor and CPU time are not priced here.",
            "All failed studies and infrastructure attempts remain; reused training is counted once.",
            "Conservative allocation seconds are not active GPU kernel time or provider billing.",
            "Future per-task savings would not by themselves recover campaign acquisition/validation costs.",
            "Admission was user-stopped with incomplete evaluator evidence; transport calls are not completed final evaluations.",
        ],
    }


if __name__ == "__main__":
    result = reconcile(Path(sys.argv[1]).resolve())
    Path(sys.argv[2]).write_text(json.dumps(result, indent=2))
    print(
        json.dumps(
            {k: v for k, v in result.items() if k not in {"entries", "limitations"}}, indent=2
        )
    )
