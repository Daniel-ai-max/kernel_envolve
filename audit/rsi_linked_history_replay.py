"""Offline losslessness/size check on real decision packets; no efficacy claim."""

import argparse
import hashlib
import json
import statistics
import time
from pathlib import Path

from kernel_envolve.rsi.research import linked_history


def restore(linked, original):
    result = json.loads(json.dumps(linked))
    assert result.pop("history_representation") == "candidate-observation-refs-v1"
    assert result["instructions"].startswith(original["instructions"])
    result["instructions"] = original["instructions"]
    for step in result["history"]:
        ref = step.pop("observation_ref", None)
        if ref is None:
            continue
        candidates = [c for c in result["candidates"] if c["id"] == ref["candidate"]]
        assert len(candidates) == 1
        observations = [
            o
            for o in candidates[0]["observations"]
            if o["run"] == ref["run"] and o["mode"] == ref["mode"]
        ]
        assert len(observations) == 1
        step["observation"] = {
            "candidate": ref["candidate"],
            "mode": ref["mode"],
            "status": ref["status"],
        }
        if ref["status"] == "evaluated":
            step["observation"].update(observations[0])
        else:
            assert ref["status"] == "duplicate_skipped"
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("calls", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    rows = []
    for file in sorted(args.calls.glob("*.json")):
        raw = file.read_bytes()
        call = json.loads(raw)
        packet = json.loads(call["request"]["input"].split("\n", 1)[1])
        before = json.dumps(packet)
        began = time.perf_counter()
        linked = linked_history(packet)
        seconds = time.perf_counter() - began
        assert restore(linked, packet) == packet
        assert json.dumps(packet) == before, "Source packet mutated"
        rows.append(
            {
                "call": file.name,
                "call_sha256": hashlib.sha256(raw).hexdigest(),
                "arm": call["trial_id"].split("-", 2)[-1],
                "call_status": call["status"],
                "original_packet_bytes": len(before.encode()),
                "linked_packet_bytes": len(json.dumps(linked).encode()),
                "view_seconds": seconds,
            }
        )
    assert rows, "No packets"
    groups = {}
    for arm in sorted({r["arm"] for r in rows}):
        group = [r for r in rows if r["arm"] == arm]
        old = sum(r["original_packet_bytes"] for r in group)
        new = sum(r["linked_packet_bytes"] for r in group)
        groups[arm] = {
            "packets": len(group),
            "original_packet_bytes": old,
            "linked_packet_bytes": new,
            "byte_reduction": 1 - new / old,
        }
    from kernel_envolve.rsi import research

    result = {
        "scope": "Offline byte-level roundtrip and size only; no model/GPU calls or measured inference/quality benefit. Raw traces and failed requests retained.",
        "research_source_sha256": hashlib.sha256(Path(research.__file__).read_bytes()).hexdigest(),
        "verified_packets": len(rows),
        "view_seconds_median": statistics.median(r["view_seconds"] for r in rows),
        "view_seconds_max": max(r["view_seconds"] for r in rows),
        "groups": groups,
        "rows": rows,
    }
    args.output.write_text(json.dumps(result, indent=2))
    print(json.dumps({k: v for k, v in result.items() if k != "rows"}, indent=2))


if __name__ == "__main__":
    main()
