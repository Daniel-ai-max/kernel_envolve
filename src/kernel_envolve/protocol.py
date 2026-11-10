"""A bounded, reproducible schedule shared by execution and evidence checking."""

import hashlib
import json


def measured_cases(cases):
    return [case for case in cases if case.get("measure", True)]


def performance_role(case):
    """A guard is measured and validated but never contributes to the objective."""
    role = case.get("performance_role", "objective")
    if role not in ("objective", "guard"):
        raise ValueError("performance_role must be objective or guard")
    if "performance_role" in case and not case.get("measure", True):
        raise ValueError("Unmeasured cases cannot declare performance_role")
    if role == "guard" and "weight" in case:
        raise ValueError("Guard cases cannot have a weight")
    return role


def objective_cases(cases):
    return [case for case in measured_cases(cases) if performance_role(case) == "objective"]


def plan(cfg, mode, selection=None):
    selection = selection or {}
    if mode not in ("quick", "full"):
        raise ValueError("Unknown evaluation mode")
    if mode == "full" and selection:
        raise ValueError("Case/seed selection is diagnostic; use quick mode")
    wanted = selection.get("cases", [])
    known = {c["id"] for c in cfg["cases"]}
    if len(set(wanted)) != len(wanted) or set(wanted) - known:
        raise ValueError("Unknown or duplicate selected cases")
    cases = (
        [c for c in cfg["cases"] if c["id"] in wanted]
        if wanted
        else [c for c in cfg["cases"] if mode == "full" or c.get("quick", True)]
    )
    seeds = selection.get("seeds", cfg[f"{mode}_seeds"])
    if (
        not seeds
        or len(set(seeds)) != len(seeds)
        or any(type(s) is not int or s < 0 for s in seeds)
    ):
        raise ValueError("Seeds must be distinct nonnegative integers")
    return cases, seeds, cfg[f"{mode}_trials"], 2 if mode == "full" else 1


def roles(result):
    if result["signal_only"]:
        return ["candidate"]
    values = ["candidate", "comparator"]
    if result["mode"] == "full" and result.get("comparator", {}).get("kind") == "best":
        values.append("baseline")
    return values


def order(implementations, batch, trial):
    # ABBA for two implementations, rotating ABC CBA for three.
    start = (batch + trial) % len(implementations)
    front = implementations[start:] + implementations[:start]
    return front + front[::-1]


def timing_seed(case, batch, trial, repeat, seeds):
    key = json.dumps(["timing-v2", case, batch, trial, repeat, seeds]).encode()
    return int.from_bytes(hashlib.sha256(key).digest()[:8], "big") % (2**63)


def post_seed(case, seeds):
    return timing_seed(case, -1, -1, -1, seeds)
