"""Native task contracts and isolated source import environments."""

import math
import os
from pathlib import Path

from .protocol import plan
from .storage import read_json


def load_config(task):
    raw = read_json(task / "native.json")
    defaults = {
        "device": "cuda:0",
        "quick_seeds": [0],
        "full_seeds": [0, 17, 101],
        "quick_trials": 3,
        "full_trials": 6,
        "warmup": 3,
        "iterations": 5,
        "rtol": 1e-5,
        "atol": 1e-6,
        "equal_nan": False,
        "bitwise": False,
        "deterministic": True,
        "min_gain": 0.01,
        "noise_multiplier": 2.0,
        "max_regression": 0.05,
        "max_baseline_regression": 0.05,
        "check_output_layout": False,
        "threads": 1,
        "env": {},
        "artifacts": {},
        "checks": [],
        "required_scopes": ["operator", "numerical", "integration"],
        "required_outputs": ["output", "gradients"],
    }
    unknown = set(raw) - (set(defaults) | {"name", "objective", "sources", "imports", "cases"})
    if unknown:
        raise ValueError(f"Unknown native settings: {sorted(unknown)}")
    cfg = defaults | raw
    if not cfg["required_outputs"] or any(
        not isinstance(p, str) or not p for p in cfg["required_outputs"]
    ):
        raise ValueError("Declare required output/gradient paths")
    if not cfg.get("name") or not cfg.get("objective") or not cfg.get("imports"):
        raise ValueError("Declare name, objective, and production package imports")
    if not (task / "adapter.py").is_file():
        raise ValueError("Missing adapter.py")
    if set(cfg["sources"]) != {"candidate", "baseline"}:
        raise ValueError("Provide candidate and fixed baseline sources")
    if (
        not cfg["sources"]["candidate"]
        or cfg["sources"]["candidate"].keys() != cfg["sources"]["baseline"].keys()
    ):
        raise ValueError("Source roles must have the same nonempty named roots")
    for roots in cfg["sources"].values():
        for name, path in roots.items():
            if not name.isidentifier():
                raise ValueError("Source root names must be identifiers")
            expanded = os.path.expandvars(path)
            if "$" in expanded:
                raise ValueError(f"Unresolved source environment variable: {path}")
            roots[name] = str((task / expanded).resolve())
    if set(cfg["imports"].values()) - cfg["sources"]["candidate"].keys():
        raise ValueError("Unknown import root")
    if cfg["device"] != "cpu" and not (
        cfg["device"].startswith("cuda:") and cfg["device"][5:].isdigit()
    ):
        raise ValueError("device must be cpu or cuda:N")
    for key in ("quick_trials", "full_trials", "warmup", "iterations", "threads"):
        if type(cfg[key]) is not int or cfg[key] < 1:
            raise ValueError(f"Invalid {key}")
    if cfg["full_trials"] < 3:
        raise ValueError("full_trials >= 3 required for noise estimation")
    for key in (
        "rtol",
        "atol",
        "min_gain",
        "max_regression",
        "max_baseline_regression",
        "noise_multiplier",
    ):
        if type(cfg[key]) not in (float, int) or not math.isfinite(cfg[key]) or cfg[key] < 0:
            raise ValueError(f"Invalid {key}")
    for key in ("min_gain", "max_regression", "max_baseline_regression"):
        if cfg[key] >= 1:
            raise ValueError(f"Invalid fraction {key}")
    for key in ("deterministic", "equal_nan", "bitwise", "check_output_layout"):
        if type(cfg[key]) is not bool:
            raise ValueError(f"Invalid boolean {key}")
    ids = [c["id"] for c in cfg["cases"]]
    if not ids or any(not isinstance(s, str) or not s for s in ids) or len(set(ids)) != len(ids):
        raise ValueError("Unique nonempty string case ids required")
    for case in cfg["cases"]:
        if set(case) - {"id", "args", "weight", "quick", "measure", "max_us", "max_peak_bytes"}:
            raise ValueError("Unknown case field")
        for flag in ("quick", "measure"):
            if flag in case and type(case[flag]) is not bool:
                raise ValueError(f"Invalid case flag: {flag}")
        if not case.get("measure", True) and {"weight", "max_us", "max_peak_bytes"} & case.keys():
            raise ValueError("Unmeasured cases cannot have weights or performance caps")
        if "max_us" in case and (
            type(case["max_us"]) not in (int, float)
            or not math.isfinite(case["max_us"])
            or case["max_us"] <= 0
        ):
            raise ValueError("max_us must be finite and positive")
        if "max_peak_bytes" in case and cfg["device"] == "cpu":
            raise ValueError("CUDA peak-memory limits require CUDA")
        if case.get("measure", True):
            if (
                type(case.get("weight")) not in (float, int)
                or not math.isfinite(case["weight"])
                or case["weight"] <= 0
            ):
                raise ValueError("Explicit positive case weights required")
        if "max_peak_bytes" in case and (
            type(case["max_peak_bytes"]) is not int or case["max_peak_bytes"] < 1
        ):
            raise ValueError("max_peak_bytes must be positive")
    for mode in ("quick", "full"):
        selected, _, _, _ = plan(cfg, mode)
        if not any(c.get("measure", True) for c in selected):
            raise ValueError("Each mode needs measured cases")
    if not set(cfg["quick_seeds"]).issubset(cfg["full_seeds"]):
        raise ValueError("quick seeds must be included in full seeds")
    if not cfg["required_scopes"] or set(cfg["required_scopes"]) - {
        "operator",
        "numerical",
        "integration",
    }:
        raise ValueError("Declare required native validation scopes")
    if set(cfg["env"]) & {"PYTHONPATH", "PYTHONOPTIMIZE", "PYTHONHOME"}:
        raise ValueError("Python import/optimization overrides are not allowed")
    check_ids = [c["id"] for c in cfg["checks"]]
    if len(check_ids) != len(set(check_ids)) or any(not s.isidentifier() for s in check_ids):
        raise ValueError("Native checks need unique identifier names")
    for check in cfg["checks"]:
        if check.get("scope") not in {"operator", "numerical", "integration"}:
            raise ValueError("Unknown check scope")
        if type(check.get("min_tests", 1)) is not int or check.get("min_tests", 1) < 1:
            raise ValueError("min_tests must be positive")
        if "devices" in check and (
            not check["devices"]
            or len(set(check["devices"])) != len(check["devices"])
            or any(type(d) is not int or d < 0 for d in check["devices"])
        ):
            raise ValueError("Check devices must be unique nonnegative visible indices")
        if check["root"] not in cfg["sources"]["candidate"]:
            raise ValueError("Unknown check root")
        if check["kind"] == "gate":
            if not check.get("operators") or len(set(check["operators"])) != len(
                check["operators"]
            ):
                raise ValueError("Declare exact unique gate operators")
            if check.get("scope") != "numerical":
                raise ValueError("Gate scope must be numerical")
        elif check["kind"] == "pytest":
            if not check.get("required_tests") or not check.get("reports") or not check.get("argv"):
                raise ValueError("pytest requires argv, exact report paths and required_tests")
            for report in check["reports"]:
                if Path(report).is_absolute() or ".." in Path(report).parts:
                    raise ValueError("JUnit paths must be relative to the fresh reports directory")
        else:
            raise ValueError("Native check kind must be gate or pytest")
    return cfg


def environment(cfg, roots):
    env = os.environ.copy()
    env.pop("PYTHONOPTIMIZE", None)
    if env.get("KERNEL_ENVOLVE_GPU_JOB"):
        for key, value in cfg["env"].items():
            if (
                key == "CUDA_VISIBLE_DEVICES" or key.startswith("KERNEL_ENVOLVE_")
            ) and value != env.get(key):
                raise ValueError(f"Managed GPU allocation cannot be overridden by task env: {key}")
    env.update(cfg["env"])
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["PYTHONPATH"] = os.pathsep.join([str(Path(__file__).resolve().parents[1]), *roots.values()])
    return env
