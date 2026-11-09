"""Small explicit source bundles. No working-directory imports in saved bests."""

import hashlib
import importlib
import importlib.abc
import importlib.machinery
import importlib.util
import json
import math
import os
import re
import shutil
import sys
import types
import uuid
from pathlib import Path

from . import __version__


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, value):
    path = Path(path)
    tmp = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    try:
        tmp.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def safe_file(root, name):
    relative = Path(name)
    if relative.is_absolute() or ".." in relative.parts or not relative.parts:
        raise ValueError(f"Expected a relative task file: {name}")
    if any(part.startswith(".") or part == "__pycache__" for part in relative.parts):
        raise ValueError(f"Hidden/generated files cannot be bundled: {name}")
    path = root / relative
    if any(p.is_symlink() for p in [path, *path.parents] if p != root.parent):
        # Resolve the root first; only task descendants should normally be symlinks.
        raise ValueError(f"Symlinks cannot be bundled: {name}")
    if not path.is_file():
        raise ValueError(f"Missing task file: {name}")
    return path


def config(root):
    cfg = read_json(root / "task.json")
    defaults = {
        "reference_files": ["task.py"],
        "candidate_files": ["candidate.py"],
        "device": "cpu",
        "rtol": 1e-5,
        "atol": 1e-6,
        "equal_nan": False,
        "mutable_inputs": [],
        "quick_seeds": [0],
        "full_seeds": [0, 1, 2],
        "quick_trials": 3,
        "full_trials": 10,
        "warmup": 3,
        "min_gain": 0.01,
        "noise_multiplier": 2.0,
        "max_regression": 0.05,
        "max_baseline_regression": 0.05,
        "deterministic": False,
        "timer": "auto",
    }
    unknown = set(cfg) - (set(defaults) | {"name", "objective", "cases"})
    if unknown:
        raise ValueError(f"Unknown task settings: {sorted(unknown)}")
    cfg = defaults | cfg
    for key in ("equal_nan", "deterministic"):
        if type(cfg[key]) is not bool:
            raise ValueError(f"{key} must be a boolean")
    if cfg["timer"] not in ("auto", "wall", "cuda_event"):
        raise ValueError("timer must be auto, wall or cuda_event")
    if cfg["device"] == "cpu" and cfg["timer"] == "cuda_event":
        raise ValueError("cuda_event requires a CUDA device")
    if not cfg.get("name") or not cfg.get("objective"):
        raise ValueError("Declare task name and objective (what the case weights represent)")
    cases = cfg.get("cases", [])
    ids = [c["id"] for c in cases]
    if not ids or len(set(ids)) != len(ids) or any(not isinstance(x, str) for x in ids):
        raise ValueError("Cases require unique string ids")
    for c in cases:
        if "quick" in c and type(c["quick"]) is not bool:
            raise ValueError("case.quick must be a boolean")
        if set(c) - {"id", "args", "weight", "quick", "max_us"}:
            raise ValueError(f"Unknown case settings: {c['id']}")
        for key in ("weight", "max_us"):
            if key in c and (
                type(c[key]) not in (int, float) or not math.isfinite(c[key]) or c[key] <= 0
            ):
                raise ValueError(f"{key} must be finite and positive")
        if "weight" not in c:
            raise ValueError("Explicit case weights are required")
    if not any(c.get("quick", True) for c in cases):
        raise ValueError("At least one quick case is required")
    for key in (
        "rtol",
        "atol",
        "min_gain",
        "noise_multiplier",
        "max_regression",
        "max_baseline_regression",
    ):
        if type(cfg[key]) not in (int, float) or not math.isfinite(cfg[key]) or cfg[key] < 0:
            raise ValueError(f"{key} must be finite and nonnegative")
    if any(cfg[k] >= 1 for k in ("min_gain", "max_regression", "max_baseline_regression")):
        raise ValueError("Gain and regression settings are fractions less than 1")
    for key in ("quick_trials", "full_trials", "warmup"):
        if type(cfg[key]) is not int or cfg[key] < 1:
            raise ValueError(f"{key} must be a positive integer")
    for key in ("quick_seeds", "full_seeds"):
        values = cfg[key]
        if (
            not values
            or len(set(values)) != len(values)
            or any(type(x) is not int or x < 0 for x in values)
        ):
            raise ValueError(f"{key} must contain distinct integer seeds")
    if cfg["full_trials"] < 3:
        raise ValueError("full_trials must be at least 3 for a noise estimate")
    if not set(cfg["quick_seeds"]).issubset(cfg["full_seeds"]):
        raise ValueError("quick_seeds must be a subset of full_seeds")
    if not isinstance(cfg["device"], str) or not re.fullmatch(
        r"cpu|cuda(?::[0-9]+)?", cfg["device"]
    ):
        raise ValueError("device must be cpu or cuda[:index]")
    if any(type(i) is not int or i < 0 for i in cfg["mutable_inputs"]):
        raise ValueError("mutable_inputs must contain nonnegative input indices")
    refs, candidates = cfg["reference_files"], cfg["candidate_files"]
    if "task.py" not in refs or "candidate.py" not in candidates:
        raise ValueError("Bundle task.py in reference_files and candidate.py in candidate_files")
    if set(refs) & set(candidates) or len(set(refs + candidates)) != len(refs + candidates):
        raise ValueError("Reference and candidate file lists must be distinct")
    for name in refs + candidates:
        safe_file(root, name)
    return cfg


def contract(root, cfg):
    settings = {k: v for k, v in cfg.items() if k != "candidate_files"}
    return digest(
        {"settings": settings, "files": {n: file_hash(root / n) for n in cfg["reference_files"]}}
    )


def bundle(root, destination, cfg):
    destination.mkdir()
    names = ["task.json", *cfg["reference_files"], *cfg["candidate_files"]]
    if sum(safe_file(root, n).stat().st_size for n in names) > 32 * 1024 * 1024:
        raise ValueError("Source bundle exceeds 32 MiB; use generated inputs for this version")
    for name in names:
        src = safe_file(root, name)
        dst = destination / name
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, dst)
    # Read settings from the snapshot as well, so identity and execution agree.
    snap_cfg = config(destination)
    return {
        "files": {n: file_hash(destination / n) for n in names},
        "contract": contract(destination, snap_cfg),
        "evaluator_version": __version__,
        "evaluator_hash": evaluator_hash(),
    }


def evaluator_hash():
    return digest({p.name: file_hash(p) for p in sorted(Path(__file__).parent.glob("*.py"))})


def candidate_identity(identity, cfg):
    return digest({name: identity["files"][name] for name in cfg["candidate_files"]})


def verify_source(run, identity):
    source = run / "source"
    actual = set()
    for p in source.rglob("*"):
        if "__pycache__" in p.relative_to(source).parts:
            continue
        if p.is_symlink():
            raise ValueError("Symlink in frozen source")
        if p.is_file():
            actual.add(p.relative_to(source).as_posix())
    if actual != set(identity["files"]):
        raise ValueError("Source inventory changed (added or missing files)")
    for name, expected in identity["files"].items():
        if file_hash(safe_file(run / "source", name)) != expected:
            raise ValueError(f"Source changed: {name}")
    if contract(run / "source", config(run / "source")) != identity["contract"]:
        raise ValueError("Task contract changed")


class SourceOnlyLoader(importlib.machinery.SourceFileLoader):
    def get_code(self, fullname):
        # Neither read nor write bytecode caches: timestamp/size validation is weaker than source hashes.
        return self.source_to_code(self.get_data(self.path), self.path, _optimize=0)


class SnapshotFinder(importlib.abc.MetaPathFinder):
    def __init__(self, prefix):
        self.prefix = prefix + "."

    def find_spec(self, fullname, path=None, target=None):
        if not fullname.startswith(self.prefix):
            return None
        spec = importlib.machinery.PathFinder.find_spec(fullname, path)
        if spec and isinstance(spec.loader, importlib.machinery.SourceFileLoader):
            spec.loader = SourceOnlyLoader(fullname, spec.origin)
        elif spec and isinstance(spec.loader, importlib.machinery.SourcelessFileLoader):
            raise ImportError("Bytecode-only snapshot modules are unsupported")
        return spec


def load_module(source, module):
    name = "_ke_" + uuid.uuid4().hex
    package = types.ModuleType(name)
    package.__path__ = [str(source)]
    sys.modules[name] = package
    sys.meta_path.insert(0, SnapshotFinder(name))
    return importlib.import_module(f"{name}.{module}")


def read_best(task_root):
    path = task_root / ".kernel_envolve" / "best.json"
    if not path.exists():
        return None
    pointer = read_json(path)
    run = (task_root / ".kernel_envolve" / pointer["run"]).resolve()
    if not run.is_relative_to((task_root / ".kernel_envolve" / "runs").resolve()):
        raise ValueError("Best pointer must reference a saved local run")
    result = read_json(run / "result.json")
    if digest(result) != pointer["result_hash"]:
        raise ValueError("Best result changed")
    verify_source(run, result["identity"])
    return pointer, run, result
