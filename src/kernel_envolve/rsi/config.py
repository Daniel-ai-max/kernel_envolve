"""Operator-owned learning and validation budgets, task scopes and frozen suites."""

from pathlib import Path

from ..manager.store import Store as Manager
from .model import argv, identifier, keys, number, profile, read, text


def validate(raw, base):
    keys(
        raw,
        {"manager", "scopes", "learner_command"},
        {
            "learn_timeout_seconds",
            "learn_budget_seconds",
            "validation_budget_seconds",
            "min_new_evidence",
            "cooldown_seconds",
            "max_evidence",
            "suites",
            "auto_validate",
            "validation_device",
            "profiles",
        },
    )
    cfg = {
        "learn_timeout_seconds": 120,
        "learn_budget_seconds": 1200,
        "validation_budget_seconds": 3600,
        "min_new_evidence": 8,
        "cooldown_seconds": 1800,
        "max_evidence": 48,
        "suites": {},
        "auto_validate": False,
        "validation_device": "cpu",
        **raw,
    }
    cfg["manager"] = str((Path(base) / cfg["manager"]).resolve())
    argv(cfg["learner_command"])
    for name in (
        "learn_timeout_seconds",
        "learn_budget_seconds",
        "validation_budget_seconds",
        "cooldown_seconds",
    ):
        number(cfg[name], name, upper=365 * 86400)
    for name in ("min_new_evidence", "max_evidence"):
        if type(cfg[name]) is not int or not 1 <= cfg[name] <= 128:
            raise ValueError(f"{name} must be 1–128")
    if cfg["min_new_evidence"] > cfg["max_evidence"]:
        raise ValueError("min_new_evidence exceeds evidence window")
    if type(cfg["auto_validate"]) is not bool:
        raise ValueError("auto_validate must be boolean")
    device = cfg["validation_device"]
    if not isinstance(device, str) or (device != "cpu" and not device.startswith("GPU-")):
        raise ValueError("validation_device must be cpu or a dedicated physical GPU UUID")
    if not isinstance(cfg["scopes"], dict) or not 1 <= len(cfg["scopes"]) <= 32:
        raise ValueError("Declare 1–32 scopes")
    assigned = set()
    manager = Manager(cfg["manager"], readonly=True)
    try:
        for scope, tasks in cfg["scopes"].items():
            identifier(scope)
            if not isinstance(tasks, list) or not 1 <= len(tasks) <= 128:
                raise ValueError("Scopes need 1–128 explicit task IDs")
            for task in tasks:
                identifier(task)
                manager.task(task)
                if task in assigned:
                    raise ValueError("Task assigned to more than one scope")
                assigned.add(task)
        if device != "cpu":
            for row in manager.rows("SELECT id FROM resources"):
                if manager.resource(row["id"])["spec"]["device"] == device:
                    raise ValueError(
                        "RSI validation GPU must be separate from production resources"
                    )
        production_paths = [manager.task(t)["spec"]["workspace"] for t in assigned]
    finally:
        manager.close()
    profiles = cfg.get("profiles", {})
    if not isinstance(profiles, dict) or set(profiles) - assigned:
        raise ValueError("Task profiles must name registered scope tasks")
    cfg["profiles"] = {task: profile(value) for task, value in profiles.items()}
    if not isinstance(cfg["suites"], dict) or set(cfg["suites"]) - set(cfg["scopes"]):
        raise ValueError("Unknown validation scope")
    if cfg["auto_validate"] and set(cfg["suites"]) != set(cfg["scopes"]):
        raise ValueError("Automatic validation needs a suite for each scope")
    for scope, suite in cfg["suites"].items():
        keys(
            suite,
            {"command", "model", "tasks"},
            {
                "repeats",
                "solver_seconds",
                "evaluation_seconds",
                "max_seconds",
                "min_quality_gain",
                "max_quality_regression",
                "min_cost_gain",
                "max_cost_regression",
                "adapter_files",
            },
        )
        suite = {
            "repeats": 2,
            "solver_seconds": 120,
            "evaluation_seconds": 300,
            "max_seconds": 1800,
            "min_quality_gain": 0.01,
            "max_quality_regression": 0.02,
            "min_cost_gain": 0.1,
            "max_cost_regression": 0.05,
            **suite,
        }
        argv(suite["command"])
        dependencies = suite.get("adapter_files", [])
        if not isinstance(dependencies, list) or len(dependencies) > 128:
            raise ValueError("adapter_files must list at most 128 local files")
        resolved = []
        for value in dependencies:
            text(value, "adapter file")
            path = (Path(base) / value).resolve()
            if not path.is_file():
                raise ValueError("Adapter dependency must be an existing file")
            resolved.append(str(path))
        suite["adapter_files"] = sorted(set(resolved))
        text(suite["model"], "fixed model identity")
        if type(suite["repeats"]) is not int or not 2 <= suite["repeats"] <= 10:
            raise ValueError("Validation requires 2–10 matched repeats")
        for key in ("solver_seconds", "evaluation_seconds", "max_seconds"):
            number(suite[key], key)
        for key in (
            "min_quality_gain",
            "max_quality_regression",
            "min_cost_gain",
            "max_cost_regression",
        ):
            number(suite[key], key, lower=-1e-12, upper=0.5)
            if suite[key] < 0:
                raise ValueError("Negative gate threshold")
        if not isinstance(suite["tasks"], list) or not 2 <= len(suite["tasks"]) <= 16:
            raise ValueError("A suite needs 2–16 distinct held-out task templates")
        paths = []
        for item in suite["tasks"]:
            keys(item, {"path", "kind"}, {"editable", "profile"})
            item["profile"] = profile(item.get("profile", {}))
            path = (Path(base) / item["path"]).resolve()
            if not path.is_dir() or item["kind"] not in ("generic", "native"):
                raise ValueError("Invalid validation task")
            if any(
                path.is_relative_to(p) or Path(p).is_relative_to(path) for p in production_paths
            ):
                raise ValueError("Validation templates must be outside production workspaces")
            if item["kind"] == "native":
                if not isinstance(item.get("editable"), list) or not item["editable"]:
                    raise ValueError(
                        "Native templates need explicit editable candidate/root/file paths"
                    )
            for name in item.get("editable", []):
                relative = Path(name)
                if (
                    relative.is_absolute()
                    or ".." in relative.parts
                    or not name.startswith("candidate/")
                ):
                    raise ValueError("Editable native files must be under candidate/")
            item["path"] = str(path)
            paths.append(str(path))
        if len(set(paths)) != len(paths):
            raise ValueError("Duplicate validation templates")
        cfg["suites"][scope] = suite
    return cfg


def initialize(root, path):
    from ..storage import digest, write_json
    from .trials import freeze_template

    root, path = Path(root).resolve(), Path(path).resolve()
    cfg = validate(read(path), path.parent)
    if root.exists():
        raise ValueError("RSI state already exists; do not overwrite an existing experiment")
    manager = Manager(cfg["manager"], readonly=True)
    try:
        for tasks in cfg["scopes"].values():
            for task in tasks:
                workspace = Path(manager.task(task)["spec"]["workspace"])
                if root.is_relative_to(workspace) or workspace.is_relative_to(root):
                    raise ValueError("RSI state and task workspaces must be separate")
        owner = Path(cfg["manager"]) / "rsi-owner.json"
        if owner.exists():
            raise ValueError("Manager already has an RSI publisher")
        root.mkdir(parents=True, mode=0o700)
        for scope, suite in cfg["suites"].items():
            for i, task in enumerate(suite["tasks"]):
                original = task["path"]
                target = root / "templates" / scope / str(i)
                spec = freeze_template(task, target)
                frozen = Path(spec["path"]) / "task"
                device_cfg = read(
                    frozen / ("task.json" if task["kind"] == "generic" else "native.json")
                )
                if (device_cfg.get("device", "cpu") == "cpu") != (
                    cfg["validation_device"] == "cpu"
                ):
                    raise ValueError("Validation device does not match the frozen task")
                task.update(spec)
                task["origin"] = original
        write_json(root / "config.json", cfg)
        with owner.open("x") as stream:
            import json

            json.dump({"state": str(root), "config_hash": digest(cfg)}, stream)
    finally:
        manager.close()
    return cfg
