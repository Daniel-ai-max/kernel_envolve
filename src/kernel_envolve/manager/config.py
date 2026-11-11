"""Validate a deployment before any task or subprocess is created."""

import datetime as dt
import math
import re
from pathlib import Path

from ..storage import digest


def positive(value, name, nullable=False):
    if nullable and value is None:
        return None
    if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be finite and positive")
    return value


def identifier(value):
    if not isinstance(value, str) or not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9_.-]{0,79}", value):
        raise ValueError("IDs must be 1–80 letters, digits, dots, underscores or hyphens")
    return value


def date(value, name):
    if value is not None:
        try:
            dt.date.fromisoformat(value)
        except (ValueError, TypeError) as exc:
            raise ValueError(f"{name} must be YYYY-MM-DD or null") from exc
    return value


def resource_devices(resource):
    """Declared rank order; lock order is independently sorted by physical UUID."""
    return resource["devices"] if "devices" in resource else [resource["device"]]


def validate(raw, base):
    if set(raw) - {"version", "resources", "tasks"} or raw.get("version") != 1:
        raise ValueError("Expected version 1 deployment with resources and tasks")
    resources, tasks = [], []
    for item in raw["resources"]:
        if set(item) - {"id", "device", "devices", "sessions"}:
            raise ValueError("Unknown resource setting")
        resource = {"sessions": 2, **item}
        identifier(resource["id"])
        if ("device" in resource) == ("devices" in resource):
            raise ValueError("Declare exactly one of device or devices")
        devices = resource_devices(resource)
        if (
            not isinstance(devices, list)
            or not devices
            or any(
                not isinstance(device, str) or not re.fullmatch(r"GPU-[a-fA-F0-9-]+", device)
                for device in devices
            )
        ) and resource.get("device") != "cpu":
            raise ValueError("Use physical NVIDIA GPU UUIDs, not indices or MIG slices")
        if len(set(devices)) != len(devices):
            raise ValueError("Physical devices must be unique within a resource")
        if type(resource["sessions"]) is not int or not 1 <= resource["sessions"] <= 2:
            raise ValueError("Each resource supports one or two agent sessions")
        resources.append(resource)
    ids = [r["id"] for r in resources]
    devices = [device for resource in resources for device in resource_devices(resource)]
    if len(set(ids)) != len(ids) or len(set(devices)) != len(devices):
        raise ValueError("Resource IDs and physical devices must be unique")
    for item in raw["tasks"]:
        allowed = {
            "id",
            "name",
            "workspace",
            "task_dir",
            "kind",
            "resource",
            "command",
            "baseline",
            "round_timeout_seconds",
            "wall_budget_seconds",
            "gpu_budget_seconds",
            "cooldown_seconds",
            "host_driven",
        }
        if set(item) - allowed:
            raise ValueError("Unknown task setting")
        task = {
            "round_timeout_seconds": 1800,
            "wall_budget_seconds": None,
            "gpu_budget_seconds": None,
            "cooldown_seconds": 10,
            "host_driven": False,
            **item,
        }
        if type(task["host_driven"]) is not bool:
            raise ValueError("host_driven must be boolean")
        identifier(task["id"])
        if not isinstance(task["name"], str) or not task["name"].strip():
            raise ValueError("Task name is required")
        if task["resource"] not in ids or task["kind"] not in ("generic", "native"):
            raise ValueError("Unknown resource or evaluator kind")
        for key in ("workspace", "task_dir"):
            path = (Path(base) / task[key]).resolve()
            if not path.is_dir():
                raise ValueError(f"{key} does not exist: {path}")
            task[key] = str(path)
        if not Path(task["task_dir"]).is_relative_to(task["workspace"]):
            raise ValueError("Task directory must be inside its dedicated workspace")
        command = task["command"]
        if (
            not isinstance(command, list)
            or not command
            or any(not isinstance(arg, str) or not arg or "\0" in arg for arg in command)
        ):
            raise ValueError("command must be a nonempty argv array, never a shell string")
        for key in ("round_timeout_seconds", "cooldown_seconds"):
            positive(task[key], key)
        for key in ("wall_budget_seconds", "gpu_budget_seconds"):
            positive(task[key], key, nullable=True)
        baseline = task["baseline"]
        if set(baseline) != {
            "repository",
            "revision",
            "entrypoint",
            "introduced_on",
            "introduction_url",
            "frozen_on",
            "hardware",
            "contract",
        }:
            raise ValueError("Baseline requires provenance, freeze date, hardware and contract")
        for key in ("repository", "revision", "entrypoint", "hardware", "contract"):
            if not isinstance(baseline[key], str) or not baseline[key].strip():
                raise ValueError(f"baseline.{key} is required")
        date(baseline["introduced_on"], "introduced_on")
        date(baseline["frozen_on"], "frozen_on")
        if baseline["frozen_on"] is None:
            raise ValueError("Baseline freeze date is required")
        url = baseline["introduction_url"]
        if url is not None and (not isinstance(url, str) or not url.startswith("https://")):
            raise ValueError("Introduction evidence must be an HTTPS URL or null")
        if baseline["introduced_on"] and not url:
            raise ValueError("An introduction date needs a source URL; otherwise use null")
        task["identity"] = digest(
            {
                k: task[k]
                for k in (
                    "workspace",
                    "task_dir",
                    "kind",
                    "baseline",
                )
            }
        )
        tasks.append(task)
    if len({t["id"] for t in tasks}) != len(tasks):
        raise ValueError("Duplicate task ID")
    for i, task in enumerate(tasks):
        for other in tasks[:i]:
            a, b = Path(task["workspace"]), Path(other["workspace"])
            if a.is_relative_to(b) or b.is_relative_to(a):
                raise ValueError("Tasks require separate, non-overlapping workspaces")
    return {"version": 1, "resources": resources, "tasks": tasks}
