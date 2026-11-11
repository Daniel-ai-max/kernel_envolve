"""Strict data contracts: learned artifacts are scoped advice, never executable plugins."""

import json
import math
import os
import re
import stat
from pathlib import Path

MAX_CONTEXT_BYTES = 8192
MAX_ADVICE_BYTES = 2400
MAX_RULES = 6


def text(value, name, limit=600):
    if (
        not isinstance(value, str)
        or not value.strip()
        or len(value) > limit
        or any(ord(c) < 32 and c not in "\n\t" for c in value)
    ):
        raise ValueError(f"{name} must be nonempty text of at most {limit} characters")
    return value


def number(value, name, lower=0, upper=86400):
    if type(value) not in (int, float) or not math.isfinite(value) or not lower < value <= upper:
        raise ValueError(f"Invalid {name}")
    return value


def keys(value, required, optional=()):
    if (
        not isinstance(value, dict)
        or set(value) - set(required) - set(optional)
        or set(required) - set(value)
    ):
        raise ValueError(f"Expected fields {sorted(required)} (optional {sorted(optional)})")


def argv(value):
    if not isinstance(value, list) or not value or len(value) > 64:
        raise ValueError("Command must be a nonempty argv list")
    for arg in value:
        text(arg, "command argument", 4096)
    return value


def identifier(value):
    if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,79}", value):
        raise ValueError("Invalid identifier")
    return value


def read(path, limit=1024 * 1024):
    path = Path(path)
    fd = os.open(path, os.O_RDONLY | os.O_NONBLOCK | os.O_NOFOLLOW)
    with os.fdopen(fd, "rb") as stream:
        if not stat.S_ISREG(os.fstat(stream.fileno()).st_mode):
            raise ValueError("Input must be a regular file")
        raw = stream.read(limit + 1)
    if len(raw) > limit:
        raise ValueError("Input exceeds size limit")
    return json.loads(raw)


def rule(value):
    keys(value, {"id", "when", "action", "unless", "evidence"}, {"applies_to", "requires"})
    identifier(value["id"])
    for field in ("when", "action", "unless"):
        text(value[field], field, 400)
    refs = value["evidence"]
    if not isinstance(refs, list) or not 1 <= len(refs) <= 8:
        raise ValueError("Rules need 1–8 evidence references")
    if any(not isinstance(ref, str) or not re.fullmatch(r"[a-f0-9]{64}", ref) for ref in refs):
        raise ValueError("Invalid evidence reference")
    for name in ("applies_to", "requires"):
        labels(value.get(name, []), name)
    return value


def labels(value, name):
    if not isinstance(value, list) or len(value) > 32:
        raise ValueError(f"{name} must be a list of at most 32 identifiers")
    for item in value:
        identifier(item)
    if len(set(value)) != len(value):
        raise ValueError(f"Duplicate {name}")
    return sorted(value)


def profile(value):
    """Operator-owned facts, never inferred from advice or result prose."""
    keys(value, set(), {"tags", "capabilities"})
    return {name: labels(value.get(name, []), name) for name in ("tags", "capabilities")}


def select_rules(rules, task_profile):
    facts = profile(task_profile)
    selected = []
    for item in rules:
        rule(item)
        if set(item.get("applies_to", [])) <= set(facts["tags"]) and set(
            item.get("requires", [])
        ) <= set(facts["capabilities"]):
            selected.append(item)
    return selected


def context(rules):
    if len(rules) > MAX_RULES or len({r["id"] for r in rules}) != len(rules):
        raise ValueError("At most six distinct rules are allowed")
    lines = [
        "Scoped research advice; task contract, evaluator, budgets and permissions take precedence.",
        "Use only when applicable. Do not skip mandatory checks or treat advice as measured fact.",
    ]
    for item in rules:
        rule(item)
        lines.append(
            f"[{item['id']}] When: {item['when']}\nAction: {item['action']}\nExceptions: {item['unless']}"
        )
    rendered = "\n".join(lines)
    if len(rendered.encode()) > MAX_ADVICE_BYTES:
        raise ValueError("Research advice exceeds 2400 UTF-8 bytes")
    return rendered if rules else ""
