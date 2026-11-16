"""Source-bound research claims, never correctness qualification or promotion."""

import hashlib
import json
import math
import os
import re
import stat
import time


def validate(value, task):
    fields = {
        "schema",
        "status",
        "promotion",
        "task",
        "baseline_revision",
        "source_sha256",
        "evidence_sha256",
        "recorded_at",
        "scope",
        "summary",
        "limitations",
        "binding",
    }

    def sha(v):
        return isinstance(v, str) and re.fullmatch(r"[a-f0-9]{64}", v)

    def text(v, limit):
        return isinstance(v, str) and bool(v.strip()) and len(v) <= limit

    if not isinstance(value, dict) or set(value) != fields:
        raise ValueError("Expected bounded research-result-v1 fields; paths/URLs are not inputs")
    if (
        value["schema"] != "research-result-v1"
        or value["status"] != "research"
        or value["promotion"] is not False
        or value["task"] != task["id"]
        or value["baseline_revision"] != task["spec"]["baseline"]["revision"]
        or not all(sha(value[k]) for k in ("source_sha256", "evidence_sha256"))
        or type(value["recorded_at"]) not in (int, float)
        or not math.isfinite(value["recorded_at"])
        or not 0 < value["recorded_at"] <= time.time()
        or not text(value["scope"], 300)
        or not text(value["summary"], 2000)
        or not isinstance(value["limitations"], list)
        or len(value["limitations"]) > 16
        or not all(text(v, 500) for v in value["limitations"])
    ):
        raise ValueError("Research identity, time or claim shape differs")
    binding = value["binding"]
    if not isinstance(binding, dict):
        raise ValueError("Research source/receipt binding required")
    if binding.get("kind") == "manager_job":
        keys = {"kind", "job", "receipt_sha256", "log_sha256"}
        identity = binding.get("job")
    elif binding.get("kind") == "transport_archive":
        keys = {
            "kind",
            "request",
            "request_sha256",
            "archive_sha256",
            "receipt_sha256",
            "log_sha256",
        }
        identity = binding.get("request")
    else:
        raise ValueError("Unknown research binding")
    if (
        set(binding) != keys
        or not isinstance(identity, str)
        or not re.fullmatch(r"[a-f0-9]{32}", identity)
        or not all(sha(binding[k]) for k in keys if k.endswith("_sha256"))
    ):
        raise ValueError("Invalid research source/receipt binding")
    return value


def validate_job(store, task, value):
    """New manager jobs additionally check the supervisor's DB-pinned source receipt.

    Legacy transport archives are checked by the explicitly pinned host verifier;
    the framework never follows a path or URL from that verifier's output.
    """
    binding = value["binding"]
    if binding["kind"] != "manager_job":
        return
    job_id = binding["job"]
    job = store.one("SELECT * FROM jobs WHERE id=? AND task=? AND kind='gpu'", (job_id, task["id"]))
    row = store.one("SELECT data FROM metadata WHERE key=?", ("research_source:" + job_id,))
    if not job or not row or job["ended"] is None or job["status"] in ("queued", "running"):
        raise ValueError("Research result requires a completed, source-bound job")
    saved = json.loads(row["data"])
    if saved["sha256"] != binding["receipt_sha256"] or value["recorded_at"] != job["ended"]:
        raise ValueError("Research job receipt or experiment time differs")
    # Fixed components only; reject symlinks at every directory and at the leaf.
    fd = os.open(store.root, os.O_RDONLY | os.O_DIRECTORY)
    try:
        for name in ("jobs", job_id):
            child = os.open(name, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=fd)
            os.close(fd)
            fd = child
        leaf = os.open(
            "research-source.json", os.O_RDONLY | os.O_NONBLOCK | os.O_NOFOLLOW, dir_fd=fd
        )
        with os.fdopen(leaf, "rb") as stream:
            if not stat.S_ISREG(os.fstat(stream.fileno()).st_mode):
                raise ValueError("Research receipt must be a regular file")
            raw = stream.read(1024 * 1024 + 1)
    finally:
        os.close(fd)
    if len(raw) > 1024 * 1024 or hashlib.sha256(raw).hexdigest() != saved["sha256"]:
        raise ValueError("Bound research receipt changed")
    receipt = json.loads(raw)
    if (
        receipt["task"] != task["id"]
        or receipt["job"] != job_id
        or receipt["command"] != json.loads(job["command"])
        or receipt["status"] != job["status"]
        or receipt["exit_code"] != job["exit_code"]
        or not receipt["endpoints_complete"]
        or receipt["source_changed"] is not False
        or receipt["before"]["baseline_revision"] != value["baseline_revision"]
        or receipt["after"]["baseline_revision"] != value["baseline_revision"]
        or receipt["before"]["fingerprint"] != value["source_sha256"]
        or receipt["after"]["fingerprint"] != value["source_sha256"]
        or receipt["logs"]["stdout.log"]["sha256"] != binding["log_sha256"]
        or not receipt["logs"]["stdout.log"]["complete"]
    ):
        raise ValueError("Research source/log identity differs from bound job")
