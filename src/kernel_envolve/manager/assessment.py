"""Durable scoped evidence from an explicitly installed, pinned host verifier.

A research report cannot call this projection a verified result. The trusted
adapter validates its native artifacts and emits scoped-assessment-v1 on stdout.
The manager validates identity/shape, preserves its input and records provenance.
"""

import hashlib
import json
import math
import os
import re
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from ..storage import digest
from .archive import durable_bytes, durable_json, hash_file
from .process import exclusive
from .store import encode


def validate_assessment(value, task):
    def positive(v):
        return type(v) in (int, float) and math.isfinite(v) and v > 0

    def sha(v):
        return isinstance(v, str) and re.fullmatch(r"[a-f0-9]{64}", v)

    if not isinstance(value, dict) or value.get("schema") != "scoped-assessment-v1":
        raise ValueError("Expected scoped-assessment-v1")
    if (
        value.get("task") != task["id"]
        or value.get("baseline_revision") != task["spec"]["baseline"]["revision"]
        or value.get("promotion") is not False
        or value.get("status") != "verified"
        or not sha(value.get("source_sha256"))
        or not sha(value.get("evidence_sha256"))
        or not positive(value.get("recorded_at"))
        or value["recorded_at"] > time.time()
    ):
        raise ValueError("Assessment identity, time or qualification differs")
    metrics = value.get("metrics")
    if not isinstance(metrics, list) or not 1 <= len(metrics) <= 8:
        raise ValueError("Expected 1–8 scoped metrics")
    for m in metrics:
        if (
            not isinstance(m, dict)
            or not isinstance(m.get("label"), str)
            or not m["label"].strip()
            or not positive(m.get("baseline_speedup"))
            or type(m.get("cases")) is not int
            or m["cases"] <= 0
            or (
                m.get("slower_cases") is not None
                and (type(m["slower_cases"]) is not int or not 0 <= m["slower_cases"] <= m["cases"])
            )
            or (
                m.get("worst_slowdown_percent") is not None
                and (
                    type(m["worst_slowdown_percent"]) not in (int, float)
                    or not math.isfinite(m["worst_slowdown_percent"])
                )
            )
        ):
            raise ValueError("Invalid scoped metric")
    for field, limit in (("checkpoints", 100), ("scenarios", 2000)):
        values = value.get(field, [])
        if not isinstance(values, list) or len(values) > limit:
            raise ValueError(f"Invalid assessment {field}")
        for row in values:
            if not isinstance(row, dict):
                raise ValueError(f"Invalid assessment {field} row")
            if field == "checkpoints":
                valid = all(positive(row.get(k)) for k in ("minutes", "eager_ratio", "graph_ratio"))
            else:
                valid = (
                    all(isinstance(row.get(k), str) for k in ("name", "mode"))
                    and all(positive(row.get(k)) for k in ("baseline_us", "candidate_us"))
                    and positive(row["candidate_us"] / row["baseline_us"])
                    and row.get("group") in (None, "leaf", "forward_backward")
                )
            if not valid:
                raise ValueError(f"Invalid assessment {field} row")
    if len(encode(value).encode()) > 2 * 1024 * 1024:
        raise ValueError("Assessment summary exceeds 2 MiB")
    return value


def watch(store, key, receipt, verifier):
    task = store.task(key)
    verifier, receipt = Path(verifier).resolve(), Path(receipt).resolve()
    if not verifier.is_file() or verifier.is_relative_to(task["spec"]["workspace"]):
        raise ValueError("Install the trusted verifier outside the candidate workspace")
    source = {
        "receipt": str(receipt),
        "verifier": str(verifier),
        "verifier_sha256": hash_file(verifier),
    }
    with store.transaction():
        old = store.one("SELECT data FROM metadata WHERE key=?", ("assessment_source:" + key,))
        if old and json.loads(old["data"]) != source:
            raise ValueError(
                "Assessment source already pinned; use a new task for a different verifier"
            )
        store.db.execute(
            "INSERT OR IGNORE INTO metadata VALUES(?,?)",
            ("assessment_source:" + key, encode(source)),
        )


def verify(source, task):
    """CPU-only host adapter, bounded output and lifetime; never run by the HTTP reader."""
    verifier = Path(source["verifier"])
    if hash_file(verifier) != source["verifier_sha256"]:
        raise ValueError("Pinned assessment verifier changed")
    context = encode(
        {"task": task["id"], "baseline": task["spec"]["baseline"], "receipt": source["receipt"]}
    )
    with tempfile.TemporaryFile() as stdout, tempfile.TemporaryFile() as stderr:
        process = subprocess.Popen(
            [
                sys.executable,
                "-I",
                "-c",
                "import resource,runpy,sys; "
                "resource.setrlimit(resource.RLIMIT_FSIZE,(2097152,2097152)); "
                "runpy.run_path(sys.argv[1],run_name='__main__')",
                str(verifier),
            ],
            stdin=subprocess.PIPE,
            stdout=stdout,
            stderr=stderr,
            start_new_session=True,
            env={**os.environ, "CUDA_VISIBLE_DEVICES": ""},
        )
        try:
            process.communicate(context.encode(), timeout=30)
            if process.returncode:
                stderr.seek(0, os.SEEK_END)
                stderr.seek(max(0, stderr.tell() - 4096))
                raise ValueError(
                    "Assessment verifier failed: " + stderr.read().decode(errors="replace")
                )
            if stdout.tell() > 2 * 1024 * 1024:
                raise ValueError("Assessment verifier output exceeds 2 MiB")
            stdout.seek(0)
            value = json.loads(stdout.read())
        finally:
            # A verifier may not leave descendants behind, including after a timeout.
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait()
    return validate_assessment(value, task)


def refresh_sources(store, *, force=False):
    with exclusive(store.root / "assessment.lock") as lock:
        if lock is None:
            return False
        _refresh_sources(store, force=force)
    return True


def _refresh_sources(store, *, force=False):
    for row in store.rows("SELECT key,data FROM metadata WHERE key LIKE 'assessment_source:%'"):
        key = row["key"].removeprefix("assessment_source:")
        source = json.loads(row["data"])
        task = store.task(key)
        receipt = Path(source["receipt"])
        previous = store.one("SELECT data FROM metadata WHERE key=?", ("assessment_sync:" + key,))
        previous = json.loads(previous["data"]) if previous else {}
        if not receipt.exists() and not previous:
            continue  # The independently running assessor has not published its receipt yet.
        signature = None
        try:
            if receipt.stat().st_size > 4 * 1024 * 1024:
                raise ValueError("Assessment input receipt exceeds 4 MiB")
            signature = digest([hash_file(receipt), hash_file(source["verifier"])])
            if not force and previous.get("signature") == signature:
                continue  # Do not repeat verification or retry an unchanged failure.
            value = verify(source, task)
            raw = receipt.read_bytes()
            if (
                digest([hashlib.sha256(raw).hexdigest(), hash_file(source["verifier"])])
                != signature
            ):
                raise ValueError("Assessment input changed during verification")
            record = {
                "assessment": value,
                "input_sha256": hashlib.sha256(raw).hexdigest(),
                "verifier_sha256": source["verifier_sha256"],
                "indexed_at": time.time(),
            }
            record_id = digest(record | {"indexed_at": None})
            existing = store.one(
                "SELECT data FROM metadata WHERE key=?",
                ("assessment_history:" + key + ":" + record_id,),
            )
            if existing:
                record = json.loads(existing["data"])
            folder = store.root / "assessments" / record_id
            folder.mkdir(parents=True, exist_ok=True)
            durable_json(folder / "record.json", record)
            # The receipt and output survive removal of the evaluator workspace.
            durable_bytes(folder / "input.json", raw)
            with store.transaction():
                store.db.execute(
                    "INSERT OR IGNORE INTO metadata VALUES(?,?)",
                    ("assessment_history:" + key + ":" + record_id, encode(record)),
                )
                latest = store.one("SELECT data FROM metadata WHERE key=?", ("assessment:" + key,))
                if (
                    not latest
                    or json.loads(latest["data"])["assessment"]["recorded_at"]
                    < value["recorded_at"]
                ):
                    store.db.execute(
                        "INSERT OR REPLACE INTO metadata VALUES(?,?)",
                        ("assessment:" + key, encode(record)),
                    )
                state = {"signature": signature, "recorded_at": time.time(), "error": None}
                store.db.execute(
                    "INSERT OR REPLACE INTO metadata VALUES(?,?)",
                    ("assessment_sync:" + key, encode(state)),
                )
                store.event(
                    key,
                    "assessment_indexed",
                    {"record": record_id, "source": value["source_sha256"]},
                )
        except (OSError, ValueError, KeyError, TypeError, subprocess.SubprocessError) as exc:
            state = {"signature": signature, "recorded_at": time.time(), "error": str(exc)}
            if previous.get("error") == state["error"] and previous.get("signature") == signature:
                continue
            store.db.execute(
                "INSERT OR REPLACE INTO metadata VALUES(?,?)",
                ("assessment_sync:" + key, encode(state)),
            )
            store.event(key, "assessment_sync_failed", state)
