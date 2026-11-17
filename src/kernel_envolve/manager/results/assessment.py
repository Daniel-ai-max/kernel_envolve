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

from kernel_envolve._io import hash_file
from kernel_envolve.evaluation.shared.storage import digest
from kernel_envolve.manager.execution.process import exclusive
from kernel_envolve.manager.results.archive import durable_bytes, durable_json
from kernel_envolve.manager.store import encode


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


def validate_validation(value, task):
    """Validate a trusted verifier's external outcome, never promote or qualify.

    The pinned host verifier must check the external terminal job, original
    receipt/log hashes and immutable source manifest. These identifiers are
    provenance only: no external path, job or executable is followed here.
    ``passed`` means only that every declared check in ``scope`` passed.
    """
    fields = {
        "schema",
        "task",
        "baseline_revision",
        "source_sha256",
        "evidence_sha256",
        "recorded_at",
        "promotion",
        "status",
        "scope",
        "summary",
        "limitations",
        "external",
        "checks",
    }
    statuses = {"passed", "no_benefit", "failed", "interrupted", "inconclusive"}

    def text(v, limit):
        return isinstance(v, str) and bool(v.strip()) and len(v) <= limit

    def sha(v):
        return isinstance(v, str) and re.fullmatch(r"[a-f0-9]{64}", v)

    if (
        not isinstance(value, dict)
        or set(value) != fields
        or value["schema"] != "validation-result-v1"
        or value["task"] != task["id"]
        or value["baseline_revision"] != task["spec"]["baseline"]["revision"]
        or value["promotion"] is not False
        or value["status"] not in statuses
        or not all(sha(value[k]) for k in ("source_sha256", "evidence_sha256"))
        or type(value["recorded_at"]) not in (float, int)
        or not math.isfinite(value["recorded_at"])
        or not 0 < value["recorded_at"] <= time.time()
        or not text(value["scope"], 300)
        or not text(value["summary"], 2000)
        or not isinstance(value["limitations"], list)
        or len(value["limitations"]) > 16
        or not all(text(v, 500) for v in value["limitations"])
    ):
        raise ValueError("Invalid external validation identity or outcome")
    external = value["external"]
    if (
        not isinstance(external, dict)
        or set(external)
        != {
            "task",
            "job",
            "receipt_sha256",
            "source_sha256",
            "baseline_revision",
            "evidence_sha256",
        }
        or not all(
            isinstance(external[k], str) and re.fullmatch(r"[A-Za-z0-9_.:-]{1,160}", external[k])
            for k in ("task", "job")
        )
        or not sha(external["receipt_sha256"])
        or any(
            external[k] != value[k]
            for k in ("baseline_revision", "source_sha256", "evidence_sha256")
        )
    ):
        raise ValueError("External validation source/receipt binding differs")
    checks = value["checks"]
    if (
        not isinstance(checks, list)
        or not 1 <= len(checks) <= 1000
        or any(
            not isinstance(c, dict)
            or set(c) != {"name", "status"}
            or not text(c["name"], 300)
            or c["status"] not in statuses | {"skipped"}
            for c in checks
        )
        or len({c["name"] for c in checks}) != len(checks)
        or (value["status"] == "passed" and any(c["status"] != "passed" for c in checks))
    ):
        raise ValueError("Invalid external validation checks")
    if len(encode(value).encode()) > 2 * 1024 * 1024:
        raise ValueError("Validation summary exceeds 2 MiB")
    return value


def watch(store, key, receipt, verifier, *, research=False, validation=False):
    if research and validation:
        raise ValueError("Choose research or validation, not both")
    task = store.task(key)
    verifier, receipt = Path(verifier).resolve(), Path(receipt).resolve()
    if not verifier.is_file() or verifier.is_relative_to(task["spec"]["workspace"]):
        raise ValueError("Install the trusted verifier outside the candidate workspace")
    source = {
        "receipt": str(receipt),
        "verifier": str(verifier),
        "verifier_sha256": hash_file(verifier),
    }
    namespace = "validation" if validation else "research_result" if research else "assessment"
    if validation:
        source["validation_only"] = True
    if research:
        source["research_only"] = True
    with store.transaction():
        old = store.one("SELECT data FROM metadata WHERE key=?", (namespace + "_source:" + key,))
        if old and json.loads(old["data"]) != source:
            raise ValueError(
                "Assessment source already pinned; use a new task for a different verifier"
            )
        store.db.execute(
            "INSERT OR IGNORE INTO metadata VALUES(?,?)",
            (namespace + "_source:" + key, encode(source)),
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
    if source.get("validation_only"):
        return validate_validation(value, task)
    if source.get("research_only"):
        from kernel_envolve.manager.results.research_result import validate

        return validate(value, task)
    return validate_assessment(value, task)


def refresh_sources(store, *, force=False):
    with exclusive(store.root / "assessment.lock") as lock:
        if lock is None:
            return False
        _refresh_sources(store, force=force)
    return True


def _refresh_sources(store, *, force=False):
    from kernel_envolve.manager.results.validation_import import sources, sync_health

    installed = store.rows(
        "SELECT key,data FROM metadata WHERE key LIKE 'assessment_source:%' "
        "OR key LIKE 'research_result_source:%' OR key LIKE 'validation_source:%'"
    )
    for row in [*installed, *sources(store)]:
        namespace = row["key"].split("_source:", 1)[0]
        key = row["key"].split("_source:", 1)[1]
        research = namespace == "research_result"
        field = namespace
        directory = (
            "validations"
            if namespace == "validation"
            else "research-results"
            if research
            else "assessments"
        )
        source = json.loads(row["data"])
        external = source.get("kind") == "external_validation"
        task = store.task(key)
        receipt = None if external else Path(source["receipt"])
        sync_key = row["sync_key"] if "sync_key" in row.keys() else namespace + "_sync:" + key
        if namespace == "validation" and not external:
            sync_key = "validation_custom_sync:" + key
        previous = store.one("SELECT data FROM metadata WHERE key=?", (sync_key,))
        previous = json.loads(previous["data"]) if previous else {}
        if not external and not receipt.exists() and not previous:
            continue  # The independently running assessor has not published its receipt yet.
        signature = None
        try:
            if external:
                from kernel_envolve.manager.results import validation_import

                if namespace != "validation":
                    raise ValueError("External validation requires its own namespace")
                signature = validation_import.signature(source)
            elif receipt.stat().st_size > 4 * 1024 * 1024:
                raise ValueError("Assessment input receipt exceeds 4 MiB")
            else:
                signature = digest([hash_file(receipt), hash_file(source["verifier"])])
            if not force and previous.get("signature") == signature:
                continue  # Do not repeat verification or retry an unchanged failure.
            proof_files = None
            if external:
                value, proof_files = validation_import.verify_bundle(source, task)
                validate_validation(value, task)
            else:
                value = verify(source, task)
            if research:
                from kernel_envolve.manager.results.research_result import validate_job

                validate_job(store, task, value)
            raw = proof_files["validation.json"] if external else receipt.read_bytes()
            after_signature = (
                validation_import.signature(source)
                if external
                else digest([hashlib.sha256(raw).hexdigest(), hash_file(source["verifier"])])
            )
            if len(raw) > 4 * 1024 * 1024 or after_signature != signature:
                raise ValueError("Assessment input changed during verification")
            record = {
                field: value,
                "input_sha256": hashlib.sha256(raw).hexdigest(),
                "verifier_sha256": (
                    digest(["builtin-external-validation-v1"])
                    if external
                    else source["verifier_sha256"]
                ),
                "indexed_at": time.time(),
            }
            if proof_files is not None:
                record["proof_files"] = {
                    name: {"sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)}
                    for name, data in proof_files.items()
                }
            record_id = digest(record | {"indexed_at": None})
            existing = store.one(
                "SELECT data FROM metadata WHERE key=?",
                (namespace + "_history:" + key + ":" + record_id,),
            )
            if existing:
                record = json.loads(existing["data"])
            folder = store.root / directory / record_id
            folder.mkdir(parents=True, exist_ok=True)
            durable_json(folder / "record.json", record)
            # The receipt and output survive removal of the evaluator workspace.
            durable_bytes(folder / "input.json", raw)
            for name, data in (proof_files or {}).items():
                (folder / "proof" / name).parent.mkdir(parents=True, exist_ok=True)
                durable_bytes(folder / "proof" / name, data)
            if external:
                validation_import.verify_saved(folder, task)
            with store.transaction():
                store.db.execute(
                    "INSERT OR IGNORE INTO metadata VALUES(?,?)",
                    (namespace + "_history:" + key + ":" + record_id, encode(record)),
                )
                latest = store.one(
                    "SELECT data FROM metadata WHERE key=?", (namespace + ":" + key,)
                )
                if (
                    not latest
                    or json.loads(latest["data"])[field]["recorded_at"] < value["recorded_at"]
                    or (
                        research
                        and json.loads(latest["data"])[field]["recorded_at"] == value["recorded_at"]
                        and json.loads(latest["data"])["indexed_at"] < record["indexed_at"]
                    )
                ):
                    store.db.execute(
                        "INSERT OR REPLACE INTO metadata VALUES(?,?)",
                        (namespace + ":" + key, encode(record)),
                    )
                state = {
                    "signature": signature,
                    "recorded_at": time.time(),
                    "error": None,
                    "target": key,
                }
                store.db.execute(
                    "INSERT OR REPLACE INTO metadata VALUES(?,?)",
                    (sync_key, encode(state)),
                )
                if external:
                    store.db.execute(
                        "INSERT OR REPLACE INTO metadata VALUES(?,?)",
                        (namespace + "_sync:" + key, encode(state)),
                    )
                if not existing:
                    store.event(
                        key,
                        namespace + "_indexed",
                        {"record": record_id, "source": value["source_sha256"]},
                    )
        except (OSError, ValueError, KeyError, TypeError, subprocess.SubprocessError) as exc:
            state = {
                "signature": signature,
                "recorded_at": time.time(),
                "error": str(exc),
                "target": key,
            }
            if previous.get("error") == state["error"] and previous.get("signature") == signature:
                continue
            store.db.execute(
                "INSERT OR REPLACE INTO metadata VALUES(?,?)",
                (sync_key, encode(state)),
            )
            if external:
                store.db.execute(
                    "INSERT OR REPLACE INTO metadata VALUES(?,?)",
                    (namespace + "_sync:" + key, encode(state)),
                )
            store.event(key, namespace + "_sync_failed", state)

    sync_health(store)
