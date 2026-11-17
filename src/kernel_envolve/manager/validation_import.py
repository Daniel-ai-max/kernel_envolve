"""Import terminal validation outcomes from one host-installed manager feed.

The feed is installed once, outside the candidate workspace. No candidate path
or executable from a receipt is followed. Imports explain outcomes, never best.
"""

import contextlib
import hashlib
import json
import os
import re
import sqlite3
import stat
from pathlib import Path

from ..storage import digest
from .store import Store, encode

LIMIT = 4 * 1024 * 1024
MARKER = b"KE_VALIDATION_RECEIPT_SHA256="


def _read(root, relative, limit=LIMIT):
    parts = Path(relative).parts
    if Path(relative).is_absolute() or ".." in parts:
        raise ValueError("Invalid validation artifact path")
    fd = os.open(root, os.O_RDONLY | os.O_DIRECTORY)
    try:
        for part in parts[:-1]:
            child = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=fd)
            os.close(fd)
            fd = child
        leaf = os.open(parts[-1], os.O_RDONLY | os.O_NONBLOCK | os.O_NOFOLLOW, dir_fd=fd)
        with os.fdopen(leaf, "rb") as stream:
            if not stat.S_ISREG(os.fstat(stream.fileno()).st_mode):
                raise ValueError("Validation artifact must be a regular file")
            raw = stream.read(limit + 1)
    finally:
        os.close(fd)
    if len(raw) > limit:
        raise ValueError("Validation artifact exceeds bounded input size")
    return raw


def _root(path):
    root = Path(path).resolve()
    if (root / "CURRENT.json").is_file():
        pointer = json.loads(_read(root, "CURRENT.json"))
        name = pointer["generation"]
        if not isinstance(name, str) or not re.fullmatch(r"snapshot-[A-Za-z0-9_-]+", name):
            raise ValueError("Invalid validation archive generation")
        if (root / name).is_symlink():
            raise ValueError("Archive generation cannot be a symlink")
        root = root / name
        if hashlib.sha256(_read(root, "manifest.json")).hexdigest() != pointer["sha256"]:
            raise ValueError("Validation archive manifest changed")
    archived = (root / "manifest.json").is_file()
    if archived:
        manifest = json.loads(_read(root, "manifest.json"))
        if manifest.get("schema") != "dashboard-archive-v1":
            raise ValueError("Unsupported validation archive")
        proof = manifest["files"]["manager.sqlite"]
        from .archive import hash_file

        db = root / "manager.sqlite"
        if (
            db.is_symlink()
            or db.stat().st_size != proof["bytes"]
            or hash_file(db) != proof["sha256"]
        ):
            raise ValueError("Archived manager database changed")
    return root, archived


@contextlib.contextmanager
def _open(source):
    root, archived = _root(source["state"])
    try:
        store = Store(root, readonly=True)
    except sqlite3.Error as exc:
        raise ValueError(f"External manager unavailable: {exc}") from exc
    try:
        yield root, archived, store
    except sqlite3.Error as exc:
        raise ValueError(f"External manager unavailable: {exc}") from exc
    finally:
        store.close()


def _names(job, archived):
    if (
        not isinstance(job, str)
        or not re.fullmatch(r"[A-Za-z0-9_.:-]{1,160}", job)
        or job in (".", "..")
    ):
        raise ValueError("Invalid external GPU job")
    base = f"{'research-jobs' if archived else 'jobs'}/{job}"
    log = f"{'logs' if archived else 'jobs'}/{job}/stdout.log"
    return base + "/validation.json", base + "/research-source.json", log


def receipt_bytes(source):
    root, archived = _root(source["state"])
    return _read(root, _names(source["job"], archived)[0])


def signature(source):
    with _open(source) as (root, archived, store):
        raw, bound, log = _names(source["job"], archived)
        job = store.one("SELECT * FROM jobs WHERE id=?", (source["job"],))
        pinned = store.one(
            "SELECT data FROM metadata WHERE key=?", ("research_source:" + source["job"],)
        )
        info = (root / log).stat()
        return digest(
            [
                dict(job) if job else None,
                dict(pinned) if pinned else None,
                hashlib.sha256(_read(root, raw)).hexdigest(),
                hashlib.sha256(_read(root, bound)).hexdigest(),
                info.st_size,
                info.st_mtime_ns,
                info.st_ctime_ns,
            ]
        )


def proof_files(source):
    """Capture fixed provenance files once; the target retains them independently."""
    from .research_source import MAX_BYTES

    with _open(source) as (root, archived, store):
        job = store.one(
            "SELECT * FROM jobs WHERE id=? AND task=? AND kind='gpu'",
            (source["job"], source["external_task"]),
        )
        pin = store.one(
            "SELECT data FROM metadata WHERE key=?", ("research_source:" + source["job"],)
        )
        if not job or not pin or job["ended"] is None or job["status"] in ("queued", "running"):
            raise ValueError("External validation requires a source-bound terminal GPU job")
        raw_name, source_name, log_name = _names(source["job"], archived)
        result = {
            "validation.json": _read(root, raw_name),
            "source.json": _read(root, source_name),
            "stdout.log": _read(root, log_name, MAX_BYTES),
            "identity.json": encode(
                {
                    "job": dict(job),
                    "pin": json.loads(pin["data"]),
                    "baseline_revision": store.task(job["task"])["spec"]["baseline"]["revision"],
                }
            ).encode(),
        }
        bound = json.loads(result["source.json"])
        for end in (bound["before"], bound["after"]):
            for blob in [end.get("tracked_patch"), *end.get("untracked_scripts", {}).values()]:
                if blob is not None:
                    sha = blob["sha256"]
                    if not isinstance(sha, str) or not re.fullmatch(r"[a-f0-9]{64}", sha):
                        raise ValueError("Invalid source blob identity")
                    result["source-blobs/" + sha] = _read(
                        root, "research-sources/" + sha, MAX_BYTES
                    )
        return result


def _check_parts(parts):
    raw, bound_raw, log = (parts[n] for n in ("validation.json", "source.json", "stdout.log"))
    identity = json.loads(parts["identity.json"])
    job = identity["job"]
    if job["kind"] != "gpu" or job["ended"] is None or job["status"] in ("running", "queued"):
        raise ValueError("External validation requires a terminal GPU job")
    bound_sha = hashlib.sha256(bound_raw).hexdigest()
    bound, report = json.loads(bound_raw), json.loads(raw)
    if bound_sha != identity["pin"]["sha256"]:
        raise ValueError("External supervisor receipt hash differs")
    if (
        bound.get("schema") != "research-job-source-v1"
        or bound["task"] != job["task"]
        or bound["job"] != job["id"]
        or bound["command"] != json.loads(job["command"])
        or bound["status"] != job["status"]
        or bound["exit_code"] != job["exit_code"]
        or bound["endpoints_complete"] is not True
        or bound["source_changed"] is not False
    ):
        raise ValueError("External job/source identity differs")
    for end in (bound["before"], bound["after"]):
        if (
            end.get("complete") is not True
            or end["baseline_revision"] != report["baseline_revision"]
            or end["fingerprint"] != report["source_sha256"]
            or digest(
                {k: end[k] for k in ("baseline_revision", "tracked_patch", "untracked_scripts")}
            )
            != end["fingerprint"]
        ):
            raise ValueError("External source endpoints differ")
    proof = bound["logs"]["stdout.log"]
    if (
        proof["complete"] is not True
        or proof["bytes"] != len(log)
        or proof["hashed_bytes"] != len(log)
        or hashlib.sha256(log).hexdigest() != proof["sha256"]
    ):
        raise ValueError("External complete log hash differs")
    markers = re.findall(rb"^" + MARKER + rb"([a-f0-9]{64})\r?$", log, re.MULTILINE)
    if not markers or markers[-1].decode() != hashlib.sha256(raw).hexdigest():
        raise ValueError("Validation receipt is not bound to the completed log")
    if report.get("schema") != "candidate-validation-v1":
        raise ValueError("Expected candidate-validation-v1")
    if identity["baseline_revision"] != report["baseline_revision"]:
        raise ValueError("External task baseline differs")
    for end in (bound["before"], bound["after"]):
        for blob in [end.get("tracked_patch"), *end.get("untracked_scripts", {}).values()]:
            if blob is not None:
                data = parts["source-blobs/" + blob["sha256"]]
                if len(data) != blob["bytes"] or hashlib.sha256(data).hexdigest() != blob["sha256"]:
                    raise ValueError("Retained source blob differs")
    return raw, report, job, bound_sha


def _verified(source):
    return _check_parts(proof_files(source))


def verify_saved(folder, task):
    """Reverify an imported outcome without its source state or workspace."""
    record = json.loads(_read(folder, "record.json"))
    parts = {}
    from .research_source import MAX_BYTES

    for name, proof in record["proof_files"].items():
        if name not in (
            "validation.json",
            "source.json",
            "stdout.log",
            "identity.json",
        ) and not re.fullmatch(r"source-blobs/[a-f0-9]{64}", name):
            raise ValueError("Invalid saved validation proof path")
        raw = _read(folder, "proof/" + name, MAX_BYTES)
        if len(raw) != proof["bytes"] or hashlib.sha256(raw).hexdigest() != proof["sha256"]:
            raise ValueError("Saved validation proof changed")
        parts[name] = raw
    if hashlib.sha256(_read(folder, "input.json")).hexdigest() != record["input_sha256"]:
        raise ValueError("Saved validation input changed")
    value = _project(*_check_parts(parts), task)
    if value != record["validation"] or parts["validation.json"] != _read(folder, "input.json"):
        raise ValueError("Saved validation projection differs")
    return value


def verify_bundle(source, task):
    parts = proof_files(source)
    return _project(*_check_parts(parts), task), parts


def verify(source, task):
    return verify_bundle(source, task)[0]


def _project(raw, report, job, bound_sha, task):
    if (
        report["target"] not in (task["id"], job["task"])
        or report["baseline_revision"] != task["spec"]["baseline"]["revision"]
    ):
        raise ValueError("External validation target/baseline differs")
    status = report["status"]
    checks = report["checks"]
    if status in ("passed", "no_benefit") and report.get("performance") is not None:
        from ..validation import performance_decision

        decision, _ = performance_decision(
            report["performance"],
            **report["policy"],
            scope=report["scope"],
            mode=report["mode"],
            baseline_revision=report["baseline_revision"],
            source_sha256=report["source_sha256"],
        )
        expected = "continue" if status == "passed" else status
        if decision != expected:
            raise ValueError("Validation performance decision differs from measurements")
    elif status in ("passed", "no_benefit"):
        raise ValueError("Validation performance measurements are missing")
    if status in ("passed", "no_benefit") and (
        job["exit_code"] != 0 or job["status"] != "succeeded"
    ):
        raise ValueError("Successful validation outcome requires successful command exit")
    value = {
        "schema": "validation-result-v1",
        "task": task["id"],
        "baseline_revision": report["baseline_revision"],
        "source_sha256": report["source_sha256"],
        "evidence_sha256": hashlib.sha256(raw).hexdigest(),
        "recorded_at": job["ended"],
        "promotion": False,
        "status": status,
        "scope": report["scope"],
        "summary": report["summary"],
        "limitations": report["limitations"],
        "checks": checks,
    }
    value["external"] = {
        "task": job["task"],
        "job": job["id"],
        "receipt_sha256": bound_sha,
        **{k: value[k] for k in ("baseline_revision", "source_sha256", "evidence_sha256")},
    }
    from .assessment import validate_validation

    return validate_validation(value, task)


def export_receipt(state_root, job):
    root, archived = _root(state_root)
    if not (root / _names(job, archived)[0]).exists():
        return None
    store = Store(root, readonly=True)
    try:
        row = store.job(job)
    finally:
        store.close()
    raw, _, _, _ = _verified({"state": str(root), "job": job, "external_task": row["task"]})
    return raw


def watch_external(store, task, state, external_task):
    target = store.task(task)
    state = Path(state).resolve()
    if state.is_relative_to(Path(target["spec"]["workspace"]).resolve()):
        raise ValueError("Install the external feed outside the candidate workspace")
    source = {"state": str(state), "external_task": external_task, "target": task}
    # Verify the host binding once, before accepting any jobs.
    with _open(source) as (_, _, external):
        if (
            external.task(external_task)["spec"]["baseline"]["revision"]
            != target["spec"]["baseline"]["revision"]
        ):
            raise ValueError("External feed baseline differs")
    key = "validation_feed:" + digest(source)
    store.db.execute("INSERT OR IGNORE INTO metadata VALUES(?,?)", (key, encode(source)))


def sources(store):
    """One registration discovers all terminal receipts, including late arrivals."""
    result = []
    feeds = store.rows("SELECT key,data FROM metadata WHERE key LIKE 'validation_feed:%'")
    # Local receipts need no installation or per-experiment reporting step.
    feeds += [
        {
            "key": "local-validation:" + row["task"],
            "data": encode(
                {"state": str(store.root), "external_task": row["task"], "target": row["task"]}
            ),
        }
        for row in store.rows(
            "SELECT DISTINCT task FROM jobs WHERE kind='gpu' AND ended IS NOT NULL"
        )
    ]
    for feed in feeds:
        source = json.loads(feed["data"])
        try:
            with _open(source) as (root, archived, external):
                jobs = external.rows(
                    "SELECT id FROM jobs WHERE task=? AND kind='gpu' AND ended IS NOT NULL ORDER BY ended,id",
                    (source["external_task"],),
                )
                for job in jobs:
                    if not (root / _names(job["id"], archived)[0]).exists():
                        continue
                    item = source | {
                        "kind": "external_validation",
                        "job": job["id"],
                        "state": str(root),
                    }
                    result.append(
                        {
                            "key": "validation_source:" + source["target"],
                            "data": encode(item),
                            "sync_key": "validation_feed_sync:" + digest([feed["key"], job["id"]]),
                        }
                    )
            _feed_health(store, feed["key"], source["target"], None)
        except (OSError, ValueError, KeyError) as exc:
            _feed_health(store, feed["key"], source["target"], str(exc))
    return result


def _feed_health(store, feed, target, error):
    import time

    key = "validation_feed_health:" + digest(feed)
    old = store.one("SELECT data FROM metadata WHERE key=?", (key,))
    if old and json.loads(old["data"])["error"] == error:
        return
    data = {"target": target, "error": error, "recorded_at": time.time(), "signature": None}
    store.db.execute("INSERT OR REPLACE INTO metadata VALUES(?,?)", (key, encode(data)))
    if error:
        store.event(target, "validation_sync_failed", data)


def sync_health(store):
    """Aggregate per-job verification and feed availability without erasing errors."""
    grouped = {}
    for row in store.rows(
        "SELECT data FROM metadata WHERE key LIKE 'validation_feed_sync:%' OR key LIKE 'validation_feed_health:%' OR key LIKE 'validation_custom_sync:%'"
    ):
        value = json.loads(row["data"])
        if value.get("target"):
            grouped.setdefault(value["target"], []).append(value)
    for target, values in grouped.items():
        errors = [v for v in values if v.get("error")]
        latest = max(errors or values, key=lambda v: v["recorded_at"])
        store.db.execute(
            "INSERT OR REPLACE INTO metadata VALUES(?,?)",
            ("validation_sync:" + target, encode(latest)),
        )
