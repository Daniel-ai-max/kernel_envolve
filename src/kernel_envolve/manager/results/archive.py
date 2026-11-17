"""Portable, read-only dashboard checkpoints; never a research resume mechanism."""

import json
import os
import shutil
import sqlite3
import tempfile
import threading
import time
import uuid
from importlib.resources import files
from pathlib import Path

from kernel_envolve._io import durable_bytes, durable_json, hash_file, sync_dir
from kernel_envolve.evaluation.shared.storage import digest
from kernel_envolve.manager.dashboard.views import detail, overview
from kernel_envolve.manager.execution.process import exclusive
from kernel_envolve.manager.store import Store, encode

ASSETS = ("index.html", "app.js", "summary.js", "sync.js", "style.css")


def capture_notes(store):
    for row in store.rows("SELECT id FROM tasks"):
        task = store.task(row["id"])
        for key in ("task_dir", "workspace"):
            path = Path(task["spec"][key]) / "notes.md"
            if path.is_file():
                with path.open(errors="replace") as stream:
                    text = stream.read(32000)
                store.db.execute(
                    "INSERT OR REPLACE INTO metadata VALUES(?,?)",
                    ("saved_notes:" + row["id"], encode(text)),
                )
                break


def export_archive(store, destination, *, attachments=(), include_logs=False, automatic=False):
    """One consistent SQLite backup, original projections, hashed assets, atomic publish."""
    destination = Path(destination).resolve()
    destination.mkdir(parents=True, exist_ok=True, mode=0o700)
    with exclusive(destination / "archive.lock") as lock:
        if lock is None:
            raise ValueError("Archive export already in progress")
        with tempfile.TemporaryDirectory(prefix=".building-", dir=destination) as temp:
            stage = Path(temp)
            with sqlite3.connect(stage / "manager.sqlite") as db:
                store.db.backup(db)
            saved = Store(stage, readonly=True)
            try:
                with saved.transaction(readonly=True):
                    assessment_records = saved.rows(
                        "SELECT key,data FROM metadata WHERE key LIKE 'assessment_history:%' "
                        "OR key LIKE 'research_result_history:%' OR key LIKE 'validation_history:%'"
                    )
                    research_bindings = {
                        row["key"].removeprefix("research_source:"): json.loads(row["data"])
                        for row in saved.rows(
                            "SELECT key,data FROM metadata WHERE key LIKE 'research_source:%'"
                        )
                    }
                    now = time.time()
                    state = overview(saved, compact=True, now=now)
                    details = {t["id"]: detail(saved, t["id"], now) for t in state["tasks"]}
                state.update(
                    archived=True,
                    as_of=now,
                    coordinator=None,
                    observer=None,
                    detail=None,
                    archive_status={"error": None, "as_of": now},
                )
                for resource in state["resources"]:
                    resource["archived"] = True
                # A snapshot of live work is an observation, never a claim it still runs here.
                for task in [*state["tasks"], *details.values()]:
                    task["archived_phase"] = task["phase"]
                    task["execution_state"] = "archived"
                    if task["desired"] == "active":
                        task["phase"] = "archived"
                    task["stale"] = False
                view = {"state": state, "details": details}
                durable_json(stage / "view.json", view)
            finally:
                saved.close()
            for name in ASSETS:
                (stage / name).write_bytes(
                    files("kernel_envolve.manager.dashboard").joinpath("static", name).read_bytes()
                )
            for row in assessment_records:
                record = json.loads(row["data"])
                record_id = row["key"].rsplit(":", 1)[1]
                if len(record_id) != 64 or any(c not in "0123456789abcdef" for c in record_id):
                    raise ValueError("Invalid assessment record identity")
                directory = (
                    "validations"
                    if row["key"].startswith("validation_")
                    else "research-results"
                    if row["key"].startswith("research_result_")
                    else "assessments"
                )
                source = store.root / directory / record_id
                if (
                    hash_file(source / "input.json") != record["input_sha256"]
                    or json.loads((source / "record.json").read_text()) != record
                ):
                    raise ValueError("Saved assessment receipt changed")
                target = stage / directory / record_id
                target.mkdir(parents=True)
                for name in ("input.json", "record.json"):
                    shutil.copyfile(source / name, target / name)
                if record.get("proof_files"):
                    from kernel_envolve.manager.results.validation_import import verify_saved

                    validation_task = store.task(record["validation"]["task"])
                    verify_saved(source, validation_task)
                    for name in record["proof_files"]:
                        destination_file = target / "proof" / name
                        destination_file.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copyfile(source / "proof" / name, destination_file)
                    verify_saved(target, validation_task)
            # Periodic dashboard snapshots keep lightweight ledger references.
            # Explicit/terminal exports also carry the bounded, deduplicated source
            # deltas; baseline Git objects are intentionally retained separately.
            unbound_research_receipts = []
            unverified_validation_receipts = []
            if not automatic:
                source_blobs = set()
                bound_research_records = {}
                for job, binding in research_bindings.items():
                    if Path(job).name != job:
                        raise ValueError("Invalid research source job identity")
                    receipt = store.root / "jobs" / job / "research-source.json"
                    if (
                        receipt.is_symlink()
                        or receipt.parent.is_symlink()
                        or not receipt.is_file()
                        or hash_file(receipt) != binding["sha256"]
                    ):
                        raise ValueError("Bound research source receipt changed or missing")
                    bound_research_records[job] = json.loads(receipt.read_text())
                    from kernel_envolve.manager.results.validation_import import (
                        export_receipt,
                        receipt_bytes,
                    )

                    try:
                        validation_raw = export_receipt(store.root, job)
                    except (OSError, ValueError, KeyError, TypeError) as exc:
                        # A killed runner can leave a partial receipt. Preserve
                        # the dashboard and diagnosis without publishing that
                        # artifact at the importer's validated receipt path.
                        validation_raw = None
                        problem = {"job": job, "reason": str(exc)}
                        try:
                            raw = receipt_bytes({"state": str(store.root), "job": job})
                            target = stage / "unverified-validations" / job / "validation.json"
                            target.parent.mkdir(parents=True, exist_ok=True)
                            durable_bytes(target, raw)
                            problem["artifact"] = str(target.relative_to(stage))
                            problem["sha256"] = hash_file(target)
                        except (OSError, ValueError, KeyError, TypeError) as read_error:
                            problem["artifact_error"] = str(read_error)
                        unverified_validation_receipts.append(problem)
                    if validation_raw is not None:
                        target = stage / "research-jobs" / job / "validation.json"
                        target.parent.mkdir(parents=True, exist_ok=True)
                        durable_bytes(target, validation_raw)
                for source in (store.root / "jobs").glob("*/research-source*.json"):
                    if source.is_symlink() or source.parent.is_symlink():
                        raise ValueError("Research receipts must be regular files")
                    value = json.loads(source.read_text())
                    bound = bound_research_records.get(source.parent.name)
                    if bound is None:
                        unbound_research_receipts.append(str(source.relative_to(store.root)))
                    elif source.name == "research-source-before.json" and value != bound["before"]:
                        raise ValueError("Bound research source before receipt changed")
                    elif (
                        source.name == "research-source.json"
                        and hash_file(source) != research_bindings[source.parent.name]["sha256"]
                    ):
                        raise ValueError("Bound research source receipt changed during archive")
                    endpoints = (
                        [value]
                        if value.get("schema") == "research-source-v1"
                        else [value["before"], value["after"]]
                    )
                    for endpoint in endpoints:
                        blobs = [
                            endpoint.get("tracked_patch"),
                            *endpoint.get("untracked_scripts", {}).values(),
                        ]
                        for blob in blobs:
                            if blob is not None:
                                source_blobs.add(blob["sha256"])
                    target = stage / "research-jobs" / source.parent.name / source.name
                    target.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copyfile(source, target)
                    if bound is not None:
                        if (
                            source.name == "research-source.json"
                            and hash_file(target) != research_bindings[source.parent.name]["sha256"]
                        ):
                            raise ValueError("Bound research source receipt changed during copy")
                        if (
                            source.name == "research-source-before.json"
                            and json.loads(target.read_text()) != bound["before"]
                        ):
                            raise ValueError(
                                "Bound research source before receipt changed during copy"
                            )
                for sha in source_blobs:
                    if len(sha) != 64 or any(c not in "0123456789abcdef" for c in sha):
                        raise ValueError("Invalid research source blob identity")
                    source = store.root / "research-sources" / sha
                    from kernel_envolve.manager.results.research_source import MAX_BYTES

                    if (
                        source.is_symlink()
                        or not source.is_file()
                        or source.stat().st_size > MAX_BYTES
                        or hash_file(source) != sha
                    ):
                        raise ValueError("Research source blob changed")
                    target = stage / "research-sources" / sha
                    target.parent.mkdir(exist_ok=True)
                    shutil.copyfile(source, target)
                    if hash_file(target) != sha:
                        raise ValueError("Research source blob changed during copy")
            copied = []
            for index, source in enumerate(attachments):
                source = Path(source)
                if not source.is_file() or source.is_symlink():
                    raise ValueError("Archive attachments must be regular files")
                target = stage / "attachments" / f"{index}-{source.name}"
                target.parent.mkdir(exist_ok=True)
                shutil.copyfile(source, target)
                copied.append(
                    {"path": str(target.relative_to(stage)), "original": str(source.resolve())}
                )
            if include_logs:
                for source in (store.root / "jobs").glob("*/*.log"):
                    if source.is_symlink() or source.parent.is_symlink() or not source.is_file():
                        raise ValueError("Job logs must be regular files")
                    target = stage / "logs" / source.relative_to(store.root / "jobs")
                    target.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copyfile(source, target)
            manifest = {
                "schema": "dashboard-archive-v1",
                "automatic": automatic,
                "research_source_artifacts_included": not automatic,
                "unbound_research_source_receipts": unbound_research_receipts,
                "unverified_validation_receipts": unverified_validation_receipts,
                "created_at": now,
                "attachments": copied,
                "files": {},
            }
            for path in sorted(stage.rglob("*")):
                if path.is_file():
                    with path.open("rb") as stream:
                        os.fsync(stream.fileno())
                    manifest["files"][str(path.relative_to(stage))] = {
                        "sha256": hash_file(path),
                        "bytes": path.stat().st_size,
                    }
            durable_json(stage / "manifest.json", manifest)
            for directory in sorted(
                (p for p in stage.rglob("*") if p.is_dir()),
                key=lambda p: len(p.parts),
                reverse=True,
            ):
                sync_dir(directory)
            sync_dir(stage)
            generation = "snapshot-" + uuid.uuid4().hex
            final = destination / generation
            os.replace(stage, final)
            sync_dir(destination)
            durable_json(
                destination / "CURRENT.json",
                {"generation": generation, "sha256": hash_file(final / "manifest.json")},
            )
            return {"archive": str(destination), "generation": generation, "as_of": now}


class Archive:
    def __init__(self, root):
        self.root = Path(root).resolve()
        self.fixed = (self.root / "manifest.json").is_file()
        self.lock = threading.RLock()
        self.pointer = None
        self.view = None
        self.assets = None
        self.read()

    def read(self):
        with self.lock:
            return self._read()

    def _read(self):
        pointer = (
            {"generation": self.root.name, "sha256": hash_file(self.root / "manifest.json")}
            if self.fixed
            else json.loads((self.root / "CURRENT.json").read_text())
        )
        if pointer == self.pointer:
            return self.view
        name = pointer["generation"]
        if not isinstance(name, str) or not name.startswith("snapshot-") or Path(name).name != name:
            raise ValueError("Invalid archive generation")
        generation = self.root if self.fixed else self.root / name
        if generation.is_symlink() or not generation.is_dir():
            raise ValueError("Invalid archive directory")
        manifest_path = generation / "manifest.json"
        if hash_file(manifest_path) != pointer["sha256"]:
            raise ValueError("Archive manifest checksum mismatch")
        manifest = json.loads(manifest_path.read_text())
        if manifest["schema"] != "dashboard-archive-v1":
            raise ValueError("Unsupported archive schema")
        if not {*ASSETS, "view.json", "manager.sqlite"}.issubset(manifest["files"]):
            raise ValueError("Archive is missing required records")
        for relative, proof in manifest["files"].items():
            path = generation / relative
            if (
                Path(relative).is_absolute()
                or ".." in Path(relative).parts
                or not path.resolve().is_relative_to(generation)
                or path.is_symlink()
            ):
                raise ValueError("Unsafe archive path")
            if path.stat().st_size != proof["bytes"] or hash_file(path) != proof["sha256"]:
                raise ValueError(f"Archive checksum mismatch: {relative}")
        view = json.loads((generation / "view.json").read_text())
        if not view["state"]["archived"] or {t["id"] for t in view["state"]["tasks"]} != set(
            view["details"]
        ):
            raise ValueError("Archive task records differ")
        self.assets = {name: (generation / name).read_bytes() for name in ASSETS}
        self.pointer, self.view = pointer, view
        return view


def checkpoint(store, *, force=False):
    """Background only; bound cost and keep the last two fully published checkpoints."""
    from kernel_envolve.manager.results.index import gpu_work_pending

    if gpu_work_pending(store):
        return False
    now = time.time()
    previous = store.one("SELECT data FROM metadata WHERE key='dashboard_archive'")
    previous = json.loads(previous["data"]) if previous else {}
    if not force and now - previous.get("attempted_at", 0) < 60:
        return False
    destination = Path(
        os.environ.get("KERNEL_ENVOLVE_ARCHIVE", str(store.root / "dashboard-archive"))
    ).resolve()
    try:
        capture_notes(store)
        signature = digest(
            {
                "tasks": store.rows(
                    "SELECT id,spec,desired,reason,wall_seconds,gpu_seconds,report FROM tasks ORDER BY id"
                ),
                "jobs": store.rows("SELECT id,status,seconds,error,ended FROM jobs ORDER BY id"),
                "runs": store.rows("SELECT path,data,error FROM runs ORDER BY path"),
                "events": store.one("SELECT MAX(seq) AS seq FROM events"),
                "records": store.rows(
                    "SELECT key,data FROM metadata WHERE key LIKE 'assessment:%' OR key LIKE 'assessment_sync:%' OR key LIKE 'allocation:%' OR key LIKE 'publication:%' OR key LIKE 'saved_notes:%' OR key LIKE 'research_result:%' OR key LIKE 'research_result_sync:%' OR key LIKE 'validation:%' OR key LIKE 'validation_sync:%' OR key LIKE 'narration:%' ORDER BY key"
                ),
            }
        )
        if (
            previous.get("signature") == signature
            and not previous.get("error")
            and previous.get("archive") == str(destination)
            and (destination / "CURRENT.json").is_file()
        ):
            store.db.execute(
                "INSERT OR REPLACE INTO metadata VALUES('dashboard_archive',?)",
                (encode(previous | {"attempted_at": now}),),
            )
            return False
        tasks = store.rows("SELECT desired FROM tasks")
        terminal = (
            bool(tasks)
            and all(t["desired"] != "active" for t in tasks)
            and not store.one("SELECT id FROM jobs WHERE status IN ('queued','running') LIMIT 1")
        )
        result = export_archive(store, destination, automatic=not terminal)
        value = {**result, "attempted_at": now, "error": None, "signature": signature}
        # Checkpoints are replaceable copies; full history remains in the SQLite ledger.
        generations = sorted(
            (
                p
                for p in destination.glob("snapshot-*")
                if p.is_dir()
                and not p.is_symlink()
                and json.loads((p / "manifest.json").read_text()).get("automatic")
            ),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        for path in generations[2:]:
            if path.is_dir() and not path.is_symlink():
                shutil.rmtree(path)
    except Exception as exc:
        value = {**previous, "attempted_at": now, "error": str(exc)}
    store.db.execute(
        "INSERT OR REPLACE INTO metadata VALUES('dashboard_archive',?)", (encode(value),)
    )
    return not value["error"]
