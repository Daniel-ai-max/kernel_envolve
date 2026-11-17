"""Derive an explicitly incomplete dashboard archive from a consistent saved ledger.

This is a presentation recovery, never a manager migration or research resume.
Original task state, policies, jobs and receipts remain byte-for-byte in the input.
"""

import math
import shutil
import sqlite3
import tempfile
from pathlib import Path

from kernel_envolve._io import hash_file
from kernel_envolve.manager.dashboard import narrator
from kernel_envolve.manager.dashboard.views import overview
from kernel_envolve.manager.results.archive import export_archive
from kernel_envolve.manager.store import Store, encode
from kernel_envolve.manager.usage import import_usage


def recover_archive(
    source,
    destination,
    *,
    observed_at,
    recorded_at,
    coverage,
    interruption,
    attachments,
    diagnostics=(),
    usage=None,
    command=None,
    call=narrator.invoke,
):
    """Keep source hashes and explicit coverage; replay only retained narration evidence.

    Attachments are local evidence receipts, not executable instructions. Usage must
    follow the normal monotonic allocation importer. Only the independent temporary
    copy gains presentation metadata, usage receipts and validated model narration.
    """
    source, destination = Path(source).resolve(), Path(destination).resolve()
    if source == destination or source.is_relative_to(destination):
        raise ValueError("Recovery destination must not contain its source")
    for value in (observed_at, recorded_at):
        if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
            raise ValueError("Recovery observations require finite positive timestamps")
    if recorded_at < observed_at:
        raise ValueError("Recovery record predates its saved observation")
    if not coverage or not interruption or not attachments:
        raise ValueError("Recovery requires coverage, interruption and source evidence")
    originals = [source / "manager.sqlite", *(Path(p).resolve() for p in attachments)]
    if any(p.is_symlink() or not p.is_file() for p in originals):
        raise ValueError("Recovery sources must be regular files")
    provenance = [{"path": str(p), "sha256": hash_file(p)} for p in originals]
    record = {
        "schema": "dashboard-recovery-v1",
        "observed_at": observed_at,
        "recorded_at": recorded_at,
        "coverage": coverage,
        "interruption": interruption,
        "sources": provenance,
    }
    diagnostic_sources = []
    for path in diagnostics:
        path = Path(path).resolve()
        if path not in originals or path.stat().st_size > 8192:
            raise ValueError("Diagnostics must be bounded evidence attachments")
        diagnostic_sources.append(
            {"path": str(path), "sha256": hash_file(path), "text": path.read_text()}
        )
    if sum(len(item["text"].encode()) for item in diagnostic_sources) > 16384:
        raise ValueError("Recovery diagnostics exceed the evidence budget")
    record["diagnostic_evidence"] = diagnostic_sources
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".recovery-", dir=destination.parent) as temp:
        root = Path(temp)
        original = Store(source, readonly=True)
        try:
            with sqlite3.connect(root / "manager.sqlite") as db:
                original.db.backup(db)
        finally:
            original.close()
        # No writable Store constructor: no migration, host reassignment or scheduling.
        saved = Store(root, readonly=True)
        saved.db.close()
        saved.db = sqlite3.connect(root / "manager.sqlite", isolation_level=None)
        saved.db.row_factory = sqlite3.Row
        try:
            if saved.db.execute("PRAGMA quick_check").fetchone()[0] != "ok":
                raise ValueError("Invalid recovery ledger")
            saved.db.execute(
                "INSERT OR REPLACE INTO metadata VALUES('archive_recovery',?)", (encode(record),)
            )
            if usage is not None:
                import_usage(saved, usage)
            if command is not None:
                changed = narrator.step(saved, command, replay=True, scan_logs=False, call=call)
                expected = sum(not t.get("delivery_parent") for t in overview(saved)["tasks"])
                if changed != expected:
                    attempt = narrator.get(saved, "narrator_attempt", {})
                    raise ValueError(
                        "Recovery narration incomplete: " + str(attempt.get("error", changed))
                    )
            # Copy available receipt directories; absent proof stays explicitly absent.
            for name in ("assessments", "research-results", "validations"):
                if (source / name).is_dir():
                    shutil.copytree(source / name, root / name, symlinks=True)
            for item in provenance:
                if hash_file(Path(item["path"])) != item["sha256"]:
                    raise ValueError("Recovery source changed during replay")
            return export_archive(
                saved, destination, automatic=True, recovery=True, attachments=originals
            )
        finally:
            saved.close()
