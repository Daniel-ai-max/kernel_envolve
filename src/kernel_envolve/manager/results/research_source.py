"""Bounded Git deltas attached to existing jobs; never evaluator snapshots."""

import hashlib
import json
import os
import selectors
import stat
import subprocess
import tempfile
import time
from pathlib import Path

from kernel_envolve.evaluation.shared.storage import digest, write_json

MAX_BYTES = 16 * 1024**2
MAX_FILE_BYTES = 1024**2
MAX_FILES = 256
GIT_SECONDS = 3
SOURCE_SUFFIXES = {
    ".py",
    ".cu",
    ".cuh",
    ".cpp",
    ".c",
    ".h",
    ".hpp",
    ".sh",
    ".bash",
    ".zsh",
    ".rs",
    ".go",
    ".js",
    ".mjs",
    ".ts",
    ".tsx",
    ".ptx",
}


def bounded_git(workspace, args, limit):
    """Stop reading and terminate Git at the byte/deadline limit, not afterward."""
    data = bytearray()
    process = subprocess.Popen(
        ["git", "-C", str(workspace), *args],
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
    )
    deadline = time.monotonic() + GIT_SECONDS
    try:
        with selectors.DefaultSelector() as selector:
            selector.register(process.stdout, selectors.EVENT_READ)
            while True:
                remaining = deadline - time.monotonic()
                if remaining <= 0 or not selector.select(remaining):
                    return bytes(data), "Git capture deadline reached"
                block = process.stdout.read1(min(65536, limit + 1 - len(data)))
                if not block:
                    code = process.wait(timeout=max(0.01, deadline - time.monotonic()))
                    return bytes(data), None if code == 0 else f"Git capture exited {code}"
                data.extend(block)
                if len(data) > limit:
                    return bytes(data[:limit]), "Git capture byte limit reached"
    finally:
        if process.poll() is None:
            process.kill()
        process.wait()
        process.stdout.close()


def capture(task, blob_root=None, *, code_only=False):
    """Capture tracked changes plus nonignored source scripts; disclose all gaps.

    The baseline commit must remain available separately. Ignored files, external
    dependencies and submodule contents are deliberately outside this local scope.
    No unbounded source walk, Git output buffering or full-checkout copy occurs.
    """
    root = Path(task["spec"]["workspace"])
    baseline = task["spec"]["baseline"]["revision"]
    record = {
        "schema": "research-source-v1",
        "baseline_revision": baseline,
        "research_only": True,
        "eligible_for_best": False,
        "complete": True,
        "tracked_patch": None,
        "untracked_scripts": {},
        "gaps": [],
        "excluded": [
            "ignored files",
            "untracked files outside the recorded source suffix list",
            "external dependencies",
            "submodule contents",
            "baseline commit objects (retain separately)",
        ],
        "limits": {"bytes": MAX_BYTES, "file_bytes": MAX_FILE_BYTES, "files": MAX_FILES},
        "capture_semantics": "bounded non-atomic endpoint observation",
        "untracked_source_suffixes": sorted(SOURCE_SUFFIXES),
    }
    remaining = MAX_BYTES

    def gap(message):
        record["complete"] = False
        record["gaps"].append(message)

    def blob(data):
        sha = hashlib.sha256(data).hexdigest()
        if blob_root is not None:
            destination = Path(blob_root) / sha
            destination.parent.mkdir(parents=True, exist_ok=True)
            # Content-addressed deltas are shared by before/after and later jobs.
            with tempfile.NamedTemporaryFile(dir=destination.parent) as stream:
                stream.write(data)
                stream.flush()
                try:
                    os.link(stream.name, destination)
                except FileExistsError:
                    if destination.is_symlink() or destination.stat().st_size != len(data):
                        raise ValueError("Research source blob changed")
                    with destination.open("rb") as existing:
                        if hashlib.sha256(existing.read(len(data) + 1)).hexdigest() != sha:
                            raise ValueError("Research source blob changed")
        return {"sha256": sha, "bytes": len(data)}

    try:
        patterns = (
            [f":(glob)**/*{suffix}" for suffix in sorted(SOURCE_SUFFIXES)] if code_only else []
        )
        patch, error = bounded_git(
            root,
            [
                "diff",
                "--binary",
                "--full-index",
                "--no-ext-diff",
                "--no-textconv",
                baseline,
                "--",
                *patterns,
            ],
            remaining,
        )
        record["tracked_patch"] = blob(patch)
        remaining -= len(patch)
        if error:
            gap(error + ": tracked diff")
        names, error = bounded_git(
            root, ["ls-files", "--others", "--exclude-standard", "-z"], 65536
        )
        if error:
            gap(error + ": untracked names")
        names = names.split(b"\0")[:-1]  # Drop a partial final path when clipped.
        if len(names) > MAX_FILES:
            gap("Untracked entry count limit reached")
        for raw in names[:MAX_FILES]:
            name = os.fsdecode(raw)
            relative = Path(name)
            if relative.suffix not in SOURCE_SUFFIXES:
                continue
            path = root / relative
            if (
                relative.is_absolute()
                or ".." in relative.parts
                or any(
                    p.is_symlink() for p in (path, *path.parents) if p != root and root in p.parents
                )
            ):
                gap(f"Symlink/unsafe path not captured: {name}")
                continue
            fd = os.open(path, os.O_RDONLY | os.O_NONBLOCK | os.O_NOFOLLOW)
            with os.fdopen(fd, "rb") as stream:
                before = os.fstat(stream.fileno())
                if not stat.S_ISREG(before.st_mode) or before.st_size > min(
                    MAX_FILE_BYTES, remaining
                ):
                    gap(f"Nonregular/oversized source not captured: {name}")
                    continue
                data = stream.read(min(MAX_FILE_BYTES, remaining) + 1)
                after = os.fstat(stream.fileno())
            if len(data) > min(MAX_FILE_BYTES, remaining):
                gap(f"Growing source exceeded capture limit: {name}")
                continue
            if (before.st_size, before.st_mtime_ns, before.st_ctime_ns) != (
                after.st_size,
                after.st_mtime_ns,
                after.st_ctime_ns,
            ):
                gap(f"Source changed while reading: {name}")
            record["untracked_scripts"][name] = blob(data)
            remaining -= len(data)
    except (OSError, ValueError, subprocess.SubprocessError) as exc:
        gap(f"Source capture incomplete: {type(exc).__name__}: {exc}")
    record["fingerprint"] = digest(
        {key: record[key] for key in ("baseline_revision", "tracked_patch", "untracked_scripts")}
    )
    return record


def bind_job(store, task, job, directory, before, status, exit_code):
    """Bind the same supervisor job identity to both source endpoints and result."""
    after = capture(task, store.root / "research-sources")
    record = {
        "schema": "research-job-source-v1",
        "task": task["id"],
        "job": job["id"],
        "command": json.loads(job["command"]),
        "status": status,
        "exit_code": exit_code,
        "before": before,
        "after": after,
        "research_only": True,
        "eligible_for_best": False,
        "source_changed": (before["fingerprint"] != after["fingerprint"])
        if before["complete"] and after["complete"]
        else None,
        "endpoints_complete": before["complete"] and after["complete"],
    }
    record["logs"] = {}
    for name in ("stdout.log", "stderr.log"):
        log = directory / name
        if log.is_file():
            with log.open("rb") as stream:
                data = stream.read(MAX_BYTES)
            record["logs"][name] = {
                "sha256": hashlib.sha256(data).hexdigest(),
                "hashed_bytes": len(data),
                "bytes": log.stat().st_size,
                "complete": log.stat().st_size <= MAX_BYTES,
            }
    path = directory / "research-source.json"
    write_json(path, record)
    binding = {
        "job": job["id"],
        "receipt": str(path),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "source_changed": record["source_changed"],
        "endpoints_complete": record["endpoints_complete"],
    }
    with store.transaction():
        store.db.execute(
            "INSERT OR REPLACE INTO metadata VALUES(?,?)",
            ("research_source:" + job["id"], json.dumps(binding)),
        )
        store.event(task["id"], "research_source_bound", binding)
