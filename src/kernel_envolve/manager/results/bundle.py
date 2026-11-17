"""Host-side evidence handoff: compact records plus explicitly retained large artifacts.

This runs once on quiescent evidence, outside the research loop. It does not change
manager state, evaluation results, source files or GPU admission.
"""

import hashlib
import io
import json
import os
import re
import stat
import tarfile
from pathlib import Path

from kernel_envolve._io import durable_json, hash_file, sync_dir


class _Writer:
    def __init__(self, stream):
        self.stream = stream
        self.digest = hashlib.sha256()
        self.bytes = 0

    def write(self, data):
        written = self.stream.write(data)
        if written != len(data):
            raise OSError("Short evidence write")
        self.digest.update(data)
        self.bytes += written
        return written


class _Reader:
    def __init__(self, stream):
        self.stream = stream
        self.digest = hashlib.sha256()
        self.bytes = 0

    def read(self, size):
        data = self.stream.read(size)
        self.digest.update(data)
        self.bytes += len(data)
        return data


def _identity(value):
    return value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns, value.st_ctime_ns


def _copy_file(tar, source, name):
    before = source.lstat()
    if not stat.S_ISREG(before.st_mode):
        raise ValueError(f"Evidence is no longer a regular file: {name}")
    fd = os.open(source, os.O_RDONLY | os.O_NOFOLLOW)
    with os.fdopen(fd, "rb") as stream:
        if _identity(os.fstat(stream.fileno())) != _identity(before):
            raise ValueError(f"Evidence changed before copy: {name}")
        # Construct regular entries explicitly: never encode hardlinks or follow symlinks.
        info = tarfile.TarInfo(name)
        info.size = before.st_size
        info.mode = 0o600
        reader = _Reader(stream)
        tar.addfile(info, reader)
        after = os.fstat(stream.fileno())
    if (
        reader.bytes != before.st_size
        or _identity(before) != _identity(after)
        or _identity(before) != _identity(source.lstat())
    ):
        raise ValueError(f"Evidence changed during copy: {name}")
    return {"bytes": reader.bytes, "sha256": reader.digest.hexdigest()}


def pack_evidence(roots, destination, *, retain=None, exclude=None):
    """Package selected roots without loading tensors into memory.

    Trusted host callbacks receive (archive path, file size) for ``retain`` or
    archive path for ``exclude`` and return a nonempty reason or None. There is
    no default size/extension filter. Retained bytes are persisted uncompressed
    before the compact gzip and final manifest are published. Download both
    evidence.json and evidence.tar.gz; keep retained.tar at its durable origin.
    A failed attempt leaves diagnostic partials, never a successful manifest.
    """
    destination = Path(destination).absolute()
    sources = {}
    for label, source in roots.items():
        if not isinstance(label, str) or not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9_.-]*", label):
            raise ValueError("Evidence root labels must be simple path components")
        source = Path(source)
        if source.is_symlink() or not source.is_dir():
            raise ValueError("Evidence roots must be real directories")
        source = source.resolve()
        target = destination.resolve()
        if source.is_relative_to(target) or target.is_relative_to(source):
            raise ValueError("Evidence output overlaps a source root")
        if any(source.is_relative_to(p) or p.is_relative_to(source) for p in sources.values()):
            raise ValueError("Evidence source roots overlap")
        sources[label] = source
    if not sources:
        raise ValueError("At least one evidence root is required")
    destination.mkdir(exist_ok=False, mode=0o700)
    sync_dir(destination.parent)
    manifest = {"schema": "evidence-bundle-v1", "files": {}, "links": [], "excluded": []}
    entries = []

    def reason(callback, *args):
        value = callback(*args) if callback else None
        if value is not None and (not isinstance(value, str) or not value.strip()):
            raise ValueError("Evidence selection must return a nonempty reason or None")
        return value

    def visit(path, name):
        value = path.lstat()
        if stat.S_ISLNK(value.st_mode):
            manifest["links"].append({"path": name, "target": os.readlink(path)})
            return
        omitted = reason(exclude, name)
        if omitted:
            manifest["excluded"].append({"path": name, "reason": omitted})
        elif stat.S_ISDIR(value.st_mode):
            for child in sorted(path.iterdir()):
                visit(child, name + "/" + child.name)
        elif stat.S_ISREG(value.st_mode):
            entries.append((path, name, reason(retain, name, value.st_size)))
        else:
            raise ValueError(f"Unsupported evidence file type: {name}")

    for label, source in sorted(sources.items()):
        visit(source, label)

    # Complete the durable retained archive first. Publication of a compact
    # manifest cannot succeed if saving its referenced bytes fails.
    retained = destination / "retained.tar"
    with retained.with_suffix(".partial").open("xb") as raw:
        writer = _Writer(raw)
        with tarfile.open(fileobj=writer, mode="w|") as tar:
            for source, name, selected in entries:
                if selected:
                    record = _copy_file(tar, source, name)
                    manifest["files"][name] = dict(record, storage="retained", reason=selected)
        raw.flush()
        os.fsync(raw.fileno())
    retained.with_suffix(".partial").replace(retained)
    sync_dir(destination)
    manifest["retained"] = {
        "path": retained.name,
        "bytes": writer.bytes,
        "sha256": writer.digest.hexdigest(),
    }
    compact = destination / "evidence.tar.gz"
    partial = compact.with_suffix(".partial")
    with tarfile.open(partial, "w:gz", compresslevel=1) as tar:
        for source, name, selected in entries:
            if not selected:
                record = _copy_file(tar, source, name)
                manifest["files"][name] = dict(record, storage="compact")
        # The transferred archive remains self-describing when its host stores
        # the outer checksum in an existing download/ACK receipt.
        raw = json.dumps(manifest, sort_keys=True).encode()
        index = tarfile.TarInfo("evidence-index.json")
        index.size = len(raw)
        index.mode = 0o600
        tar.addfile(index, io.BytesIO(raw))
    with partial.open("rb") as stream:
        os.fsync(stream.fileno())
    partial.replace(compact)
    manifest["compact"] = {
        "path": compact.name,
        "bytes": compact.stat().st_size,
        "sha256": hash_file(compact),
    }
    sync_dir(destination)
    durable_json(destination / "evidence.json", manifest)
    return manifest


def verify_evidence(directory, *, include_retained=False):
    """Verify transferred bytes and every declared member without extracting files.

    The caller must authenticate evidence.json through its existing transfer
    receipt. This verifies consistency, never candidate correctness or best.
    """
    directory = Path(directory)
    manifest = json.loads((directory / "evidence.json").read_text())
    if manifest.get("schema") != "evidence-bundle-v1":
        raise ValueError("Unknown evidence bundle")
    records = manifest["files"]
    for name, item in records.items():
        if (
            not isinstance(name, str)
            or name.startswith("/")
            or ".." in name.split("/")
            or item.get("storage") not in ("compact", "retained")
        ):
            raise ValueError("Invalid evidence member declaration")
    verified = {}
    for kind in ("compact", "retained") if include_retained else ("compact",):
        record = manifest[kind]
        filename = "evidence.tar.gz" if kind == "compact" else "retained.tar"
        path = directory / filename
        if (
            record["path"] != filename
            or path.is_symlink()
            or path.stat().st_size != record["bytes"]
            or hash_file(path) != record["sha256"]
        ):
            raise ValueError(f"Evidence archive differs: {kind}")
        expected = {name: item for name, item in records.items() if item["storage"] == kind}
        index_seen = False
        with tarfile.open(path, "r|*") as tar:
            for member in tar:
                if kind == "compact" and member.name == "evidence-index.json":
                    if index_seen or not member.isfile():
                        raise ValueError("Invalid embedded evidence index")
                    raw = json.dumps(
                        {k: v for k, v in manifest.items() if k != "compact"}, sort_keys=True
                    ).encode()
                    if member.size != len(raw) or tar.extractfile(member).read(len(raw) + 1) != raw:
                        raise ValueError("Embedded evidence index differs")
                    index_seen = True
                    continue
                if not member.isfile() or member.name not in expected:
                    raise ValueError("Unexpected or duplicate evidence member")
                item = expected.pop(member.name)
                digest = hashlib.sha256()
                stream = tar.extractfile(member)
                for block in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(block)
                if member.size != item["bytes"] or digest.hexdigest() != item["sha256"]:
                    raise ValueError(f"Evidence member differs: {member.name}")
        if expected:
            raise ValueError("Evidence archive is missing declared members")
        if kind == "compact" and not index_seen:
            raise ValueError("Embedded evidence index missing")
        verified[kind] = record["sha256"]
    return {"verified": verified, "retained_verified": include_retained}
