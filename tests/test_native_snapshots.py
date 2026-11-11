"""Snapshot identity and isolation survive metadata-operation reductions."""

import os
import shutil
import subprocess
from pathlib import Path

import pytest

from kernel_envolve import native
from kernel_envolve.storage import file_hash


@pytest.fixture
def source(tmp_path):
    root = tmp_path / "repo"
    root.mkdir()
    (root / "src").mkdir()
    (root / "src" / "op.py").write_text("result = 1\n")
    (root / "deleted.py").write_text("deleted = 1\n")
    (root / ".gitignore").write_text("ignored.so\n__pycache__/\n")
    for argv in (
        ["git", "init", "-q"],
        ["git", "add", "."],
        [
            "git",
            "-c",
            "user.name=Fixture",
            "-c",
            "user.email=test@example.test",
            "commit",
            "-qm",
            "snapshot fixture",
        ],
    ):
        subprocess.run(argv, cwd=root, check=True, capture_output=True)
    (root / "deleted.py").unlink()
    (root / "src" / "op.py").write_text("result = 2\n")
    (root / "new.py").write_text("new = 3\n")
    (root / "ignored.so").write_bytes(b"required compiled artifact")
    (root / "__pycache__").mkdir()
    (root / "__pycache__" / "op.pyc").write_bytes(b"stale cache")
    return root


def test_snapshot_preserves_all_contents_and_role_isolation(source, tmp_path):
    original = {
        name: file_hash(source / name)
        for name in (".gitignore", "src/op.py", "new.py", "ignored.so")
    }
    roles = {}
    for role in ("baseline", "candidate"):
        dest = tmp_path / "run" / "sources" / role / "core"
        identity = native.snapshot(source, dest, ["ignored.so"])
        assert identity["files"] == original
        assert (
            identity["git_head"]
            == subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=source, text=True).strip()
        )
        roles[role] = dest
        native.verify_roots(tmp_path / "run", {"core": identity}, role)
    for name in original:
        a, b = roles["baseline"] / name, roles["candidate"] / name
        assert not os.path.samefile(a, b)
        assert not os.path.samefile(a, source / name)
    (roles["candidate"] / "src/op.py").write_text("result = 9\n")
    assert file_hash(roles["baseline"] / "src/op.py") == original["src/op.py"]
    assert file_hash(source / "src/op.py") == original["src/op.py"]


@pytest.fixture
def frozen(source, tmp_path):
    run = tmp_path / "run"
    dest = run / "sources" / "candidate" / "core"
    identity = native.snapshot(source, dest, ["ignored.so"])
    return run, dest, {"core": identity}


def test_every_verification_rehashes_all_files_even_with_unchanged_metadata(frozen, monkeypatch):
    run, dest, identities = frozen
    hashes = []

    def traced(path):
        hashes.append(Path(path).relative_to(dest).as_posix())
        return file_hash(path)

    monkeypatch.setattr(native, "file_hash", traced)
    for _ in range(2):
        hashes.clear()
        native.verify_roots(run, identities, "candidate")
        assert sorted(hashes) == sorted(identities["core"]["files"])
    path = dest / "src/op.py"
    before = path.stat()
    path.write_text("result = 9\n")
    os.utime(path, ns=(before.st_atime_ns, before.st_mtime_ns))
    assert path.stat().st_size == before.st_size
    assert path.stat().st_mtime_ns == before.st_mtime_ns
    with pytest.raises(ValueError, match="Frozen source changed"):
        native.verify_roots(run, identities, "candidate")


@pytest.mark.parametrize(
    "mutation", ["delete", "directory", "file-link", "directory-link", "broken-link", "root-link"]
)
def test_changed_types_and_links_are_rejected(frozen, tmp_path, mutation):
    run, dest, identities = frozen
    path = dest / "src/op.py"
    if mutation == "delete":
        path.unlink()
    elif mutation == "directory":
        path.unlink()
        path.mkdir()
    elif mutation == "file-link":
        target = tmp_path / "same-content.py"
        target.write_bytes(path.read_bytes())
        path.unlink()
        path.symlink_to(target)
    elif mutation == "directory-link":
        (dest / "injected").symlink_to(tmp_path, target_is_directory=True)
    elif mutation == "broken-link":
        (dest / "dangling").symlink_to(tmp_path / "absent")
    else:
        target = tmp_path / "moved-source"
        dest.rename(target)
        dest.symlink_to(target, target_is_directory=True)
    with pytest.raises(ValueError):
        native.verify_roots(run, identities, "candidate")


@pytest.mark.parametrize("suffix", [".py", ".so", ".cu", ".cuh", ".cpp", ".pyc", ".pyo"])
def test_unrecorded_executable_is_rejected_at_any_depth(frozen, suffix):
    run, dest, identities = frozen
    nested = dest / "generated" / "nested"
    nested.mkdir(parents=True)
    (nested / ("injected" + suffix)).write_bytes(b"unrecorded")
    with pytest.raises(ValueError, match="Unrecorded executable source"):
        native.verify_roots(run, identities, "candidate")


def test_reports_and_empty_directories_are_still_allowed(frozen):
    run, dest, identities = frozen
    (dest / "reports").mkdir()
    (dest / "reports" / "result.json").write_text('{"passed": true}')
    (dest / "empty.py").mkdir()
    native.verify_roots(run, identities, "candidate")


def test_scan_errors_propagate(frozen, monkeypatch):
    run, _, identities = frozen

    def fail(_):
        raise PermissionError("cannot read source directory")

    monkeypatch.setattr(native.os, "scandir", fail)
    with pytest.raises(PermissionError, match="cannot read source"):
        native.verify_roots(run, identities, "candidate")


def test_cancellation_after_last_hash_is_not_lost(frozen, monkeypatch):
    run, _, identities = frozen
    count = 0

    def traced(path):
        nonlocal count
        count += 1
        return file_hash(path)

    def checkpoint():
        if count == len(identities["core"]["files"]):
            raise InterruptedError("cancelled after final hash")

    monkeypatch.setattr(native, "file_hash", traced)
    with pytest.raises(InterruptedError, match="final hash"):
        native.verify_roots(run, identities, "candidate", checkpoint)


def test_source_symlinks_and_missing_required_artifacts_rejected(source, tmp_path):
    (source / "new.py").unlink()
    (source / "new.py").symlink_to(source / "src/op.py")
    with pytest.raises(ValueError, match="Source symlink"):
        native.snapshot(source, tmp_path / "linked")
    with pytest.raises(ValueError, match="Required compiled artifact missing"):
        native.snapshot(source, tmp_path / "missing", ["absent.so"])


def test_verification_detects_removed_root(frozen):
    run, dest, identities = frozen
    shutil.rmtree(dest)
    with pytest.raises(FileNotFoundError):
        native.verify_roots(run, identities, "candidate")
