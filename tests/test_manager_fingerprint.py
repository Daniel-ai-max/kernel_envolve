"""Fast metadata invalidation must still send changed evidence through content hashes."""

import contextlib
import os

import pytest

from kernel_envolve.cli import evaluate, promote
from kernel_envolve.manager import evidence
from kernel_envolve.manager.views import detail
from kernel_envolve.storage import digest

pytest_plugins = ["test_manager"]


def previous_entries(root):
    return [
        (str(p.relative_to(root)), p.stat().st_size, p.stat().st_mtime_ns)
        for p in sorted(root.rglob("*"))
        if p.is_file() and "__pycache__" not in p.parts
    ]


def test_same_file_coverage_and_order_independent_signature(tmp_path, monkeypatch):
    for name in ("a.py", "a/deep/kernel.py", ".git/index", "数据/参数.json"):
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(name)
    cache = tmp_path / "__pycache__/nested/ignored.pyc"
    cache.parent.mkdir(parents=True)
    cache.write_bytes(b"generated")
    (tmp_path / "empty").mkdir()
    observed = []

    def capture(rows):
        observed.extend(rows)
        return digest(rows)

    monkeypatch.setattr(evidence, "digest", capture)
    first = evidence.fingerprint(tmp_path)
    assert sorted(row[:3] for row in observed) == sorted(previous_entries(tmp_path))
    original_scandir = os.scandir

    @contextlib.contextmanager
    def reverse_scandir(path):
        with original_scandir(path) as entries:
            yield iter(reversed(list(entries)))

    monkeypatch.setattr(evidence.os, "scandir", reverse_scandir)
    assert evidence.fingerprint(tmp_path) == first
    cache.write_bytes(b"a different generated cache")
    assert evidence.fingerprint(tmp_path) == first


@pytest.mark.parametrize(
    "change", ["write", "preserve_mtime", "replace", "add", "delete", "rename"]
)
def test_metadata_changes_invalidate(tmp_path, change):
    path = tmp_path / "kernel.py"
    path.write_text("return 1\n")
    before_stat = path.stat()
    before = evidence.fingerprint(tmp_path)
    if change in {"write", "preserve_mtime"}:
        path.write_text("return 2\n")  # Same length must not evade invalidation.
        if change == "preserve_mtime":
            os.utime(path, ns=(before_stat.st_atime_ns, before_stat.st_mtime_ns))
            assert path.stat().st_mtime_ns == before_stat.st_mtime_ns
    elif change == "replace":
        other = tmp_path / "replacement"
        other.write_text("return 2\n")
        os.utime(other, ns=(before_stat.st_atime_ns, before_stat.st_mtime_ns))
        other.replace(path)
    elif change == "add":
        (tmp_path / "injected.py").write_text("return 2\n")
    elif change == "delete":
        path.unlink()
    else:
        path.rename(tmp_path / "renamed.py")
    assert evidence.fingerprint(tmp_path) != before
    assert evidence.fingerprint(tmp_path) == evidence.fingerprint(tmp_path)


@pytest.mark.parametrize("kind", ["file", "directory", "dangling", "cache"])
def test_symlinks_are_rejected_without_following_them(tmp_path, kind):
    root = tmp_path / "run"
    root.mkdir()
    target = tmp_path / "external"
    target.mkdir()
    (target / "kernel.py").write_text("return 1\n")
    link = root / ("__pycache__" if kind == "cache" else "link")
    destination = target / "kernel.py" if kind == "file" else target
    if kind == "dangling":
        destination = tmp_path / "missing"
    link.symlink_to(destination, target_is_directory=kind in {"directory", "cache"})
    with pytest.raises(ValueError, match="Symlink introduced"):
        evidence.fingerprint(root)


def test_run_root_cannot_be_replaced_with_a_symlink(tmp_path):
    root = tmp_path / "run"
    root.mkdir()
    (root / "kernel.py").write_text("return 1\n")
    evidence.fingerprint(root)
    moved = tmp_path / "moved"
    root.rename(moved)
    root.symlink_to(moved, target_is_directory=True)
    with pytest.raises(ValueError, match="real directory"):
        evidence.fingerprint(root)


def test_existing_directory_replaced_with_link_is_rejected(tmp_path):
    root = tmp_path / "run"
    source = root / "source"
    source.mkdir(parents=True)
    (source / "kernel.py").write_text("return 1\n")
    evidence.fingerprint(root)
    external = tmp_path / "external"
    source.rename(external)
    source.symlink_to(external, target_is_directory=True)
    with pytest.raises(ValueError, match="Symlink introduced"):
        evidence.fingerprint(root)


def test_deep_directory_and_missing_tree(tmp_path):
    path = tmp_path
    for _ in range(150):
        path /= "d"
        path.mkdir()
    leaf = path / "kernel.py"
    leaf.write_text("return 1\n")
    before = evidence.fingerprint(tmp_path)
    leaf.write_text("return 2\n")
    assert evidence.fingerprint(tmp_path) != before
    with pytest.raises(FileNotFoundError):
        evidence.fingerprint(tmp_path / "missing")


def test_each_entry_has_one_metadata_read_and_cache_subtrees_are_pruned(
    tmp_path, monkeypatch, record_property
):
    for directory in range(10):
        folder = tmp_path / str(directory)
        folder.mkdir()
        for item in range(20):
            (folder / f"{item}.py").write_text("source")
    cache = tmp_path / "__pycache__/nested"
    cache.mkdir(parents=True)
    (cache / "ignored.pyc").write_bytes(b"generated")
    counts = {"path_stat": 0, "entry_stat": 0, "scandir": 0}
    original_stat, original_scandir = os.stat, os.scandir

    def path_stat(*args, **kwargs):
        counts["path_stat"] += 1
        return original_stat(*args, **kwargs)

    class Entry:
        def __init__(self, entry):
            self.entry = entry

        def __getattr__(self, name):
            return getattr(self.entry, name)

        def stat(self, *, follow_symlinks=True):
            counts["entry_stat"] += 1
            assert follow_symlinks is False
            return self.entry.stat(follow_symlinks=False)

    @contextlib.contextmanager
    def scandir(path):
        counts["scandir"] += 1
        with original_scandir(path) as entries:
            yield (Entry(entry) for entry in entries)

    monkeypatch.setattr(os, "stat", path_stat)
    previous_entries(tmp_path)
    old_reads = counts["path_stat"]
    assert old_reads >= 600  # is_file plus two stat calls for each of 200 source files.
    counts.update(path_stat=0, entry_stat=0, scandir=0)
    monkeypatch.setattr(os, "scandir", scandir)
    evidence.fingerprint(tmp_path)
    assert counts == {"path_stat": 1, "entry_stat": 211, "scandir": 11}
    new_reads = counts["path_stat"] + counts["entry_stat"]
    assert new_reads < old_reads / 2
    record_property("previous_metadata_reads", old_reads)
    record_property("scandir_metadata_reads", new_reads)
    record_property("scandir_directory_reads", counts["scandir"])


@pytest.mark.parametrize("change", ["preserve_mtime", "symlink", "add", "delete"])
def test_cached_verified_best_rechecks_real_changes(managed, monkeypatch, change):
    store, _, work = managed
    run, result = evaluate(work, "full")
    assert result["decision"]["eligible_for_best"], result["decision"]
    promote(run)
    inspect = evidence.inspect_run
    calls = []

    def counted_inspect(path, spec, checkpoint=None):
        calls.append(path)
        return inspect(path, spec, checkpoint)

    monkeypatch.setattr(evidence, "inspect_run", counted_inspect)
    evidence.refresh(store)
    assert detail(store, "toy")["best"] is not None
    evidence.refresh(store)
    assert len(calls) == 1  # An unchanged tree reuses verified evidence.
    source = run / "source/candidate.py"
    before = source.stat()
    if change == "preserve_mtime":
        source.write_text(source.read_text().replace("x+1", "x+2"))
        os.utime(source, ns=(before.st_atime_ns, before.st_mtime_ns))
        assert source.stat().st_size == before.st_size
        assert source.stat().st_mtime_ns == before.st_mtime_ns
    elif change == "symlink":
        saved = work / "external.py"
        saved.write_bytes(source.read_bytes())
        os.utime(saved, ns=(before.st_atime_ns, before.st_mtime_ns))
        source.unlink()
        source.symlink_to(saved)
    elif change == "add":
        (run / "source/injected.py").write_text("new = True\n")
    else:
        source.unlink()
    evidence.refresh(store)
    assert detail(store, "toy")["best"] is None
    assert store.task("toy")["evidence_error"]
    assert len(calls) == (1 if change == "symlink" else 2)
