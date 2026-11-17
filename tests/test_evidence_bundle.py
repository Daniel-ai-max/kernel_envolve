"""Evidence transport preserves bytes without making evaluation claims."""

import json
import os
import shutil
import tarfile

import pytest

from kernel_envolve._io import hash_file
from kernel_envolve.manager.results import bundle


def fixture(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    (source / "checkpoint.bin").write_bytes(os.urandom(2 * 1024 * 1024))
    (source / "candidate.py").write_text("def kernel(x): return x\n")
    (source / "stdout.log").write_text("native test passed\n")
    (source / "small.pt").write_bytes(b"small native proof")
    return source


def select(name, size):
    return "generated checkpoint retained at origin" if name == "native/checkpoint.bin" else None


def test_retained_bytes_verified_before_portable_handoff(tmp_path):
    source = fixture(tmp_path)
    before = {p.name: hash_file(p) for p in source.iterdir()}
    output = tmp_path / "bundle"
    manifest = bundle.pack_evidence({"native": source}, output, retain=select)
    assert {p.name: hash_file(p) for p in source.iterdir()} == before
    assert manifest["files"]["native/checkpoint.bin"]["storage"] == "retained"
    assert manifest["files"]["native/small.pt"]["storage"] == "compact"
    assert manifest["compact"]["bytes"] < 10000
    assert bundle.verify_evidence(output, include_retained=True)["retained_verified"]
    moved = tmp_path / "host"
    moved.mkdir()
    for name in ("evidence.json", "evidence.tar.gz"):
        shutil.copyfile(output / name, moved / name)
    assert bundle.verify_evidence(moved)["retained_verified"] is False
    with pytest.raises(FileNotFoundError):
        bundle.verify_evidence(moved, include_retained=True)
    with tarfile.open(output / "retained.tar") as tar:
        assert (
            tar.extractfile("native/checkpoint.bin").read()
            == (source / "checkpoint.bin").read_bytes()
        )


def test_no_implicit_size_or_suffix_filter_and_no_symlink_traversal(tmp_path):
    source = fixture(tmp_path)
    (source / "alias").symlink_to(source / "candidate.py")
    (source / "external").symlink_to(tmp_path)
    os.link(source / "stdout.log", source / "hardlink.log")
    result = bundle.pack_evidence({"native": source}, tmp_path / "out")
    assert all(x["storage"] == "compact" for x in result["files"].values())
    assert {x["path"] for x in result["links"]} == {"native/alias", "native/external"}
    with tarfile.open(tmp_path / "out/evidence.tar.gz") as tar:
        assert all(m.isfile() for m in tar)
    assert bundle.verify_evidence(tmp_path / "out", include_retained=True)["retained_verified"]


def test_exclusions_are_explicit_and_output_never_overlaps_inputs(tmp_path):
    source = fixture(tmp_path)
    cache = source / "cache"
    cache.mkdir()
    (cache / "file").write_bytes(b"generated cache")
    with pytest.raises(ValueError, match="overlaps"):
        bundle.pack_evidence({"source": source}, source / "out")
    with pytest.raises(ValueError, match="overlap"):
        bundle.pack_evidence({"source": source, "again": cache}, tmp_path / "out")
    result = bundle.pack_evidence(
        {"source": source},
        tmp_path / "out",
        exclude=lambda name: "rebuildable cache" if name == "source/cache" else None,
    )
    assert result["excluded"] == [{"path": "source/cache", "reason": "rebuildable cache"}]
    assert not any("cache" in name for name in result["files"])
    with pytest.raises(FileExistsError):
        bundle.pack_evidence({"source": source}, tmp_path / "out")


def test_retained_write_failure_does_not_publish_success(tmp_path, monkeypatch):
    source = fixture(tmp_path)

    def fail(self, data):
        raise OSError("storage full")

    monkeypatch.setattr(bundle._Writer, "write", fail)
    with pytest.raises(OSError, match="storage full"):
        bundle.pack_evidence({"native": source}, tmp_path / "out", retain=select)
    assert not (tmp_path / "out/evidence.json").exists()
    assert not (tmp_path / "out/evidence.tar.gz").exists()


def test_source_mutation_during_copy_is_rejected(tmp_path, monkeypatch):
    source = fixture(tmp_path)
    original = bundle._Reader.read

    def mutate(self, size):
        value = original(self, size)
        with (source / "checkpoint.bin").open("ab") as stream:
            stream.write(b"changed")
        return value

    monkeypatch.setattr(bundle._Reader, "read", mutate)
    with pytest.raises(ValueError, match="changed during copy"):
        bundle.pack_evidence({"native": source}, tmp_path / "out", retain=select)
    assert not (tmp_path / "out/evidence.json").exists()


@pytest.mark.parametrize("kind", ["compact", "retained"])
def test_archive_corruption_is_detected(tmp_path, kind):
    source = fixture(tmp_path)
    output = tmp_path / "out"
    result = bundle.pack_evidence({"native": source}, output, retain=select)
    with (output / result[kind]["path"]).open("ab") as stream:
        stream.write(b"corruption")
    with pytest.raises(ValueError, match="archive differs"):
        bundle.verify_evidence(output, include_retained=True)


def test_missing_member_and_bad_selection_are_rejected(tmp_path):
    source = fixture(tmp_path)
    output = tmp_path / "out"
    result = bundle.pack_evidence({"native": source}, output)
    result["files"]["native/missing.log"] = dict(result["files"]["native/stdout.log"])
    (output / "evidence.json").write_text(json.dumps(result))
    with pytest.raises(ValueError, match="index differs"):
        bundle.verify_evidence(output)
    with pytest.raises(ValueError, match="nonempty reason"):
        bundle.pack_evidence({"native": source}, tmp_path / "bad", retain=lambda *_: True)


@pytest.mark.parametrize("label", ["../bad", "a/b", "", ".", ".."])
def test_unsafe_root_labels_are_rejected(tmp_path, label):
    source = fixture(tmp_path)
    with pytest.raises(ValueError):
        bundle.pack_evidence({label: source}, tmp_path / "out")


def test_parent_sync_failure_does_not_publish_success(tmp_path, monkeypatch):
    source = fixture(tmp_path)
    original = bundle.sync_dir

    def sync(path):
        if path == tmp_path:
            raise OSError("parent directory sync failed")
        return original(path)

    monkeypatch.setattr(bundle, "sync_dir", sync)
    with pytest.raises(OSError, match="parent directory sync failed"):
        bundle.pack_evidence({"native": source}, tmp_path / "out", retain=select)
    assert not (tmp_path / "out/evidence.json").exists()


def test_missing_tar_member_is_rejected_even_with_updated_archive_checksum(tmp_path):
    source = fixture(tmp_path)
    output = tmp_path / "out"
    result = bundle.pack_evidence({"native": source}, output, retain=select)
    changed = output / "changed.tar.gz"
    with tarfile.open(output / "evidence.tar.gz") as original, tarfile.open(changed, "w:gz") as new:
        for member in original:
            if member.name != "native/stdout.log":
                new.addfile(member, original.extractfile(member))
    changed.replace(output / "evidence.tar.gz")
    result["compact"].update(
        bytes=(output / "evidence.tar.gz").stat().st_size,
        sha256=hash_file(output / "evidence.tar.gz"),
    )
    (output / "evidence.json").write_text(json.dumps(result))
    with pytest.raises(ValueError, match="missing declared"):
        bundle.verify_evidence(output)
