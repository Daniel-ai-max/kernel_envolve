"""Harness construction preserves the trusted overlay and obeys the run budget."""

import os
import shutil
from pathlib import Path

import pytest

from kernel_envolve import native_harness
from kernel_envolve.storage import file_hash


@pytest.fixture
def frozen(tmp_path):
    contents = {
        "baseline": {
            "tests/test_op.py": "trusted test",
            "pkg/op.py": "old implementation",
            "pkg/deleted.py": "removed implementation",
            "README.md": "trusted documentation",
            ".git/index": "fixed git index",
        },
        "candidate": {
            "tests/test_op.py": "trusted test",
            "pkg/op.py": "new implementation",
            "pkg/added.py": "new helper",
            "README.md": "candidate documentation",
            ".git/index": "fixed git index",
        },
    }
    identities = {}
    for role, files in contents.items():
        base = tmp_path / "sources" / role / "core"
        for name, value in files.items():
            path = base / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(value)
        identities[role] = {
            "core": {
                "git_head": role,
                "files": {name: file_hash(base / name) for name in files},
            }
        }
    cfg = {"implementation_paths": {"core": ["pkg"]}}
    return tmp_path, cfg, identities


def test_harness_matches_overlay_without_recopying(frozen, monkeypatch):
    run, cfg, identities = frozen
    baseline = run / "sources/baseline/core"
    candidate = run / "sources/candidate/core"
    # Independently reconstruct the previous overlay semantics for comparison.
    expected = run / "expected"
    shutil.copytree(baseline, expected)
    shutil.rmtree(expected / "pkg")
    shutil.copytree(candidate / "pkg", expected / "pkg")
    expected_files = {
        p.relative_to(expected).as_posix(): file_hash(p) for p in expected.rglob("*") if p.is_file()
    }
    real_copy, copied = shutil.copy2, []

    def tracked(source, target):
        copied.append(Path(source))
        return real_copy(source, target)

    monkeypatch.setattr(native_harness.shutil, "copy2", tracked)
    result, _ = native_harness.prepare_harness(run, cfg, identities)
    dest = run / "sources/validation/core"
    assert result["core"]["files"] == expected_files
    assert len(copied) == len(expected_files)
    assert not (dest / "pkg/deleted.py").exists()
    for name in expected_files:
        origin = candidate if name.startswith("pkg/") else baseline
        assert not os.path.samefile(dest / name, origin / name)
    (dest / "pkg/op.py").write_text("edited validation")
    assert (candidate / "pkg/op.py").read_text() == "new implementation"


@pytest.mark.parametrize("phase", ["before", "copy", "hash"])
def test_harness_budget_expires_without_copying_remaining_files(frozen, monkeypatch, phase):
    run, cfg, identities = frozen
    expired, copies = phase == "before", []
    real_copy, real_hash = shutil.copy2, file_hash

    def checkpoint():
        if expired:
            raise TimeoutError("budget exhausted")

    def copy(source, target):
        nonlocal expired
        copies.append(source)
        result = real_copy(source, target)
        expired = phase == "copy"
        return result

    def hash_file(path):
        nonlocal expired
        result = real_hash(path)
        expired = phase == "hash"
        return result

    monkeypatch.setattr(native_harness.shutil, "copy2", copy)
    monkeypatch.setattr(native_harness, "file_hash", hash_file)
    with pytest.raises(TimeoutError, match="budget exhausted"):
        native_harness.prepare_harness(run, cfg, identities, checkpoint)
    assert len(copies) == (0 if phase == "before" else 1)


@pytest.mark.parametrize("mutation", ["bytes", "symlink", "directory_symlink", "missing"])
def test_harness_rejects_changed_frozen_inputs(frozen, mutation):
    run, cfg, identities = frozen
    target = run / "sources/baseline/core/tests/test_op.py"
    if mutation == "bytes":
        target.write_text("changed test")
    elif mutation == "directory_symlink":
        target.parent.rename(target.parent.with_name("moved-tests"))
        target.parent.symlink_to(target.parent.with_name("moved-tests"), target_is_directory=True)
    else:
        target.unlink()
        if mutation == "symlink":
            target.symlink_to(run / "sources/candidate/core/tests/test_op.py")
    with pytest.raises((ValueError, FileNotFoundError)):
        native_harness.prepare_harness(run, cfg, identities)


def test_harness_preserves_deleting_whole_implementation(frozen):
    run, cfg, identities = frozen
    files = identities["candidate"]["core"]["files"]
    identities["candidate"]["core"]["files"] = {
        name: value for name, value in files.items() if not name.startswith("pkg/")
    }
    shutil.rmtree(run / "sources/candidate/core/pkg")
    result, _ = native_harness.prepare_harness(run, cfg, identities)
    assert not (run / "sources/validation/core/pkg").exists()
    assert "tests/test_op.py" in result["core"]["files"]
