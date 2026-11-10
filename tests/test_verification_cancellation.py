"""Cancellation propagates between source files without weakening content verification."""

from pathlib import Path

import pytest

from kernel_envolve import native, storage


class Cancelled(Exception):
    pass


@pytest.mark.parametrize("kind", ["native", "generic"])
def test_cancel_between_source_hashes(tmp_path, monkeypatch, kind):
    module = native if kind == "native" else storage
    source = tmp_path / ("sources/candidate/core" if kind == "native" else "source")
    source.mkdir(parents=True)
    files = {}
    for name in ("first.py", "second.py"):
        (source / name).write_text(name)
        files[name] = storage.file_hash(source / name)
    hashed = []
    original = module.file_hash

    def hash_once(path):
        hashed.append(Path(path).name)
        return original(path)

    def checkpoint():
        if hashed:
            raise Cancelled

    monkeypatch.setattr(module, "file_hash", hash_once)
    with pytest.raises(Cancelled):
        if kind == "native":
            native.verify_roots(tmp_path, {"core": {"files": files}}, "candidate", checkpoint)
        else:
            storage.verify_source(tmp_path, {"files": files}, checkpoint)
    assert hashed == ["first.py"]
