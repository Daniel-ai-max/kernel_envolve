import json
import shutil
from pathlib import Path

import pytest


@pytest.fixture(autouse=True)
def local_evaluator(monkeypatch):
    # Tests exercise the working tree, never the public main or a user's cache.
    monkeypatch.setenv("KERNEL_ENVOLVE_AUTO_UPDATE", "0")
    monkeypatch.delenv("KERNEL_ENVOLVE_RELEASE", raising=False)
    monkeypatch.delenv("KERNEL_ENVOLVE_UPDATE_SESSION", raising=False)


@pytest.fixture
def task(tmp_path):
    source = Path(__file__).parents[1] / "examples" / "vector_add"
    for p in source.glob("*.py"):
        shutil.copy(p, tmp_path / p.name)
    cfg = json.loads((source / "task.json").read_text())
    cfg.update(full_trials=3, quick_trials=1, warmup=1)
    (tmp_path / "task.json").write_text(json.dumps(cfg))
    return tmp_path
