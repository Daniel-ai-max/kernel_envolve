import json
import shutil
from pathlib import Path

import pytest


@pytest.fixture
def task(tmp_path):
    source = Path(__file__).parents[1] / "examples" / "vector_add"
    for p in source.glob("*.py"):
        shutil.copy(p, tmp_path / p.name)
    cfg = json.loads((source / "task.json").read_text())
    cfg.update(full_trials=3, quick_trials=1, warmup=1)
    (tmp_path / "task.json").write_text(json.dumps(cfg))
    return tmp_path
