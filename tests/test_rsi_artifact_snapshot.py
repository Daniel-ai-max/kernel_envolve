"""Corruption and compatibility boundaries of the one-read artifact view."""

import copy
import json
import os

import pytest

from kernel_envolve.rsi.artifact_snapshot import MAX_SNAPSHOT_BYTES, proposal
from kernel_envolve.rsi.research import source_id
from kernel_envolve.storage import digest


@pytest.fixture
def snapshot(tmp_path):
    cfg = {
        "reference_files": ["task.py"],
        "candidate_files": ["candidate.py"],
        "device": "cpu",
        "rtol": 1e-5,
        "atol": 1e-8,
        "equal_nan": False,
        "mutable_inputs": False,
        "deterministic": True,
    }
    environment = {
        "packages": {"numpy": "fixture"},
        "gpu": None,
        "runtime_settings": {"numpy_errors": {"over": "warn"}},
    }
    result = {
        "mode": "full",
        "execution": "completed",
        "correctness": "pass",
        "signal_only": False,
        "check_only": False,
        "selection": {},
        "identity": {"evaluator_hash": "e" * 64, "files": {"task.py": "a" * 64}},
        "environment": environment,
    }
    source = "def run(x): return x + 1\n"
    value = {
        "schema": 1,
        "profile": {"tags": ["pointwise"], "capabilities": []},
        "compatibility": cfg,
        "runtime": environment,
        "evaluator_hash": "e" * 64,
        "origin_contract": "b" * 64,
        "confirmation_result_hash": "c" * 64,
        "proposal": {
            "episode_id": "d" * 64,
            "source_hash": source_id(source),
            "task_source_hash": "a" * 64,
            "source": source,
        },
    }
    path = tmp_path / "snapshot.json"
    path.write_text(json.dumps(value))
    return path, value, cfg, result


def test_one_read_without_loading_archive_or_probing_runtime(snapshot, monkeypatch):
    from kernel_envolve.rsi import artifact_snapshot as module

    path, value, cfg, result = snapshot
    original = module.read
    calls = []

    def counted(*args):
        calls.append(args)
        return original(*args)

    monkeypatch.setattr(module, "read", counted)
    monkeypatch.setattr(
        module, "build", lambda *a, **kw: pytest.fail("No foreground episode verification")
    )
    actual, reason = proposal(path, digest(value), {"tags": ["pointwise", "strided"]}, cfg, result)
    assert actual == value["proposal"] and reason == "compatible_proposal"
    assert calls == [(path, MAX_SNAPSHOT_BYTES)]


@pytest.mark.parametrize(
    "change",
    [
        "runtime",
        "evaluator",
        "reference",
        "precision",
        "profile",
        "wrong-initial",
        "selected-subset",
    ],
)
def test_incompatible_current_result_falls_back(snapshot, change):
    path, value, cfg, result = snapshot
    cfg, result = copy.deepcopy(cfg), copy.deepcopy(result)
    facts = {"tags": ["pointwise"]}
    if change == "runtime":
        result["environment"]["packages"]["numpy"] = "changed"
    elif change == "evaluator":
        result["identity"]["evaluator_hash"] = "f" * 64
    elif change == "reference":
        result["identity"]["files"]["task.py"] = "f" * 64
    elif change == "precision":
        cfg["rtol"] = 0.1
    elif change == "profile":
        facts = {}
    elif change == "wrong-initial":
        result["correctness"] = "fail"
    else:
        result["selection"] = {"cases": ["subset"]}
    actual, reason = proposal(path, digest(value), facts, cfg, result)
    assert actual is None and reason != "compatible_proposal"


@pytest.mark.parametrize(
    "kind",
    [
        "missing",
        "truncated",
        "oversized",
        "symlink",
        "fifo",
        "changed-source",
        "wrong-pin",
        "untrusted-field",
    ],
)
def test_corrupt_optional_input_never_executes_or_blocks(snapshot, kind):
    path, value, cfg, result = snapshot
    identity = digest(value)
    if kind == "missing":
        path.unlink()
    elif kind == "truncated":
        path.write_text("{")
    elif kind == "oversized":
        path.write_text("x" * (MAX_SNAPSHOT_BYTES + 1))
    elif kind == "symlink":
        target = path.with_suffix(".target")
        path.rename(target)
        path.symlink_to(target)
    elif kind == "fifo":
        path.unlink()
        os.mkfifo(path)
    elif kind == "changed-source":
        value["proposal"]["source"] = 'raise AssertionError("never execute")'
        path.write_text(json.dumps(value))
    elif kind == "wrong-pin":
        identity = "f" * 64
    else:
        value["run_command"] = "not a permitted capability"
        path.write_text(json.dumps(value))
        identity = digest(value)
    assert proposal(path, identity, {"tags": ["pointwise"]}, cfg, result)[0] is None


def test_unicode_source_must_fit_actual_research_transport(snapshot):
    path, value, cfg, result = snapshot
    source = "# " + "算" * 6000 + "\ndef run(x): return x+1\n"
    value["proposal"].update(source=source, source_hash=source_id(source))
    raw = json.dumps(value, ensure_ascii=False)
    assert len(raw.encode()) < MAX_SNAPSHOT_BYTES
    assert len(json.dumps(value["proposal"]).encode()) > MAX_SNAPSHOT_BYTES
    path.write_text(raw)
    assert proposal(path, digest(value), {"tags": ["pointwise"]}, cfg, result) == (
        None,
        "source_mismatch",
    )
