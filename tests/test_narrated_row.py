"""Atomic, source-bound row explanations without authority over research state."""

import copy
import os

import pytest

from kernel_envolve.manager import narrator as n
from kernel_envolve.manager.row_facts import row_facts
from kernel_envolve.manager.views import task_view

pytest_plugins = ["test_manager"]


def reply(command, request, stop):
    result = {}
    for b in request["tasks"]:
        claim = {"text": "保留转置；局部改善，整链无收益。", "sources": [next(iter(b["sources"]))]}
        result[b["task"]] = dict(
            direction=None,
            finding=claim,
            failure=None,
            next=None,
            row=dict(
                revision=b["row_facts"]["revision"],
                benefit="system:benefit",
                optimization_pr="system:optimization_pr",
                current_progress="finding",
                latest_result=dict(ref="system:latest_result", **claim),
            ),
        )
    return result


def test_final_summary_once_and_archive_revision(managed):
    store, _, _ = managed
    store.event("toy", "research_context", {"text": "保留转置；局部改善，整链无收益。"})
    assert n.step(store, [], call=reply) == 1
    assert n.step(store, [], call=reply) == 0
    original = task_view(store, "toy")
    store.db.execute("UPDATE tasks SET desired='paused' WHERE id='toy'")
    calls = []

    def final(*args):
        calls.append(args[1])
        return reply(*args)

    assert n.step(store, [], call=final) == 1
    view = task_view(store, "toy")
    assert not calls[0]["tasks"][0]["can_continue"]
    assert view["narration"]["row"]["revision"] == view["row_facts"]["revision"]
    assert view["row_facts"]["revision"] != original["row_facts"]["revision"]
    assert n.step(store, [], call=final) == 0
    assert len(calls) == 1
    assert view["best"] is None and view["implementation_pr"] is None


@pytest.mark.parametrize(
    "change",
    [
        lambda r: r["row"].update(revision="stale"),
        lambda r: r["row"].update(benefit="2.0×"),
        lambda r: r["row"].update(optimization_pr="https://example.com/pr/9"),
        lambda r: r["row"]["latest_result"].update(ref="system:other"),
        lambda r: r["finding"].update(text="加速99.3%"),
        lambda r: r["finding"].update(text="结论" * 80),
        lambda r: r["row"]["latest_result"].update(sources=[["invalid"]]),
    ],
)
def test_fabricated_row_rejected(managed, change):
    store, _, _ = managed
    store.event("toy", "research_context", {"text": "局部改善"})

    def model(*args):
        v = reply(*args)
        change(v["toy"])
        return v

    assert n.step(store, [], call=model) == 0
    assert n.get(store, "narration:toy") is None
    assert n.get(store, "narrator_attempt")["status"] == "failed"


def test_late_model_response_is_history_not_current(managed):
    store, _, _ = managed
    store.event("toy", "research_context", {"text": "局部改善"})

    def model(*args):
        store.db.execute("UPDATE tasks SET desired='paused' WHERE id='toy'")
        return reply(*args)

    assert n.step(store, [], call=model) == 0
    assert n.get(store, "narration:toy") is None
    assert len(store.rows("SELECT * FROM metadata WHERE key LIKE 'narration_history:%'")) == 1
    assert n.step(store, [], call=reply) == 1


def test_new_validation_required_but_not_unrelated_older_one(managed):
    store, _, _ = managed
    store.event("toy", "research_context", {"text": "局部改善"})
    batch = n.collect(store)[0]
    batch["sources"]["v"] = dict(kind="validation", at=100, text="完整调用无收益")
    batch["row_facts"]["latest_result"] = dict(kind="validation", value={"recorded_at": 100})
    output = reply([], {"tasks": [batch]}, None)
    with pytest.raises(ValueError, match="New validation"):
        n.validate(output, [batch])
    output["toy"]["finding"]["sources"] = ["v"]
    output["toy"]["row"]["latest_result"]["sources"] = ["v"]
    n.validate(output, [batch])
    batch["row_facts"]["latest_result"] = dict(kind="research", value={"recorded_at": 200})
    n.validate(reply([], {"tasks": [batch]}, None), [batch])


def test_copy_mtime_does_not_reorder_experiment(managed):
    store, _, _ = managed
    job = store.schedule()[0]
    root = store.root / "jobs" / job
    root.mkdir(parents=True)
    path = root / "stdout.log"
    path.write_text("local measurement")
    os.utime(path, (2000000000, 2000000000))
    sources = n.collect(store)[0]["sources"]
    source = next(s for s in sources.values() if s["kind"] == "trace")
    actual = store.one("SELECT COALESCE(ended,started,created) AS at FROM jobs WHERE id=?", (job,))
    assert source["at"] == actual["at"] != path.stat().st_mtime


def test_real_best_is_not_removed_by_later_failed_attempt():
    task = dict(
        id="toy",
        spec={"baseline": {"revision": "base"}},
        best={"baseline_speedup": 1.2, "run": "accepted"},
        latest_validation={"recorded_at": 200, "status": "no_benefit", "source_sha256": "other"},
    )
    facts = row_facts(task)
    assert facts["benefit"]["kind"] == "retained_best"
    assert facts["benefit"]["value"]["baseline_speedup"] == 1.2
    assert facts["latest_result"]["value"]["status"] == "no_benefit"
    newer = copy.deepcopy(task)
    newer["implementation_pr"] = {"url": "https://example.com/pr/1", "updated_at": 201}
    assert row_facts(newer)["revision"] != facts["revision"]


def test_authoritative_source_survives_log_window_and_batch_is_bounded(managed):
    store, deployment, _ = managed
    # Separate tasks share one lightweight service; a noisy task cannot force a huge call.
    for index in range(3):
        other = copy.deepcopy(deployment)
        key = f"other{index}"
        other["tasks"][0]["id"] = key
        other["tasks"][0]["workspace"] += key
        store.register(other)
        store.event(key, "research_context", {"text": "结论" * 3900})
    store.event("toy", "research_context", {"text": "保留候选"})
    value = {"recorded_at": 100, "status": "no_benefit", "summary": "整链无收益"}
    n.put(store, "validation:toy", {"validation": value})
    for _ in range(30):
        store.event("toy", "bridge_execution", {"text": "log" * 2000})
    batches = n.collect(store)
    assert len(batches) < 4
    assert sum(len(n.encode(s).encode()) for b in batches for s in b["sources"].values()) <= 48000
    assert any(s["kind"] == "validation" for s in batches[0]["sources"].values())


def test_policy_refresh_recovers_context_after_a_failed_model_call(managed, monkeypatch):
    store, _, _ = managed
    store.event("toy", "research_context", {"text": "保留转置，放弃更慢分块"})

    def fail(*args):
        raise TimeoutError()

    assert n.step(store, [], call=fail) == 0
    assert n.collect(store) == []
    monkeypatch.setattr(n, "POLICY_REVISION", "new-policy")
    batch = n.collect(store)[0]
    assert any("放弃更慢分块" in source["text"] for source in batch["sources"].values())
    assert n.step(store, [], call=reply) == 1
