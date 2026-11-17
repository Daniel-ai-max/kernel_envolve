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


def test_failed_inference_retains_context_for_next_new_event(managed):
    store, _, _ = managed
    store.event("toy", "research_context", {"text": "转置候选保持原加法顺序"})

    def failed(*args):
        raise TimeoutError()

    assert n.step(store, [], call=failed) == 0
    assert n.collect(store) == []  # no automatic retry of identical input
    for _ in range(35):
        store.event("toy", "bridge_execution", {"text": "new experiment"})
    batch = n.collect(store)[0]
    assert any("保持原加法顺序" in s["text"] for s in batch["sources"].values())
    assert n.step(store, [], call=reply) == 1
    assert n.get(store, "narrator_context:toy")["sources"]
    assert n.collect(store) == []


def test_row_version_ignores_polling_duration_and_large_execution_payloads():
    task = dict(
        id="toy",
        spec={"baseline": {"revision": "base"}},
        recent_execution={
            "id": "job",
            "status": "running",
            "recorded_at": 10,
            "seconds": 20,
            "cases": [{"large": "x" * 10000}],
        },
    )
    first = row_facts(task)
    task["recent_execution"]["seconds"] = 21
    assert row_facts(task) == first
    assert "cases" not in first["execution"]
    task["recent_execution"]["status"] = "succeeded"
    assert row_facts(task)["revision"] != first["revision"]


@pytest.mark.parametrize(
    "source,display",
    [
        ("88.65015983581543", "88.65"),
        ("74.19519901275635", "74.20"),
        ("88.65015983581543", "88.7"),
        ("1.235", "1.24"),  # Decimal half-up, not binary float or ties-to-even.
        ("1.245", "1.25"),
        ("9.999", "10.0"),
        ("123456789012345678901234567890.125", "123456789012345678901234567890.13"),
    ],
)
def test_cited_display_rounding_preserves_sources_and_is_idempotent(source, display):
    batch = dict(
        task="toy",
        sources={"measurement": dict(kind="trace", at=1, text=f"耗时{source}微秒。")},
        row_facts={"revision": "revision", "latest_result": {"kind": "none"}},
    )
    request = {"tasks": [batch]}
    output = reply([], request, None)
    for claim in (output["toy"]["finding"], output["toy"]["row"]["latest_result"]):
        claim["text"] = f"局部耗时约{display}微秒。"
    original = copy.deepcopy((batch, output))
    assert n.validate(output, [batch]) == output
    assert n.validate(output, [batch]) == output
    assert (batch, output) == original


@pytest.mark.parametrize(
    "source,display",
    [
        ("88.65015983581543", "88.64"),  # Wrong neighbor, not tolerance matching.
        ("88.65015983581543", "88.66"),
        ("74.19519901275635", "74.19"),  # Truncation is not rounding.
        ("88.65015983581543", "88.650159"),
        ("88.65015983581543", "89"),  # Integer display not admitted.
        ("1.2345", "1.2"),  # Fewer than three significant digits.
        ("0.10499", "0.10"),
        ("88.65015983581543 74.19519901275635", "16.31%"),  # Derived gain.
        ("88.65015983581543微秒", "0.09毫秒"),  # Unit conversion.
        ("88.65015983581543微秒", "88650.16纳秒"),
        ("-88.65015983581543", "88.65"),
        ("88.65015983581543", "-88.65"),
        ("v88.65015983581543", "88.65"),
        ("88.65015983581543e2", "88.65"),
        ("88.65015983581543", "kernel88.65"),
        ("88.65015983581543", "88.65suffix"),
        ("88.65015983581543 4", "88.65e4"),
        ("88.65015983581543 4", "88.65E+4"),
    ],
)
def test_rounding_does_not_admit_arithmetic_truncation_or_number_fragments(source, display):
    assert not n.cited_numbers(display, source)


@pytest.mark.parametrize("sign", ["-", "+", "−", "±", "∓", "﹣", "－", "＋"])
@pytest.mark.parametrize("space", ["", " "])
def test_signed_rounding_is_rejected_in_both_citation_and_claim(sign, space):
    assert not n.cited_numbers(f"{sign}{space}88.65", "88.65015983581543")
    assert not n.cited_numbers("88.65", f"{sign}{space}88.65015983581543")


def test_exact_numbers_remain_valid_at_any_original_precision():
    assert n.cited_numbers("1.2 0.10 89 88.650159", "1.2 0.1 89 88.650159")
    # Only the new rounding path checks token boundaries; legacy exact behavior is unchanged.
    assert n.cited_numbers("kernel88.65 88.65e4 −88.65", "88.65 4")


def test_rounded_numbers_are_task_local_and_keep_revision_guard():
    batch = dict(
        task="toy",
        sources={"own": dict(kind="trace", at=1, text="耗时99.123微秒。")},
        row_facts={"revision": "revision", "latest_result": {"kind": "none"}},
    )
    other = copy.deepcopy(batch)
    other.update(task="other", sources={"foreign": dict(kind="trace", at=1, text="88.650159")})
    output = reply([], {"tasks": [batch, other]}, None)
    output["toy"]["finding"].update(text="耗时约88.65微秒。", sources=["foreign"])
    with pytest.raises(ValueError, match="task-local"):
        n.validate(output, [batch, other])
    output["toy"]["finding"]["sources"] = ["own"]
    with pytest.raises(n.UncitedNumberError):
        n.validate(output, [batch, other])
    output["toy"]["finding"]["text"] = "耗时约99.12微秒。"
    output["toy"]["row"]["revision"] = "stale"
    with pytest.raises(ValueError, match="current system facts"):
        n.validate(output, [batch, other])


def test_invalid_number_records_only_safe_error_code(managed):
    store, _, _ = managed
    store.event("toy", "research_context", {"text": "耗时74.195199微秒。"})

    def model(*args):
        output = reply(*args)
        output["toy"]["finding"]["text"] = "耗时约74.19微秒。"
        return output

    assert n.step(store, [], call=model) == 0
    attempt = n.get(store, "narrator_attempt")
    assert attempt["error_code"] == "numeric_not_cited"
    assert "74.19" not in n.encode(attempt)
    assert n.get(store, "narration:toy") is None
