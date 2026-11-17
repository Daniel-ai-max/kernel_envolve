"""Public synthetic regressions for completed evidence versus live execution state."""

import copy

import pytest
from test_narrated_row import reply

from kernel_envolve.manager import narrator as n
from kernel_envolve.manager.row_facts import row_facts
from kernel_envolve.manager.views import task_view

pytest_plugins = ["test_manager"]


def gpu(store, status="queued", started=None, ended=None):
    key = store.enqueue(store.task("toy"), "gpu", ["python", "synthetic_probe.py"])
    store.db.execute(
        "UPDATE jobs SET status=?,started=?,ended=? WHERE id=?", (status, started, ended, key)
    )
    return key


def revision(store):
    return task_view(store, "toy")["row_facts"]["revision"]


def explain(store):
    store.event("toy", "research_context", {"text": "合成局部测量完成，保留候选；尚无整链收益。"})
    assert n.step(store, [], call=reply) == 1
    return revision(store)


@pytest.mark.parametrize("terminal", ["succeeded", "failed", "cancelled", "interrupted"])
def test_running_job_keeps_conclusion_and_terminal_job_invalidates(managed, terminal):
    store, _, _ = managed
    gpu(store, "succeeded", 10, 20)
    before = explain(store)
    key = gpu(store)
    assert revision(store) == before
    assert n.collect(store) == []  # queued phase alone is not new model input
    store.db.execute("UPDATE jobs SET status='running',started=30 WHERE id=?", (key,))
    assert task_view(store, "toy")["phase"] == "gpu_running"
    assert revision(store) == before
    assert n.collect(store) == []
    view = task_view(store, "toy")
    assert view["narration"]["row"]["revision"] == view["row_facts"]["revision"]
    # A trace delta still triggers inference while the stable conclusion version stays unchanged.
    folder = store.root / "jobs" / key
    folder.mkdir(parents=True)
    (folder / "stdout.log").write_text("新的合成局部测量，未形成正式收益。")
    batch = n.collect(store)[0]
    assert any(s["job"] == key for s in batch["sources"].values() if s["kind"] == "trace")
    assert batch["row_facts"]["revision"] == before

    def trace_reply(*args):
        value = reply(*args)
        sources = args[1]["tasks"][0]["sources"]
        ref = next(k for k, s in sources.items() if s.get("job") == key)
        value["toy"]["finding"] = {"text": "新的合成局部测量，未形成正式收益。", "sources": [ref]}
        return value

    assert n.step(store, [], call=trace_reply) == 1
    store.db.execute(
        "UPDATE jobs SET status=?,ended=40,error=? WHERE id=?",
        (terminal, "Timed out" if terminal == "failed" else None, key),
    )
    assert revision(store) != before
    assert n.collect(store)[0]["row_facts"]["revision"] == revision(store)
    assert task_view(store, "toy")["narration"]["row"]["revision"] != revision(store)


def test_cancelled_before_start_is_a_new_terminal_observation(managed):
    store, _, _ = managed
    before = explain(store)
    gpu(store, "cancelled", None, 30)
    assert revision(store) != before


def test_late_inference_cannot_replace_a_new_completed_result(managed):
    store, _, _ = managed
    before = explain(store)
    key = gpu(store, "running", 30)
    store.event("toy", "research_context", {"text": "新的合成实验正在执行。"})

    def late(*args):
        store.db.execute("UPDATE jobs SET status='failed',ended=40 WHERE id=?", (key,))
        return reply(*args)

    assert n.step(store, [], call=late) == 0
    assert revision(store) != before
    assert n.get(store, "narration:toy")["row"]["revision"] == before
    assert len(store.rows("SELECT * FROM metadata WHERE key LIKE 'narration_history:%'")) == 2
    assert n.step(store, [], call=reply) == 1
    assert n.get(store, "narration:toy")["row"]["revision"] == revision(store)


def cpu(store, parent, request, at, status, *, saved=True):
    value = dict(
        task="toy",
        kind="cpu_request",
        round=parent,
        request=request,
        status=status,
        recorded_at=at,
        command=["python", "synthetic_cpu.py"],
        returncode=0 if status == "succeeded" else None,
    )
    n.put(store, "cpu_activity:toy", value)
    if saved and status != "running":
        n.put(store, "cpu_completed:toy", value)
    return value


@pytest.mark.parametrize("terminal", ["succeeded", "failed", "cancelled", "interrupted"])
def test_cpu_gpu_completion_order_survives_new_running_requests(managed, terminal):
    store, _, _ = managed
    parent = store.enqueue(store.task("toy"), "agent", ["python", "synthetic_agent.py"])
    store.db.execute("UPDATE jobs SET status='running',started=1 WHERE id=?", (parent,))
    gpu(store, "succeeded", 5, 20)
    cpu(store, parent, "first", 25, terminal)
    before = explain(store)
    assert task_view(store, "toy")["latest_completed_execution"]["request"] == "first"
    cpu(store, parent, "second", 35, "running")
    assert revision(store) == before
    # A GPU that started earlier but completes later is the new latest outcome.
    gpu(store, "failed", 10, 40)
    after_gpu = revision(store)
    assert after_gpu != before
    assert task_view(store, "toy")["latest_completed_execution"]["kind"] == "gpu_job"
    cpu(store, parent, "second", 45, "succeeded")
    assert revision(store) != after_gpu
    assert task_view(store, "toy")["latest_completed_execution"]["request"] == "second"


def test_legacy_cpu_completion_event_remains_stable_after_new_request(managed):
    store, _, _ = managed
    parent = store.enqueue(store.task("toy"), "agent", ["python"])
    store.db.execute("UPDATE jobs SET status='running',started=1 WHERE id=?", (parent,))
    value = cpu(store, parent, "old", 20, "succeeded", saved=False)
    store.event("toy", "bridge_execution", value | {"cleanup_confirmed": True})
    # Legacy events did not include recorded_at. Their own durable timestamp is used consistently.
    store.db.execute(
        "UPDATE events SET data=json_remove(data,'$.recorded_at'),at=21 WHERE kind='bridge_execution'"
    )
    before = explain(store)
    assert task_view(store, "toy")["latest_completed_execution"]["recorded_at"] == 21
    cpu(store, parent, "new", 30, "running")
    assert revision(store) == before
    assert n.collect(store) == []


@pytest.mark.parametrize("desired", ["paused", "budget", "blocked", "superseded"])
def test_stopped_or_finished_research_invalidates_conclusion(managed, desired):
    store, _, _ = managed
    before = explain(store)
    store.db.execute("UPDATE tasks SET desired=? WHERE id='toy'", (desired,))
    assert revision(store) != before


@pytest.mark.parametrize(
    "field,value",
    [
        ("best", {"run": "new", "baseline_speedup": 1.1}),
        ("implementation_pr", {"url": "https://example.test/pr/1", "head": "new"}),
        ("latest_validation", {"recorded_at": 25, "status": "no_benefit", "source_sha256": "new"}),
        ("evidence_error", "candidate changed"),
        ("delivery", {"task": "delivery", "status": "failed", "policy_phase": "exhausted"}),
        ("research_result", {"recorded_at": 26, "summary": "new completed conclusion"}),
    ],
)
def test_authoritative_result_changes_remain_strongly_bound(field, value):
    task = dict(
        id="toy", desired="active", phase="agent_running", spec={"baseline": {"revision": "base"}}
    )
    before = row_facts(task)["revision"]
    updated = copy.deepcopy(task)
    updated[field] = value
    assert row_facts(updated)["revision"] != before


def test_killed_cpu_parent_retains_interruption_before_next_round(managed):
    store, _, _ = managed
    parent = store.enqueue(store.task("toy"), "agent", ["python"])
    store.db.execute("UPDATE jobs SET status='running',started=1 WHERE id=?", (parent,))
    gpu(store, "succeeded", 5, 20)
    before = explain(store)
    cpu(store, parent, "shared-request-id", 30, "running")
    assert revision(store) == before
    store.finish(parent, "interrupted", "synthetic parent termination")
    terminal = n.get(store, "cpu_completed:toy")
    assert terminal["status"] == "interrupted"
    assert terminal["recorded_at"] == store.job(parent)["ended"]
    after = revision(store)
    assert after != before
    next_parent = store.enqueue(store.task("toy"), "agent", ["python"])
    store.db.execute("UPDATE jobs SET status='running',started=40 WHERE id=?", (next_parent,))
    cpu(store, next_parent, "shared-request-id", 50, "running")
    assert revision(store) == after
    cpu(store, next_parent, "shared-request-id", terminal["recorded_at"] + 10, "succeeded")
    assert revision(store) != after


def test_late_inference_cannot_cover_a_new_acceptance(managed):
    store, _, _ = managed
    before = explain(store)
    store.event("toy", "research_context", {"text": "待验收的合成局部探索。"})

    def late(*args):
        n.put(
            store,
            "validation:toy",
            {
                "validation": {
                    "recorded_at": 30,
                    "status": "no_benefit",
                    "scope": "synthetic full call",
                    "summary": "完整调用无收益",
                    "source_sha256": "new-candidate",
                }
            },
        )
        return reply(*args)

    assert n.step(store, [], call=late) == 0
    assert revision(store) != before
    assert n.get(store, "narration:toy")["row"]["revision"] == before
    assert n.step(store, [], call=reply) == 1
    current = n.get(store, "narration:toy")
    assert current["row"]["revision"] == revision(store)
    assert any(
        current["sources"][ref]["kind"] == "validation"
        for ref in current["claims"]["finding"]["sources"]
    )
