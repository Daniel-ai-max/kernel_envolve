"""Real Manager admission with CPU evaluator children; no GPU benefit is simulated."""

import json
import os
import signal
import sys
import time

import pytest
from test_manager import launch, replace_command

from kernel_envolve.manager.process import alive, exclusive, identity, lock_path
from kernel_envolve.rsi.managed import evaluate

pytest_plugins = ["test_manager"]


@pytest.fixture
def active(managed, monkeypatch):
    store, _, work = managed
    parent = store.schedule()[0]
    assert store.claim(parent, os.getpid(), identity(os.getpid()))
    for name, value in {
        "KERNEL_ENVOLVE_STATE": str(store.root),
        "KERNEL_ENVOLVE_TASK": "toy",
        "KERNEL_ENVOLVE_ROUND": parent,
    }.items():
        monkeypatch.setenv(name, value)
    yield store, work, parent
    store.finish(parent, "succeeded")


@pytest.mark.parametrize("adaptive,snapshot_enabled", [(False, False), (True, False), (True, True)])
def test_search_releases_device_for_decisions_and_requires_separate_final(
    managed, adaptive, snapshot_enabled
):
    store, _, work = managed
    (work / "candidate.py").write_text("import time\ndef run(x):\n time.sleep(.001)\n return x+1\n")
    out = work.parent / "research"
    view = None
    if snapshot_enabled:
        import shutil

        from kernel_envolve.cli import evaluate as direct
        from kernel_envolve.rsi.artifact_snapshot import materialize
        from kernel_envolve.rsi.research import run

        history = work.parent / "historical-task"
        history.mkdir()
        for name in ("task.py", "candidate.py", "task.json"):
            shutil.copy2(work / name, history / name)
        evidence = work.parent / "historical-evidence"
        packet = dict(
            task=str(history),
            kind="generic",
            editable_files=[str(history / "candidate.py")],
            model="no-model",
            seed=0,
            strategy="",
            seconds=15,
        )
        run(
            packet,
            evidence,
            lambda *a: dict(
                action="propose",
                source="def run(x): return x+1\n",
                mode="full",
                hypothesis="CPU historical fixture",
            ),
            max_actions=1,
        )
        confirmation, _ = direct(history, "full", timeout=5)
        view = materialize(
            history, evidence, {}, work.parent / "views", confirmation_run=confirmation
        ) | {"profile": {}}
    script = work / "agent.py"
    script.write_text(
        "import os,json,ast\nfrom pathlib import Path\n"
        "from kernel_envolve.rsi.research import run,source_id\n"
        "from kernel_envolve.rsi.managed import evaluate\n"
        "from kernel_envolve.manager.process import exclusive,lock_path\n"
        "from kernel_envolve.manager.store import Store\n"
        "os.environ['PYTHONPATH']='deliberately-missing-relative-source'\n"
        "def decide(request,timeout):\n"
        " assert not os.environ.get('KERNEL_ENVOLVE_GPU_JOB')\n"
        " with exclusive(lock_path('gpu','cpu')) as lock:\n"
        "  assert lock is not None, 'model decision held device lock'\n"
        " s=Store(os.environ['KERNEL_ENVOLVE_STATE'])\n"
        " try: assert not s.rows(\"SELECT id FROM jobs WHERE kind='gpu' AND status='running'\")\n"
        " finally: s.close()\n"
        " return dict(action='propose',source='def run(x): return x+1\\n',mode='full',hypothesis='Remove fixture sleep')\n"
        f"task=Path({str(work)!r}); out=Path({str(out)!r})\n"
        "packet=dict(task=str(task),kind='generic',editable_files=[str(task/'candidate.py')],model='no-model',seed=0,strategy='',seconds=25)\n"
        "source='def run(x): return x+1\\n'\n"
        "proposal=dict(episode_id='a'*64,source_hash=source_id(source),task_source_hash=source_id((task/'task.py').read_text()),source=source)\n"
        f"trace=run(packet,out,decide,max_actions={4 if adaptive else 1},evaluation_seconds=5,managed_evaluations=True,adaptive_artifact_budget={adaptive!r},artifact_proposal=proposal if {bool(adaptive and not snapshot_enabled)!r} else None,artifact_snapshot={view!r},source_validator=ast.parse)\n"
        "final,result=evaluate(task,'full',5,out/'final')\n"
        "assert result['correctness']=='pass'\n"
        "assert str(final) != trace['steps'][0]['observation']['run']\n"
        "assert not (task/'.kernel_envolve/best.json').exists()\n"
        "Path(os.environ['KERNEL_ENVOLVE_REPORT']).write_text(json.dumps({'next_action':'inspect independent final receipt before promotion'}))\n"
    )
    replace_command(store, [sys.executable, str(script)], round_timeout_seconds=30)
    parent = store.schedule()[0]
    process = launch(store, parent)
    stdout, stderr = process.communicate(timeout=35)
    assert process.returncode == 0, (stdout, stderr)
    jobs = store.rows("SELECT * FROM jobs WHERE kind='gpu'")
    assert len(jobs) == 3
    assert all(job["status"] == "succeeded" for job in jobs)
    assert store.task("toy")["gpu_seconds"] > 0
    trace = json.loads((out / "search-trace.json").read_text())
    assert trace["managed_evaluations"] is True
    if snapshot_enabled:
        assert trace["artifact_snapshot"]["reason"] == "compatible_proposal"
    if adaptive:
        assert trace["effective_actions"] == 2
        assert trace["allocation"]["reason"] == "within_margin"
        assert len(trace["steps"]) == 2
    assert trace["selected_quality"] > 0.8
    assert len(list((out / "managed-evaluations").glob("*/cost.json"))) == 2


def test_queue_deadline_cancels_without_starting_child(active):
    store, work, _ = active
    previous = signal.getsignal(signal.SIGALRM)
    with exclusive(lock_path("gpu", "cpu")) as handle:
        assert handle is not None
        with pytest.raises(TimeoutError, match="queue/execution"):
            evaluate(work, "full", 0.2, work.parent / "queued")
    job = store.one("SELECT * FROM jobs WHERE kind='gpu'")
    assert job["status"] == "cancelled"
    assert job["child_pid"] is None
    assert not (work / ".kernel_envolve/runs").exists()
    assert signal.getsignal(signal.SIGALRM) == previous
    assert signal.getitimer(signal.ITIMER_REAL) == (0, 0)


def test_expired_execution_cleans_child_and_preserves_failure(active):
    store, work, _ = active
    (work / "candidate.py").write_text("import time\ndef run(x):\n time.sleep(60)\n return x+1\n")
    with pytest.raises(TimeoutError, match="queue/execution"):
        evaluate(work, "full", 1.5, work.parent / "expired")
    job = store.one("SELECT * FROM jobs WHERE kind='gpu'")
    assert job["status"] in {"cancelled", "failed"}
    assert job["child_pid"] is not None
    assert not alive(job["child_pid"], job["child_identity"])
    for child in store.rows("SELECT * FROM processes WHERE job=?", (job["id"],)):
        assert not alive(child["pid"], child["identity"])
    with exclusive(lock_path("gpu", "cpu")) as handle:
        assert handle is not None


def test_device_probe_cannot_swallow_queue_deadline(active, monkeypatch):
    store, work, _ = active
    probes = []

    def busy(device):
        probes.append(device)
        time.sleep(5)
        pytest.fail("The device probe swallowed the deadline")

    monkeypatch.setattr("kernel_envolve.manager.gpu.gpu_idle", busy)
    with pytest.raises(TimeoutError, match="queue/execution"):
        evaluate(work, "full", 0.2, work.parent / "probe-deadline")
    assert probes == ["cpu"]
    job = store.one("SELECT * FROM jobs WHERE kind='gpu'")
    assert job["status"] == "cancelled"
    assert job["child_pid"] is None


def test_duration_admission_is_not_bypassed(active):
    store, work, _ = active
    with pytest.raises(ValueError, match="duration admission"):
        evaluate(work, "full", 20, work.parent / "too-long")
    assert not store.rows("SELECT id FROM jobs WHERE kind='gpu'")


def test_source_changed_during_admission_is_not_evaluated(active, monkeypatch):
    from kernel_envolve.rsi import managed as adapter

    store, work, _ = active
    original = adapter.gpu_run

    def changed(*args, **kwargs):
        (work / "candidate.py").write_text("def run(x): return x-1\n")
        return original(*args, **kwargs)

    monkeypatch.setattr(adapter, "gpu_run", changed)
    with pytest.raises(ValueError, match="did not finish successfully"):
        evaluate(work, "full", 5, work.parent / "changed")
    assert not (work / ".kernel_envolve/runs").exists()
    assert store.one("SELECT status FROM jobs WHERE kind='gpu'")["status"] == "failed"


def test_locator_cannot_redirect_to_another_source(active, monkeypatch):
    from kernel_envolve.rsi import managed as adapter
    from kernel_envolve.storage import digest, write_json

    _, work, _ = active
    original = adapter.gpu_run
    directory = work.parent / "redirected"

    def changed(*args, **kwargs):
        # Change both the request and candidate consistently so the real worker
        # evaluates another valid source. The parent still owns the original hash.
        (work / "candidate.py").write_text("def run(x): return 1+x\n")
        request = json.loads((directory / "request.json").read_text())
        original_request = dict(request)
        request["source_hash"] = adapter.file_hash(work / "candidate.py")
        write_json(directory / "request.json", request)
        code = original(*args, **kwargs)
        locator = json.loads((directory / "result.json").read_text())
        locator["request_hash"] = digest(original_request)
        write_json(directory / "result.json", locator)
        return code

    monkeypatch.setattr(adapter, "gpu_run", changed)
    with pytest.raises(ValueError, match="source, mode, or receipt mismatch"):
        evaluate(work, "full", 5, directory)


def test_wrong_task_and_contract_rejected_before_enqueue(active):
    store, work, _ = active
    with pytest.raises(ValueError, match="mismatch"):
        evaluate(work.parent, "full", 1, work.parent / "wrong-task")
    with (work / "task.py").open("a") as stream:
        stream.write("\n# changed contract\n")
    with pytest.raises(ValueError, match="mismatch"):
        evaluate(work, "full", 1, work.parent / "wrong-contract")
    assert not store.rows("SELECT id FROM jobs WHERE kind='gpu'")


def test_existing_timer_is_not_displaced(active):
    store, work, _ = active
    signal.setitimer(signal.ITIMER_REAL, 60)
    try:
        with pytest.raises(ValueError, match="existing process deadline"):
            evaluate(work, "full", 1, work.parent / "timer")
        assert signal.getitimer(signal.ITIMER_REAL)[0] > 59
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
    assert not store.rows("SELECT id FROM jobs WHERE kind='gpu'")


def test_nested_research_rejected_before_any_evaluation(active, monkeypatch):
    from kernel_envolve.rsi.research import run

    store, work, _ = active
    packet = {"kind": "generic"}
    with pytest.raises(ValueError, match="managed_evaluations=True"):
        run(packet, work.parent / "unmanaged", lambda *args: pytest.fail("decision called"))
    monkeypatch.setenv("KERNEL_ENVOLVE_GPU_JOB", "nested")
    with pytest.raises(ValueError, match="outside the Manager GPU"):
        run(packet, work.parent / "nested", lambda *args: pytest.fail("decision called"))
    assert not store.rows("SELECT id FROM jobs WHERE kind='gpu'")


def test_commit_boundary_deadline_keeps_timeout_and_cleans_real_child(active, monkeypatch):
    from test_transaction_interrupt import InterruptedTransaction

    from kernel_envolve.manager.store import Store
    from kernel_envolve.rsi.managed import _DeadlineExpired

    store, work, _ = active
    (work / "candidate.py").write_text("import time\ndef run(x):\n time.sleep(60)\n return x+1\n")
    original = Store.heartbeat
    injected = []

    def deadline_after_first_gpu_heartbeat(self, key, seconds):
        if injected or self.job(key)["kind"] != "gpu":
            return original(self, key, seconds)
        injected.append(key)
        connection = self.db
        self.db = InterruptedTransaction(
            connection, _DeadlineExpired("queue/execution commit deadline")
        )
        try:
            return original(self, key, seconds)
        finally:
            self.db = connection

    monkeypatch.setattr(Store, "heartbeat", deadline_after_first_gpu_heartbeat)
    with pytest.raises(TimeoutError, match="queue/execution commit deadline"):
        evaluate(work, "full", 5, work.parent / "commit-interrupt")
    assert len(injected) == 1
    job = store.job(injected[0])
    assert job["status"] in {"failed", "cancelled"}
    assert job["child_pid"] is not None
    assert not alive(job["child_pid"], job["child_identity"])
    for child in store.rows("SELECT * FROM processes WHERE job=?", (job["id"],)):
        assert not alive(child["pid"], child["identity"])
    with exclusive(lock_path("gpu", "cpu")) as handle:
        assert handle is not None
    assert not store.db.in_transaction
    assert signal.getitimer(signal.ITIMER_REAL) == (0, 0)


@pytest.mark.parametrize("after", [False, True])
def test_enqueue_interruption_cleans_only_a_committed_waiter(active, monkeypatch, after):
    from test_transaction_interrupt import InterruptedTransaction

    from kernel_envolve.manager.store import Store
    from kernel_envolve.rsi.managed import _DeadlineExpired

    store, work, _ = active
    original = Store.enqueue
    injected = []

    def enqueue(self, task, kind, command, parent=None):
        key = original(self, task, kind, command, parent)
        if kind == "gpu":
            injected.append(key)
            self.db = InterruptedTransaction(
                self.db, _DeadlineExpired("queue/execution enqueue deadline"), after=after
            )
        return key

    monkeypatch.setattr(Store, "enqueue", enqueue)
    with pytest.raises(TimeoutError, match="queue/execution enqueue deadline"):
        evaluate(work, "full", 5, work.parent / "enqueue-interrupt")
    assert len(injected) == 1
    jobs = store.rows("SELECT * FROM jobs WHERE kind='gpu'")
    assert len(jobs) == int(after)
    if after:
        assert jobs[0]["status"] == "cancelled" and jobs[0]["child_pid"] is None
    assert not (work / ".kernel_envolve/runs").exists()
    assert not store.db.in_transaction
