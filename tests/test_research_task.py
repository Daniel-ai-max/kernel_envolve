"""Exploration uses the real manager without constructing an evaluator."""

import json
import subprocess
import sys

import pytest

from kernel_envolve.cli import promote
from kernel_envolve.manager import cli
from kernel_envolve.manager.config import validate
from kernel_envolve.manager.evidence import inspect_run, refresh, state_path
from kernel_envolve.manager.publication import sync_pr
from kernel_envolve.manager.research_task import baseline_revision, source_identity
from kernel_envolve.manager.single import parser, prepare
from kernel_envolve.manager.store import Store
from kernel_envolve.manager.views import best, task_view
from kernel_envolve.manager.worker import checkpoint
from kernel_envolve.native import promote_native
from kernel_envolve.storage import digest


@pytest.fixture
def research(tmp_path):
    work = tmp_path / "source"
    work.mkdir()
    (work / "kernel.py").write_text("def run(x): return x + 1\n")
    for command in (
        ["git", "init", "-q"],
        ["git", "add", "."],
        [
            "git",
            "-c",
            "user.name=Test",
            "-c",
            "user.email=test@example.test",
            "commit",
            "-qm",
            "baseline",
        ],
    ):
        subprocess.run(command, cwd=work, check=True, capture_output=True)
    store = Store(tmp_path / "state")
    try:
        yield store, work
    finally:
        store.close()


def arguments(work, *extra):
    return parser().parse_args(["--research", str(work), *extra])


def test_source_only_observer_has_no_evaluator_or_gpu_probe(research, monkeypatch):
    store, work = research

    def forbidden(*args, **kwargs):
        pytest.fail("Source-only registration must not load an evaluator or snapshot sources")

    monkeypatch.setattr("kernel_envolve.manager.single.config", forbidden)
    monkeypatch.setattr("kernel_envolve.native_config.load_config", forbidden)
    monkeypatch.setattr("kernel_envolve.native.snapshot", forbidden)
    monkeypatch.setattr("kernel_envolve.manager.single.device_for", forbidden)
    key = prepare(store, arguments(work))
    task = store.task(key)
    assert task["spec"]["kind"] == "research"
    assert task["spec"]["baseline"]["revision"] == baseline_revision(work)
    assert task["spec"]["baseline"]["contract"] == source_identity(work, baseline_revision(work))
    assert not store.schedule()
    assert refresh(store)
    view = task_view(store, key)
    assert view["phase"] == "observing"
    assert view["evidence_status"] == "not_attached" and not view["evidence_stale"]
    assert view["best"] is None and view["latest_evaluation"] is None
    assert not any(
        (work / name).exists() for name in ("task.py", "task.json", "adapter.py", "native.json")
    )


def test_research_host_uses_existing_bridge_and_retains_budgets(research):
    store, work = research
    key = prepare(store, arguments(work))
    store.db.execute("UPDATE tasks SET wall_seconds=3,gpu_seconds=2")
    host = [sys.executable, "-c", "pass"]
    assert prepare(store, arguments(work, "--device", "cpu", "--seconds", "60", "--", *host)) == key
    spec = store.task(key)["spec"]
    assert spec["host_command"] == host and "bridge" in spec["command"]
    assert spec["wall_budget_seconds"] == 63
    deadline = spec["deadline_at"]
    assert prepare(store, arguments(work)) == key
    assert store.task(key)["spec"]["deadline_at"] == deadline
    assert store.task(key)["gpu_seconds"] == 2
    jobs = store.schedule()
    assert len(jobs) == 1 and not store.schedule()
    text = checkpoint(store, store.task(key), store.job(jobs[0]))
    assert "research without an evaluator" in text and "manage gpu-run" in text
    assert "--run <verified-best-run>" not in text
    store.control(key, "pause")
    prepare(store, arguments(work))
    assert store.task(key)["desired"] == "paused"
    with pytest.raises(ValueError, match="retained"):
        prepare(store, arguments(work, "--seconds", "100"))


def test_baseline_is_pinned_once_and_kind_cannot_silently_change(research):
    store, work = research
    key = prepare(store, arguments(work, "--baseline", "HEAD"))
    previous = store.task(key)["spec"]["baseline"]
    (work / "kernel.py").write_text("def run(x): return x + 2\n")
    subprocess.run(
        [
            "git",
            "-c",
            "user.name=Test",
            "-c",
            "user.email=test@example.test",
            "commit",
            "-qam",
            "candidate",
        ],
        cwd=work,
        check=True,
        capture_output=True,
    )
    assert prepare(store, arguments(work)) == key
    assert store.task(key)["spec"]["baseline"] == previous
    with pytest.raises(ValueError, match="baseline changed"):
        prepare(store, arguments(work, "--baseline", "HEAD"))
    with pytest.raises(ValueError, match="original"):
        prepare(store, parser().parse_args(["--task", str(work)]))


def test_manager_init_accepts_research_without_evaluator(research, tmp_path):
    store, work = research
    key = prepare(store, arguments(work))
    task = store.task(key)["spec"].copy()
    task.pop("identity")
    deployment = {
        "version": 1,
        "tasks": [task],
        "resources": [store.resource(task["resource"])["spec"]],
    }
    file = tmp_path / "deployment.json"
    file.write_text(json.dumps(deployment))
    target = tmp_path / "other-state"
    assert cli.main(["--state", str(target), "init", str(file)]) == 0
    other = Store(target)
    try:
        assert other.task(key)["spec"]["kind"] == "research"
    finally:
        other.close()
    task["baseline"]["revision"] = "HEAD"
    with pytest.raises(ValueError, match="full commit"):
        validate(deployment, tmp_path)


def test_research_contract_command_resolves_revision_without_copy(research, capsys):
    _, work = research
    assert (
        cli.main(["contract", "--task", str(work), "--kind", "research", "--baseline", "HEAD"]) == 0
    )
    assert capsys.readouterr().out.strip() == source_identity(work, baseline_revision(work))


def test_forged_research_receipt_and_best_are_never_admitted(research):
    store, work = research
    key = prepare(store, arguments(work))
    task = store.task(key)
    fake = work / ".kernel_envolve" / "runs" / "forged"
    fake.mkdir(parents=True)
    result = {
        "research_only": True,
        "task_root": str(work),
        "mode": "full",
        "decision": {"eligible_for_best": True},
        "baseline_speedup": 900,
    }
    (fake / "result.json").write_text(json.dumps(result))
    (fake / "receipt.json").write_text(json.dumps({"result_hash": digest(result)}))
    (fake.parent.parent / "best.json").write_text(json.dumps({"run": "runs/forged"}))
    assert refresh(store, force=True)
    assert not store.rows("SELECT * FROM runs")
    assert not store.rows("SELECT * FROM promotions")
    with pytest.raises(ValueError, match="Research records"):
        inspect_run(fake, task["spec"])
    with pytest.raises(ValueError, match="no evaluator state"):
        state_path(task["spec"])
    for promotion in (promote, promote_native):
        with pytest.raises(ValueError, match="Research results"):
            promotion(fake)
    # Even an accidentally stale best projection must not turn this task into
    # an evaluator; the public view and publication gate enforce the kind too.
    store.db.execute(
        "INSERT INTO metadata VALUES(?,?)", ("best:" + key, json.dumps({"run": str(fake)}))
    )
    assert best(store, store.task(key)) is None
    assert task_view(store, key)["implementation_pr"] is None
    with pytest.raises(ValueError, match="Research tasks"):
        sync_pr(
            store,
            key,
            str(fake),
            "https://github.com/example/project/pull/1",
            source_root="core",
            base_branch="main",
        )


@pytest.mark.parametrize("ref", ["missing", "--help"])
def test_invalid_baseline_is_rejected_before_registration(research, ref):
    store, work = research
    with pytest.raises(ValueError, match="local Git commit"):
        prepare(store, arguments(work, "--baseline=" + ref))
    assert not store.rows("SELECT * FROM tasks")


def test_nested_workspace_and_conflicting_options_are_rejected(research):
    store, work = research
    nested = work / "nested"
    nested.mkdir()
    with pytest.raises(ValueError, match="own Git root"):
        prepare(store, arguments(nested))
    with pytest.raises(ValueError, match="already selects"):
        prepare(store, arguments(work, "--workspace", str(work)))
    with pytest.raises(ValueError, match="requires --research"):
        prepare(store, parser().parse_args(["--task", str(work), "--baseline", "HEAD"]))


@pytest.mark.parametrize("large_source", [False, True])
def test_research_dashboard_runs_real_host_and_cpu_gpu_queue_without_evaluator(
    research, large_source
):
    import urllib.request

    from test_manager import wait_for
    from test_manager_continuation import HOST

    from kernel_envolve.manager.process import terminate

    store, work = research
    host = work / "host.py"
    host.write_text(HOST.replace("candidate.py", "kernel.py"))
    if large_source:
        from kernel_envolve.manager.research_source import MAX_FILE_BYTES

        (work / "oversized.py").write_bytes(b"#" * (MAX_FILE_BYTES + 1))
    process = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "kernel_envolve.cli",
            "dashboard",
            "--research",
            str(work),
            "--state",
            str(store.root),
            "--device",
            "cpu",
            "--port",
            "0",
            "--seconds",
            "90",
            "--",
            sys.executable,
            str(host),
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=True,
    )
    info = None
    try:
        line = process.stdout.readline()
        assert line, process.stderr.read()
        info = json.loads(line)
        key = info["task"]
        wait_for(lambda: store.task(key)["desired"] == "blocked", timeout=60)
        with urllib.request.urlopen(info["url"] + "/api/state", timeout=5) as response:
            view = json.load(response)["tasks"][0]
        assert view["spec"]["kind"] == "research" and view["best"] is None
        assert view["evidence_status"] == "not_attached"
        jobs = store.rows(
            "SELECT * FROM jobs WHERE task=? AND kind='agent' ORDER BY created", (key,)
        )
        assert [job["status"] for job in jobs] == ["succeeded", "succeeded", "failed"]
        assert jobs[-1]["exit_code"] == 78
        assert store.task(key)["wall_seconds"] == pytest.approx(sum(job["seconds"] for job in jobs))
        assert (work / "iteration.txt").read_text() == "3"
        gpu = store.rows("SELECT * FROM jobs WHERE task=? AND kind='gpu'", (key,))
        assert len(gpu) == 1 and gpu[0]["status"] == "succeeded"
        binding = store.one(
            "SELECT data FROM metadata WHERE key=?", ("research_source:" + gpu[0]["id"],)
        )
        assert binding and json.loads(binding["data"])["endpoints_complete"] is (not large_source)
        source = json.loads(
            (store.root / "jobs" / gpu[0]["id"] / "research-source.json").read_text()
        )
        assert source["exit_code"] == 0 and source["status"] == "succeeded"
        assert source["source_changed"] is (None if large_source else False)
        assert source["logs"]["stdout.log"]["complete"]
        assert not (work / "task.json").exists() and not (work / "native.json").exists()
        assert not store.rows("SELECT * FROM runs")
    finally:
        if info:
            store.control(info["task"], "pause")
        terminate(process)
        process.communicate(timeout=10)


def test_source_edits_count_as_recovery_but_notes_do_not(research):
    import os

    from kernel_envolve.manager.process import identity
    from kernel_envolve.manager.recovery import begin

    store, work = research
    key = prepare(
        store,
        arguments(work, "--device", "cpu", "--seconds", "60", "--", sys.executable, "-c", "pass"),
    )
    for index, edit in enumerate(("kernel.py", "notes.md")):
        store.db.execute("UPDATE tasks SET next_at=0")
        job = store.schedule()[0]
        store.claim(job, os.getpid(), identity(os.getpid()))
        begin(store, store.task(key), store.job(job))
        with (work / edit).open("a") as stream:
            stream.write("\n# repair or note\n")
        store.finish(job, "succeeded", None, 0)
        record = json.loads(
            store.one("SELECT data FROM metadata WHERE key=?", ("recovery:" + key,))["data"]
        )
        assert record["implementation_changed"] is (index == 0)
        assert record["observable_progress"] is (index == 0)
    store.control(key, "pause")
    (work / "kernel.py").write_text("# another repair\n")
    assert not store.schedule()  # Source changes never revive explicit stops.


def test_source_delta_is_deduplicated_and_archive_reconstructs_changes(research, tmp_path):
    import hashlib
    import shutil

    from kernel_envolve.manager.archive import Archive, export_archive
    from kernel_envolve.manager.research_source import bind_job, capture

    store, work = research
    key = prepare(store, arguments(work))
    task = store.task(key)
    (work / "kernel.py").write_text("def run(x): return x + 7\n")
    (work / "probe.py").write_text("print('new probe')\n")
    blobs = store.root / "research-sources"
    before = capture(task, blobs)
    assert before["complete"] and "probe.py" in before["untracked_scripts"]
    files = set(blobs.iterdir())
    assert capture(task, blobs) == before and set(blobs.iterdir()) == files
    patch = blobs / before["tracked_patch"]["sha256"]
    assert b"return x + 7" in patch.read_bytes()
    (work / "kernel.py").write_text("def run(x): return x + 8\n")
    directory = store.root / "jobs" / "source-fixture"
    directory.mkdir(parents=True)
    (directory / "stdout.log").write_text("measured\n")
    bind_job(
        store,
        task,
        {"id": "source-fixture", "command": '["python", "probe.py"]'},
        directory,
        before,
        "succeeded",
        0,
    )
    receipt = json.loads((directory / "research-source.json").read_text())
    assert receipt["source_changed"] and receipt["endpoints_complete"]
    assert receipt["command"] == ["python", "probe.py"]
    assert receipt["logs"]["stdout.log"]["sha256"] == hashlib.sha256(b"measured\n").hexdigest()
    root = tmp_path / "archive"
    exported = export_archive(store, root)
    generation = root / exported["generation"]
    assert (generation / "research-jobs/source-fixture/research-source.json").is_file()
    assert (generation / "research-sources" / patch.name).read_bytes() == patch.read_bytes()
    shutil.rmtree(work)
    shutil.rmtree(blobs)
    assert Archive(root).read()["details"][key]["best"] is None


def test_source_limits_disclose_gaps_and_do_not_read_oversized_scripts(research, monkeypatch):
    from kernel_envolve.manager import research_source as source

    store, work = research
    key = prepare(store, arguments(work))
    monkeypatch.setattr(source, "MAX_FILE_BYTES", 8)
    (work / "large.py").write_bytes(b"x" * 1000)
    (work / "ignored.py").write_text("secret ignored source\n")
    (work / ".gitignore").write_text("ignored.py\n")
    result = source.capture(store.task(key))
    assert not result["complete"] and any("large.py" in gap for gap in result["gaps"])
    assert "ignored.py" not in result["untracked_scripts"]
    (work / "kernel.py").write_bytes(b"a" * 20000)
    monkeypatch.setattr(source, "MAX_BYTES", 64)
    result = source.capture(store.task(key))
    assert result["tracked_patch"]["bytes"] == 64
    assert any("byte limit" in gap for gap in result["gaps"])


def test_git_output_and_deadline_limits_stop_the_actual_process(research, monkeypatch):
    from kernel_envolve.manager import research_source as source

    _, work = research
    real_popen = subprocess.Popen
    children = []

    def spawn(*args, **kwargs):
        process = real_popen(
            [sys.executable, "-c", "import os; os.write(1,b'x'*1000000)"], **kwargs
        )
        children.append(process)
        return process

    monkeypatch.setattr(source.subprocess, "Popen", spawn)
    output, error = source.bounded_git(work, [], 31)
    assert len(output) == 31 and "byte limit" in error
    assert children[-1].poll() is not None and children[-1].stdout.closed

    def waiting(*args, **kwargs):
        process = real_popen([sys.executable, "-c", "import time; time.sleep(60)"], **kwargs)
        children.append(process)
        return process

    monkeypatch.setattr(source.subprocess, "Popen", waiting)
    monkeypatch.setattr(source, "GIT_SECONDS", 0.05)
    output, error = source.bounded_git(work, [], 31)
    assert not output and "deadline" in error
    assert children[-1].poll() is not None and children[-1].stdout.closed


def test_research_can_attach_trusted_scoped_assessment_but_not_reported_gains(research, tmp_path):
    import time

    from kernel_envolve.manager.assessment import refresh_sources, watch

    store, work = research
    key = prepare(store, arguments(work))
    value = dict(
        schema="scoped-assessment-v1",
        status="verified",
        promotion=False,
        task=key,
        baseline_revision=store.task(key)["spec"]["baseline"]["revision"],
        source_sha256="a" * 64,
        evidence_sha256="b" * 64,
        recorded_at=time.time(),
        metrics=[dict(label="独立测试范围", baseline_speedup=1.2, cases=2)],
    )
    receipt, verifier = tmp_path / "trusted-receipt.json", tmp_path / "trusted-verifier.py"
    receipt.write_text(json.dumps(value))
    # Test-only trusted verifier; production verifiers check their raw evidence.
    verifier.write_text(
        "import json,sys\nfrom pathlib import Path\nc=json.load(sys.stdin)\nv=json.loads(Path(c['receipt']).read_text())\nassert v['task']==c['task']\nprint(json.dumps(v))\n"
    )
    unsafe = work / "verifier.py"
    unsafe.write_text(verifier.read_text())
    with pytest.raises(ValueError, match="outside"):
        watch(store, key, receipt, unsafe)
    with pytest.raises(ValueError, match="unknown fields"):
        store.report(key, "none", {"verified_assessment": value})
    assert task_view(store, key)["verified_assessment"] is None
    watch(store, key, receipt, verifier)
    refresh_sources(store)
    view = task_view(store, key)
    assert view["verified_assessment"] == value and view["best"] is None
    assert view["implementation_pr"] is None
    verifier.write_text(verifier.read_text() + "# changed verifier\n")
    refresh_sources(store, force=True)
    assert "Pinned" in task_view(store, key)["assessment_sync"]["error"]
    assert not store.rows("SELECT * FROM promotions")


def test_archive_rechecks_original_job_binding_and_labels_before_only(
    research, tmp_path, monkeypatch
):
    from kernel_envolve.manager.archive import export_archive
    from kernel_envolve.manager.research_source import bind_job, capture
    from kernel_envolve.storage import write_json

    store, work = research
    key = prepare(store, arguments(work))
    task = store.task(key)
    before = capture(task, store.root / "research-sources")
    directory = store.root / "jobs" / "bound"
    directory.mkdir(parents=True)
    write_json(directory / "research-source-before.json", before)
    bind_job(
        store, task, {"id": "bound", "command": '["probe"]'}, directory, before, "succeeded", 0
    )
    receipt = directory / "research-source.json"
    original = receipt.read_bytes()
    receipt.write_bytes(original + b"\n")
    with pytest.raises(ValueError, match="Bound research source receipt changed"):
        export_archive(store, tmp_path / "tampered")
    receipt.write_bytes(original)
    write_json(directory / "research-source-before.json", before | {"complete": False})
    with pytest.raises(ValueError, match="before receipt changed"):
        export_archive(store, tmp_path / "tampered-before")
    write_json(directory / "research-source-before.json", before)
    orphan = store.root / "jobs" / "interrupted" / "research-source-before.json"
    orphan.parent.mkdir()
    write_json(orphan, before)
    exported = export_archive(store, tmp_path / "archive")
    manifest = json.loads(
        (tmp_path / "archive" / exported["generation"] / "manifest.json").read_text()
    )
    assert manifest["unbound_research_source_receipts"] == [
        "jobs/interrupted/research-source-before.json"
    ]
    import shutil

    copy = shutil.copyfile

    def race(source, target, *args, **kwargs):
        result = copy(source, target, *args, **kwargs)
        if str(target).endswith("research-source.json"):
            with open(target, "ab") as stream:
                stream.write(b"\n")
        return result

    with monkeypatch.context() as context:
        context.setattr(shutil, "copyfile", race)
        with pytest.raises(ValueError, match="changed during copy"):
            export_archive(store, tmp_path / "racing")
    receipt.unlink()
    with pytest.raises(ValueError, match="missing"):
        export_archive(store, tmp_path / "missing")
