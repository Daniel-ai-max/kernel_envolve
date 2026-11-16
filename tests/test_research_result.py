"""Stage claims survive archival without becoming evaluator facts or new experiments."""

import copy
import hashlib
import json
import time

import pytest

from kernel_envolve.manager.archive import Archive, export_archive
from kernel_envolve.manager.assessment import refresh_sources, watch
from kernel_envolve.manager.research_result import validate, validate_job
from kernel_envolve.manager.views import detail, task_view

pytest_plugins = ["test_manager"]


def install(store, tmp_path):
    receipt, verifier = tmp_path / "index.json", tmp_path / "host.py"
    artifacts = tmp_path / "host-artifacts.json"
    actual = {
        name: hashlib.sha256(name.encode()).hexdigest()
        for name in ("request", "archive", "source", "log", "receipt")
    }
    artifacts.write_text(json.dumps(actual))
    value = {
        "schema": "research-result-v1",
        "status": "research",
        "promotion": False,
        "task": "toy",
        "baseline_revision": store.task("toy")["spec"]["baseline"]["revision"],
        "source_sha256": actual["source"],
        "evidence_sha256": "e" * 64,
        "recorded_at": time.time() - 60,
        "scope": "Gradient stage, Graph",
        "summary": "Synthetic stage: 100 → 80 µs",
        "limitations": ["No public forward/backward qualification"],
        "binding": {
            "kind": "transport_archive",
            "request": "1" * 32,
            **{name + "_sha256": actual[name] for name in ("request", "archive", "receipt", "log")},
        },
    }
    # The real host adapter hashes its bounded, allowlisted legacy artifacts.
    # This fixture exercises the same rejection boundary without any internal data.
    verifier.write_text(
        "import json,sys\nfrom pathlib import Path\n"
        "ctx=json.load(sys.stdin)\nr=json.loads(Path(ctx['receipt']).read_text())\n"
        f"a=json.loads(Path({str(artifacts)!r}).read_text())\n"
        "assert r['source_sha256']==a['source']\n"
        "assert all(r['binding'][k+'_sha256']==a[k] for k in ('request','archive','receipt','log'))\n"
        "print(json.dumps(r))\n"
    )
    receipt.write_text(json.dumps(value))
    watch(store, "toy", receipt, verifier, research=True)
    return receipt, verifier, artifacts, value


def test_terminal_research_is_visible_without_rewriting_experiment_or_best(managed, tmp_path):
    store, _, _ = managed
    _, _, _, value = install(store, tmp_path)
    store.control("toy", "pause")
    before = store.task("toy")
    view_before = task_view(store, "toy")
    refresh_sources(store)
    refresh_sources(store)
    assert store.task("toy") == before
    view = detail(store, "toy")
    assert view["research_result"] == value
    assert view["best"] is None and view["verified_assessment"] is None
    assert not view["runs"] and not view["assessment_history"] and not view["history"]
    assert len(view["research_result_history"]) == 1
    assert view["research_result_history"][0]["indexed_at"] > value["recorded_at"]
    for field in ("experiment_as_of", "progress_as_of", "gpu_seconds", "wall_seconds"):
        assert view[field] == view_before[field]
    assert not store.schedule()
    assert not store.rows("SELECT * FROM promotions")


@pytest.mark.parametrize("field", ["source", "request", "archive", "receipt", "log"])
def test_legacy_source_receipt_mismatch_is_rejected_by_pinned_host(managed, tmp_path, field):
    store, _, _ = managed
    _, _, artifacts, _ = install(store, tmp_path)
    value = json.loads(artifacts.read_text())
    value[field] = "f" * 64
    artifacts.write_text(json.dumps(value))
    refresh_sources(store)
    view = task_view(store, "toy")
    assert view["research_result"] is None and view["research_result_sync"]["error"]
    assert view["verified_assessment"] is None


@pytest.mark.parametrize(
    "change",
    [
        {"status": "verified"},
        {"schema": "scoped-assessment-v1"},
        {"promotion": True},
        {"task": "other"},
        {"recorded_at": float("nan")},
        {"recorded_at": time.time() + 3600},
        {"path": "/etc/passwd"},
        {"url": "https://example.invalid/evidence"},
        {"binding": {"kind": "manager_job", "job": "../escape"}},
    ],
)
def test_research_shape_cannot_be_upgraded_or_redirect_reader(managed, tmp_path, change):
    store, _, _ = managed
    _, _, _, value = install(store, tmp_path)
    with pytest.raises(ValueError):
        validate(value | change, store.task("toy"))


def test_failed_update_and_changed_verifier_keep_research_history(managed, tmp_path):
    store, _, _ = managed
    receipt, verifier, _, value = install(store, tmp_path)
    refresh_sources(store)
    verifier.write_text(verifier.read_text() + "\n# modified\n")
    receipt.write_text(json.dumps(value | {"summary": "Unverified replacement"}))
    refresh_sources(store)
    view = detail(store, "toy")
    assert view["research_result"] == value
    assert "Pinned" in view["research_result_sync"]["error"]
    assert len(view["research_result_history"]) == 1


def test_research_archive_survives_removed_host_input_and_rejects_tamper(managed, tmp_path):
    store, _, _ = managed
    receipt, verifier, artifacts, value = install(store, tmp_path)
    refresh_sources(store)
    destination = tmp_path / "archive"
    export_archive(store, destination)
    for path in (receipt, verifier, artifacts):
        path.unlink()
    archived = Archive(destination)
    assert archived.read()["details"]["toy"]["research_result"] == value
    assert archived.read()["details"]["toy"]["best"] is None
    record = store.rows("SELECT data FROM metadata WHERE key LIKE 'research_result_history:%'")[0]
    folder = next((store.root / "research-results").iterdir())
    (folder / "record.json").write_text(json.dumps(json.loads(record["data"]) | {"indexed_at": 0}))
    with pytest.raises(ValueError, match="receipt changed"):
        export_archive(store, tmp_path / "bad-archive")


def test_manager_binding_rejects_source_receipt_time_and_symlink_mismatches(managed, tmp_path):
    store, _, _ = managed
    _, _, _, value = install(store, tmp_path)
    task = store.task("toy")
    job = store.enqueue(task, "gpu", ["python", "experiment.py"])
    at = time.time() - 5
    store.db.execute(
        "UPDATE jobs SET status='succeeded',started=?,ended=?,exit_code=0 WHERE id=?",
        (at - 1, at, job),
    )
    value.update(
        recorded_at=at,
        binding={
            "kind": "manager_job",
            "job": job,
            "receipt_sha256": "a" * 64,
            "log_sha256": "b" * 64,
        },
    )
    endpoint = {
        "fingerprint": value["source_sha256"],
        "baseline_revision": value["baseline_revision"],
    }
    source = {
        "task": "toy",
        "job": job,
        "command": ["python", "experiment.py"],
        "status": "succeeded",
        "exit_code": 0,
        "endpoints_complete": True,
        "source_changed": False,
        "before": endpoint,
        "after": endpoint,
        "logs": {"stdout.log": {"sha256": "b" * 64, "complete": True}},
    }
    folder = store.root / "jobs" / job
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / "research-source.json"
    path.write_text(json.dumps(source))
    sha = hashlib.sha256(path.read_bytes()).hexdigest()
    value["binding"]["receipt_sha256"] = sha
    store.db.execute(
        "INSERT INTO metadata VALUES(?,?)", ("research_source:" + job, json.dumps({"sha256": sha}))
    )
    validate_job(store, task, validate(value, task))
    for change in ({"source_sha256": "f" * 64}, {"recorded_at": at + 1}):
        with pytest.raises(ValueError):
            validate_job(store, task, value | change)
    changed = copy.deepcopy(value)
    changed["binding"]["log_sha256"] = "f" * 64
    with pytest.raises(ValueError):
        validate_job(store, task, changed)
    path.unlink()
    path.symlink_to(tmp_path / "index.json")
    with pytest.raises(OSError):
        validate_job(store, task, value)


def test_late_audit_amends_same_experiment_without_changing_its_time(managed, tmp_path):
    store, _, _ = managed
    receipt, _, _, value = install(store, tmp_path)
    refresh_sources(store)
    before = store.task("toy")
    receipt.write_text(json.dumps(value | {"summary": "More precise stage scope"}))
    refresh_sources(store)
    view = detail(store, "toy")
    assert view["research_result"]["summary"] == "More precise stage scope"
    assert view["research_result"]["recorded_at"] == value["recorded_at"]
    assert len(view["research_result_history"]) == 2
    assert store.task("toy") == before
    receipt.write_text(json.dumps(value))
    refresh_sources(store)
    assert detail(store, "toy")["research_result"]["summary"] == "More precise stage scope"
