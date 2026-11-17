"""Fixed research clocks, real continuation, and evidence-only completion policy."""

import json
import sys
import time
from pathlib import Path

import pytest
from test_manager import launch
from test_research_task import arguments

from kernel_envolve.manager import research_policy as policy
from kernel_envolve.manager.single import prepare
from kernel_envolve.manager.store import Store
from kernel_envolve.storage import digest

pytest_plugins = ["test_research_task", "test_native_confirmation"]


@pytest.fixture
def lane(research):
    store, work = research
    key = prepare(
        store,
        arguments(
            work, "--device", "cpu", "--seconds", "100", "--", sys.executable, "-c", "print('host')"
        ),
    )
    store.control(key, "pause")
    store.db.execute("UPDATE tasks SET wall_seconds=31,gpu_seconds=17 WHERE id=?", (key,))
    start = time.time()
    value = policy.install(
        store, key, started_at=start, min_gain=0.05, deadline_command=["host-deadline"]
    )
    return store, key, start, value


def clock(monkeypatch, now):
    monkeypatch.setattr(policy.time, "time", lambda: now)


def ready(command, request):
    return request | {
        "status": "ready",
        "allocation_deadline_at": request["deadline_at"],
        "allocation_gpu_seconds_used": max(0, time.time() - request["started_at"]),
        "allocation_gpu_seconds_limit": 2 * policy.WINDOW,
    }


def activate(lane):
    store, key, _, _ = lane
    return policy.tick(store, key, sync=ready)


def test_install_requires_no_evaluator_and_never_resets_existing_policy(lane):
    store, key, start, value = lane
    assert value["delivery"] is None and value["phase"] == "starting"
    assert not (Path(store.task(key)["spec"]["workspace"]) / "native.json").exists()
    with pytest.raises(ValueError, match="already installed"):
        policy.install(store, key, started_at=start, min_gain=0.05, deadline_command=["adapter"])
    activate(lane)
    assert store.task(key)["wall_seconds"] == 31 and store.task(key)["gpu_seconds"] == 17
    assert store.task(key)["spec"]["wall_budget_seconds"] == 31 + policy.WINDOW
    assert store.task(key)["spec"]["deadline_at"] == start + policy.WINDOW
    assert store.task(key)["desired"] == "active" and store.schedule()


def test_no_benefit_extends_once_to_original_end_and_keeps_counters(lane, monkeypatch):
    store, key, start, _ = lane
    activate(lane)
    clock(monkeypatch, start + policy.WINDOW + 7)
    store.db.execute("UPDATE tasks SET wall_seconds=600,gpu_seconds=500 WHERE id=?", (key,))
    assert policy.tick(store, key, sync=ready)["phase"] == "extending"
    assert store.task(key)["desired"] == "paused"
    value = policy.tick(store, key, sync=ready)
    assert value["phase"] == "extended" and value["extension_used"]
    assert store.task(key)["spec"]["deadline_at"] == start + 2 * policy.WINDOW
    assert store.task(key)["spec"]["wall_budget_seconds"] == 31 + 2 * policy.WINDOW
    assert (store.task(key)["wall_seconds"], store.task(key)["gpu_seconds"]) == (600, 500)
    jobs = store.schedule()
    assert len(jobs) == 1  # Actual manager-owned host round, not a status-only extension.
    with StoreContext(store.root) as reopened:
        assert policy.get(reopened, key) == value
        clock(monkeypatch, start + 2 * policy.WINDOW)
        assert policy.tick(reopened, key, sync=ready)["phase"] == "exhausted"
        assert not reopened.schedule()


class StoreContext:
    def __init__(self, root):
        self.store = Store(root)

    def __enter__(self):
        return self.store

    def __exit__(self, *args):
        self.store.close()


def test_host_pending_failure_and_cumulative_allocation_cap_hold_pause(lane):
    store, key, _, _ = lane

    def pending(command, request):
        assert policy.get(store, key)["pending_transition"] == request
        return request | {"status": "pending"}

    assert policy.tick(store, key, sync=pending)["phase"] == "starting"
    assert store.task(key)["desired"] == "paused" and not store.schedule()

    def overbudget(command, request):
        return ready(command, request) | {"allocation_gpu_seconds_used": 28000}

    with pytest.raises(ValueError, match="exceeds"):
        policy.tick(store, key, sync=overbudget)
    assert store.task(key)["desired"] == "paused"


@pytest.mark.parametrize("action", ["pause", "resume", "configure"])
def test_manual_control_during_admission_never_auto_resumes(lane, action):
    store, key, _, _ = lane

    def interrupted(command, request):
        store.control(key, action, {} if action == "configure" else None)
        return ready(command, request)

    with pytest.raises(ValueError, match="policy changed"):
        policy.tick(store, key, sync=interrupted)
    assert policy.get(store, key)["phase"] == "suspended"
    before = store.task(key)
    assert policy.tick(store, key, sync=ready)["phase"] == "suspended"
    assert store.task(key) == before


def test_drain_before_extension_and_guard_margin(lane, monkeypatch):
    store, key, start, _ = lane

    def margin(command, request):
        return ready(command, request) | {"deadline_at": request["deadline_at"] - 120}

    policy.tick(store, key, sync=margin)
    job = store.schedule()[0]
    store.db.execute("UPDATE jobs SET status='running' WHERE id=?", (job,))
    clock(monkeypatch, start + policy.WINDOW - 119)
    assert policy.tick(store, key, sync=margin)["phase"] == "extending"

    def forbidden(*args):
        pytest.fail("Cannot extend host deadline while prior jobs still run")

    assert policy.tick(store, key, sync=forbidden)["phase"] == "extending"
    store.db.execute("UPDATE jobs SET status='cancelled',ended=? WHERE id=?", (time.time(), job))
    assert policy.tick(store, key, sync=margin)["phase"] == "extended"
    assert store.task(key)["spec"]["deadline_at"] == start + 2 * policy.WINDOW - 120


def test_empty_or_unrecoverable_host_is_suspended_not_extended(lane):
    store, key, _, _ = lane
    activate(lane)
    store.db.execute(
        "UPDATE tasks SET desired='blocked',reason='no execution in three rounds' WHERE id=?",
        (key,),
    )
    assert policy.tick(store, key, sync=ready)["phase"] == "suspended"
    assert store.task(key)["desired"] == "blocked" and not store.schedule()


def test_scoped_summary_does_not_stop_research(lane):
    store, key, _, _ = lane
    activate(lane)
    store.db.execute(
        "INSERT INTO metadata VALUES(?,?)",
        ("validation:" + key, json.dumps({"status": "passed", "speedup": 100})),
    )
    assert policy.tick(store, key, sync=ready)["phase"] == "initial"
    assert store.task(key)["desired"] == "active"


@pytest.fixture
def bound_sources(lane, tmp_path, monkeypatch):
    """Synthetic native config, real Git sources and persistent binding state."""
    import shutil

    from kernel_envolve.manager.config import validate

    store, key, _, _ = lane
    research = store.task(key)["spec"]
    native = tmp_path / "native"
    roots = {role: {} for role in ("baseline", "candidate")}
    for role in roots:
        for name in ("core", "support"):
            path = native / role / name
            shutil.copytree(research["workspace"], path)
            roots[role][name] = str(path)
    cfg = {
        "sources": roots,
        "artifacts": {},
        "required_scopes": ["operator", "numerical", "integration"],
        "confirmation": {},
    }
    monkeypatch.setattr("kernel_envolve.native_config.load_config", lambda _: cfg)
    (native / "task").mkdir()
    delivery = {
        "id": "delivery",
        "name": "delivery",
        "kind": "native",
        "workspace": str(native),
        "task_dir": str(native / "task"),
        "resource": research["resource"],
        "baseline": research["baseline"],
        "command": [],
        "observe_only": True,
    }
    store.register(
        validate(
            {
                "version": 1,
                "tasks": [delivery],
                "resources": [store.resource(research["resource"])["spec"]],
            },
            tmp_path,
        )
    )
    policy.bind_delivery(store, key, "delivery", "core")
    return store, key, roots


@pytest.mark.parametrize("role", ["baseline", "candidate"])
@pytest.mark.parametrize("root", ["core", "support"])
def test_bound_attempt_cannot_silently_replace_any_source_root(bound_sources, role, root):
    store, key, roots = bound_sources
    old = policy.get(store, key)
    assert policy.bind_delivery(store, key, "delivery", "core") == old
    (Path(roots[role][root]) / "kernel.py").write_text("# different frozen input")
    if (role, root) == ("candidate", "core"):
        work = Path(store.task(key)["spec"]["workspace"])
        (work / "kernel.py").write_text("# different frozen input")
    with pytest.raises(ValueError, match="source binding changed"):
        policy.bind_delivery(store, key, "delivery", "core")
    assert policy.get(store, key) == old


def test_legacy_binding_upgrades_only_after_explicit_source_match(bound_sources):
    store, key, _ = bound_sources
    old = policy.get(store, key)
    old["delivery"].pop("source_binding")
    store.db.execute(
        "UPDATE metadata SET data=? WHERE key=?", (json.dumps(old), "research_policy:" + key)
    )
    row = store.one("SELECT data FROM metadata WHERE key='delivery_link:delivery'")
    relation = json.loads(row["data"])
    relation.pop("source_binding")
    store.db.execute(
        "UPDATE metadata SET data=? WHERE key='delivery_link:delivery'", (json.dumps(relation),)
    )
    work = Path(store.task(key)["spec"]["workspace"]) / "kernel.py"
    original = work.read_bytes()
    work.write_text("# continued research")
    with pytest.raises(ValueError, match="contents"):
        policy.bind_delivery(store, key, "delivery", "core")
    assert policy.get(store, key) == old
    work.write_bytes(original)
    upgraded = policy.bind_delivery(store, key, "delivery", "core")
    assert upgraded["delivery"]["source_binding"]
    assert {k: v for k, v in upgraded.items() if k != "delivery"} == {
        k: v for k, v in old.items() if k != "delivery"
    }
    row = store.one("SELECT data FROM metadata WHERE key='delivery_link:delivery'")
    linked = json.loads(row["data"])
    assert linked["source_binding"] == upgraded["delivery"]["source_binding"]
    assert linked["bound_at"] == relation["bound_at"]


def test_success_waits_for_drain_and_rechecks_proof(lane, monkeypatch):
    store, key, _, _ = lane
    activate(lane)
    proof = {"run": "native-full", "confirmation": "independent", "publication": "bound"}
    monkeypatch.setattr(policy, "delivery_proof", lambda *args: proof)
    job = store.schedule()[0]
    store.db.execute("UPDATE jobs SET status='running' WHERE id=?", (job,))
    assert policy.tick(store, key)["phase"] == "stopping"
    assert policy.tick(store, key)["phase"] == "stopping"
    store.db.execute("UPDATE jobs SET status='cancelled',ended=? WHERE id=?", (time.time(), job))
    assert policy.tick(store, key)["phase"] == "succeeded"
    assert store.task(key)["desired"] == "paused" and not store.schedule()
    # No research best or evaluator qualification was manufactured.
    assert not store.one("SELECT data FROM metadata WHERE key=?", ("best:" + key,))


def test_real_host_runs_after_extension_without_notes_or_consume(lane, monkeypatch):
    store, key, start, _ = lane
    activate(lane)
    clock(monkeypatch, start + policy.WINDOW)
    policy.tick(store, key, sync=ready)
    policy.tick(store, key, sync=ready)
    # Child uses real time; the fixed extended deadline is safely in its future.
    marker = Path(store.task(key)["spec"]["workspace"]) / "executed.txt"
    spec = store.task(key)["spec"]
    spec["host_command"] = [
        sys.executable,
        "-c",
        f"from pathlib import Path;Path({str(marker)!r}).write_text('real host')",
    ]
    store.db.execute("UPDATE tasks SET spec=? WHERE id=?", (json.dumps(spec), key))
    job = store.schedule()[0]
    child = launch(store, job)
    child.communicate(timeout=15)
    assert marker.read_text() == "real host"
    assert store.job(job)["status"] == "succeeded"


def test_real_native_full_confirmation_is_required_before_delivery(
    confirmation_task, tmp_path, monkeypatch
):
    """Real CPU native receipts; only the already-tested remote PR binding is supplied."""
    import shutil
    import subprocess

    from kernel_envolve.manager import linked_delivery
    from kernel_envolve.manager.cli import task_contract
    from kernel_envolve.manager.config import validate
    from kernel_envolve.manager.evidence import refresh_task
    from kernel_envolve.manager.views import best, detail, overview, task_view
    from kernel_envolve.native import evaluate_native, promote_native
    from kernel_envolve.native_confirmation import confirm_native
    from kernel_envolve.storage import read_json, write_json

    task, candidate = confirmation_task
    cfg = read_json(task / "native.json")
    baseline = Path(cfg["sources"]["baseline"]["core"])
    work = tmp_path / "research-source"
    subprocess.run(["git", "clone", "-q", str(baseline), str(work)], check=True)
    for name in ("toyop.py", "test_native_contract.py"):
        shutil.copy2(candidate / name, work / name)
    native_workspace = tmp_path / "native-workspace"
    native_workspace.mkdir()
    shutil.move(str(task), native_workspace / "task")
    shutil.move(str(candidate), native_workspace / "candidate")
    task, candidate = native_workspace / "task", native_workspace / "candidate"
    cfg["sources"]["candidate"]["core"] = str(candidate)
    write_json(task / "native.json", cfg)
    store = Store(tmp_path / "policy-state")
    try:
        key = prepare(store, arguments(work, "--device", "cpu", "--", sys.executable, "-c", "pass"))
        store.control(key, "pause")
        policy.install(
            store, key, started_at=time.time(), min_gain=0.05, deadline_command=["adapter"]
        )
        policy.tick(store, key, sync=ready)
        spec = store.task(key)["spec"]
        delivery = {
            "id": "delivery",
            "name": "delivery",
            "workspace": str(native_workspace),
            "task_dir": str(task),
            "kind": "native",
            "resource": spec["resource"],
            "command": [],
            "observe_only": True,
            "baseline": spec["baseline"] | {"contract": task_contract(task, "native")},
        }
        store.register(
            validate(
                {
                    "version": 1,
                    "resources": [store.resource(spec["resource"])["spec"]],
                    "tasks": [delivery],
                },
                tmp_path,
            )
        )
        value = policy.bind_delivery(store, key, "delivery", "core")
        assert policy.delivery_proof(store, value) is None
        pending = task_view(store, key)
        assert pending["best"] is None and pending["delivery"]["status"] == "pending"
        assert pending["implementation_pr"] is None
        assert task_view(store, "delivery")["delivery_parent"] == key
        assert len([t for t in overview(store)["tasks"] if not t.get("delivery_parent")]) == 1
        original = {k: store.task(key)[k] for k in ("desired", "gpu_seconds", "wall_seconds")}
        run, result = evaluate_native(task, "full", timeout=90)
        assert result["decision"]["eligible_for_best"], result
        refresh_task(store, store.task("delivery"))
        assert policy.delivery_proof(store, value) is None  # Missing independent confirm and PR.
        confirm_native(run, 90)
        promote_native(run)
        refresh_task(store, store.task("delivery"))
        verified = best(store, store.task("delivery"))
        assert verified and verified["confirmation_status"] == "passed"
        assert policy.delivery_proof(store, value) is None  # PR still required.
        publication = {
            "schema": 2,
            "verification": "verified",
            "run": str(run),
            "result_hash": verified["result_hash"],
            "confirmation": verified["confirmation"],
            "checked_at": time.time(),
            "head_repository": "fixture/source",
            "head": "a" * 40,
            "base_head": "b" * 40,
            "source_root": "core",
            "implementation_digest": "bound",
            "url": "https://github.com/fixture/source/pull/1",
            "evaluated_source_hash": digest(verified["source"]),
        }
        store.db.execute(
            "INSERT INTO metadata VALUES('publication:delivery',?)", (json.dumps(publication),)
        )
        assert policy.delivery_proof(store, value)
        proof = policy.delivery_proof(store, value)
        # A witness binds the owner, native identity, selected root and every
        # frozen source root; another otherwise valid best cannot substitute.
        from copy import deepcopy

        for field, replacement in (
            ("source_binding", "0" * 64),
            ("source_binding", None),
            ("source_root", "another-root"),
        ):
            changed = deepcopy(value)
            changed["delivery"][field] = replacement
            assert policy.delivery_proof(store, changed) is None
        wrong_owner = deepcopy(value)
        wrong_owner["identity"]["baseline"]["revision"] = "f" * 40
        assert policy.delivery_proof(store, wrong_owner) is None
        for task_id in (key, "delivery"):
            original_spec = store.task(task_id)["spec"]
            changed_spec = deepcopy(original_spec)
            changed_spec["baseline"]["revision"] = "f" * 40
            store.db.execute(
                "UPDATE tasks SET spec=? WHERE id=?", (json.dumps(changed_spec), task_id)
            )
            try:
                if task_id == key:
                    assert policy.delivery_proof(store, value) is None
                    with pytest.raises(ValueError, match="Research policy identity changed"):
                        policy.bind_delivery(store, key, "delivery", "core")
                else:
                    with pytest.raises(ValueError, match="Delivery identity changed"):
                        policy.delivery_proof(store, value)
            finally:
                store.db.execute(
                    "UPDATE tasks SET spec=? WHERE id=?", (json.dumps(original_spec), task_id)
                )
        legacy = deepcopy(value)
        legacy["delivery"].pop("source_binding")
        legacy_proof = policy.delivery_proof(store, legacy)
        assert legacy_proof and "source_binding" not in legacy_proof
        for target in (
            run / "result.json",
            run / "config.json",
            run / "sources/baseline/core/toyop.py",
            run / "sources/candidate/core/toyop.py",
        ):
            saved = target.read_bytes()
            target.write_text("{}" if target.suffix == ".json" else "# altered frozen source")
            try:
                with pytest.raises((ValueError, KeyError)):
                    policy.delivery_proof(store, value)
            finally:
                target.write_bytes(saved)
        from kernel_envolve.manager import evidence

        with monkeypatch.context() as isolated:
            isolated.setattr(
                evidence,
                "inspect_run",
                lambda *a, **kw: pytest.fail("Display must reuse the deep-audited index"),
            )
            linked_delivery.refresh(store, task_id="delivery")
        delivered = task_view(store, key)
        assert delivered["best"] is None and delivered["implementation_pr"] is None
        assert delivered["delivery"]["best"]["baseline_speedup"] == verified["baseline_speedup"]
        assert delivered["delivery"]["implementation_pr"]["url"] == publication["url"]
        assert delivered["row_facts"]["benefit"]["kind"] == "native_delivery"
        from kernel_envolve.manager import narrator

        calls = []

        def narrate(command, request, stop):
            calls.append(request)
            return {t["task"]: None for t in request["tasks"]}

        narrator.step(store, [], call=narrate)
        assert [t["task"] for t in calls[0]["tasks"]] == [key]
        assert any(s["kind"] == "delivery" for s in calls[0]["tasks"][0]["sources"].values())
        linked_delivery.refresh(store, task_id="delivery")
        assert task_view(store, key)["row_facts"]["revision"] == delivered["row_facts"]["revision"]
        assert narrator.collect(store) == []  # A repeated successful audit is not new research.
        assert detail(store, key)["delivery_attempts"][0]["task"] == "delivery"
        assert detail(store, "delivery")["runs"]
        assert original == {k: store.task(key)[k] for k in original}
        assert not store.one("SELECT data FROM metadata WHERE key=?", ("best:" + key,))
        assert not policy.delivery_proof(store, value | {"min_gain": 0.99999999})
        # A small receipt change is rejected even before the next deep index
        # scan. Historical display must not bypass missing/revoked index rows.
        receipt_path = run / "receipt.json"
        receipt_raw = receipt_path.read_bytes()
        receipt_path.write_text('{"result_hash":"changed"}')
        linked_delivery.refresh(store, task_id="delivery")
        assert task_view(store, key)["delivery"]["best"] is None
        receipt_path.write_bytes(receipt_raw)
        linked_delivery.refresh(store, task_id="delivery")
        indexed = store.one("SELECT * FROM runs WHERE path=?", (str(run),))
        store.db.execute("UPDATE runs SET error='revoked' WHERE path=?", (str(run),))
        assert task_view(store, key)["delivery"]["best"] is None
        store.db.execute("DELETE FROM runs WHERE path=?", (str(run),))
        assert task_view(store, key)["delivery"]["best"] is None
        store.db.execute(
            "INSERT INTO runs VALUES(?,?,?,?,?)",
            tuple(indexed[k] for k in ("path", "task", "imported", "data", "error")),
        )
        assert task_view(store, key)["delivery"]["best"]["run"] == str(run)
        # Temporarily unavailable evidence is not evidence of corruption and
        # must not erase the last verified historical record.
        unavailable = run.with_name(run.name + "-temporarily-unavailable")
        run.rename(unavailable)
        try:
            linked_delivery.refresh(store, task_id="delivery")
            assert task_view(store, key)["delivery"]["best"]["run"] == str(run)
        finally:
            unavailable.rename(run)
        raw = (run / "confirmation/receipt.json").read_bytes()
        (run / "confirmation/receipt.json").write_text("{}")
        with pytest.raises((ValueError, KeyError)):
            policy.delivery_proof(store, value)
        refresh_task(store, store.task("delivery"))
        assert task_view(store, key)["delivery"]["best"] is None
        (run / "confirmation/receipt.json").write_bytes(raw)
        refresh_task(store, store.task("delivery"))
        previous = (work / "toyop.py").read_bytes()
        (work / "toyop.py").write_text("# different research candidate")
        assert policy.delivery_proof(store, value) == proof
        assert policy.delivery_proof(store, legacy) is None
        linked_delivery.refresh(store, task_id="delivery")
        retained = task_view(store, key)
        assert retained["delivery"]["best"]["run"] == str(run)
        assert retained["row_facts"]["benefit"] == delivered["row_facts"]["benefit"]
        with pytest.raises(ValueError, match="contents"):
            policy.bind_delivery(store, key, "delivery", "core")
        candidate_previous = (candidate / "toyop.py").read_bytes()
        (candidate / "toyop.py").write_bytes((work / "toyop.py").read_bytes())
        changed_link = policy.source_link(store, key, "delivery", "core")
        assert policy.delivery_proof(store, value | {"delivery": changed_link}) is None
        with pytest.raises(ValueError, match="source binding changed"):
            policy.bind_delivery(store, key, "delivery", "core")
        (candidate / "toyop.py").write_bytes(candidate_previous)
        (work / "toyop.py").write_bytes(previous)
        linked_delivery.refresh(store, task_id="delivery")
        assert task_view(store, key)["delivery"]["status"] == "verified"
        assert task_view(store, key, now=time.time() + 10**6)["delivery"]["best"]["run"] == str(run)
        publication["checked_at"] = 1
        store.db.execute(
            "UPDATE metadata SET data=? WHERE key='publication:delivery'",
            (json.dumps(publication),),
        )
        assert policy.delivery_proof(store, value) is None
        stale_pr = task_view(store, key)
        assert stale_pr["delivery"]["best"]["run"] == str(run)
        assert stale_pr["row_facts"]["optimization_pr"]["value"] is None
        assert stale_pr["row_facts"]["revision"] != delivered["row_facts"]["revision"]
        assert stale_pr["delivery"]["implementation_pr"]["head"] == publication["head"]
        assert not stale_pr["delivery"]["implementation_pr"]["current_head_verified"]
        assert not store.one("SELECT * FROM metadata WHERE key=?", ("best:" + key,))
        # A new attempt remains separate and auditable, while both children stay
        # attached to the same research row. Names do not establish ownership.
        another = tmp_path / "another-native"
        shutil.copytree(native_workspace, another, ignore=shutil.ignore_patterns(".kernel_native"))
        next_task = another / "task"
        next_cfg = read_json(next_task / "native.json")
        next_cfg["sources"]["candidate"]["core"] = str(another / "candidate")
        write_json(next_task / "native.json", next_cfg)
        next_spec = delivery | {
            "id": "unrelated-name",
            "workspace": str(another),
            "task_dir": str(next_task),
            "baseline": spec["baseline"] | {"contract": task_contract(next_task, "native")},
        }
        store.register(
            validate(
                {
                    "version": 1,
                    "resources": [store.resource(spec["resource"])["spec"]],
                    "tasks": [next_spec],
                },
                tmp_path,
            )
        )
        policy.bind_delivery(store, key, "unrelated-name", "core")
        projected = task_view(store, key)
        assert {x["task"] for x in projected["delivery_attempts"]} == {"delivery", "unrelated-name"}
        assert len([t for t in overview(store)["tasks"] if not t.get("delivery_parent")]) == 1
        assert detail(store, "delivery")["runs"]  # Old real evidence remains intact.
        assert projected["current_delivery_task"] == "unrelated-name"
        assert projected["delivery"]["task"] == "delivery"
        assert projected["delivery"]["best"]["run"] == str(run)
        other_work = tmp_path / "other-research"
        shutil.copytree(work, other_work)
        other_key = prepare(
            store, arguments(other_work, "--device", "cpu", "--", sys.executable, "-c", "pass")
        )
        store.control(other_key, "pause")
        policy.install(
            store, other_key, started_at=time.time(), min_gain=0.05, deadline_command=["adapter"]
        )
        with pytest.raises(ValueError, match="belongs to another"):
            policy.bind_delivery(store, other_key, "unrelated-name", "core")
        # A successful stopped policy and its archive retain the real result;
        # reopening needs neither the research checkout nor a live verifier.
        publication["checked_at"] = time.time()
        store.db.execute(
            "UPDATE metadata SET data=? WHERE key='publication:delivery'",
            (json.dumps(publication),),
        )
        policy.bind_delivery(store, key, "delivery", "core")
        job = store.schedule()[0]
        store.db.execute("UPDATE jobs SET status='running' WHERE id=?", (job,))
        # Continuing research neither overwrites the accepted snapshot nor
        # blocks early success. A final host edit during drain is preserved.
        (work / "toyop.py").write_text("# unfinished next candidate")
        counters = {k: store.task(key)[k] for k in ("gpu_seconds", "wall_seconds")}
        clocks = {k: policy.get(store, key)[k] for k in ("started_at", "maximum_end")}
        assert policy.tick(store, key)["phase"] == "stopping"
        (work / "toyop.py").write_text("# unfinished next candidate during drain")
        assert policy.tick(store, key)["phase"] == "stopping"
        store.db.execute("UPDATE jobs SET status='cancelled' WHERE id=?", (job,))
        assert policy.tick(store, key)["phase"] == "succeeded"
        assert (work / "toyop.py").read_text() == "# unfinished next candidate during drain"
        assert counters == {k: store.task(key)[k] for k in counters}
        assert clocks == {k: policy.get(store, key)[k] for k in clocks}
        from kernel_envolve.manager.archive import Archive, export_archive

        destination = tmp_path / "delivery-archive"
        export_archive(store, destination)
        shutil.rmtree(work)
        reopened = Archive(destination).read()["details"][key]
        assert reopened["delivery"]["policy_phase"] == "succeeded"
        assert reopened["delivery"]["best"]["run"] == str(run)
        assert len(reopened["delivery_attempts"]) == 2
        assert reopened["best"] is None
    finally:
        store.close()


def test_invalid_delivery_cannot_prevent_authorized_extension(lane, monkeypatch):
    store, key, start, _ = lane
    activate(lane)

    def invalid(*args):
        raise ValueError("confirmation evidence changed")

    monkeypatch.setattr(policy, "delivery_proof", invalid)
    clock(monkeypatch, start + policy.WINDOW)
    assert policy.tick(store, key, sync=ready)["phase"] == "extending"
    assert policy.tick(store, key, sync=ready)["phase"] == "extended"


def test_changed_proof_during_drain_never_succeeds(lane, monkeypatch):
    store, key, _, _ = lane
    activate(lane)
    monkeypatch.setattr(policy, "delivery_proof", lambda *args: {"result": "verified"})
    assert policy.tick(store, key)["phase"] == "stopping"
    monkeypatch.setattr(policy, "delivery_proof", lambda *args: None)
    assert policy.tick(store, key)["phase"] == "suspended"
    assert not store.schedule()


def test_observer_never_invokes_policy_but_coordinator_does(lane, monkeypatch):
    from kernel_envolve.manager import service

    store, key, _, _ = lane
    calls = []
    monkeypatch.setattr(policy, "refresh", lambda local: calls.append(local.root))
    service.run(store, once=True, observe=True)
    assert not calls
    service.run(store, once=True)
    assert calls == [store.root]


@pytest.mark.parametrize("phase", ["succeeded", "exhausted"])
def test_owner_cleanup_pause_preserves_terminal_outcome(lane, phase):
    store, key, _, _ = lane
    value = policy.get(store, key)
    policy.save(store, key, value, value | {"phase": phase})
    store.control(key, "pause")
    assert policy.get(store, key)["phase"] == phase
    assert policy.tick(store, key, sync=ready)["phase"] == phase


def test_deadline_policy_does_not_wait_behind_evidence_scan(lane, monkeypatch):
    import threading

    from kernel_envolve.manager import evidence, service

    store, _, _, _ = lane
    advanced = threading.Event()
    observed = []

    def slow_scan(*args, **kwargs):
        observed.append(advanced.wait(2))

    monkeypatch.setattr(evidence, "refresh", slow_scan)
    monkeypatch.setattr(policy, "refresh", lambda local: advanced.set())
    service.run(store, once=True)
    assert observed == [True]
