"""Replacement preserves the budget ledger and permanently retires old identities."""

import concurrent.futures
import copy
import json
import sqlite3
import threading

import pytest

from kernel_envolve.manager.cli import main
from kernel_envolve.manager.config import validate
from kernel_envolve.manager.store import Store
from kernel_envolve.manager.views import detail, overview


@pytest.fixture
def replaced(tmp_path):
    tasks = []
    for name in ("old", "new", "third"):
        workspace = tmp_path / name
        workspace.mkdir()
        tasks.append(
            {
                "id": name,
                "name": name,
                "workspace": str(workspace),
                "task_dir": str(workspace),
                "kind": "generic",
                "resource": "gpu",
                "command": ["python", "-c", "pass"],
                "wall_budget_seconds": 100 if name == "old" else 70,
                "gpu_budget_seconds": 80 if name == "old" else 60,
                "baseline": {
                    "repository": "https://example.test/repo",
                    "revision": "fixed",
                    "entrypoint": "kernel",
                    "introduced_on": None,
                    "introduction_url": None,
                    "frozen_on": "2026-11-10",
                    "hardware": "B300",
                    "contract": name,
                },
            }
        )
    cfg = validate(
        {
            "version": 1,
            "resources": [
                {"id": "gpu", "device": "GPU-123", "sessions": 1},
            ],
            "tasks": tasks,
        },
        tmp_path,
    )
    store = Store(tmp_path / "state")
    store.register(cfg)
    store.db.execute("UPDATE tasks SET desired='paused'")
    store.db.execute("UPDATE tasks SET wall_seconds=20,gpu_seconds=10 WHERE id='old'")
    store.db.execute("UPDATE tasks SET wall_seconds=5,gpu_seconds=3 WHERE id='new'")
    yield store
    store.close()


def change_spec(store, key, **changes):
    spec = copy.deepcopy(store.task(key)["spec"])
    spec.update(changes)
    store.db.execute("UPDATE tasks SET spec=? WHERE id=?", (json.dumps(spec), key))


def test_cli_retirement_preserves_ledger_and_history(replaced, capsys):
    store = replaced
    store.db.execute("INSERT INTO runs VALUES('historical','old',1,'{}',NULL)")
    for task, seconds in (("old", 10), ("new", 3)):
        job = store.enqueue(store.task(task), "gpu", ["python"])
        store.db.execute("UPDATE jobs SET status='succeeded',seconds=? WHERE id=?", (seconds, job))
    jobs = store.rows("SELECT * FROM jobs ORDER BY id")
    original = store.rows("SELECT id,spec,wall_seconds,gpu_seconds FROM tasks ORDER BY id")
    main(["--state", str(store.root), "supersede", "old", "--by", "new"])
    receipt = json.loads(capsys.readouterr().out)
    assert receipt["superseded_by"] == "new"
    assert receipt["predecessor"]["wall_seconds"] == 20
    assert original == store.rows("SELECT id,spec,wall_seconds,gpu_seconds FROM tasks ORDER BY id")
    assert detail(store, "old")["runs"][0]["path"] == "historical"
    assert store.task("old")["desired"] == "superseded"
    assert store.task("old")["superseded_by"] == "new"
    assert store.task("new")["desired"] == "paused"  # No implicit launch.
    assert jobs == store.rows("SELECT * FROM jobs ORDER BY id")
    assert overview(store)["resources"][0]["gpu_seconds"] == 13
    events = store.rows("SELECT * FROM events")
    assert store.supersede("old", "new") == receipt
    assert events == store.rows("SELECT * FROM events")
    reopened = Store(store.root)
    try:
        assert reopened.task("old")["superseded_by"] == "new"
        reopened.control("old", "pause")
        assert reopened.task("old")["desired"] == "superseded"
        for action in ("resume", "configure"):
            with pytest.raises(ValueError, match="Superseded"):
                reopened.control("old", action, {"gpu_budget_seconds": 200})
        assert not reopened.schedule()
    finally:
        reopened.close()


@pytest.mark.parametrize(
    "field,value",
    [
        ("wall_budget_seconds", 81),
        ("gpu_budget_seconds", 71),
        ("wall_budget_seconds", None),
        ("gpu_budget_seconds", None),
    ],
)
def test_rejects_total_limit_reset_even_after_successor_spending(replaced, field, value):
    store = replaced
    change_spec(store, "new", **{field: value})
    store.db.execute("UPDATE tasks SET wall_seconds=60,gpu_seconds=50 WHERE id='new'")
    with pytest.raises(ValueError, match="unused budget"):
        store.supersede("old", "new")
    assert store.task("old")["desired"] == "paused"
    assert not store.one("SELECT * FROM metadata WHERE key LIKE 'superseded:%'")


@pytest.mark.parametrize("status", ["queued", "running"])
def test_live_predecessor_cannot_be_replaced(replaced, status):
    store = replaced
    key = store.enqueue(store.task("old"), "agent", ["python"])
    store.db.execute("UPDATE jobs SET status=? WHERE id=?", (status, key))
    with pytest.raises(ValueError, match="jobs to finish"):
        store.supersede("old", "new")
    assert not store.task("old")["superseded_by"]


def test_rechecks_usage_after_an_earlier_migration_plan(replaced):
    store = replaced
    # A previously valid 70-second replacement no longer fits the current ledger.
    store.db.execute("UPDATE tasks SET wall_seconds=31 WHERE id='old'")
    with pytest.raises(ValueError, match="unused budget"):
        store.supersede("old", "new")
    assert store.task("old")["wall_seconds"] == 31
    assert store.task("new")["spec"]["wall_budget_seconds"] == 70


def test_current_successor_and_resource_quarantine_remain_unchanged(replaced):
    store = replaced
    key = store.enqueue(store.task("new"), "agent", ["python"])
    store.db.execute("UPDATE jobs SET status='running' WHERE id=?", (key,))
    store.db.execute("UPDATE resources SET problem='Unconfirmed cleanup'")
    store.supersede("old", "new")
    assert store.job(key)["status"] == "running"
    assert store.resource("gpu")["problem"] == "Unconfirmed cleanup"


def test_rejects_self_cycles_fan_in_and_alternative_successors(replaced):
    store = replaced
    with pytest.raises(ValueError, match="itself"):
        store.supersede("old", "old")
    store.supersede("old", "new")
    with pytest.raises(ValueError, match="different replacement"):
        store.supersede("old", "third")
    with pytest.raises(ValueError, match="cycles"):
        store.supersede("new", "old")
    with pytest.raises(ValueError, match="predecessor"):
        store.supersede("third", "new")
    change_spec(store, "third", wall_budget_seconds=65, gpu_budget_seconds=57)
    store.supersede("new", "third")
    assert store.task("new")["superseded_by"] == "third"
    with pytest.raises(ValueError, match="cycles"):
        store.supersede("third", "old")


@pytest.mark.parametrize("mutation", ["resource", "kind", "baseline", "overlap", "missing"])
def test_rejects_unrelated_or_unsafe_workspaces(replaced, mutation):
    store = replaced
    spec = store.task("new")["spec"]
    if mutation in ("resource", "kind"):
        change_spec(store, "new", **{mutation: "different"})
    elif mutation == "baseline":
        change_spec(store, "new", baseline=spec["baseline"] | {"entrypoint": "other"})
    elif mutation == "overlap":
        old = store.task("old")["spec"]
        change_spec(store, "new", workspace=old["workspace"], task_dir=old["task_dir"])
    else:
        change_spec(store, "new", task_dir=spec["task_dir"] + "/missing")
    with pytest.raises(ValueError):
        store.supersede("old", "new")
    assert not store.task("old")["superseded_by"]


def test_schedule_race_has_one_atomic_winner(replaced):
    store = replaced
    store.control("old", "resume")
    ready = threading.Barrier(2)

    def compete(action):
        local = Store(store.root)
        try:
            ready.wait(timeout=5)
            if action == "schedule":
                return local.schedule()
            try:
                return local.supersede("old", "new")
            except ValueError:
                return None
        finally:
            local.close()

    with concurrent.futures.ThreadPoolExecutor(2) as pool:
        scheduled, receipt = list(pool.map(compete, ("schedule", "supersede")))
    assert bool(scheduled) != bool(receipt)
    assert bool(store.task("old")["superseded_by"]) == bool(receipt)
    assert len(store.rows("SELECT * FROM jobs WHERE task='old'")) == len(scheduled)


def test_transaction_rolls_back_if_event_write_fails(replaced):
    store = replaced
    store.db.execute("""CREATE TRIGGER reject_superseded BEFORE INSERT ON events
        WHEN NEW.kind='superseded' BEGIN SELECT RAISE(ABORT,'test failure'); END""")
    with pytest.raises(sqlite3.IntegrityError, match="test failure"):
        store.supersede("old", "new")
    assert store.task("old")["desired"] == "paused"
    assert not store.task("old")["superseded_by"]
