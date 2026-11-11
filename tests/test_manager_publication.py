"""PR metadata is explicitly linked to best, never inferred from candidate reports."""

import json

import pytest

from kernel_envolve.manager.publication import FIELDS, refresh_publications, sync_pr
from kernel_envolve.manager.views import detail, overview

pytest_plugins = ["test_manager_supersede"]
URL = "https://github.com/example/kernels/pull/7"


def verified(store, run="run-1"):
    data = {
        "run": run,
        "baseline_speedup": 1.2,
        "decision": {"eligible_for_best": True},
        "mode": "full",
        "result_hash": "hash-" + run,
        "source": {},
    }
    store.db.execute("INSERT INTO runs VALUES(?,?,?,?,NULL)", (run, "new", 1, json.dumps(data)))
    store.db.execute("INSERT OR REPLACE INTO metadata VALUES('best:new',?)", (json.dumps(data),))


def github(monkeypatch, callback=None):
    callbacks = [callback] if callback else []

    def fetch(argv, **kwargs):
        assert argv == ["pr", "view", URL, "--json", FIELDS]
        if callbacks:
            callbacks.pop()()
        return {
            "url": URL,
            "updatedAt": "2026-11-10T12:00:00Z",
            "headRefOid": "a" * 40,
            "baseRefOid": "b" * 40,
            "baseRefName": "integration",
            "headRepository": {"nameWithOwner": "example/kernels"},
            "state": "OPEN",
        }

    monkeypatch.setattr("kernel_envolve.manager.publication.github", fetch)
    monkeypatch.setattr(
        "kernel_envolve.manager.publication.bind_source",
        lambda *args: {"source_root": "core", "head_repository": "example/kernels"},
    )
    monkeypatch.setattr(
        "kernel_envolve.manager.evidence.inspect_run",
        lambda path, spec: {
            "mode": "full",
            "decision": {"eligible_for_best": True},
            "result_hash": "hash-" + str(path),
            "source": {},
        },
    )


def test_pr_uses_github_time_and_only_current_verified_best(replaced, monkeypatch):
    verified(replaced)
    github(monkeypatch)
    sync_pr(replaced, "new", "run-1", URL, source_root="core", base_branch="integration")
    row = detail(replaced, "new")["implementation_pr"]
    assert row["url"] == URL and row["updated_at"] == 1794312000
    current = next(t for t in overview(replaced, compact=True)["tasks"] if t["id"] == "new")
    assert current["implementation_pr"] == row
    verified(replaced, "run-2")
    assert detail(replaced, "new")["implementation_pr"] is None


def test_candidate_or_invalidated_best_cannot_be_published(replaced, monkeypatch):
    github(monkeypatch)
    with pytest.raises(ValueError, match="verified best"):
        sync_pr(replaced, "new", "candidate", URL, source_root="core", base_branch="integration")
    verified(replaced)
    sync_pr(replaced, "new", "run-1", URL, source_root="core", base_branch="integration")
    replaced.db.execute("UPDATE runs SET error='source changed' WHERE path='run-1'")
    assert detail(replaced, "new")["implementation_pr"] is None


def test_best_change_during_network_fetch_is_rejected(replaced, monkeypatch):
    verified(replaced)
    github(monkeypatch, lambda: verified(replaced, "run-2"))
    with pytest.raises(ValueError, match="changed"):
        sync_pr(replaced, "new", "run-1", URL, source_root="core", base_branch="integration")
    assert detail(replaced, "new")["implementation_pr"] is None


def test_bad_url_never_calls_github(replaced, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("must not fetch malformed URL")

    monkeypatch.setattr("kernel_envolve.manager.publication.subprocess.run", forbidden)
    with pytest.raises(ValueError, match="URL"):
        sync_pr(
            replaced,
            "new",
            "candidate",
            "https://evil.test/pull/1",
            source_root="core",
            base_branch="integration",
        )


def age_binding(store):
    row = json.loads(store.one("SELECT data FROM metadata WHERE key='publication:new'")["data"])
    row["checked_at"] = 1
    store.db.execute("UPDATE metadata SET data=? WHERE key='publication:new'", (json.dumps(row),))


def test_expired_or_offline_binding_keeps_last_verified_commit(replaced, monkeypatch):
    verified(replaced)
    github(monkeypatch)
    sync_pr(replaced, "new", "run-1", URL, source_root="core", base_branch="integration")
    age_binding(replaced)
    row = detail(replaced, "new")["implementation_pr"]
    assert not row["current_head_verified"] and row["commit_url"].endswith("a" * 40)

    def unavailable(*args):
        raise OSError("offline")

    monkeypatch.setattr("kernel_envolve.manager.publication.github", unavailable)
    refresh_publications(replaced)
    row = detail(replaced, "new")["implementation_pr"]
    assert not row["current_head_verified"] and row["verification"] == "unavailable"
    assert row["url"] == URL and row["updated_at"] == 1794312000


@pytest.mark.parametrize(
    "field,value",
    [
        ("headRefOid", "c" * 40),
        ("baseRefOid", "c" * 40),
        ("baseRefName", "wrong-branch"),
        ("state", "CLOSED"),
    ],
)
def test_changed_pr_is_historical_until_explicit_resync(replaced, monkeypatch, field, value):
    verified(replaced)
    github(monkeypatch)
    sync_pr(replaced, "new", "run-1", URL, source_root="core", base_branch="integration")
    age_binding(replaced)
    from kernel_envolve.manager import publication

    original = publication.github
    monkeypatch.setattr(publication, "github", lambda args: original(args) | {field: value})
    refresh_publications(replaced)
    row = detail(replaced, "new")["implementation_pr"]
    assert not row["current_head_verified"] and row["verification"] == "head_changed"
    assert row["head"] == "a" * 40
    monkeypatch.setattr(
        publication,
        "github",
        lambda args: (_ for _ in ()).throw(
            AssertionError("changed head needs explicit source verification")
        ),
    )
    refresh_publications(replaced)


def test_failed_rebind_never_keeps_old_binding_current(replaced, monkeypatch):
    verified(replaced)
    github(monkeypatch)
    sync_pr(replaced, "new", "run-1", URL, source_root="core", base_branch="integration")

    def bad_source(*args):
        raise ValueError("implementation changed")

    monkeypatch.setattr("kernel_envolve.manager.publication.bind_source", bad_source)
    with pytest.raises(ValueError, match="implementation changed"):
        sync_pr(replaced, "new", "run-1", URL, source_root="core", base_branch="integration")
    row = detail(replaced, "new")["implementation_pr"]
    assert not row["current_head_verified"] and row["head"] == "a" * 40


def test_source_binding_requires_declared_target_branch(replaced, monkeypatch):
    verified(replaced)
    github(monkeypatch)
    with pytest.raises(ValueError, match="target branch"):
        sync_pr(replaced, "new", "run-1", URL, source_root="core", base_branch="other")


def test_refresh_is_bounded_and_preserves_concurrent_resync(replaced, monkeypatch):
    verified(replaced)
    github(monkeypatch)
    sync_pr(replaced, "new", "run-1", URL, source_root="core", base_branch="integration")
    age_binding(replaced)
    original = replaced.one("SELECT data FROM metadata WHERE key='publication:new'")["data"]
    replaced.db.execute("INSERT INTO metadata VALUES('publication:other',?)", (original,))
    from kernel_envolve.manager import publication

    fetch = publication.github
    calls = []
    newer = json.loads(original) | {"head": "d" * 40, "checked_at": 123456}

    def race(args):
        calls.append(args)
        replaced.db.execute(
            "UPDATE metadata SET data=? WHERE key='publication:new'", (json.dumps(newer),)
        )
        return fetch(args)

    monkeypatch.setattr(publication, "github", race)
    refresh_publications(replaced)
    assert len(calls) == 1
    stored = json.loads(
        replaced.one("SELECT data FROM metadata WHERE key='publication:new'")["data"]
    )
    assert stored == newer


def test_confirmation_receipt_changes_invalidate_pr_binding(replaced, monkeypatch):
    verified(replaced)
    github(monkeypatch)
    sync_pr(replaced, "new", "run-1", URL, source_root="core", base_branch="integration")
    data = json.loads(replaced.one("SELECT data FROM metadata WHERE key='best:new'")["data"])
    data["confirmation"] = {"receipt_hash": "new-confirmation"}
    replaced.db.execute("UPDATE runs SET data=? WHERE path='run-1'", (json.dumps(data),))
    assert detail(replaced, "new")["implementation_pr"] is None
