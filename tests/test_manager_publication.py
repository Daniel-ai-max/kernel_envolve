"""PR metadata is explicitly linked to best, never inferred from candidate reports."""

import json
from types import SimpleNamespace

import pytest

from kernel_envolve.manager.publication import sync_pr
from kernel_envolve.manager.views import detail, overview

pytest_plugins = ["test_manager_supersede"]
URL = "https://github.com/example/kernels/pull/7"


def verified(store, run="run-1"):
    data = {"run": run, "baseline_speedup": 1.2, "decision": {}}
    store.db.execute("INSERT INTO runs VALUES(?,?,?,?,NULL)", (run, "new", 1, json.dumps(data)))
    store.db.execute("INSERT OR REPLACE INTO metadata VALUES('best:new',?)", (json.dumps(data),))


def github(monkeypatch, callback=None):
    def fetch(argv, **kwargs):
        assert argv == ["gh", "pr", "view", URL, "--json", "url,updatedAt,headRefOid"]
        if callback:
            callback()
        return SimpleNamespace(
            stdout=json.dumps(
                {
                    "url": URL,
                    "updatedAt": "2026-11-10T12:00:00Z",
                    "headRefOid": "a" * 40,
                }
            )
        )

    monkeypatch.setattr("kernel_envolve.manager.publication.subprocess.run", fetch)


def test_pr_uses_github_time_and_only_current_verified_best(replaced, monkeypatch):
    verified(replaced)
    github(monkeypatch)
    sync_pr(replaced, "new", "run-1", URL)
    row = detail(replaced, "new")["implementation_pr"]
    assert row["url"] == URL and row["updated_at"] == 1794312000
    current = next(t for t in overview(replaced, compact=True)["tasks"] if t["id"] == "new")
    assert current["implementation_pr"] == row
    verified(replaced, "run-2")
    assert detail(replaced, "new")["implementation_pr"] is None


def test_candidate_or_invalidated_best_cannot_be_published(replaced, monkeypatch):
    github(monkeypatch)
    with pytest.raises(ValueError, match="verified best"):
        sync_pr(replaced, "new", "candidate", URL)
    verified(replaced)
    sync_pr(replaced, "new", "run-1", URL)
    replaced.db.execute("UPDATE runs SET error='source changed' WHERE path='run-1'")
    assert detail(replaced, "new")["implementation_pr"] is None


def test_best_change_during_network_fetch_is_rejected(replaced, monkeypatch):
    verified(replaced)
    github(monkeypatch, lambda: verified(replaced, "run-2"))
    with pytest.raises(ValueError, match="changed"):
        sync_pr(replaced, "new", "run-1", URL)
    assert detail(replaced, "new")["implementation_pr"] is None


def test_bad_url_never_calls_github(replaced, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("must not fetch malformed URL")

    monkeypatch.setattr("kernel_envolve.manager.publication.subprocess.run", forbidden)
    with pytest.raises(ValueError, match="URL"):
        sync_pr(replaced, "new", "candidate", "https://evil.test/pull/1")
