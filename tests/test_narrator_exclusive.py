"""A replacement narrator owns the old lock without owning research scheduling."""

import os
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest

from kernel_envolve.manager import narrator as n
from kernel_envolve.manager.cli import parser
from kernel_envolve.manager.process import exclusive
from kernel_envolve.manager.store import Store

pytest_plugins = ["test_manager"]


def reply(command, request, stop):
    output = {}
    for batch in request["tasks"]:
        claim = {"text": "局部测量已记录。", "sources": [next(iter(batch["sources"]))]}
        output[batch["task"]] = dict(
            direction=None,
            finding=claim,
            failure=None,
            next=None,
            row=dict(
                revision=batch["row_facts"]["revision"],
                benefit="system:benefit",
                optimization_pr="system:optimization_pr",
                current_progress="finding",
                latest_result=dict(ref="system:latest_result", **claim),
            ),
        )
    return output


def record(store):
    store.event("toy", "research_context", {"text": "局部测量已记录。"})


def test_takeover_waits_for_inflight_step_and_old_steps_skip_without_consuming(
    managed, monkeypatch
):
    store, _, _ = managed
    record(store)
    entered, release, acquired, stop = (threading.Event() for _ in range(4))
    calls, errors = [], []
    original = n._step_locked

    def old_model(*args):
        calls.append("old")
        entered.set()
        assert release.wait(3)
        return reply(*args)

    def new_model(*args):
        calls.append("new")
        return reply(*args)

    def owned_step(*args, **kwargs):
        result = original(*args, **{**kwargs, "call": new_model})
        acquired.set()
        return result

    def old_worker():
        local = Store(store.root)
        try:
            n.step(local, [], call=old_model)
        except BaseException as exc:
            errors.append(exc)
        finally:
            local.close()

    def service():
        try:
            n.serve(store.root, [], stop, exclusive_service=True)
        except BaseException as exc:
            errors.append(exc)

    old = threading.Thread(target=old_worker)
    replacement = threading.Thread(target=service)
    old.start()
    try:
        assert entered.wait(3)
        monkeypatch.setattr(n, "_step_locked", owned_step)
        replacement.start()
        assert not acquired.wait(0.15)
        release.set()
        old.join(3)
        assert acquired.wait(3)
        # Same cursor means takeover does not repeat the old model request.
        assert calls == ["old"]
        record(store)
        metadata = store.rows("SELECT * FROM metadata ORDER BY key")
        began = time.monotonic()
        assert n.step(store, [], call=new_model) == 0
        assert time.monotonic() - began < 0.5
        assert store.rows("SELECT * FROM metadata ORDER BY key") == metadata
        assert store.schedule()  # Real research admission is independent of the narrator lock.
    finally:
        release.set()
        stop.set()
        old.join(3)
        if replacement.ident:
            replacement.join(3)
    assert not old.is_alive() and not replacement.is_alive() and not errors
    monkeypatch.setattr(n, "_step_locked", original)
    assert n.step(store, [], call=new_model) == 1
    assert calls == ["old", "new"]


def test_exclusive_service_reuses_cursor_and_releases_on_normal_or_exceptional_exit(
    managed, monkeypatch
):
    store, _, _ = managed
    record(store)
    original, calls = n._step_locked, []

    def model(*args):
        calls.append(1)
        return reply(*args)

    monkeypatch.setattr(
        n, "_step_locked", lambda *args, **kw: original(*args, **{**kw, "call": model})
    )
    for _ in range(2):
        n.serve(store.root, [], threading.Event(), once=True, exclusive_service=True)
    assert calls == [1]
    with exclusive(store.root / "narrator.lock") as handle:
        assert handle is not None

    def crash(*args, **kwargs):
        raise RuntimeError("synthetic service failure")

    monkeypatch.setattr(n, "_step_locked", crash)
    with pytest.raises(RuntimeError, match="synthetic"):
        n.serve(store.root, [], threading.Event(), once=True, exclusive_service=True)
    with exclusive(store.root / "narrator.lock") as handle:
        assert handle is not None


def test_waiting_exclusive_service_can_stop_without_consuming_input(managed):
    store, _, _ = managed
    record(store)
    stop = threading.Event()
    with exclusive(store.root / "narrator.lock"):
        thread = threading.Thread(
            target=n.serve, args=(store.root, [], stop), kwargs={"exclusive_service": True}
        )
        thread.start()
        stop.set()
        thread.join(3)
        assert not thread.is_alive()
        assert n.get(store, "narrator_attempt") is None


def test_exclusive_response_still_checks_current_row_revision(managed, monkeypatch):
    store, _, _ = managed
    record(store)
    original = n._step_locked

    def model(*args):
        store.db.execute("UPDATE tasks SET desired='paused' WHERE id='toy'")
        return reply(*args)

    monkeypatch.setattr(
        n, "_step_locked", lambda *args, **kw: original(*args, **{**kw, "call": model})
    )
    n.serve(store.root, [], threading.Event(), once=True, exclusive_service=True)
    assert n.get(store, "narration:toy") is None
    assert len(store.rows("SELECT * FROM metadata WHERE key LIKE 'narration_history:%'")) == 1


def test_cli_parses_exclusive_service_without_changing_adapter_arguments():
    args = parser().parse_args(["narrate", "--exclusive-service", "--", "adapter", "--flag"])
    assert args.exclusive_service and args.command == ["--", "adapter", "--flag"]


def test_cli_sigterm_releases_service_lock_without_stopping_research(managed, tmp_path):
    store, _, _ = managed
    record(store)
    adapter = tmp_path / "adapter.py"
    adapter.write_text(
        "import json,sys; r=json.load(sys.stdin); "
        "print(json.dumps({b['task']:None for b in r['tasks']}))"
    )
    process = subprocess.Popen(
        [
            sys.executable,
            "-c",
            "from kernel_envolve.manager.cli import main; raise SystemExit(main())",
            "--state",
            str(store.root),
            "narrate",
            "--exclusive-service",
            "--",
            sys.executable,
            str(adapter),
        ],
        env={**os.environ, "PYTHONPATH": str(Path(n.__file__).resolve().parents[2])},
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
    )
    try:
        deadline = time.monotonic() + 5
        while (n.get(store, "narrator_attempt") or {}).get("status") != "complete":
            assert process.poll() is None
            assert time.monotonic() < deadline
            time.sleep(0.02)
        assert n.step(store, [], call=reply) == 0
        assert store.task("toy")["desired"] == "active"
        process.terminate()
        assert process.wait(3) == 0
        with exclusive(store.root / "narrator.lock") as handle:
            assert handle is not None
        assert store.task("toy")["desired"] == "active"
        assert store.schedule()
    finally:
        if process.poll() is None:
            process.kill()
        process.communicate(timeout=3)
