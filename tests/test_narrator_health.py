"""Missing narration must be visible without changing research control or results."""

import json
import sys
import threading

import pytest

from kernel_envolve.manager.dashboard import narrator as n
from kernel_envolve.manager.dashboard.views import snapshot

pytest_plugins = ["test_manager"]
KEYS = (
    "KERNEL_ENVOLVE_NARRATOR_COMMAND",
    "NARRATOR_MODEL",
    "NARRATOR_RESPONSES_URL",
    "NARRATOR_API_KEY",
    "NARRATOR_TIMEOUT_SECONDS",
)


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    for key in KEYS:
        monkeypatch.delenv(key, raising=False)


def test_unconfigured_start_is_visible_and_does_not_change_research(managed, capsys):
    store, _, _ = managed
    tasks = store.rows("SELECT * FROM tasks")
    thread = n.start(store.root, threading.Event())
    thread.join(5)
    assert not thread.is_alive()
    assert snapshot(store)["narrator"]["status"] == "disabled"
    assert "Narrator disabled" in capsys.readouterr().err
    assert tasks == store.rows("SELECT * FROM tasks")
    assert not store.rows("SELECT * FROM jobs")


@pytest.mark.parametrize(
    "setting,value",
    [
        ("NARRATOR_API_KEY", "secret-fixture"),
        ("KERNEL_ENVOLVE_NARRATOR_COMMAND", "not-json-secret-fixture"),
    ],
)
def test_bad_configuration_visible_without_leaking_values(
    managed, monkeypatch, capsys, setting, value
):
    store, _, _ = managed
    monkeypatch.setenv(setting, value)
    thread = n.start(store.root, threading.Event())
    thread.join(5)
    state = snapshot(store)["narrator"]
    assert state["status"] == "configuration_error"
    assert value not in json.dumps(state) + capsys.readouterr().err


def test_partial_configuration_is_not_silently_disabled(monkeypatch):
    monkeypatch.setenv("NARRATOR_MODEL", "fixture")
    with pytest.raises(ValueError, match="NARRATOR_API_KEY"):
        n.command_from_env()


def test_health_distinguishes_running_calls_output_and_stale_service(managed, monkeypatch):
    store, _, _ = managed
    n.service_status(store, "running", owns_lock=True)
    assert n.health(store)["status"] == "waiting"
    n.put(store, "narrator_attempt", {"status": "started"})
    assert n.health(store)["status"] == "generating"
    n.put(store, "narrator_attempt", {"status": "failed", "error": "TimeoutError"})
    assert n.health(store)["status"] == "call_failed"
    n.put(store, "narrator_attempt", {"status": "complete", "changed": 0})
    assert n.health(store)["status"] == "no_output"
    n.put(store, "narration:toy", {"fixture": True})
    assert n.health(store)["status"] == "ready"
    service = n.get(store, "narrator_service")
    assert n.health(store, service["heartbeat"] + service["lease"] + 1)["status"] == "unavailable"
    monkeypatch.setattr(n, "alive", lambda *args: False)
    assert n.health(store)["status"] == "unavailable"


def test_disabled_observer_cannot_overwrite_external_narrator_health(managed):
    store, _, _ = managed
    n.service_status(store, "running", owns_lock=True)
    owner = n.get(store, "narrator_service")["owner"]
    thread = n.start(store.root, threading.Event())
    thread.join(5)
    assert n.get(store, "narrator_service")["owner"] == owner
    assert n.health(store)["status"] == "waiting"


@pytest.mark.parametrize(
    "stderr,code", [("HTTPError:404", "HTTPError:404"), ("secret provider body", None)]
)
def test_adapter_exposes_only_safe_http_status(stderr, code):
    with pytest.raises(n.NarratorAdapterError) as caught:
        n.invoke(
            [sys.executable, "-c", f"import sys;sys.stderr.write({stderr!r});sys.exit(1)"],
            {},
            threading.Event(),
        )
    assert caught.value.code == code
    assert stderr not in str(caught.value)


def test_failed_call_keeps_prior_narration_but_never_reports_ready(managed):
    from test_narrator_exclusive import record, reply

    store, _, _ = managed
    record(store)
    assert n.step(store, [], call=reply) == 1
    previous = n.get(store, "narration:toy")
    assert n.health(store)["status"] == "ready"
    record(store)
    calls = []

    def rejected(*args):
        calls.append(True)
        raise n.NarratorAdapterError(1, "HTTPError:404")

    assert n.step(store, [], call=rejected) == 0
    assert n.get(store, "narration:toy") == previous
    assert n.health(store)["status"] == "call_failed"
    assert n.health(store)["last_attempt"]["error_code"] == "HTTPError:404"
    assert n.step(store, [], call=rejected) == 0
    assert len(calls) == 1  # A known failure still cannot cause blind automatic replay.


def test_configured_start_produces_validated_output_and_records_shutdown(
    managed, tmp_path, monkeypatch
):
    import time

    from test_narrator_exclusive import record

    store, _, _ = managed
    record(store)
    before = store.rows("SELECT * FROM tasks")
    adapter = tmp_path / "adapter.py"
    adapter.write_text(
        "import json,sys\n"
        "request=json.load(sys.stdin); output={}\n"
        "for b in request['tasks']:\n"
        " claim={'text':'局部测量已记录。','sources':[next(iter(b['sources']))]}\n"
        " output[b['task']]={'direction':None,'finding':claim,'failure':None,'next':None,"
        "'row':{'revision':b['row_facts']['revision'],'benefit':'system:benefit',"
        "'optimization_pr':'system:optimization_pr','current_progress':'finding',"
        "'latest_result':dict(ref='system:latest_result',**claim)}}\n"
        "print(json.dumps(output))\n"
    )
    monkeypatch.setenv(
        "KERNEL_ENVOLVE_NARRATOR_COMMAND", json.dumps([sys.executable, str(adapter)])
    )
    stop = threading.Event()
    thread = n.start(store.root, stop)
    try:
        deadline = time.monotonic() + 10
        while n.health(store)["status"] != "ready" and time.monotonic() < deadline:
            time.sleep(0.05)
        state = snapshot(store)
        assert state["narrator"]["status"] == "ready"
        assert state["narrator"]["last_attempt"]["changed"] == 1
        assert state["tasks"][0]["narration"]["claims"]["finding"]["text"] == "局部测量已记录。"
    finally:
        stop.set()
        thread.join(5)
    assert not thread.is_alive()
    assert n.health(store)["status"] == "stopped"
    assert store.rows("SELECT * FROM tasks") == before
    assert not store.rows("SELECT * FROM jobs")
