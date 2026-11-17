"""Narration is asynchronous explanation; no evaluator or scheduling authority."""

import copy
import os
import sys
import threading
import time

import pytest

from kernel_envolve.manager import narrator as n
from kernel_envolve.manager.archive import Archive, export_archive
from kernel_envolve.manager.store import Store
from kernel_envolve.manager.views import task_view

pytest_plugins = ["test_manager"]


def explain(command, request, stop):
    return {
        b["task"]: {
            "direction": None,
            "finding": {
                "text": "长序列局部测试更快，完整验收尚未完成。",
                "sources": [next(iter(b["sources"]))],
            },
            "failure": None,
            "next": None,
        }
        for b in request["tasks"]
    }


def record(store, task="toy"):
    store.event(
        task, "bridge_execution", {"command": ["python", "probe.py"], "result": "local faster"}
    )


def test_shared_batch_deduplicates_restarts_and_preserves_real_state(managed, tmp_path):
    store, deployment, _ = managed
    other = copy.deepcopy(deployment)
    other["tasks"][0]["id"] = "second"
    other["tasks"][0]["workspace"] += "-second"
    store.register(other)
    record(store)
    record(store, "second")
    before = store.rows("SELECT * FROM tasks")
    clocks = task_view(store, "toy")["progress_as_of"]
    calls = []

    def model(*args):
        calls.append(args[1])
        return explain(*args)

    assert n.step(store, [], call=model) == 2
    assert len(calls) == 1
    assert n.step(store, [], call=model) == 0
    reopened = Store(store.root)
    try:
        assert n.step(reopened, [], call=model) == 0
    finally:
        reopened.close()
    assert len(calls) == 1
    assert before == store.rows("SELECT * FROM tasks")
    view = task_view(store, "toy")
    assert view["progress_as_of"] == clocks
    assert view["best"] is None and view["verified_assessment"] is None
    assert not store.rows("SELECT * FROM jobs")
    assert not store.rows("SELECT * FROM promotions")
    assert view["narration"]["claims"]["finding"]["text"].startswith("长序列")
    output = tmp_path / "archive"
    export_archive(store, output)
    restored = Archive(output).read()
    assert restored["details"]["toy"]["narration"] == view["narration"]
    assert restored["state"]["tasks"][0]["narration"]


def test_failure_retains_previous_and_never_replays_same_input(managed):
    store, _, _ = managed
    record(store)
    n.step(store, [], call=explain)
    saved = n.get(store, "narration:toy")
    record(store)
    calls = []

    def broken(*args):
        calls.append(1)
        raise TimeoutError("secret must not be stored")

    assert n.step(store, [], call=broken) == 0
    assert n.get(store, "narration:toy") == saved
    assert n.step(store, [], call=broken) == 0 and len(calls) == 1
    assert n.get(store, "narrator_attempt")["error"] == "TimeoutError"
    record(store)

    def unchanged(*args):
        return {"toy": saved["claims"]}

    assert n.step(store, [], call=unchanged) == 0  # identical cited finding retains its timestamp


@pytest.mark.parametrize(
    "mutate",
    [
        lambda v: v["toy"].update(best={"baseline_speedup": 2}),
        lambda v: v["toy"]["finding"].update(sources=["event:not-in-input"]),
        lambda v: v["toy"]["finding"].update(text="x" * 161),
        lambda v: v.update(other=v.pop("toy")),
    ],
)
def test_untrusted_output_cannot_change_facts_or_invent_sources(managed, mutate):
    store, _, _ = managed
    record(store)

    def bad(*args):
        value = explain(*args)
        mutate(value)
        return value

    assert n.step(store, [], call=bad) == 0
    assert n.get(store, "narration:toy") is None
    assert n.get(store, "narrator_attempt")["status"] == "failed"


def test_incremental_trace_reads_rotation_and_no_information(managed, tmp_path):
    store, _, _ = managed
    path = tmp_path / "trace"
    path.write_text("first\n")
    value, cursor = n.read_delta(path, {})
    assert value["start"] == 0
    assert n.read_delta(path, cursor)[0] is None
    with path.open("a") as stream:
        stream.write("second\n")
    value, cursor = n.read_delta(path, cursor)
    assert value["text"] == "second\n"
    path.write_text("new")
    value, cursor = n.read_delta(path, cursor)
    assert value["start"] == 0 and value["text"] == "new"
    path.write_text("x" * 10000)
    value, _ = n.read_delta(path, cursor)
    assert len(value["text"]) == 8192 and value["truncated"]
    link = tmp_path / "symlink"
    link.symlink_to(path)
    assert n.read_delta(link, {})[0] is None
    fifo = tmp_path / "fifo"
    os.mkfifo(fifo)
    assert n.read_delta(fifo, {})[0] is None
    assert not n.collect(store)


def test_slow_narrator_does_not_hold_database_or_research_lock(managed):
    store, _, _ = managed
    record(store)
    entered, release = threading.Event(), threading.Event()

    def model(*args):
        entered.set()
        assert release.wait(3)
        return explain(*args)

    def run():
        local = Store(store.root)
        try:
            n.step(local, [], call=model)
        finally:
            local.close()

    thread = threading.Thread(target=run)
    thread.start()
    try:
        assert entered.wait(3)
        store.event("toy", "bridge_execution", {"new": "work continues"})
        assert store.schedule()  # actual admission can progress while model is waiting
        assert task_view(store, "toy")["narration"] is None
    finally:
        release.set()
        thread.join(3)
    assert not thread.is_alive()


def test_adapter_timeout_cancellation_and_gpu_environment(tmp_path, monkeypatch):
    adapter = tmp_path / "adapter.py"
    adapter.write_text(
        "import json,os,sys; json.load(sys.stdin); print(json.dumps({'gpu':os.environ.get('CUDA_VISIBLE_DEVICES'),'state':os.environ.get('KERNEL_ENVOLVE_STATE')}))"
    )
    monkeypatch.setenv("KERNEL_ENVOLVE_STATE", "/private/state")
    assert n.invoke([sys.executable, str(adapter)], {}, threading.Event()) == {
        "gpu": "",
        "state": None,
    }
    adapter.write_text("import time;time.sleep(30)")
    start = time.monotonic()
    with pytest.raises(TimeoutError):
        n.invoke([sys.executable, str(adapter)], {}, threading.Event(), timeout=0.1)
    assert time.monotonic() - start < 3


def test_trace_only_change_is_detected_without_new_event(managed):
    store, _, _ = managed
    job = store.schedule()[0]
    directory = store.root / "jobs" / job
    directory.mkdir(parents=True)
    trace = directory / "stdout.log"
    trace.write_text("checking layouts\n")
    assert n.step(store, [], call=explain) == 1
    assert n.step(store, [], call=explain) == 0
    with trace.open("a") as stream:
        stream.write("different layout improves\n")
    batches = n.collect(store)
    assert any("different layout" in source["text"] for source in batches[0]["sources"].values())


def test_responses_adapter_is_tool_free_bounded_and_rejects_incomplete(monkeypatch):
    import http.server
    import json

    from kernel_envolve.manager.narrator_client import respond

    seen = []
    incomplete = [False]

    class Handler(http.server.BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_POST(self):
            seen.append(json.loads(self.rfile.read(int(self.headers["Content-Length"]))))
            assert self.headers["Authorization"] == "Bearer test-only"
            reply = {
                "status": "incomplete" if incomplete[0] else "completed",
                "output": [
                    {
                        "type": "message",
                        "content": [{"type": "output_text", "text": '{"toy":null}'}],
                    }
                ],
            }
            body = json.dumps(reply).encode()
            self.send_response(200)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    with http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler) as server:
        thread = threading.Thread(target=server.serve_forever)
        thread.start()
        monkeypatch.setenv(
            "NARRATOR_RESPONSES_URL", f"http://127.0.0.1:{server.server_port}/v1/responses"
        )
        monkeypatch.setenv("NARRATOR_MODEL", "fixture-model")
        monkeypatch.setenv("NARRATOR_REASONING_EFFORT", "low")
        monkeypatch.setenv("NARRATOR_API_KEY", "test-only")
        try:
            assert respond({"instructions": "JSON only", "tasks": []}) == {"toy": None}
            assert seen[0]["tools"] == [] and seen[0]["store"] is False
            assert seen[0]["stream"] is False
            assert seen[0]["reasoning"] == {"effort": "low"}
            assert seen[0]["text"]["format"]["strict"] is True
            assert seen[0]["text"]["format"]["type"] == "json_schema"
            assert seen[0]["max_output_tokens"] == 8192
            incomplete[0] = True
            with pytest.raises(ValueError, match="incomplete"):
                respond({"instructions": "JSON only", "tasks": []})
            assert len(seen) == 2  # no retry loops
        finally:
            server.shutdown()
            thread.join()


def test_shared_service_stops_inflight_model_without_holding_research(
    managed, tmp_path, monkeypatch
):
    import json

    store, _, _ = managed
    record(store)
    marker = tmp_path / "started"
    adapter = tmp_path / "adapter.py"
    adapter.write_text(
        f"import json,sys,time\nfrom pathlib import Path\njson.load(sys.stdin)\nPath({str(marker)!r}).write_text('started')\ntime.sleep(30)\n"
    )
    monkeypatch.setenv(
        "KERNEL_ENVOLVE_NARRATOR_COMMAND", json.dumps([sys.executable, str(adapter)])
    )
    stopped = threading.Event()
    worker = n.start(store.root, stopped)
    deadline = time.monotonic() + 5
    try:
        while not marker.exists() and time.monotonic() < deadline:
            time.sleep(0.02)
        assert marker.exists()
        assert store.schedule()
    finally:
        stopped.set()
        worker.join(2)
    assert not worker.is_alive()
    assert n.get(store, "narrator_attempt")["status"] == "failed"


def test_chat_adapter_is_explicit_tool_free_and_rejects_tool_calls(monkeypatch):
    import json

    from kernel_envolve.manager.narrator_client import respond

    seen = []
    reply = {"choices": [{"finish_reason": "stop", "message": {"content": '{"toy":null}'}}]}

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def read(self, size):
            return json.dumps(reply).encode()

    class Opener:
        def open(self, request, timeout):
            seen.append(json.loads(request.data))
            return Response()

    monkeypatch.setattr("urllib.request.build_opener", lambda *args: Opener())
    monkeypatch.setenv("NARRATOR_RESPONSES_URL", "https://example.test/v1/chat/completions")
    monkeypatch.setenv("NARRATOR_MODEL", "fixture-model")
    monkeypatch.setenv("NARRATOR_API_KEY", "test-only")
    monkeypatch.setenv("NARRATOR_API_STYLE", "chat_completions")
    monkeypatch.setenv("NARRATOR_REASONING_EFFORT", "low")
    assert respond({"instructions": "JSON only", "tasks": []}) == {"toy": None}
    assert seen[0]["tools"] == [] and seen[0]["store"] is False
    assert seen[0]["response_format"]["json_schema"]["strict"] is True
    assert seen[0]["reasoning_effort"] == "low"
    reply["choices"][0]["message"]["tool_calls"] = [{"name": "forbidden"}]
    with pytest.raises(ValueError, match="only return text"):
        respond({"instructions": "JSON only", "tasks": []})
    reply["choices"][0]["finish_reason"] = "length"
    with pytest.raises(ValueError, match="incomplete"):
        respond({"instructions": "JSON only", "tasks": []})
