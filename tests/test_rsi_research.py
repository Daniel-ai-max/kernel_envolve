"""Bounded research actions with real CPU evaluator receipts, never synthetic GPU scores."""

import json
import signal
from pathlib import Path

import pytest

from kernel_envolve.rsi.research import run, source_id
from kernel_envolve.storage import write_json


@pytest.fixture
def research(tmp_path):
    task = tmp_path / "task"
    task.mkdir()
    (task / "task.py").write_text(
        "import numpy as np\nimport time\ndef make_inputs(case,seed): return (np.ones(16),)\n"
        "def reference(x): return x+1\ndef baseline(x):\n time.sleep(.006)\n return x+1\n"
    )
    initial = "import time\ndef run(x):\n time.sleep(.003)\n return x+1\n"
    (task / "candidate.py").write_text(initial)
    write_json(
        task / "task.json",
        {
            "name": "research",
            "objective": "Synthetic CPU research lifecycle only",
            "device": "cpu",
            "cases": [{"id": "a", "args": {}, "weight": 1}],
            "warmup": 1,
            "full_trials": 3,
            "quick_trials": 1,
            "full_seeds": [0],
            "quick_seeds": [0],
        },
    )
    packet = {
        "task": str(task),
        "kind": "generic",
        "editable_files": [str(task / "candidate.py")],
        "model": "deterministic-no-model",
        "seed": 0,
        "strategy": "",
        "seconds": 30,
    }
    return packet, tmp_path / "trace", initial


def test_quick_cannot_replace_full_but_can_inform_next_action(research):
    packet, out, initial = research
    source = "def run(x): return x+1\n"

    def decide(request, timeout):
        if request["remaining_actions"] == 2:
            return {
                "action": "propose",
                "source": source,
                "mode": "quick",
                "hypothesis": "Remove synthetic sleep",
            }
        assert request["selected"] == source_id(initial)
        assert request["history"][0]["observation"]["mode"] == "quick"
        return {
            "action": "evaluate",
            "candidate": source_id(source),
            "mode": "full",
            "hypothesis": "Confirm diagnostic with full evaluation",
        }

    trace = run(packet, out, decide)
    assert trace["selected"] == source_id(source)
    assert trace["selected_quality"] > 0.8
    assert Path(packet["editable_files"][0]).read_text() == source
    assert len(trace["steps"]) == 2
    assert not (Path(packet["task"]) / ".kernel_envolve/best.json").exists()


def test_quick_only_keeps_prior_candidate(research):
    packet, out, initial = research
    trace = run(
        packet,
        out,
        lambda *a: {
            "action": "propose",
            "source": "def run(x): return x+1\n",
            "mode": "quick",
            "hypothesis": "Diagnostic only",
        },
        max_actions=1,
    )
    assert trace["selected"] == source_id(initial)
    assert Path(packet["editable_files"][0]).read_text() == initial


def test_duplicate_measurement_is_skipped(research):
    packet, out, initial = research
    trace = run(
        packet,
        out,
        lambda *a: {
            "action": "evaluate",
            "candidate": source_id(initial),
            "mode": "full",
            "hypothesis": "Same experiment",
        },
    )
    assert all(x["observation"]["status"] == "duplicate_skipped" for x in trace["steps"])
    assert len(list((Path(packet["task"]) / ".kernel_envolve/runs").iterdir())) == 1


def test_failed_full_candidate_does_not_replace_known_good(research):
    packet, out, initial = research
    trace = run(
        packet,
        out,
        lambda *a: {
            "action": "propose",
            "source": "def run(x): return x-1\n",
            "mode": "full",
            "hypothesis": "Deliberate incorrect fixture",
        },
        max_actions=1,
    )
    assert trace["steps"][0]["observation"]["correctness"] == "fail"
    assert trace["selected"] == source_id(initial)


def test_transport_failure_preserves_source_and_signal_handlers(research):
    packet, out, initial = research
    old = signal.getsignal(signal.SIGTERM)

    def fail(*args):
        raise TimeoutError("fixture transport timeout")

    with pytest.raises(TimeoutError):
        run(packet, out, fail)
    assert Path(packet["editable_files"][0]).read_text() == initial
    assert signal.getsignal(signal.SIGTERM) == old
    assert json.loads((out / "search-trace.json").read_text())["error"]["type"] == "TimeoutError"
    with pytest.raises(ValueError, match="immutable"):
        run(packet, out, fail)


def test_unknown_action_cannot_change_budget(research):
    packet, out, initial = research
    with pytest.raises(ValueError, match="Expected fields"):
        run(packet, out, lambda *a: {"action": "stop", "hypothesis": "stop", "seconds": 100000})
    assert Path(packet["editable_files"][0]).read_text() == initial


def test_protected_mutation_is_rejected_before_next_evaluation(research):
    packet, out, initial = research

    def mutate(*args):
        (Path(packet["task"]) / "task.py").write_text("tampered")
        return {"action": "stop", "hypothesis": "stop"}

    with pytest.raises(ValueError, match="protected"):
        run(packet, out, mutate)
    assert len(list((Path(packet["task"]) / ".kernel_envolve/runs").iterdir())) == 1


def test_input_limit_stops_before_model_call(research):
    packet, out, initial = research
    packet["strategy"] = "x" * 140000
    trace = run(packet, out, lambda *a: pytest.fail("Unbounded model request"))
    assert trace["stop_reason"] == "decision_input_limit"
    assert trace["selected"] == source_id(initial)


def test_interrupt_during_evaluation_reaps_worker(research):
    import os
    import subprocess
    import sys
    import time

    packet, out, _ = research
    Path(packet["editable_files"][0]).write_text(
        "import time\ndef run(x):\n time.sleep(60)\n return x+1\n"
    )
    pidfile = out.parent / "worker.pid"
    script = """import json,sys
from pathlib import Path
import kernel_envolve.cli as cli
from kernel_envolve.rsi.research import run
original=cli.spawn
def spawn(*args,**kwargs):
    child=original(*args,**kwargs)
    Path(sys.argv[2]).write_text(str(child.pid))
    return child
cli.spawn=spawn
run(json.load(sys.stdin),sys.argv[1],lambda *args: {"action":"stop","hypothesis":"done"})
"""
    child = subprocess.Popen(
        [sys.executable, "-c", script, str(out), str(pidfile)],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=True,
    )
    try:
        child.stdin.write(json.dumps(packet))
        child.stdin.close()
        deadline = time.monotonic() + 10
        while not pidfile.exists() and child.poll() is None and time.monotonic() < deadline:
            time.sleep(0.02)
        assert pidfile.exists()
        worker = int(pidfile.read_text())
        child.send_signal(signal.SIGTERM)
        child.wait(timeout=5)
        with pytest.raises(ProcessLookupError):
            os.kill(worker, 0)
        trace = json.loads((out / "search-trace.json").read_text())
        assert trace["error"]["type"] == "InterruptedError"
    finally:
        if child.poll() is None:
            child.kill()
            child.wait()
        child.stdout.close()
        child.stderr.close()


def test_controller_layout_keeps_trace_outside_nested_task(research):
    packet, out, initial = research
    # trials.trial uses cwd=trial and task=trial/workspace/task. An ancestor output
    # directory is valid, provided evidence files themselves stay outside task.
    trace = run(packet, out.parent, lambda *a: {"action": "stop", "hypothesis": "done"})
    assert trace["selected"] == source_id(initial)
    assert (out.parent / "search-trace.json").exists()
    assert not (Path(packet["task"]) / "search-trace.json").exists()


def test_example_research_adapter_inside_real_trial_controller(research):
    import sys

    from kernel_envolve.rsi.trials import freeze_template, trial

    packet, out, _ = research
    root = out.parent
    adapter = root / "decision.py"
    adapter.write_text("""import hashlib,json,sys
from pathlib import Path
p=json.load(sys.stdin)
s="def run(x): return x+1\\n"
if p['remaining_actions']==2:
    a={'action':'propose','source':s,'mode':'quick','hypothesis':'Remove artificial sleep'}
else:
    a={'action':'evaluate','candidate':hashlib.sha256(s.encode()).hexdigest(),'mode':'full','hypothesis':'Confirm with full'}
Path('decision.json').write_text(json.dumps(a))
""")
    template = freeze_template({"path": packet["task"], "kind": "generic"}, root / "template")
    template["kind"] = "generic"
    example = Path(__file__).parents[1] / "examples/rsi/research_solver.py"
    suite = {
        "model": "deterministic-no-model",
        "solver_seconds": 15,
        "evaluation_seconds": 15,
        "command": [sys.executable, str(example), "--model-command", sys.executable, str(adapter)],
    }
    destination = root / "controller-trial"
    destination.mkdir()
    result = trial(destination, template, suite, [], 0, 30, "cpu")
    assert result["passed"] and result["quality"] > 0.8
    trace = json.loads((destination / "search-trace.json").read_text())
    assert trace["steps"][0]["observation"]["mode"] == "quick"
    assert trace["steps"][1]["observation"]["mode"] == "full"
