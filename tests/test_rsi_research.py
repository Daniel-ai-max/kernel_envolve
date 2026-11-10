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


def test_reports_bind_attempted_source_not_restored_finalist(research):
    from kernel_envolve.rsi.research import report_records

    packet, out, initial = research
    wrong = "def run(x): return x-1\r\n"
    trace = run(
        packet,
        out,
        lambda *a: {
            "action": "propose",
            "source": wrong,
            "mode": "full",
            "hypothesis": "Deliberate failing attempt",
        },
        max_actions=1,
    )
    assert Path(packet["editable_files"][0]).read_text() == initial
    record = report_records(packet["task"], trace, out)[0]
    assert record["attempted_candidate_id"] == source_id(wrong)
    assert record["final_selected_candidate_id"] == source_id(initial)
    assert record["tested_source_excerpt"] == wrong
    assert record["evaluator_correctness"] == "fail"
    (out / f"candidate-{source_id(wrong)}.py").write_text(initial)
    with pytest.raises(ValueError, match="identity mismatch"):
        report_records(packet["task"], trace, out)


def test_stop_reports_do_not_attribute_final_source_to_an_experiment(research):
    from kernel_envolve.rsi.research import report_records

    packet, out, _ = research
    trace = run(packet, out, lambda *a: {"action": "stop", "hypothesis": "No useful experiment"})
    record = report_records(packet["task"], trace, out)[0]
    assert record["action"] == "stop"
    assert "tested_source_excerpt" not in record and "evaluator_run" not in record


@pytest.fixture
def measured_episode(research):
    packet, out, initial = research
    correct = "def run(x): return x+1\n" + "# complete source, beyond old excerpt\n" * 40
    answers = iter(
        [
            {
                "action": "propose",
                "source": "def run(x): return x-1\n",
                "mode": "full",
                "hypothesis": "Deliberate CPU failure",
            },
            {
                "action": "propose",
                "source": correct,
                "mode": "full",
                "hypothesis": "Test independent correction",
            },
        ]
    )
    trace = run(packet, out, lambda *args: next(answers), max_actions=2)
    return packet, out, trace, correct


def test_episode_preserves_failed_and_complete_successful_code(measured_episode, tmp_path):
    from kernel_envolve.rsi.episodes import build, load, save

    packet, out, trace, correct = measured_episode
    episode = build(packet["task"], out, {"tags": ["cpu-fixture"]})
    assert len(episode["observations"]) == 3
    assert episode["observations"][1]["correctness"] == "fail"
    assert episode["observations"][1]["failed_checks"]["count"] > 0
    assert episode["sources"][trace["selected"]] == correct
    assert len(correct) > 1000
    assert episode["observations"][2]["quality"] > 0.8
    assert episode["actions"][0]["untrusted_hypothesis"] == "Deliberate CPU failure"
    directory = tmp_path / "episodes"
    identity = save(directory, episode)
    assert save(directory, episode) == identity
    assert load(directory, identity) == episode
    changed = dict(episode, selected="0" * 64)
    write_json(directory / f"{identity}.json", changed)
    with pytest.raises(ValueError, match="content changed"):
        load(directory, identity)
    with pytest.raises(ValueError, match="content changed"):
        save(directory, episode)
    assert json.loads((directory / f"{identity}.json").read_text()) == changed


def test_episode_recomputes_scores_and_rejects_forged_selection(measured_episode):
    from kernel_envolve.rsi.episodes import build

    packet, out, trace, _ = measured_episode
    trace["steps"][0]["observation"]["quality"] = 999
    trace["steps"][0]["observation"]["correctness"] = "pass"
    write_json(out / "search-trace.json", trace)
    episode = build(packet["task"], out, {})
    assert episode["observations"][1]["quality"] == -1
    assert episode["observations"][1]["correctness"] == "fail"
    trace["selected"] = trace["steps"][0]["observation"]["candidate"]
    write_json(out / "search-trace.json", trace)
    with pytest.raises(ValueError, match="best observed full"):
        build(packet["task"], out, {})


def test_episode_rejects_tampered_artifact_and_incomplete_trace(measured_episode):
    from kernel_envolve.rsi.episodes import build

    packet, out, trace, correct = measured_episode
    source = out / f"candidate-{trace['selected']}.py"
    source.write_text("def run(x): return x-1\n")
    with pytest.raises(ValueError, match="identity mismatch"):
        build(packet["task"], out, {})
    source.write_text(correct)
    trace["error"] = {"type": "TimeoutError", "message": "Synthetic adapter failure"}
    write_json(out / "search-trace.json", trace)
    with pytest.raises(ValueError, match="Incomplete research trace"):
        build(packet["task"], out, {})


def test_episode_import_maps_only_one_task_and_checks_original_identity(measured_episode, tmp_path):
    import shutil

    from kernel_envolve.rsi.episodes import build

    packet, out, trace, _ = measured_episode
    imported = tmp_path / "imported-task"
    shutil.copytree(packet["task"], imported)
    original = build(packet["task"], out, {})
    assert build(imported, out, {}, origin_task=packet["task"]) == original
    with pytest.raises(ValueError):
        build(imported, out, {})
    trace["steps"][0]["observation"]["run"] = str(tmp_path / "outside")
    write_json(out / "search-trace.json", trace)
    with pytest.raises(ValueError):
        build(imported, out, {}, origin_task=packet["task"])


def test_episode_archive_bounds_and_symlinks_fail_closed(tmp_path):
    from kernel_envolve.rsi.episodes import MAX_EPISODE_BYTES, load, save
    from kernel_envolve.storage import digest

    with pytest.raises(ValueError, match="archive limit"):
        save(tmp_path / "large", {"payload": "x" * MAX_EPISODE_BYTES})
    assert not (tmp_path / "large").exists()
    value = {"payload": "data"}
    identity = digest(value)
    target = tmp_path / "external.json"
    write_json(target, value)
    (tmp_path / f"{identity}.json").symlink_to(target)
    with pytest.raises(OSError):
        load(tmp_path, identity)
    with pytest.raises(ValueError, match="identity"):
        load(tmp_path, "../external")


def test_episode_confirmation_is_a_separate_verified_selected_source(measured_episode):
    from kernel_envolve.cli import evaluate
    from kernel_envolve.rsi.episodes import build

    packet, out, trace, _ = measured_episode
    assert build(packet["task"], out, {})["final_confirmation"] is None
    with pytest.raises(ValueError, match="separate evaluation"):
        build(packet["task"], out, {}, confirmation_run=trace["steps"][-1]["observation"]["run"])
    run_path, _ = evaluate(Path(packet["task"]), "full", timeout=10)
    value = build(packet["task"], out, {}, confirmation_run=run_path)
    observed = value["observations"][value["final_confirmation"]]
    assert observed["source"] == trace["selected"]
    assert observed["mode"] == "full" and observed["correctness"] == "pass"


def test_episode_view_keeps_every_source_and_case_or_rejects(
    measured_episode, tmp_path, monkeypatch
):
    from kernel_envolve.rsi.episodes import build, save, view

    packet, out, _, _ = measured_episode
    episode = build(packet["task"], out, {})
    identity = save(tmp_path / "library", episode)
    projected = view(tmp_path / "library", identity)
    assert projected["sources"] == episode["sources"]
    assert len(projected["observations"]) == len(episode["observations"])
    for actual, original in zip(projected["observations"], episode["observations"]):
        assert len(actual["cases"]) == len(original["decision"].get("cases", []))
    monkeypatch.setattr("kernel_envolve.rsi.episodes.MAX_VIEW_BYTES", 100)
    with pytest.raises(ValueError, match="model input limit"):
        view(tmp_path / "library", identity)


def test_episode_import_preserves_producer_path_spelling(measured_episode, tmp_path):
    from kernel_envolve.rsi.episodes import build

    packet, out, trace, _ = measured_episode
    task = Path(packet["task"])
    alias = tmp_path / "alias-parent"
    alias.symlink_to(task.parent, target_is_directory=True)
    origin = alias / task.name
    observations = [trace["initial_observation"]] + [step["observation"] for step in trace["steps"]]
    for observed in observations:
        observed["run"] = str(origin / Path(observed["run"]).relative_to(task))
    write_json(out / "search-trace.json", trace)
    episode = build(task, out, {}, origin_task=origin)
    assert episode["origin_task"] == str(origin)
    assert len(episode["observations"]) == 3


def test_historical_experience_is_data_not_an_automatic_candidate(research):
    from kernel_envolve.storage import digest

    packet, out, initial = research
    experience = {"source": "def run(x): raise AssertionError('must not execute')"}

    def decide(request, timeout):
        assert request["historical_experience"] == experience
        assert request["selected"] == source_id(initial)
        assert len(request["candidates"]) == 1
        return {"action": "stop", "hypothesis": "No candidate requested"}

    trace = run(packet, out, decide, experience=experience)
    assert trace["selected"] == source_id(initial)
    assert trace["experience_hash"] == digest(experience)
    assert json.loads((out / "experience.json").read_text()) == experience
    assert Path(packet["editable_files"][0]).read_text() == initial


def test_changed_experience_is_rejected_without_changing_candidate(research):
    packet, out, initial = research

    def decide(request, timeout):
        write_json(out / "experience.json", {"fabricated": "changed"})
        return {"action": "stop", "hypothesis": "Tampering fixture"}

    with pytest.raises(ValueError, match="experience snapshot"):
        run(packet, out, decide, experience={"facts": "original"})
    assert Path(packet["editable_files"][0]).read_text() == initial


def test_oversized_experience_fails_before_evaluation(research):
    packet, out, _ = research
    with pytest.raises(ValueError, match="Experience exceeds"):
        run(
            packet,
            out,
            lambda *args: pytest.fail("Unexpected model call"),
            experience={"facts": "x" * 32768},
        )
    assert not (Path(packet["task"]) / ".kernel_envolve/runs").exists()


def artifact(packet, source):
    return {
        "episode_id": "a" * 64,
        "source_hash": source_id(source),
        "task_source_hash": source_id((Path(packet["task"]) / "task.py").read_text()),
        "source": source,
    }


def test_artifact_spends_one_action_and_requires_current_full_success(research):
    import ast

    packet, out, _ = research
    source = "def run(x): return x+1\n"
    calls = []

    def decide(request, timeout):
        calls.append(request)
        assert request["remaining_actions"] == 1
        assert request["selected"] == source_id(source)
        prior = request["history"][0]
        assert prior["decision_kind"] == "artifact_proposal"
        assert prior["model_seconds"] == 0
        assert prior["observation"]["mode"] == "full"
        return {"action": "stop", "hypothesis": "Stop deterministic CPU fixture"}

    trace = run(
        packet,
        out,
        decide,
        max_actions=2,
        artifact_proposal=artifact(packet, source),
        source_validator=ast.parse,
    )
    assert len(calls) == 1 and len(trace["steps"]) == 2
    assert trace["selected_quality"] > 0.8
    assert Path(packet["editable_files"][0]).read_text() == source
    assert not (Path(packet["task"]) / ".kernel_envolve/best.json").exists()


@pytest.mark.parametrize(
    "source",
    [
        "def run(x): return x-1\n",
        "import time\ndef run(x):\n time.sleep(.009)\n return x+1\n",
    ],
)
def test_bad_or_slower_artifact_keeps_measured_incumbent(research, source):
    import ast

    packet, out, initial = research
    trace = run(
        packet,
        out,
        lambda *a: pytest.fail("The single action belongs to the artifact"),
        max_actions=1,
        artifact_proposal=artifact(packet, source),
        source_validator=ast.parse,
    )
    assert len(trace["steps"]) == 1
    assert trace["selected"] == source_id(initial)
    assert Path(packet["editable_files"][0]).read_text() == initial
    assert len(list((Path(packet["task"]) / ".kernel_envolve/runs").iterdir())) == 2


def test_artifact_needs_explicit_validator_and_matching_identity(research):
    import ast

    packet, out, initial = research
    proposal = artifact(packet, "def run(x): return x+1\n")
    with pytest.raises(ValueError, match="explicit source validator"):
        run(packet, out, lambda *a: None, artifact_proposal=proposal)
    with pytest.raises(ValueError, match="identity mismatch"):
        run(
            packet,
            out,
            lambda *a: None,
            artifact_proposal=proposal | {"source_hash": "0" * 64},
            source_validator=ast.parse,
        )
    with pytest.raises(ValueError, match="identity mismatch"):
        run(
            packet,
            out,
            lambda *a: None,
            artifact_proposal=proposal | {"task_source_hash": "0" * 64},
            source_validator=ast.parse,
        )
    assert not (Path(packet["task"]) / ".kernel_envolve/runs").exists()
    assert Path(packet["editable_files"][0]).read_text() == initial


def test_validator_rejects_artifact_before_execution(research):
    packet, out, initial = research

    def reject(source):
        raise ValueError("adapter rejects this artifact")

    with pytest.raises(ValueError, match="adapter rejects"):
        run(
            packet,
            out,
            lambda *a: pytest.fail("Unexpected model call"),
            artifact_proposal=artifact(packet, "raise AssertionError('must not execute')\n"),
            source_validator=reject,
        )
    assert Path(packet["editable_files"][0]).read_text() == initial
    assert len(list((Path(packet["task"]) / ".kernel_envolve/runs").iterdir())) == 1


def test_frozen_artifact_cannot_be_replaced_after_measurement(research):
    import ast

    packet, out, _ = research
    source = "def run(x): return x+1\n"

    def tamper(request, timeout):
        write_json(out / "artifact-proposal.json", artifact(packet, "def run(x): return x-1\n"))
        return {"action": "stop", "hypothesis": "tampering fixture"}

    with pytest.raises(ValueError, match="frozen artifact"):
        run(
            packet,
            out,
            tamper,
            artifact_proposal=artifact(packet, source),
            source_validator=ast.parse,
        )
    assert Path(packet["editable_files"][0]).read_text() == source


def test_artifact_selector_requires_exact_task_runtime_and_confirmed_source(
    measured_episode, tmp_path
):
    import copy

    from kernel_envolve.cli import evaluate
    from kernel_envolve.rsi.episodes import build, save
    from kernel_envolve.rsi.reuse import select

    packet, out, _, correct = measured_episode
    task = Path(packet["task"])
    confirmation, _ = evaluate(task, "full", timeout=10)
    episode = build(task, out, {"tags": ["fixture"]}, confirmation_run=confirmation)
    identity = save(tmp_path / "library", episode)
    environment = episode["observations"][-1]["environment"]
    facts = {"tags": ["fixture"]}
    proposal = select(tmp_path / "library", [identity], task, facts, environment)
    assert proposal["source"] == correct
    assert proposal["source_hash"] == source_id(correct)
    assert proposal["episode_id"] == identity
    assert select(tmp_path / "library", [identity], task, {}, environment) is None
    changed = copy.deepcopy(environment)
    changed["packages"]["numpy"] = "different-version"
    assert select(tmp_path / "library", [identity], task, facts, changed) is None
    cfg = json.loads((task / "task.json").read_text())
    cfg["cases"][0]["args"] = {"new_case_configuration": True}
    write_json(task / "task.json", cfg)
    assert select(tmp_path / "library", [identity], task, facts, environment) == proposal
    cfg["atol"] = 1.0
    write_json(task / "task.json", cfg)
    assert select(tmp_path / "library", [identity], task, facts, environment) is None
    cfg.pop("atol")
    write_json(task / "task.json", cfg)
    (task / "task.py").write_text((task / "task.py").read_text() + "# changed oracle module\n")
    assert select(tmp_path / "library", [identity], task, facts, environment) is None


def test_unconfirmed_episode_does_not_supply_an_artifact(measured_episode, tmp_path):
    from kernel_envolve.rsi.episodes import build, save
    from kernel_envolve.rsi.reuse import select

    packet, out, _, _ = measured_episode
    episode = build(packet["task"], out, {})
    identity = save(tmp_path / "library", episode)
    env = episode["observations"][-1]["environment"]
    assert select(tmp_path / "library", [identity], packet["task"], {}, env) is None
    for ids in ([], [identity, identity]):
        with pytest.raises(ValueError, match="unique approved"):
            select(tmp_path / "library", ids, packet["task"], {}, env)


@pytest.mark.parametrize("recover_timeout", [False, True])
def test_artifact_trial_still_requires_independent_final_evaluation(research, recover_timeout):
    import sys

    from kernel_envolve.rsi.trials import freeze_template, trial

    packet, out, _ = research
    root = out.parent
    source = "def run(x): return x+1\n"
    proposal_file = root / "approved-artifact.json"
    write_json(proposal_file, artifact(packet, source))
    adapter = root / "reuse-adapter.py"
    adapter.write_text(
        "import ast,json,sys\nfrom pathlib import Path\n"
        "from kernel_envolve.rsi.research import run\n"
        f"proposal=json.loads(Path({str(proposal_file)!r}).read_text())\n"
        f"run(json.load(sys.stdin),Path.cwd(),lambda *a: (_ for _ in ()).throw(TimeoutError('fixture')),max_actions={2 if recover_timeout else 1},artifact_proposal=proposal,source_validator=ast.parse,stop_on_decision_timeout={recover_timeout!r})\n"
    )
    template = freeze_template({"path": packet["task"], "kind": "generic"}, root / "template")
    template["kind"] = "generic"
    suite = {
        "model": "deterministic-no-model",
        "solver_seconds": 15,
        "evaluation_seconds": 15,
        "command": [sys.executable, str(adapter)],
        "adapter_files": [str(proposal_file)],
    }
    destination = root / "artifact-trial"
    destination.mkdir()
    result = trial(destination, template, suite, [], 0, 30, "cpu")
    trace = json.loads((destination / "search-trace.json").read_text())
    assert result["passed"] and result["quality"] > 0.8
    assert trace["selected"] == source_id(source)
    assert result["run"] != trace["initial_observation"]["run"]
    assert result["run"] != trace["steps"][0]["observation"]["run"]
    assert (Path(result["run"]) / "source/candidate.py").read_text() == source


@pytest.mark.parametrize("mode", ["quick", "full"])
def test_opt_in_decision_timeout_keeps_only_full_incumbent(research, mode):
    from kernel_envolve.rsi.research import report_records

    packet, out, initial = research
    source = "def run(x): return x+1\n"
    calls = []

    def decide(request, timeout):
        calls.append(request["remaining_actions"])
        if len(calls) == 1:
            return {
                "action": "propose",
                "source": source,
                "mode": mode,
                "hypothesis": "CPU fixture",
            }
        raise TimeoutError("private transport detail must not become archive content")

    old = signal.getsignal(signal.SIGTERM)
    trace = run(packet, out, decide, max_actions=4, stop_on_decision_timeout=True)
    assert calls == [4, 3]  # Failure consumes its action, with no retry or extra decisions.
    assert trace["stop_reason"] == "decision_timeout" and "error" not in trace
    assert trace["selected"] == source_id(source if mode == "full" else initial)
    assert Path(packet["editable_files"][0]).read_text() == (source if mode == "full" else initial)
    stop = trace["steps"][-1]
    assert stop["decision_error"] == {"type": "TimeoutError", "tokens": None}
    assert stop["stop_origin"] == "controller" and "observation" not in stop
    assert stop["model_seconds"] >= 0
    assert signal.getsignal(signal.SIGTERM) == old
    records = report_records(packet["task"], trace, out)
    assert records[-1]["reported_decision_error"] == stop["decision_error"]
    assert "evaluator_run" not in records[-1]
    assert "private transport detail" not in (out / "search-trace.json").read_text()


@pytest.mark.parametrize(
    "failure", ["invalid_incumbent", "protected_mutation", "cancel", "other_error"]
)
def test_timeout_fallback_does_not_bypass_fail_closed_boundaries(research, failure):
    packet, out, _ = research
    if failure == "invalid_incumbent":
        Path(packet["editable_files"][0]).write_text("def run(x): return x-1\n")

    def fail(*args):
        if failure == "protected_mutation":
            (Path(packet["task"]) / "task.py").write_text("tampered")
        if failure == "cancel":
            raise InterruptedError("cancelled")
        if failure == "other_error":
            raise ValueError("invalid transport result")
        raise TimeoutError("timeout")

    expected = {
        "invalid_incumbent": TimeoutError,
        "protected_mutation": ValueError,
        "cancel": InterruptedError,
        "other_error": ValueError,
    }[failure]
    with pytest.raises(expected):
        run(packet, out, fail, stop_on_decision_timeout=True)
    assert "error" in json.loads((out / "search-trace.json").read_text())


def test_timeout_fallback_does_not_swallow_source_validation_timeout(research):
    packet, out, _ = research

    def validator(source):
        raise TimeoutError("source validation did not complete")

    with pytest.raises(TimeoutError, match="source validation"):
        run(
            packet,
            out,
            lambda *a: {
                "action": "propose",
                "source": "def run(x): return x+1\n",
                "mode": "full",
                "hypothesis": "fixture",
            },
            source_validator=validator,
            stop_on_decision_timeout=True,
        )
    assert "error" in json.loads((out / "search-trace.json").read_text())


def test_timeout_fallback_retains_failure_in_independently_confirmed_episode(research):
    import ast

    from kernel_envolve.cli import evaluate
    from kernel_envolve.rsi.episodes import build, save, view

    packet, out, _ = research
    source = "def run(x): return x+1\n"

    def timeout(*args):
        raise TimeoutError("synthetic decision failure")

    trace = run(
        packet,
        out,
        timeout,
        max_actions=4,
        artifact_proposal=artifact(packet, source),
        source_validator=ast.parse,
        stop_on_decision_timeout=True,
    )
    assert len(trace["steps"]) == 2 and trace["selected"] == source_id(source)
    confirmation, result = evaluate(Path(packet["task"]), "full", timeout=15)
    assert result["correctness"] == "pass"
    assert str(confirmation) != trace["steps"][0]["observation"]["run"]
    episode = build(packet["task"], out, {}, confirmation_run=confirmation)
    assert episode["actions"][-1]["reported_decision_error"] == {
        "type": "TimeoutError",
        "tokens": None,
    }
    library = out.parent / "library"
    identity = save(library, episode)
    assert view(library, identity)["actions"][-1]["reported_stop_origin"] == "controller"
