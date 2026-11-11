"""Real CPU/source receipts, synthetic transport tokens and controller durations.

Sleep removal is an engineering fixture, not GPU/model-learning efficacy. No API
calls are made and no noisy host-duration improvement is asserted in these tests.
"""

import ast
import copy
import json
import shutil
import sys
import time
from pathlib import Path

import pytest

from kernel_envolve.cli import evaluate
from kernel_envolve.rsi import artifact_lineage as lineage
from kernel_envolve.rsi import artifact_validation as validation
from kernel_envolve.rsi.artifact_registry import Registry
from kernel_envolve.rsi.model import profile
from kernel_envolve.rsi.research import run
from kernel_envolve.rsi.trials import freeze_template
from kernel_envolve.storage import digest, write_json

INITIAL = "import time\ndef run(x):\n time.sleep(.004)\n return x+1\n"
SEED = "import time\ndef run(x):\n time.sleep(.002)\n return x+1\n"
CHILD = "import time\ndef run(x):\n time.sleep(.001)\n return x+1\n"
CONTROL = "import time\ndef run(x):\n time.sleep(.003)\n return x+1\n"
PREFIX = "Synthetic CPU admission transport, not a model."


def make_task(path, n):
    path.mkdir(parents=True)
    (path / "task.py").write_text(
        "import numpy as np\nimport time\n"
        "def make_inputs(case,seed): return (np.ones(case['n']),)\n"
        "def reference(x): return x+1\n"
        "def baseline(x):\n time.sleep(.008)\n return x+1\n"
    )
    (path / "candidate.py").write_text(INITIAL)
    write_json(
        path / "task.json",
        {
            "name": f"cpu-admission-{n}",
            "objective": "Engineering fixture only",
            "device": "cpu",
            "cases": [{"id": "a", "args": {"n": n}, "weight": 1}],
            "warmup": 1,
            "full_trials": 3,
            "quick_trials": 1,
            "full_seeds": [0],
            "quick_seeds": [0],
        },
    )


def perform(task, out, source, snapshot=None, *, repeat=0, tid=None, calls=None):
    packet = {
        "task": str(task),
        "kind": "generic",
        "editable_files": [str(task / "candidate.py")],
        "model": "deterministic-no-model",
        "seed": repeat,
        "strategy": "",
        "seconds": 30,
    }

    def decide(request, timeout):
        action = (
            {"action": "stop", "hypothesis": "Synthetic stop"}
            if request["history"] and (calls is not None or snapshot is None)
            else {
                "action": "propose",
                "source": source,
                "mode": "full",
                "hypothesis": "Synthetic proposal",
            }
        )
        if calls is not None:
            path = out / f"call-{len(calls)}.json"
            write_json(
                path,
                {
                    "started_at": time.time(),
                    "trial_id": tid,
                    "status": "completed",
                    "request": {
                        "model": packet["model"],
                        "reasoning": {"effort": "medium"},
                        "input": PREFIX + "\n" + json.dumps(request),
                    },
                    "response": {
                        "id": f"synthetic-response-{len(calls)}",
                        "status": "completed",
                        "usage": {"input_tokens": 10, "output_tokens": 2, "total_tokens": 12},
                        "output": [
                            {
                                "type": "message",
                                "content": [{"type": "output_text", "text": json.dumps(action)}],
                            }
                        ],
                    },
                },
            )
            calls.append(str(path))
        return action

    run(
        packet,
        out,
        decide,
        max_actions=2,
        artifact_snapshot=snapshot,
        source_validator=ast.parse,
        compact_history=True,
    )
    final, _ = evaluate(task, "full", timeout=10)
    return {
        "task": str(task),
        "artifacts": str(out),
        "task_profile": {},
        "confirmation_run": str(final),
    }


@pytest.fixture(scope="module")
def complete(tmp_path_factory):
    root = tmp_path_factory.mktemp("artifact-admission")
    archive = root / "archive"
    training = root / "training"
    make_task(training, 16)
    seed_provenance = perform(training, root / "seed-trace", SEED)
    seed = lineage.collect(archive, **seed_provenance)
    scope = lineage.audit(archive, seed)["scope"]
    registry = Registry.create(
        root / "registry", archive, [seed], max_attempts=3, max_seconds=10000, max_tokens=100000
    )
    seed_catalog = registry._get("active")
    learning_receipt = root / "learning-receipt.json"
    phase = registry.declare_learning([{"scope": scope, "receipt": str(learning_receipt)}])
    (training / "candidate.py").write_text(INITIAL)
    child_provenance = perform(
        training, root / "child-trace", CHILD, lineage.validation_snapshot(archive, seed)
    )
    write_json(learning_receipt, {"provenance": child_provenance})
    candidate = registry.collect_learning(phase)["candidate"]
    assert candidate is not None
    child = registry.catalog(candidate)[0]["slots"][scope]
    tasks = []
    for n, partition in [(16, "retention"), (64, "fresh")]:
        task = root / f"validation-{n}"
        make_task(task, n)
        tasks.append(
            {
                **freeze_template({"kind": "generic", "path": str(task)}, root / f"template-{n}"),
                "kind": "generic",
                "scope": scope,
                "profile": profile({}),
                "partition": partition,
            }
        )
    suite = {
        "tasks": tasks,
        "repeats": 2,
        "max_actions": 2,
        "max_quality_regression": 0.02,
        "max_cost_regression": 0.05,
        "min_quality_gain": 0.01,
        "min_cost_gain": 0.1,
        "min_relative_quality_gain": 0.1,
        "min_wall_saving": 0.1,
        "max_token_regression": 0.05,
        "max_seconds": 1000,
        "max_tokens": 10000,
        "model": "deterministic-no-model",
        "reasoning": {"effort": "medium"},
        "request_options": {},
        "prompt_prefix_hash": digest(PREFIX),
        "command": [sys.executable],
        "adapter_files": [],
    }
    attempt = registry.reserve(candidate, suite)
    evidence = {
        "attempt": attempt,
        "trials": [],
        "calls": [],
        "release_receipt": str(root / "released.json"),
    }
    write_json(
        root / "released.json",
        {"phase": "Completed", "pods": [], "live_read_status": {"pods": "available"}},
    )
    for repeat in range(2):
        for index, task in enumerate(tasks):
            for arm in (
                validation.ARMS if (repeat + index) % 2 == 0 else tuple(reversed(validation.ARMS))
            ):
                tid = f"{repeat}-{index}-{arm}"
                directory = root / "validation" / tid
                workspace = directory / "workspace"
                shutil.copytree(task["path"], workspace)
                artifact = (
                    None
                    if arm == "control"
                    else lineage.validation_snapshot(archive, child if arm == "candidate" else seed)
                )
                provenance = perform(
                    workspace / "task",
                    directory,
                    CONTROL,
                    artifact,
                    repeat=repeat,
                    tid=tid,
                    calls=evidence["calls"],
                )
                evidence["trials"].append(
                    {
                        "repeat": repeat,
                        "task": index,
                        "arm": arm,
                        "provenance": provenance,
                        "seconds": 10,
                        "solver_seconds": 9,
                    }
                )
    release = json.loads((root / "released.json").read_text())
    release["at"] = time.time()
    write_json(root / "released.json", release)
    registry.close()
    return {
        "root": root,
        "archive": archive,
        "scope": scope,
        "seed": seed,
        "seed_catalog": seed_catalog,
        "candidate": candidate,
        "child": child,
        "suite": suite,
        "attempt": attempt,
        "evidence": evidence,
        "phase": phase,
    }


@pytest.fixture
def case(complete, tmp_path):
    shutil.copytree(complete["root"] / "registry", tmp_path / "registry")
    registry = Registry(tmp_path / "registry")
    yield registry, complete
    registry.close()


def test_old_evaluator_seed_rejected_before_registry_creation(complete, tmp_path, monkeypatch):
    from kernel_envolve.rsi import artifact_registry

    monkeypatch.setattr(artifact_registry, "evaluator_hash", lambda: "f" * 64)
    target = tmp_path / "incompatible-registry"
    with pytest.raises(ValueError, match="Seed evaluator differs.*revalidate"):
        Registry.create(
            target,
            complete["archive"],
            [complete["seed"]],
            max_attempts=1,
            max_seconds=1000,
            max_tokens=10000,
        )
    assert not target.exists()


def test_real_raw_validation_adoption_disable_and_rollback(case):
    registry, data = case
    result = registry.finish(data["attempt"], data["evidence"])
    assert result["error"] is None
    assert result["decision"]["accepted"], result
    assert result["decision"]["tokens"] == 20 * 12
    assert registry._get("active") == data["seed_catalog"]
    assert registry.adopt(data["candidate"]) == data["candidate"]
    assert (
        registry.snapshot(data["scope"])["identity"]
        == lineage.audit(data["archive"], data["child"])["snapshot"]
    )
    with pytest.raises(ValueError, match="consumed|admitted"):
        registry.adopt(data["candidate"])
    registry.set_enabled(False)
    assert registry.snapshot(data["scope"]) is None
    registry.set_enabled(True)
    assert registry.rollback("Engineering rollback test") == data["seed_catalog"]
    assert registry.catalog(data["candidate"])[1] == "revoked"
    with pytest.raises(ValueError, match="not admitted"):
        registry.adopt(data["candidate"])


@pytest.mark.parametrize(
    "mutation,reason",
    [
        ("missing_trial", "four-arm"),
        ("missing_call", "model requests"),
        ("unknown_usage", "usage"),
        ("reused_response", "reused transport"),
        ("wrong_model", "configuration"),
        ("wrong_action", "executed decision"),
        ("wrong_history", "history"),
        ("wrong_source", "source/observation"),
        ("not_released", "release"),
        ("extra_call", "Duplicate model"),
    ],
)
def test_invalid_evidence_consumes_attempt_without_publication(case, tmp_path, mutation, reason):
    registry, data = case
    evidence = copy.deepcopy(data["evidence"])
    if mutation == "missing_trial":
        evidence["trials"].pop()
    elif mutation == "missing_call":
        evidence["calls"].pop()
    elif mutation == "not_released":
        write_json(
            tmp_path / "release.json", {"phase": "Running", "pods": ["worker"], "at": time.time()}
        )
        evidence["release_receipt"] = str(tmp_path / "release.json")
    elif mutation == "extra_call":
        evidence["calls"].append(evidence["calls"][0])
    else:
        call = json.loads(Path(evidence["calls"][0]).read_text())
        if mutation == "unknown_usage":
            call["response"]["usage"] = None
        elif mutation == "reused_response":
            call["response"]["id"] = json.loads(Path(evidence["calls"][1]).read_text())["response"][
                "id"
            ]
        elif mutation == "wrong_model":
            call["request"]["model"] = "other-model"
        elif mutation == "wrong_action":
            call["response"]["output"][0]["content"][0]["text"] = json.dumps(
                {"action": "stop", "hypothesis": "Other"}
            )
        else:
            packet = json.loads(call["request"]["input"].split("\n", 1)[1])
            if mutation == "wrong_history":
                packet["history"].append({"action": "stop"})
            else:
                packet["candidates"][0]["source"] += "# Changed\n"
            call["request"]["input"] = PREFIX + "\n" + json.dumps(packet)
        write_json(tmp_path / "call.json", call)
        evidence["calls"][0] = str(tmp_path / "call.json")
    result = registry.finish(data["attempt"], evidence)
    assert reason in result["error"]
    assert registry._get("active") == data["seed_catalog"]
    assert registry.catalog(data["candidate"])[1] == "rejected"
    row, _ = registry._attempt(data["attempt"])
    assert row["tokens"] == data["suite"]["max_tokens"]
    with pytest.raises(ValueError, match="consumed"):
        registry.finish(data["attempt"], data["evidence"])


def test_drift_after_validation_cannot_be_adopted(case, tmp_path):
    registry, data = case
    evidence = copy.deepcopy(data["evidence"])
    path = tmp_path / "transport.json"
    shutil.copyfile(evidence["calls"][0], path)
    evidence["calls"][0] = str(path)
    assert registry.finish(data["attempt"], evidence)["decision"]["accepted"]
    path.write_text("{}")
    with pytest.raises((KeyError, ValueError)):
        registry.adopt(data["candidate"])
    assert registry._get("active") == data["seed_catalog"]


def test_adopted_drift_revokes_and_restores_seed(case, tmp_path):
    registry, data = case
    evidence = copy.deepcopy(data["evidence"])
    path = tmp_path / "transport.json"
    shutil.copyfile(evidence["calls"][0], path)
    evidence["calls"][0] = str(path)
    assert registry.finish(data["attempt"], evidence)["decision"]["accepted"]
    registry.adopt(data["candidate"])
    path.unlink()
    assert (
        registry.snapshot(data["scope"])["identity"]
        == lineage.audit(data["archive"], data["seed"])["snapshot"]
    )
    assert registry.catalog(data["candidate"])[1] == "revoked"


def test_abandoned_attempt_stays_charged_after_reopen(case):
    registry, data = case
    row, _ = registry._attempt(data["attempt"])
    assert row["status"] == "reserved" and row["seconds"] == data["suite"]["max_seconds"]
    with pytest.raises(ValueError, match="untried"):
        registry.reserve(data["candidate"], data["suite"])
    with pytest.raises(ValueError, match="consumed"):
        registry.collect_learning(data["phase"])


@pytest.mark.parametrize(
    "field,value",
    [
        ("min_quality_gain", 0.001),
        ("min_cost_gain", 0.01),
        ("max_quality_regression", 0.03),
        ("max_cost_regression", 0.1),
    ],
)
def test_operator_cannot_weaken_original_protections(complete, field, value):
    suite = copy.deepcopy(complete["suite"])
    suite[field] = value
    with pytest.raises(ValueError):
        validation.check_suite(suite, {complete["scope"]: complete["child"]})


def test_repeated_collection_cannot_buy_another_attempt(case, tmp_path):
    registry, data = case
    receipt = tmp_path / "repeated-learning.json"
    phase = registry.declare_learning([{"scope": data["scope"], "receipt": str(receipt)}])
    task = tmp_path / "different-learning-task"
    make_task(task, 128)
    provenance = perform(
        task,
        tmp_path / "different-learning-trace",
        CHILD,
        lineage.validation_snapshot(data["archive"], data["seed"]),
    )
    write_json(receipt, {"provenance": provenance})
    candidate = registry.collect_learning(phase)["candidate"]
    assert candidate is not None and candidate != data["candidate"]
    with pytest.raises(ValueError, match="already attempted"):
        registry.reserve(candidate, data["suite"])
    assert registry.db.execute("SELECT count(*) FROM attempts").fetchone()[0] == 1


def test_validation_contract_cannot_feed_next_learning(case, tmp_path):
    registry, data = case
    receipt = tmp_path / "leaked-learning.json"
    phase = registry.declare_learning([{"scope": data["scope"], "receipt": str(receipt)}])
    write_json(receipt, {"provenance": data["evidence"]["trials"][0]["provenance"]})
    result = registry.collect_learning(phase)
    assert result["candidate"] is None
    assert "Validation data" in result["observations"][0]["error"]


def test_swapping_candidate_with_parent_is_not_activation(case):
    registry, data = case
    evidence = copy.deepcopy(data["evidence"])
    candidate = next(t for t in evidence["trials"] if t["arm"] == "candidate")
    parent = next(t for t in evidence["trials"] if t["arm"] == "parent")
    candidate["provenance"] = parent["provenance"]
    result = registry.finish(data["attempt"], evidence)
    assert "reused across trials" in result["error"] or "actually activated" in result["error"]


def test_final_raw_result_tamper_is_detected(case):
    registry, data = case
    final = Path(data["evidence"]["trials"][0]["provenance"]["confirmation_run"]) / "result.json"
    original = final.read_bytes()
    result = json.loads(original)
    result["correctness"] = "fail"
    try:
        write_json(final, result)
        report = registry.finish(data["attempt"], data["evidence"])
        assert "receipt mismatch" in report["error"]
    finally:
        final.write_bytes(original)


def test_failure_charges_known_overrun_and_unknown_usage(case, tmp_path):
    registry, data = case
    evidence = copy.deepcopy(data["evidence"])
    call = json.loads(Path(evidence["calls"][0]).read_text())
    call["response"]["usage"] = {"input_tokens": 20000, "output_tokens": 0, "total_tokens": 20000}
    call["request"]["model"] = "invalid-model"
    write_json(tmp_path / "expensive.json", call)
    evidence["calls"][0] = str(tmp_path / "expensive.json")
    report = registry.finish(data["attempt"], evidence)
    assert report["error"] and report["charge"]["tokens"] >= 20000
    assert registry._attempt(data["attempt"])[0]["tokens"] >= 20000


def test_disabled_and_stale_parent_cannot_adopt(case):
    registry, data = case
    assert registry.finish(data["attempt"], data["evidence"])["decision"]["accepted"]
    registry.set_enabled(False)
    with pytest.raises(ValueError, match="Disabled"):
        registry.adopt(data["candidate"])
    registry.set_enabled(True)
    registry.rollback("Withdraw seed; pending descendant also revoked")
    assert registry.snapshot(data["scope"]) is None
    assert registry.catalog(data["candidate"])[1] == "revoked"
    with pytest.raises(ValueError, match="not admitted"):
        registry.adopt(data["candidate"])


def test_foreign_reservation_is_rejected(case):
    registry, data = case
    evidence = copy.deepcopy(data["evidence"])
    evidence["attempt"] = "0" * 64
    result = registry.finish(data["attempt"], evidence)
    assert "different admission reservation" in result["error"]


def test_invalidated_during_check_cannot_be_resurrected(case, monkeypatch):
    registry, data = case
    original = validation.verify

    def revoke_during_check(*args):
        result = original(*args)
        registry.rollback("Concurrent operator revocation")
        return result

    monkeypatch.setattr(validation, "verify", revoke_during_check)
    assert registry.finish(data["attempt"], data["evidence"])["decision"]["accepted"]
    assert registry.catalog(data["candidate"])[1] == "revoked"
    assert registry.snapshot(data["scope"]) is None


def test_quality_gain_does_not_excuse_cost_regression(case):
    registry, data = case
    evidence = copy.deepcopy(data["evidence"])
    for trial in evidence["trials"]:
        if trial["arm"] == "candidate":
            trial.update(seconds=20, solver_seconds=19)
    result = registry.finish(data["attempt"], evidence)
    assert result["error"] is None
    assert not result["decision"]["accepted"]
    assert any("seconds regression" in s for s in result["decision"]["reasons"])
    assert registry._get("active") == data["seed_catalog"]


def test_old_transport_is_not_a_new_validation(case, tmp_path):
    registry, data = case
    evidence = copy.deepcopy(data["evidence"])
    call = json.loads(Path(evidence["calls"][0]).read_text())
    call["started_at"] = 1
    write_json(tmp_path / "old.json", call)
    evidence["calls"][0] = str(tmp_path / "old.json")
    result = registry.finish(data["attempt"], evidence)
    assert "predates admission" in result["error"]


def test_model_options_are_frozen(case, tmp_path):
    registry, data = case
    evidence = copy.deepcopy(data["evidence"])
    call = json.loads(Path(evidence["calls"][0]).read_text())
    call["request"]["tools"] = [{"type": "unapproved-tool"}]
    write_json(tmp_path / "different-options.json", call)
    evidence["calls"][0] = str(tmp_path / "different-options.json")
    result = registry.finish(data["attempt"], evidence)
    assert "transport options" in result["error"]


def test_second_generation_retains_previous_cases_and_rolls_back(case, tmp_path):
    registry, data = case
    assert registry.finish(data["attempt"], data["evidence"])["decision"]["accepted"]
    registry.adopt(data["candidate"])
    receipt = tmp_path / "generation2-learning.json"
    phase = registry.declare_learning([{"scope": data["scope"], "receipt": str(receipt)}])
    task = tmp_path / "generation2-training"
    make_task(task, 128)
    provenance = perform(
        task,
        tmp_path / "learning-trace",
        "def run(x): return x+1\n",
        lineage.validation_snapshot(data["archive"], data["child"]),
    )
    write_json(receipt, {"provenance": provenance})
    candidate = registry.collect_learning(phase)["candidate"]
    assert candidate is not None
    catalog = registry.catalog(candidate)[0]
    assert catalog["generation"] == 2 and catalog["parent"] == data["candidate"]
    child = catalog["slots"][data["scope"]]
    suite = copy.deepcopy(data["suite"])
    fresh = tmp_path / "fresh"
    make_task(fresh, 256)
    suite["tasks"][1]["partition"] = "retention"
    suite["tasks"].append(
        {
            **freeze_template({"kind": "generic", "path": str(fresh)}, tmp_path / "fresh-template"),
            "kind": "generic",
            "profile": profile({}),
            "scope": data["scope"],
            "partition": "fresh",
        }
    )
    incomplete = copy.deepcopy(suite)
    incomplete["tasks"].pop(1)  # Drops generation one's newly demonstrated capability.
    with pytest.raises(ValueError, match="Retention omits"):
        registry.reserve(candidate, incomplete)
    attempt = registry.reserve(candidate, suite)
    evidence = {
        "attempt": attempt,
        "trials": [],
        "calls": [],
        "release_receipt": str(tmp_path / "released.json"),
    }
    for repeat, index, arm in validation.trial_order(suite):
        tid = f"{repeat}-{index}-{arm}"
        out = tmp_path / tid
        workspace = out / "workspace"
        shutil.copytree(suite["tasks"][index]["path"], workspace)
        records = {"seed": data["seed"], "parent": data["child"], "candidate": child}
        snapshot = (
            lineage.validation_snapshot(data["archive"], records[arm]) if arm != "control" else None
        )
        provenance = perform(
            workspace / "task",
            out,
            CONTROL,
            snapshot,
            repeat=repeat,
            tid=tid,
            calls=evidence["calls"],
        )
        evidence["trials"].append(
            {
                "repeat": repeat,
                "task": index,
                "arm": arm,
                "provenance": provenance,
                "seconds": 10,
                "solver_seconds": 9,
            }
        )
    write_json(
        tmp_path / "released.json",
        {
            "phase": "Completed",
            "pods": [],
            "live_read_status": {"pods": "available"},
            "at": time.time(),
        },
    )
    report = registry.finish(attempt, evidence)
    assert report["error"] is None
    assert report["decision"]["accepted"], report
    registry.adopt(candidate)
    assert (
        registry.snapshot(data["scope"])["identity"]
        == lineage.audit(data["archive"], child)["snapshot"]
    )
    assert registry.rollback("Second generation engineering rollback") == data["candidate"]
    assert (
        registry.snapshot(data["scope"])["identity"]
        == lineage.audit(data["archive"], data["child"])["snapshot"]
    )


def test_original_screen_cannot_replace_frozen_practical_threshold(case):
    registry, data = case
    suite = copy.deepcopy(data["suite"])
    suite.update(min_relative_quality_gain=0.9, min_wall_saving=0.9)
    report = validation.verify(
        data["archive"], registry._catalogs(data["candidate"]), suite, data["evidence"]
    )
    assert all(d["accepted"] for d in report["comparisons"].values())
    assert not report["accepted"]
    assert any("No practical benefit" in reason for reason in report["reasons"])


def test_practical_threshold_cannot_be_relaxed(complete):
    suite = copy.deepcopy(complete["suite"])
    suite["min_relative_quality_gain"] = 0.05
    with pytest.raises(ValueError, match="ten percent"):
        validation.check_suite(suite, {complete["scope"]: complete["child"]})
