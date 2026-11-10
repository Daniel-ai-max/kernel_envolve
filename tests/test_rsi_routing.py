"""Conditional advice must be executable, matched and covered before publication."""

import importlib.util
import json
import shutil
from pathlib import Path

import pytest

from kernel_envolve.manager.config import validate as validate_manager
from kernel_envolve.manager.store import Store as Manager
from kernel_envolve.rsi.config import initialize
from kernel_envolve.rsi.config import validate as validate_config
from kernel_envolve.rsi.context import for_task
from kernel_envolve.rsi.learning import behavior_identity, collect, learn, stage
from kernel_envolve.rsi.model import context, select_rules
from kernel_envolve.rsi.process import invoke
from kernel_envolve.rsi.store import Store
from kernel_envolve.rsi.trials import adopt, validate
from kernel_envolve.storage import write_json


def advice(**conditions):
    return {
        "id": "avoid-sleep",
        "when": "Synthetic CPU workload with artificial sleep",
        "action": "Remove artificial sleep",
        "unless": "Never change task semantics",
        "evidence": ["a" * 64],
        **conditions,
    }


def test_missing_facts_do_not_match_and_legacy_advice_is_preserved():
    generic = advice()
    scoped = {**advice(applies_to=["reduction"], requires=["full"]), "id": "specific"}
    assert select_rules([generic, scoped], {}) == [generic]
    assert select_rules([scoped], {"tags": ["pointwise"], "capabilities": ["full"]}) == []
    assert select_rules([scoped], {"tags": ["reduction"], "capabilities": []}) == []
    assert select_rules([scoped], {"tags": ["reduction"], "capabilities": ["full"]}) == [scoped]


@pytest.fixture
def routing(tmp_path):
    spec = importlib.util.spec_from_file_location(
        "routing_demo", Path(__file__).parents[1] / "examples/rsi/demo.py"
    )
    demo = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(demo)
    cfg_path = demo.create(tmp_path / "demo")
    cfg = json.loads(cfg_path.read_text())
    manager = Manager(cfg["manager"])
    try:
        second = {k: v for k, v in manager.task("demo")["spec"].items() if k != "identity"}
        location = tmp_path / "second-production"
        shutil.copytree(second["workspace"], location)
        second.update(id="other", workspace=str(location), task_dir=str(location))
        manager.register(
            validate_manager(
                {
                    "version": 1,
                    "resources": [manager.resource("cpu")["spec"]],
                    "tasks": [second],
                },
                tmp_path,
            )
        )
    finally:
        manager.close()
    cfg["scopes"]["demo"].append("other")
    cfg["profiles"] = {
        "demo": {"tags": ["reduction"], "capabilities": ["full"]},
        "other": {"tags": ["pointwise"], "capabilities": []},
    }
    cfg["suites"]["demo"]["tasks"][0]["profile"] = cfg["profiles"]["demo"]
    cfg["suites"]["demo"]["tasks"][1]["profile"] = {"tags": ["pointwise"], "capabilities": ["full"]}
    # Keep the untouched fixture unpromotable in both arms so timing noise cannot
    # change its quality. The two matched fixtures still execute real improvements.
    untouched = Path(cfg["suites"]["demo"]["tasks"][1]["path"])
    (untouched / "candidate.py").write_text(
        "import time\ndef run(x):\n time.sleep(.012)\n return x+1\n"
    )
    third = tmp_path / "third-validation"
    shutil.copytree(cfg["suites"]["demo"]["tasks"][0]["path"], third)
    third_config = json.loads((third / "task.json").read_text())
    third_config["name"] = "third-validation"
    third_config["cases"][0]["args"]["n"] = 48
    write_json(third / "task.json", third_config)
    cfg["suites"]["demo"]["tasks"].append(
        {
            "path": str(third),
            "kind": "generic",
            "profile": cfg["profiles"]["demo"],
        }
    )
    write_json(cfg_path, cfg)
    initialize(tmp_path / "state", cfg_path)
    store = Store(tmp_path / "state")
    collect(store)
    ref = store.db.execute("SELECT id FROM evidence LIMIT 1").fetchone()[0]
    proposal = {
        "scope": "demo",
        "parent": "base",
        "hypothesis": "Test scoped synthetic improvement",
        "upsert": [dict(advice(applies_to=["reduction"], requires=["full"]), evidence=[ref])],
        "remove": [],
    }
    yield store, proposal, cfg, cfg_path
    store.close()


def test_validation_and_publication_use_identical_routing(routing, monkeypatch):
    store, proposal, _, _ = routing
    seen = []

    def measured(*args, **kwargs):
        elapsed = invoke(*args, **kwargs)
        if "kind" in args[2]:
            seen.append(args[2])
            return 0.1
        return elapsed

    monkeypatch.setattr("kernel_envolve.rsi.trials.invoke", measured)
    identity = stage(store, proposal)
    result = validate(store, identity)
    assert result["accepted"], result
    assert sum(bool(packet["strategy"]) for packet in seen) == 4
    assert all("Remove artificial sleep" in p["strategy"] for p in seen if p["strategy"])
    adopt(store, result["receipt"])
    manager = Manager(store.config["manager"], readonly=True)
    try:
        text = for_task(manager.root, manager.task("demo"))
        assert context(proposal["upsert"]) in text
        assert for_task(manager.root, manager.task("other")) == ""
    finally:
        manager.close()


@pytest.mark.parametrize(
    "conditions",
    [
        {"applies_to": ["pointwise"], "requires": ["full"]},
        {"applies_to": ["reduction"], "requires": ["profiler"]},
        {},
    ],
)
def test_unusable_or_implicit_scoped_rule_is_rejected(routing, conditions):
    store, proposal, _, _ = routing
    ref = proposal["upsert"][0]["evidence"]
    proposal["upsert"] = [dict(advice(**conditions), evidence=ref)]
    with pytest.raises(ValueError, match="profile|applicable|capabilit|conditions"):
        stage(store, proposal)
    assert store.db.execute("SELECT COUNT(*) FROM candidates").fetchone()[0] == 0


def test_rule_combination_without_validation_coverage_is_rejected(routing):
    store, proposal, _, _ = routing
    # Both individual predicates occur in the suite, but their production combination
    # does not. Testing rules in isolation must not authorize deploying both together.
    store.config["profiles"]["demo"] = {
        "tags": ["reduction", "pointwise"],
        "capabilities": ["full"],
    }
    proposal["upsert"].append({**proposal["upsert"][0], "id": "other", "applies_to": ["pointwise"]})
    identity = stage(store, proposal)
    result = validate(store, identity)
    assert not result["accepted"] and result["preflight"]
    assert "coverage" in result["reasons"][0]
    assert store.candidate(identity)["status"] == "rejected"
    assert store.get(f"preflight_rejection:{identity}") == result
    with pytest.raises(ValueError, match="already been evaluated"):
        validate(store, identity)
    assert not store.db.execute("SELECT * FROM operations").fetchall()


def test_one_matching_template_is_not_sufficient_validation_coverage(routing):
    store, proposal, _, _ = routing
    store.config["suites"]["demo"]["tasks"].pop()
    identity = stage(store, proposal)
    result = validate(store, identity)
    assert not result["accepted"] and "coverage needs two" in result["reasons"][0]
    assert not store.db.execute("SELECT * FROM operations").fetchall()


def test_behavior_identity_includes_conditions_but_not_order(routing):
    store, proposal, _, _ = routing
    first = stage(store, proposal)
    proposal["upsert"][0]["requires"] = []
    second = stage(store, proposal)
    assert second != first
    proposal["upsert"][0]["id"] = "renamed"
    assert stage(store, proposal) == second


def test_profile_unknown_task_and_capability_types_are_rejected(routing):
    _, _, cfg, cfg_path = routing
    cfg["profiles"]["unregistered"] = {}
    with pytest.raises(ValueError, match="profile"):
        validate_config(cfg, cfg_path.parent)
    cfg["profiles"].pop("unregistered")
    cfg["profiles"]["demo"]["capabilities"] = "full"
    with pytest.raises(ValueError, match="capabilities"):
        validate_config(cfg, cfg_path.parent)


def test_learner_sees_only_production_profiles_not_validation_metadata(routing, monkeypatch):
    store, _, _, _ = routing
    observed = []

    def learner(command, directory, packet, *args, **kwargs):
        observed.append(packet)
        write_json(directory / "proposals.json", [])

    monkeypatch.setattr("kernel_envolve.rsi.process.invoke", learner)
    assert learn(store)["status"] == "proposed"
    assert observed[0]["profiles"] == {"demo": {"tags": ["reduction"], "capabilities": ["full"]}}
    assert "pointwise" not in json.dumps(observed[0])
    assert "heldout" not in json.dumps(observed[0])


def test_predicate_order_cannot_bypass_behavior_deduplication():
    rule = advice(applies_to=["reduction", "strided"], requires=["full", "quick"])
    reordered = {
        **rule,
        "id": "renamed",
        "applies_to": ["strided", "reduction"],
        "requires": ["quick", "full"],
    }
    assert behavior_identity([rule]) == behavior_identity([reordered])


@pytest.mark.parametrize(
    "invalid",
    [
        {"tags": ["same", "same"]},
        {"capabilities": [True]},
        {"capabilities": ["profile"] * 33},
        {"can_execute": True},
    ],
)
def test_invalid_operator_profiles_are_rejected(invalid):
    with pytest.raises(ValueError):
        select_rules([advice()], invalid)
