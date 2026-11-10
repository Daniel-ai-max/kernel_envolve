"""Actual CPU receipts establish lineage, never synthetic GPU improvement scores."""

import ast
import copy
import json
import shutil
from pathlib import Path

import pytest

from kernel_envolve.cli import evaluate
from kernel_envolve.rsi import artifact_lineage as lineage
from kernel_envolve.rsi.episodes import load
from kernel_envolve.rsi.research import run, source_id
from kernel_envolve.storage import digest, write_json

pytest_plugins = ["test_rsi_research"]

SOURCES = [
    "import time\ndef run(x):\n time.sleep(.002)\n return x+1\n",
    "import time\ndef run(x):\n time.sleep(.001)\n return x+1\n",
    "def run(x): return x+1\n",
]


@pytest.fixture
def rounds(research):
    packet, original_output, initial = research
    library = original_output.parent / "lineage"
    count = 0

    def perform(source, parent=None, *, current_packet=None, snapshot=None):
        nonlocal count
        current_packet = current_packet or packet
        count += 1
        output = original_output.parent / f"round-{count}"
        task = Path(current_packet["task"])
        (task / "candidate.py").write_text(initial)
        if snapshot is None and parent is not None:
            snapshot = lineage.validation_snapshot(library, parent)
        trace = run(
            current_packet,
            output,
            lambda *a: {
                "action": "propose",
                "source": source,
                "mode": "full",
                "hypothesis": "Synthetic sleep removal for lineage engineering only",
            },
            max_actions=2 if snapshot is not None else 1,
            artifact_snapshot=snapshot,
            source_validator=ast.parse,
        )
        final, _ = evaluate(task, "full", timeout=10)
        identity = lineage.collect(library, task, output, {}, confirmation_run=final, parent=parent)
        return identity, trace, final, output

    return perform, library, packet


def test_two_actual_descendants_are_automatically_formed_but_never_adopted(rounds, monkeypatch):
    perform, library, packet = rounds
    seed, *_ = perform(SOURCES[0])
    child, trace, final, output = perform(SOURCES[1], seed)
    grandchild, *_ = perform(SOURCES[2], child)
    assert len({seed, child, grandchild}) == 3
    for identity, parent, source in [
        (seed, None, SOURCES[0]),
        (child, seed, SOURCES[1]),
        (grandchild, child, SOURCES[2]),
    ]:
        record = lineage.audit(library, identity)
        assert record["parent"] == parent
        assert record["parent_activated"] is (parent is not None)
        assert record["kind"] == ("seed" if parent is None else "candidate")
        episode = load(library / "episodes", record["episode"])
        assert episode["selected"] == source_id(source)
        assert episode["observations"][episode["final_confirmation"]]["run"] not in {
            o["run"] for o in episode["observations"][: episode["final_confirmation"]]
        }
    # Collection and ancestry verification must not trigger a new evaluation.
    from kernel_envolve.rsi import research as research_module

    monkeypatch.setattr(
        research_module, "evaluate", lambda *a, **k: pytest.fail("No new measurement")
    )
    assert (
        lineage.collect(library, packet["task"], output, {}, confirmation_run=final, parent=seed)
        == child
    )
    assert set(p.name for p in library.iterdir()) == {"episodes", "views", "records"}
    assert not (Path(packet["task"]) / ".kernel_envolve/best.json").exists()
    assert trace["artifact_snapshot"]["identity"] == lineage.audit(library, seed)["snapshot"]
    monkeypatch.setattr(lineage, "MAX_ANCESTORS", 2)
    with pytest.raises(ValueError, match="audit bound"):
        lineage.audit(library, grandchild)
    with pytest.raises(ValueError, match="audit bound"):
        lineage.collect(library, packet["task"], output, {}, confirmation_run=final, parent=child)


def test_unchanged_parent_is_an_observation_not_a_new_generation(rounds):
    perform, library, _ = rounds
    seed, *_ = perform(SOURCES[0])
    observed, _, final, output = perform(SOURCES[0], seed)
    record = lineage.audit(library, observed)
    assert record["kind"] == "observation" and record["snapshot"] is None
    assert "unchanged_implementation" in record["reasons"]
    with pytest.raises(ValueError, match="did not form"):
        lineage.validation_snapshot(library, observed)
    with pytest.raises(ValueError, match="observation cannot"):
        lineage.collect(
            library, rounds[2]["task"], output, {}, confirmation_run=final, parent=observed
        )


def test_unactivated_parent_cannot_be_claimed_for_an_independently_good_source(rounds):
    perform, library, _ = rounds
    seed, *_ = perform(SOURCES[0])
    snapshot = lineage.validation_snapshot(library, seed)
    snapshot["path"] = str(library / "absent.json")
    identity, trace, *_ = perform(SOURCES[1], seed, snapshot=snapshot)
    record = lineage.audit(library, identity)
    assert trace["artifact_snapshot"]["reason"] == "snapshot_unavailable"
    assert not record["parent_activated"] and record["kind"] == "observation"
    assert "parent_not_activated" in record["reasons"]


@pytest.mark.parametrize("mutation", ["initial_receipt", "parent_pin", "offer"])
def test_materialized_hashes_do_not_hide_modified_original_lineage(rounds, mutation):
    perform, library, _ = rounds
    seed, *_ = perform(SOURCES[0])
    child, _, _, output = perform(SOURCES[1], seed)
    if mutation == "offer":
        path = output / "artifact-proposal.json"
        value = json.loads(path.read_text())
        value["episode_id"] = "0" * 64
    else:
        path = output / "search-trace.json"
        value = json.loads(path.read_text())
        key = "initial_result_hash" if mutation == "initial_receipt" else "identity"
        value["artifact_snapshot"][key] = "0" * 64
    write_json(path, value)
    with pytest.raises(ValueError, match="lineage or original"):
        lineage.audit(library, child)


def test_changed_ancestor_receipt_invalidates_descendant_audit(rounds):
    perform, library, _ = rounds
    seed, _, final, _ = perform(SOURCES[0])
    child, *_ = perform(SOURCES[1], seed)
    (final / "source/candidate.py").write_text("def run(x): return x-1\n")
    with pytest.raises(ValueError, match="Source changed"):
        lineage.audit(library, child)


def test_parent_failure_is_retained_only_for_its_actual_new_contract(rounds):
    perform, library, packet = rounds
    task = Path(packet["task"])
    reference = task / "task.py"
    reference.write_text(reference.read_text().replace("np.ones(16)", "np.ones(case.get('n',16))"))
    specialized = "def run(x): return x+1 if len(x)==16 else x-1\n"
    seed, *_ = perform(specialized)
    target = task.parent / "new-case"
    target.mkdir()
    for name in ("task.py", "task.json", "candidate.py"):
        shutil.copyfile(task / name, target / name)
    cfg = json.loads((target / "task.json").read_text())
    cfg["cases"][0]["args"] = {"n": 32}
    write_json(target / "task.json", cfg)
    child_packet = packet | {"task": str(target), "editable_files": [str(target / "candidate.py")]}
    child, trace, _, _ = perform(SOURCES[2], seed, current_packet=child_packet)
    assert trace["steps"][0]["observation"]["correctness"] == "fail"
    record = lineage.audit(library, child)
    assert record["kind"] == "candidate" and record["parent_activated"]
    episode = load(library / "episodes", record["episode"])
    old_episode = load(library / "episodes", lineage.audit(library, seed)["episode"])
    assert episode["contract"] != old_episode["contract"]
    assert record["scope"] == lineage.audit(library, seed)["scope"]
    facts = record["parent_observations"]
    assert len(facts) == 1 and facts[0]["correctness"] == "fail"
    assert facts[0]["contract"] == episode["contract"]
    # A verified old success is not erased or turned into a universal rejection.
    assert lineage.audit(library, seed)["kind"] == "seed"


def test_a_failed_independent_final_remains_evidence_without_a_snapshot(research):
    packet, output, _ = research
    task = Path(packet["task"])
    (task / "candidate.py").write_text("import time\ndef run(x):\n time.sleep(.02)\n return x+1\n")
    run(
        packet,
        output,
        lambda *a: {"action": "stop", "hypothesis": "Keep slow CPU fixture"},
        max_actions=1,
    )
    final, result = evaluate(task, "full", timeout=10)
    assert result["correctness"] == "pass" and not result["decision"]["eligible_for_best"]
    directory = output.parent / "lineage"
    identity = lineage.collect(directory, task, output, {}, confirmation_run=final)
    record = lineage.audit(directory, identity)
    assert record["reasons"] == ["independent_final_not_eligible"]
    assert record["kind"] == "observation" and record["snapshot"] is None
    assert not (directory / "views").exists()


def test_repeated_receipts_do_not_create_a_new_validation_identity(rounds):
    perform, library, _ = rounds
    seed, *_ = perform(SOURCES[0])
    first, *_ = perform(SOURCES[1], seed)
    second, *_ = perform(SOURCES[1], seed)
    assert first != second
    assert (
        lineage.audit(library, first)["proposal_key"]
        == lineage.audit(library, second)["proposal_key"]
    )


@pytest.mark.parametrize("change", ["evaluator", "runtime", "missing_runtime"])
def test_final_context_drift_cannot_form_a_mislabelled_snapshot(rounds, monkeypatch, change):
    perform, library, packet = rounds
    _, _, final, output = perform(SOURCES[0])
    original = lineage.build

    def changed_context(*args, **kwargs):
        # Mutate inspection metadata at the trust boundary, after actual raw CPU
        # receipts have been verified. This is not a measured environment change.
        episode = copy.deepcopy(original(*args, **kwargs))
        observed = episode["observations"][episode["final_confirmation"]]
        if change == "evaluator":
            observed["evaluator_hash"] = "0" * 64
        elif change == "runtime":
            observed["environment"]["packages"]["numpy"] = "different-context"
        else:
            observed["environment"] = None
        return episode

    monkeypatch.setattr(lineage, "build", changed_context)
    target = library / "context-probe"
    identity = lineage.collect(target, packet["task"], output, {}, confirmation_run=final)
    record = lineage.audit(target, identity)
    assert record["kind"] == "observation" and record["snapshot"] is None
    assert "independent_final_context_changed" in record["reasons"]


def test_second_trace_read_cannot_replace_the_actually_verified_parent_action(rounds, monkeypatch):
    perform, library, packet = rounds
    seed, *_ = perform(SOURCES[0])
    _, trace, final, output = perform(SOURCES[1])  # No parent supplied or used.
    seed_record = lineage.audit(library, seed)
    view = load(library / "views", seed_record["snapshot"])
    forged = copy.deepcopy(trace)
    initial_run = Path(trace["initial_observation"]["run"])
    forged["artifact_snapshot"] = {
        "identity": seed_record["snapshot"],
        "reason": "compatible_proposal",
        "initial_run": str(initial_run),
        "initial_result_hash": digest(json.loads((initial_run / "result.json").read_text())),
    }
    forged["artifact_proposal_hash"] = digest(view["proposal"])
    forged["steps"][0].update(
        decision_kind="artifact_proposal", artifact_episode_id=view["proposal"]["episode_id"]
    )
    forged["steps"][0]["observation"]["candidate"] = view["proposal"]["source_hash"]
    original = lineage.read

    def changed_between_reads(path, limit):
        if Path(path) == output / "search-trace.json":
            return forged
        if Path(path) == output / "artifact-proposal.json":
            return view["proposal"]
        return original(path, limit)

    # episodes.build has its own reader and still verifies the actual agent
    # action and raw source. A later metadata read must not override that fact.
    monkeypatch.setattr(lineage, "read", changed_between_reads)
    identity = lineage.collect(
        library, packet["task"], output, {}, confirmation_run=final, parent=seed
    )
    record = lineage.audit(library, identity)
    assert record["kind"] == "observation" and not record["parent_activated"]


def test_formation_version_drift_rejects_old_records(rounds, monkeypatch):
    from kernel_envolve.rsi import trials

    perform, library, _ = rounds
    seed, *_ = perform(SOURCES[0])
    monkeypatch.setattr(trials, "validator_identity", lambda: "0" * 64)
    with pytest.raises(ValueError, match="formation code changed"):
        lineage.audit(library, seed)
