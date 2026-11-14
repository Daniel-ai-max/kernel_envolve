"""Post-measurement reuse and determinism must be checked, not only executed."""

import json

import pytest

from kernel_envolve import native
from kernel_envolve.protocol import post_seed

pytest_plugins = ["test_native"]


@pytest.fixture(autouse=True)
def cpu_only(monkeypatch):
    monkeypatch.setenv("KERNEL_ENVOLVE_NATIVE_DEVICE", "cpu")


@pytest.mark.parametrize("role", ["candidate", "comparator"])
@pytest.mark.parametrize("index", [1, 2])
def test_rejects_wrong_postcheck_reuse_and_repeat(native_task, monkeypatch, role, index):
    task, _ = native_task
    original = native.Worker.call
    changed = []

    def call(self, kind, case, seed):
        meta, observed = original(self, kind, case, seed)
        if self.folder.name == role and kind == "check" and seed == post_seed(case["id"], [0]):
            observed[index]["output"]["output"].add_(1)
            changed.append(index)
        return meta, observed

    monkeypatch.setattr(native.Worker, "call", call)
    _, result = native.evaluate_native(task, timeout=60)
    assert changed == [index]
    assert result["correctness"] == "fail"
    assert result["failed_phase"] == "postcheck"
    assert result["samples"]  # The fault is first observed after real timing.
    assert not result["decision"]["eligible_for_best"]


def test_postcheck_repeat_remains_bitwise_under_loose_oracle_tolerance(native_task, monkeypatch):
    task, _ = native_task
    cfg = json.loads((task / "native.json").read_text())
    cfg.update(atol=0.1, bitwise=False)
    (task / "native.json").write_text(json.dumps(cfg))
    original = native.Worker.call

    def call(self, kind, case, seed):
        meta, observed = original(self, kind, case, seed)
        if (
            self.folder.name == "candidate"
            and kind == "check"
            and seed == post_seed(case["id"], [0])
        ):
            observed[2]["output"]["output"].add_(0.01)
        return meta, observed

    monkeypatch.setattr(native.Worker, "call", call)
    _, result = native.evaluate_native(task, timeout=60)
    assert result["correctness"] == "fail"
    assert result["failed_phase"] == "postcheck"
    assert not result["decision"]["eligible_for_best"]


@pytest.mark.parametrize("deterministic", [True, False])
def test_postcheck_cannot_accept_uniformly_truncated_observations(
    native_task, monkeypatch, deterministic
):
    task, _ = native_task
    cfg = json.loads((task / "native.json").read_text())
    cfg["deterministic"] = deterministic
    (task / "native.json").write_text(json.dumps(cfg))
    original = native.Worker.call

    def call(self, kind, case, seed):
        meta, observed = original(self, kind, case, seed)
        if kind == "check" and seed == post_seed(case["id"], [0]):
            observed = observed[:1]
        return meta, observed

    monkeypatch.setattr(native.Worker, "call", call)
    _, result = native.evaluate_native(task, timeout=60)
    assert result["correctness"] == "fail"
    assert result["failed_phase"] == "postcheck"
    assert not result["decision"]["eligible_for_best"]


@pytest.mark.parametrize("deterministic", [True, False])
def test_native_decision_requires_each_postcheck_record(native_task, deterministic):
    import copy

    task, _ = native_task
    cfg = json.loads((task / "native.json").read_text())
    cfg["deterministic"] = deterministic
    (task / "native.json").write_text(json.dumps(cfg))
    _, result = native.evaluate_native(task, timeout=60)
    assert result["correctness"] == "pass", result.get("error")
    cfg = native.load_config(task)
    expected = {"post"}
    post = [c for c in result["checks"] if c["kind"].startswith("post")]
    assert {c["kind"] for c in post} == expected
    assert len(post) == len(expected) * 2
    for record in post:
        missing = copy.deepcopy(result)
        missing["checks"].remove(record)
        with pytest.raises(ValueError, match="Incomplete or duplicate correctness coverage"):
            native.native_decision(missing, cfg)
    duplicate = copy.deepcopy(result)
    duplicate["checks"].append(post[-1])
    with pytest.raises(ValueError, match="Incomplete or duplicate correctness coverage"):
        native.native_decision(duplicate, cfg)
    # The existing scoring schema remains unchanged.
    assert native.score(result, cfg)["coverage"] == result["decision"]["coverage"]
