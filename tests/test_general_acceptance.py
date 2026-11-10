"""Independent final acceptance for the broad comparison tasks."""

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest


def load(name):
    path = Path(__file__).parents[1] / "audit" / (name + ".py")
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def modules():
    torch = pytest.importorskip("torch")
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        yield (
            torch,
            load("build_general_tasks"),
            load("general_workloads"),
            load("general_final_acceptance"),
        )
    finally:
        torch.set_num_threads(threads)


def test_public_and_sample_inputs_obey_the_same_numerical_contract(modules):
    torch, builder, factory, verifier = modules
    for name in builder.NAMES:
        public = builder.public_cases(name, "cpu")
        sample = builder.sample_cases(name, "cpu", 20261110)
        assert len({json.dumps(c["args"], sort_keys=True) for c in public + sample}) == len(
            public + sample
        )
        for case in public + sample:
            x = factory.make_inputs(name, case["args"], 7)
            y = factory.make_inputs(name, case["args"], 8)
            assert all(
                a.untyped_storage().data_ptr() != b.untyped_storage().data_ptr()
                for a, b in zip(x, y)
            )
            assert [(a.shape, a.stride(), a.dtype) for a in x] == [
                (a.shape, a.stride(), a.dtype) for a in y
            ]
            # Factory/oracle consistency is independent of a CPU BF16 GEMM backend.
            # Native target-device baselines are checked by probe_general_tasks.
            output = factory.reference(name, *x)
            factory.validate(name, x, output)
            rtol, atol = verifier.TOLERANCES[name][case["args"]["dtype"]]
            np.testing.assert_allclose(
                output.double().numpy(), verifier.golden(name, x), rtol=rtol, atol=atol
            )
            assert all(torch.isnan(a._base[-1]) for a in x)


def test_both_validators_reject_writes_outside_the_input_view(modules):
    _, builder, factory, verifier = modules
    for name in builder.NAMES:
        case = builder.public_cases(name, "cpu")[1]
        inputs = factory.make_inputs(name, case["args"], 0)
        expected = verifier.golden(name, inputs)

        def bad(*x):
            output = factory.baseline(name, *x)
            x[0]._base[-1] = 0
            return output

        with pytest.raises(AssertionError, match="padding changed"):
            verifier.check_call(name, bad, inputs, expected, "cpu")
        with pytest.raises(AssertionError, match="padding was overwritten"):
            factory.validate(name, inputs, factory.baseline(name, *inputs))


def test_independent_acceptance_checks_actual_timed_outputs(modules, tmp_path):
    _, builder, _, verifier = modules
    builder.materialize(tmp_path / "tasks", "cpu")
    folder = tmp_path / "tasks" / "row_sum"
    cfg = json.loads((folder / "task.json").read_text())
    cfg["cases"] = [cfg["cases"][1]]
    (folder / "task.json").write_text(json.dumps(cfg))
    # Fresh/reuse and all warmups pass; only the actual timed call is wrong.
    (folder / "candidate.py").write_text(
        "calls=0\ndef run(x):\n global calls\n calls+=1\n out=x.sum(dim=1)\n return out+1 if calls>=6 else out\n"
    )
    result = verifier.accept(folder, folder / "candidate.py", [0], trials=1)
    assert not result["correct"]
    assert result["error"]["context"]["kind"] == "timed"
    assert len(result["checks"]) == 4
    (folder / "candidate.py").write_text(builder.BASELINES["row_sum"])
    result = verifier.accept(folder, folder / "candidate.py", [0], trials=1)
    assert result["correct"], result
    assert len(result["samples"]) == 8


def test_independent_acceptance_detects_precision_policy_leak(modules):
    torch, builder, factory, verifier = modules
    case = builder.public_cases("matmul", "cpu")[1]
    inputs = factory.make_inputs("matmul", case["args"], 0)
    policy = torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction

    def bad(a, b):
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = not policy
        return a @ b

    try:
        with pytest.raises(AssertionError, match="Changed precision settings"):
            verifier.check_call("matmul", bad, inputs, verifier.golden("matmul", inputs), "cpu")
    finally:
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = policy
