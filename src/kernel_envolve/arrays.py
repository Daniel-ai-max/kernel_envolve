"""Array checks shared by fresh-input and same-object reuse verification."""

import numpy as np


class Mismatch(AssertionError):
    """Completed correctness rejection, rather than a worker failure."""


def is_torch(x):
    return type(x).__module__.startswith("torch") and hasattr(x, "detach")


def host(x):
    if isinstance(x, np.ndarray):
        return x
    if is_torch(x):
        # NumPy does not represent bfloat16; comparison uses fp64 without changing dtype checks.
        torch = __import__("torch")
        dtype = torch.complex128 if x.is_complex() else torch.float64
        return x.detach().cpu().to(dtype=dtype).numpy()
    raise TypeError("Task inputs/outputs must be NumPy arrays or PyTorch tensors")


def signature(x):
    if isinstance(x, np.ndarray):
        return (str(x.dtype), tuple(x.shape), tuple(x.strides), "cpu")
    if is_torch(x):
        return (str(x.dtype), tuple(x.shape), tuple(x.stride()), str(x.device))
    raise TypeError("Inputs must be arrays or tensors in a flat tuple")


def aliases(inputs):
    def same(a, b):
        if isinstance(a, np.ndarray) and isinstance(b, np.ndarray):
            return bool(np.shares_memory(a, b))
        if is_torch(a) and is_torch(b):
            return a.untyped_storage().data_ptr() == b.untyped_storage().data_ptr()
        return False

    return {(i, j): same(a, b) for i, a in enumerate(inputs) for j, b in enumerate(inputs) if i < j}


def validate_inputs(inputs, device):
    if not isinstance(inputs, tuple) or not inputs:
        raise TypeError("make_inputs must return a nonempty flat tuple of arrays/tensors")
    for x in inputs:
        sig = signature(x)
        if sig[-1].split(":")[0] != device.split(":")[0]:
            raise ValueError("Input device differs from task device")
        if device.startswith("cuda"):
            index = int(device.split(":")[1]) if ":" in device else 0
            if sig[-1] != f"cuda:{index}":
                raise ValueError("Input CUDA index differs from the timing device")
        if isinstance(x, np.ndarray) and x.dtype.kind not in "biufc":
            raise TypeError("Only numeric/bool arrays are supported")
        if is_torch(x) and (x.is_complex() or x.requires_grad):
            raise ValueError(
                "This evaluator expects real tensors without autograd; wrap backward explicitly"
            )


def assert_close(actual, expected, cfg, path="output"):
    if isinstance(expected, (tuple, list, dict)):
        if type(actual) is not type(expected) or len(actual) != len(expected):
            raise Mismatch(f"{path}: structure differs")
        if isinstance(expected, dict) and actual.keys() != expected.keys():
            raise Mismatch(f"{path}: keys differ")
        keys = expected if isinstance(expected, dict) else range(len(expected))
        if not len(expected):
            raise TypeError("Reference must return observable, nonempty outputs")
        for key in keys:
            assert_close(actual[key], expected[key], cfg, f"{path}[{key}]")
        return
    if expected is None:
        raise TypeError("Reference must return observable outputs, including for in-place tasks")
    try:
        actual_sig, expected_sig = signature(actual), signature(expected)
    except TypeError as exc:
        raise Mismatch(f"{path}: expected array/tensor output") from exc
    if actual_sig[:2] != expected_sig[:2] or actual_sig[-1] != expected_sig[-1]:
        raise Mismatch(f"{path}: dtype, shape or device differs: {actual_sig} vs {expected_sig}")
    a, e = host(actual), host(expected)
    exact = (isinstance(expected, np.ndarray) and expected.dtype.kind in "biu") or (
        is_torch(expected) and not (expected.is_floating_point() or expected.is_complex())
    )
    try:
        if exact:
            # Preserve torch int64 values: converting large integers to float loses bits.
            if is_torch(expected):
                if not __import__("torch").equal(actual, expected):
                    raise AssertionError("integer/bool values differ")
            else:
                np.testing.assert_array_equal(a, e, strict=True)
        else:
            np.testing.assert_allclose(
                a, e, rtol=cfg["rtol"], atol=cfg["atol"], equal_nan=cfg["equal_nan"]
            )
    except AssertionError as exc:
        raise Mismatch(f"{path}: {exc}") from exc


def capture(inputs):
    # deep value copies only; the original objects/layout/aliasing remain untouched.
    return [
        (signature(x), x.detach().clone() if is_torch(x) else x.copy()) for x in inputs
    ], aliases(inputs)


def unchanged(inputs, before, mutable):
    saved, alias = before
    if aliases(inputs) != alias:
        raise Mismatch("Input alias relationships changed")
    for i, (x, (sig, value)) in enumerate(zip(inputs, saved)):
        if signature(x) != sig:
            raise Mismatch(f"Input {i}: shape/dtype/layout/device changed")
        if i not in mutable:
            assert_close(
                x, value, {"rtol": 0, "atol": 0, "equal_nan": True}, f"input[{i}] mutation"
            )


def copy_values(destination, source):
    if len(destination) != len(source) or aliases(destination) != aliases(source):
        raise ValueError("Input factory changes its alias structure between seeds")
    for a, b in zip(destination, source):
        if signature(a) != signature(b):
            raise ValueError("Input factory changes shape/layout/dtype/device between seeds")
        if isinstance(a, np.ndarray):
            np.copyto(a, b)
        else:
            a.copy_(b)
    # Overlapping inputs must be consistent after all copies, not just each copy.
    for a, b in zip(destination, source):
        assert_close(a, b, {"rtol": 0, "atol": 0, "equal_nan": True}, "reuse input reset")
