"""Array checks shared by fresh-input and same-object reuse verification."""

import copy

import numpy as np


class Mismatch(AssertionError):
    """Completed correctness rejection, rather than a worker failure."""


def is_torch(x):
    return type(x).__module__.startswith("torch") and hasattr(x, "detach")


def host(x):
    if isinstance(x, np.ndarray):
        if x.dtype.kind in "fc":
            # Compare fp16/fp32 differences in a wider dtype: tolerance arithmetic can overflow.
            return x.astype(np.result_type(x.dtype, np.float64), copy=False)
        return x
    if is_torch(x):
        # NumPy does not represent bfloat16; comparison uses fp64 without changing dtype checks.
        torch = __import__("torch")
        dtype = torch.complex128 if x.is_complex() else torch.float64
        return x.detach().resolve_conj().resolve_neg().cpu().to(dtype=dtype).numpy()
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
            return (
                a.untyped_storage().nbytes() > 0
                and b.untyped_storage().nbytes() > 0
                and a.untyped_storage().data_ptr() == b.untyped_storage().data_ptr()
            )
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
    exact = (isinstance(expected, np.ndarray) and expected.dtype.kind in "biu") or (
        is_torch(expected) and not (expected.is_floating_point() or expected.is_complex())
    )
    if is_torch(expected):
        torch = __import__("torch")
        # Most factory/mutation checks compare identical values. Reduce on the device
        # for large tensors. Small CUDA comparisons retain the host path to avoid
        # loading extra reduction kernels into every short-lived worker.
        use_equal = exact or not expected.is_cuda or expected.numel() >= 1_048_576
        same = use_equal and torch.equal(actual, expected)
        if exact:
            if not same:
                raise Mismatch(f"{path}: integer/bool values differ")
            return
        # Do not rely on equality's alias shortcuts to implement the NaN policy.
        if same and (cfg["equal_nan"] or not torch.isnan(expected).any().item()):
            return
    a, e = host(actual), host(expected)
    try:
        if exact:
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
        (signature(x), binding(x), x.detach().clone() if is_torch(x) else x.copy()) for x in inputs
    ], aliases(inputs)


def unchanged(inputs, before, mutable):
    saved, alias = before
    if aliases(inputs) != alias:
        raise Mismatch("Input alias relationships changed")
    for i, (x, (sig, address, value)) in enumerate(zip(inputs, saved)):
        if signature(x) != sig or binding(x) != address:
            raise Mismatch(f"Input {i}: shape/dtype/layout/device/storage binding changed")
        if i not in mutable:
            assert_close(
                x, value, {"rtol": 0, "atol": 0, "equal_nan": True}, f"input[{i}] mutation"
            )


def binding(x):
    return x.data_ptr() if is_torch(x) else x.__array_interface__["data"][0]


def freeze(value):
    """Preserve oracle values before candidate code can reuse an output buffer."""
    if isinstance(value, dict):
        frozen = copy.copy(value)
        for k, v in value.items():
            frozen[k] = freeze(v)
        return frozen
    if isinstance(value, list):
        frozen = copy.copy(value)
        for i, v in enumerate(value):
            frozen[i] = freeze(v)
        return frozen
    if isinstance(value, tuple):
        items = tuple(freeze(v) for v in value)
        return type(value)(*items) if hasattr(value, "_fields") else type(value)(items)
    if isinstance(value, np.ndarray):
        return value.copy()
    if is_torch(value):
        return value.detach().clone()
    raise TypeError("Reference must return arrays/tensors or their nested containers")


def independent(left, right):
    relations = aliases(left + right)
    if any(
        relations[(i, j)]
        for i in range(len(left))
        for j in range(len(left), len(left) + len(right))
    ):
        raise ValueError("Input factory must allocate independent storage on each call")


def copy_values(destination, source):
    if len(destination) != len(source) or aliases(destination) != aliases(source):
        raise ValueError("Input factory changes its alias structure between seeds")
    for a, b in zip(destination, source):
        if signature(a) != signature(b):
            raise ValueError("Input factory changes shape/layout/dtype/device between seeds")
        if isinstance(a, np.ndarray):
            np.copyto(a, b)
        else:
            # copy_ rejects expanded zero-stride dimensions; update their underlying slice once.
            index = tuple(
                0 if size > 1 and stride == 0 else slice(None)
                for size, stride in zip(a.shape, a.stride())
            )
            a[index].copy_(b[index])
    # Overlapping inputs must be consistent after all copies, not just each copy.
    for a, b in zip(destination, source):
        assert_close(a, b, {"rtol": 0, "atol": 0, "equal_nan": True}, "reuse input reset")
