import numpy as np
import pytest

from kernel_envolve.arrays import Mismatch, assert_close, capture, copy_values, freeze, unchanged

CFG = {"rtol": 1e-5, "atol": 1e-6, "equal_nan": False}


def test_output_structure_and_dtype():
    a = np.arange(4, dtype=np.float32)
    assert_close({"out": (a, [a])}, {"out": (a.copy(), [a.copy()])}, CFG)
    with pytest.raises(Mismatch, match="structure"):
        assert_close([a], (a,), CFG)
    with pytest.raises(Mismatch, match="dtype"):
        assert_close(a.astype(np.float64), a, CFG)
    with pytest.raises(Mismatch):
        assert_close(None, a, CFG)


def test_nonfinite_and_integer_checks():
    with pytest.raises(Mismatch):
        assert_close(np.array([np.nan]), np.array([1.0]), CFG)
    with pytest.raises(Mismatch):
        assert_close(np.array([np.nan]), np.array([np.nan]), CFG)
    assert_close(np.array([np.inf]), np.array([np.inf]), CFG)
    with pytest.raises(Mismatch):
        assert_close(np.array([2**60 + 1]), np.array([2**60]), CFG)


def test_reuse_preserves_strides_and_aliases():
    storage = np.arange(12, dtype=np.float32)
    other = storage + 10
    args = (storage[::2], storage[2::2])
    replacement = (other[::2], other[2::2])
    ids = [id(a) for a in args]
    copy_values(args, replacement)
    assert [id(a) for a in args] == ids
    assert args[0].strides == (8,)
    assert np.shares_memory(*args)
    assert_close(args, replacement, CFG)


def test_mutation_detected_through_alias():
    storage = np.arange(12, dtype=np.float32)
    args = (storage[::2], storage[2::2])
    before = capture(args)
    args[0][1] = -10
    with pytest.raises(Mismatch, match=r"input\[1\] mutation"):
        unchanged(args, before, mutable=[0])


def test_torch_large_integer_and_bfloat16():
    torch = pytest.importorskip("torch")
    with pytest.raises(Mismatch):
        assert_close(torch.tensor([2**60 + 1]), torch.tensor([2**60]), CFG)
    a = torch.tensor([0.25, 1.0], dtype=torch.bfloat16)
    assert_close(a, a.clone(), CFG)
    with pytest.raises(Mismatch):
        assert_close(a + 1, a, CFG)


def test_torch_complex_output_keeps_imaginary_component():
    torch = pytest.importorskip("torch")
    a = torch.tensor([1 + 1j], dtype=torch.complex64)
    assert_close(a + 1e-7j, a, CFG)
    with pytest.raises(Mismatch):
        assert_close(a + 1j, a, CFG)


def test_torch_rebinding_storage_is_not_legal_inplace_mutation():
    torch = pytest.importorskip("torch")
    a = torch.ones(8)
    before = capture((a,))
    a.set_(torch.ones(8))
    with pytest.raises(Mismatch, match="storage binding"):
        unchanged((a,), before, mutable=[0])


def test_freezing_preserves_named_tuple_and_mapping_types():
    from collections import OrderedDict, namedtuple

    pair = namedtuple("Pair", "values auxiliary")
    output = OrderedDict(out=pair(np.ones(3, dtype=np.float32), [np.zeros(2, dtype=np.float32)]))
    saved = freeze(output)
    assert type(saved) is OrderedDict and type(saved["out"]) is pair
    assert_close(output, saved, CFG)
    output["out"].values.fill(0)
    assert np.all(saved["out"].values == 1)


def test_freezing_preserves_torch_structured_return_types():
    torch = pytest.importorskip("torch")
    output = torch.max(torch.tensor([[1.0, 2.0]]), dim=1)
    saved = freeze(output)
    assert type(saved) is type(output)
    assert_close(output, saved, CFG)


def test_equal_tensors_and_integer_rejections_need_no_host_copy(monkeypatch):
    torch = pytest.importorskip("torch")
    from kernel_envolve import arrays

    def no_host_copy(value):
        raise AssertionError("A full tensor host copy is unnecessary here")

    monkeypatch.setattr(arrays, "host", no_host_copy)
    for dtype in (torch.float16, torch.float32, torch.float64, torch.int64, torch.bool):
        value = torch.tensor([0, 1], dtype=dtype)
        assert_close(value, value.clone(), CFG)
    with pytest.raises(Mismatch, match="integer/bool"):
        assert_close(torch.tensor([2**60 + 1]), torch.tensor([2**60]), CFG)


def tensor_comparison_matrix(device, large=False):
    """The fast path must preserve the original wide-precision NumPy comparison."""
    import torch

    pairs = [
        ([0.0, -0.0], [-0.0, 0.0]),
        ([1.0, 2.0], [1.0, 2.0]),
        ([1.000001], [1.0]),
        ([1.01], [1.0]),
        ([float("nan")], [float("nan")]),
        ([float("nan")], [0.0]),
        ([float("inf")], [float("inf")]),
        ([float("inf")], [-float("inf")]),
        ([1 + 2j], [1 + 2j]),
        ([1 + 2j], [1 + 3j]),
        ([complex(float("nan"), 1)], [complex(2, float("nan"))]),
    ]
    for left, right in pairs:
        dtype = torch.complex64 if isinstance(left[0], complex) else torch.float32
        for equal_nan in (False, True):
            actual, expected = torch.tensor(left, dtype=dtype), torch.tensor(right, dtype=dtype)
            cfg = CFG | {"equal_nan": equal_nan}
            np_dtype = np.complex128 if dtype == torch.complex64 else np.float64
            oracle_pass = True
            try:
                np.testing.assert_allclose(
                    actual.numpy().astype(np_dtype),
                    expected.numpy().astype(np_dtype),
                    rtol=cfg["rtol"],
                    atol=cfg["atol"],
                    equal_nan=equal_nan,
                )
            except AssertionError:
                oracle_pass = False
            actual, expected = actual.to(device), expected.to(device)
            if large:
                actual = actual.repeat(1_048_576 // actual.numel() + 1)
                expected = expected.repeat(1_048_576 // expected.numel() + 1)
            if oracle_pass:
                assert_close(actual, expected, cfg)
            else:
                with pytest.raises(Mismatch):
                    assert_close(actual, expected, cfg)
    value = torch.tensor([float("nan")], device=device)
    if large:
        value = value.repeat(1_048_576)
    with pytest.raises(Mismatch):
        assert_close(value, value, CFG)
    assert_close(value, value, CFG | {"equal_nan": True})


def test_tensor_fast_path_preserves_comparison_semantics():
    pytest.importorskip("torch")
    tensor_comparison_matrix("cpu")
