import numpy as np
import pytest

from kernel_envolve.arrays import Mismatch, assert_close, capture, copy_values, unchanged

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
