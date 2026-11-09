import numpy as np


def make_inputs(case, seed):
    rng = np.random.default_rng(seed)
    n, stride = case["n"], case["stride"]
    return tuple(rng.normal(size=n * stride).astype(np.float32)[::stride] for _ in range(2))


def reference(x, y):
    return np.add(x, y)


def baseline(x, y):
    # Deliberately scalar: a visible improvement for exercising the complete workflow on a laptop.
    return np.array([a + b for a, b in zip(x, y)], dtype=x.dtype)
