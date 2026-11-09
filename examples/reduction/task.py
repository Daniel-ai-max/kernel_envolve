"""Demonstrate a task-specific accuracy gate beyond output allclose.

The 1e-6 floor is for this small float32 sum example; it is not a universal kernel threshold.
"""

import math

import numpy as np


def make_inputs(case, seed):
    rng = np.random.default_rng(seed)
    return ((rng.normal(size=case["n"]) * case["scale"]).astype(np.float32),)


def reference(x):
    return np.asarray(np.sum(x, dtype=np.float32))


baseline = reference


def validate(inputs, output):
    (x,) = inputs
    # Python float represents each float32 input exactly; fsum supplies a distinct summation method.
    golden = math.fsum(float(v) for v in x)
    reference_error = abs(float(reference(x)) - golden)
    candidate_error = abs(float(output) - golden)
    limit = 2 * reference_error + 1e-6
    assert candidate_error <= limit, (
        f"golden error {candidate_error:.9g} > {limit:.9g}; reference error {reference_error:.9g}"
    )
