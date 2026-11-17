"""Compatibility import for :mod:`kernel_envolve.evaluation.native.evaluator`.

New implementation code must import the canonical module directly.
"""

import importlib
import sys

sys.modules[__name__] = importlib.import_module("kernel_envolve.evaluation.native.evaluator")
