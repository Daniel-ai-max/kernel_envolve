"""Compatibility import for :mod:`kernel_envolve.evaluation.native.worker`.

New implementation code must import the canonical module directly.
"""

import importlib
import sys

if __name__ == "__main__":
    import runpy

    runpy.run_module("kernel_envolve.evaluation.native.worker", run_name="__main__")
else:
    sys.modules[__name__] = importlib.import_module("kernel_envolve.evaluation.native.worker")
