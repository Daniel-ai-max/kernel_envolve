"""Compatibility import for :mod:`kernel_envolve.manager.dashboard.narrator_client`.

New implementation code must import the canonical module directly.
"""

import importlib
import sys

if __name__ == "__main__":
    import runpy

    runpy.run_module("kernel_envolve.manager.dashboard.narrator_client", run_name="__main__")
else:
    sys.modules[__name__] = importlib.import_module(
        "kernel_envolve.manager.dashboard.narrator_client"
    )
