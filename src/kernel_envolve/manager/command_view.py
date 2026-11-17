"""Compatibility import for :mod:`kernel_envolve.manager.dashboard.command_view`.

New implementation code must import the canonical module directly.
"""

import importlib
import sys

sys.modules[__name__] = importlib.import_module("kernel_envolve.manager.dashboard.command_view")
