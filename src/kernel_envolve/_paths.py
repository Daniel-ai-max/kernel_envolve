"""Installed package locations; independent of implementation module depth."""

from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parent
IMPORT_ROOT = PACKAGE_ROOT.parent
