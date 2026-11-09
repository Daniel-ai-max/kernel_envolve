"""Agent-driven kernel iteration, with independently measured evidence."""

__version__ = "0.3.1"


def load_best(task):
    """Load the verified snapshot, including its helpers, through a stable public entry."""
    from pathlib import Path

    from .storage import load_module, read_best

    best = read_best(Path(task).resolve())
    if best is None:
        raise ValueError("No verified best exists for this task")
    return load_module(best[1] / "source", "candidate").run
