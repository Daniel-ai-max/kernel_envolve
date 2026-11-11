"""Fail-open, bounded read on the foreground path: no scans, network, locks or LLMs."""

import sys

from ..storage import digest
from .model import MAX_ADVICE_BYTES, MAX_CONTEXT_BYTES, read


def for_task(root, task):
    path = root / "rsi-context" / f"{task['id']}.json"
    try:
        if (root / "rsi-context" / "DISABLED").exists():
            return ""
        value = read(path, MAX_CONTEXT_BYTES)
        if not isinstance(value, dict):
            raise ValueError("RSI context must be an object")
        expected = value.pop("hash")
        if digest(value) != expected or value["schema"] != 1:
            raise ValueError("Invalid RSI context receipt")
        if value["task_identity"] != task["spec"]["identity"]:
            raise ValueError("RSI context belongs to another task contract")
        if not isinstance(value["text"], str) or len(value["text"].encode()) > MAX_ADVICE_BYTES:
            raise ValueError("Invalid RSI context length")
        if not value["text"]:
            return ""
        return f"\nRSI strategy snapshot: {value['version']}\n{value['text']}\n"
    except FileNotFoundError:
        return ""
    except (OSError, ValueError, KeyError, TypeError, RecursionError) as exc:
        print(
            f"kernel-envolve: RSI advice unavailable; using original workflow ({exc})",
            file=sys.stderr,
        )
        return ""
