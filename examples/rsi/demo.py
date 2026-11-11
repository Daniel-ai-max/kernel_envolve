"""CPU-only RSI mechanics demo. Deterministic adapters, no model calls or GPU claims."""

import sys
from pathlib import Path

from kernel_envolve.manager.config import validate
from kernel_envolve.manager.store import Store
from kernel_envolve.storage import config, contract, write_json


def create(root):
    root = Path(root).resolve()
    root.mkdir(parents=True, exist_ok=False)
    paths = []
    for name in ("production", "heldout-a", "heldout-b"):
        task = root / name
        task.mkdir()
        write_json(
            task / "task.json",
            {
                "name": name,
                "objective": "synthetic CPU demonstration only",
                "device": "cpu",
                "cases": [
                    {"id": "small", "args": {"n": 16 if name != "heldout-b" else 32}, "weight": 1}
                ],
                "quick_trials": 1,
                "full_trials": 3,
                "warmup": 1,
                "full_seeds": [0],
            },
        )
        (task / "task.py").write_text(
            "import numpy as np\nimport time\ndef make_inputs(case,seed): return (np.ones(case['n']),)\ndef reference(x): return x+1\ndef baseline(x):\n time.sleep(.006)\n return x+1\n"
        )
        (task / "candidate.py").write_text(
            "import time\ndef run(x):\n time.sleep(.003)\n return x+1\n"
        )
        paths.append(task)
    adapter = root / "adapters.py"
    adapter.write_text("""import json,sys,time
from pathlib import Path
packet=json.load(sys.stdin)
if sys.argv[1]=='learn':
    evidence=packet['evidence']
    current=packet['current']['demo']
    proposal={'scope':'demo','parent':current['id'],'hypothesis':'Synthetic demo: remove deliberate sleep; not evidence of general RSI efficacy.','upsert':[{'id':'avoid-sleep','when':'Synthetic demo tasks only','action':'Remove artificial sleep from the candidate.','unless':'Never change real workload semantics or evaluation.','evidence':[evidence[0]['id']]}],'remove':[]}
    Path('proposals.json').write_text(json.dumps([proposal]))
else:
    if packet['strategy']:
        Path(packet['editable_files'][0]).write_text('def run(x): return x+1\\n')
""")
    manager = Store(root / "manager")
    deployment = validate(
        {
            "version": 1,
            "resources": [{"id": "cpu", "device": "cpu", "sessions": 1}],
            "tasks": [
                {
                    "id": "demo",
                    "name": "Synthetic RSI demo",
                    "workspace": str(paths[0]),
                    "task_dir": str(paths[0]),
                    "kind": "generic",
                    "resource": "cpu",
                    "command": [sys.executable, "-c", "pass"],
                    "baseline": {
                        "repository": "https://example.invalid/demo",
                        "revision": "demo",
                        "entrypoint": "task.py:baseline",
                        "introduced_on": None,
                        "introduction_url": None,
                        "frozen_on": "2026-11-10",
                        "hardware": "CPU demo",
                        "contract": contract(paths[0], config(paths[0])),
                    },
                }
            ],
        },
        root,
    )
    manager.register(deployment)
    manager.event(
        "demo",
        "report",
        {
            "hypothesis": "Artificial sleep wastes time",
            "outcome": "Demo observation, not a real optimizer result",
        },
    )
    manager.close()
    cfg = {
        "manager": str(root / "manager"),
        "scopes": {"demo": ["demo"]},
        "learner_command": [sys.executable, str(adapter), "learn"],
        "min_new_evidence": 1,
        "cooldown_seconds": 1,
        "auto_validate": True,
        "suites": {
            "demo": {
                "model": "deterministic-demo-no-llm",
                "command": [sys.executable, str(adapter), "solve"],
                "tasks": [{"path": str(p), "kind": "generic"} for p in paths[1:]],
                "solver_seconds": 5,
                "evaluation_seconds": 15,
                "max_seconds": 120,
                "min_quality_gain": 0.05,
            }
        },
    }
    write_json(root / "rsi.json", cfg)
    return root / "rsi.json"


if __name__ == "__main__":
    print(create(sys.argv[1]))
