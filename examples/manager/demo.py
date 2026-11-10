"""Create a local CPU demonstration using actual evaluator receipts, never production numbers.

Run once: python examples/manager/demo.py /tmp/kernel-manager-demo
Then use the printed coordinator and dashboard commands.
"""

import datetime as dt
import json
import sys
from pathlib import Path

from kernel_envolve.manager.cli import task_contract
from kernel_envolve.manager.config import validate
from kernel_envolve.manager.store import Store

AGENT = """import json, os, subprocess, sys, time
from pathlib import Path
task = Path.cwd()
best = task / ".kernel_envolve/best.json"
if not best.exists():
    command = [sys.executable, "-m", "kernel_envolve.cli", "manage", "gpu-run", "--",
               sys.executable, "-m", "kernel_envolve.cli", "evaluate", "--task", str(task), "--mode", "full"]
    reply = subprocess.run(command, check=True, text=True, capture_output=True)
    result = json.loads(reply.stdout)
    if result["eligible_for_best"]:
        subprocess.run([sys.executable, "-m", "kernel_envolve.cli", "promote", result["run"]], check=True)
    outcome = "CPU 示例评测已记录"
else:
    time.sleep(3)
    outcome = "保留已验证 best"
report = {"phase": "分析", "hypothesis": "CPU 演示：去除人为等待", "outcome": outcome,
          "next_action": "演示已就绪；替换为真实算子和 coding agent"}
Path(os.environ["KERNEL_ENVOLVE_REPORT"]).write_text(json.dumps(report, ensure_ascii=False))
(task / "notes.md").write_text("# CPU 演示\\n\\n这不是训练算子性能结果。\\n" + outcome)
"""


def main(destination):
    root = Path(destination).resolve()
    root.mkdir(parents=True, exist_ok=False)
    tasks = []
    for name in ("demo_vector", "demo_reduction", "demo_activation"):
        work = root / name
        work.mkdir()
        (work / "task.json").write_text(
            json.dumps(
                {
                    "name": name,
                    "objective": "CPU demonstration only",
                    "device": "cpu",
                    "cases": [
                        {"id": "small", "args": {"n": 64}, "weight": 1},
                        {"id": "padded", "args": {"n": 257}, "weight": 1},
                    ],
                    "quick_trials": 1,
                    "full_trials": 3,
                    "warmup": 1,
                }
            )
        )
        (work / "task.py").write_text(
            "import numpy as np\nimport time\ndef make_inputs(case,seed): return (np.ones(case['n']),)\n"
            "def reference(x): return x+1\ndef baseline(x):\n time.sleep(.002)\n return x+1\n"
        )
        (work / "candidate.py").write_text("def run(x): return x+1\n")
        (work / "agent.py").write_text(AGENT)
        tasks.append(
            {
                "id": name,
                "name": name,
                "workspace": str(work),
                "task_dir": str(work),
                "kind": "generic",
                "resource": "cpu-demo",
                "command": [sys.executable, str(work / "agent.py")],
                "round_timeout_seconds": 30,
                "wall_budget_seconds": 180,
                "cooldown_seconds": 20,
                "baseline": {
                    "repository": "CPU demonstration",
                    "revision": "demo-v1",
                    "entrypoint": "task.py:baseline",
                    "introduced_on": None,
                    "introduction_url": None,
                    "frozen_on": dt.date.today().isoformat(),
                    "hardware": "CPU · 演示",
                    "contract": task_contract(work, "generic"),
                },
            }
        )
    deployment = {
        "version": 1,
        "resources": [{"id": "cpu-demo", "device": "cpu", "sessions": 2}],
        "tasks": tasks,
    }
    (root / "deployment.json").write_text(json.dumps(deployment, ensure_ascii=False, indent=2))
    store = Store(root / "state")
    try:
        store.register(validate(deployment, root))
    finally:
        store.close()
    print(f"kernel-envolve manage --state {root / 'state'} run")
    print(f"kernel-envolve manage --state {root / 'state'} serve")


if __name__ == "__main__":
    main(sys.argv[1])
