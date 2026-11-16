"""Real manager subprocess/receipt path on its CPU test resource, without a GPU."""

import os
import sys

from kernel_envolve.manager.assessment import refresh_sources
from kernel_envolve.manager.gpu import run
from kernel_envolve.manager.process import identity
from kernel_envolve.manager.single import parser, prepare
from kernel_envolve.manager.views import task_view

pytest_plugins = ["test_research_task"]


def test_real_managed_command_auto_imports_no_benefit_and_keeps_task_active(research):
    store, work = research
    script = work / "validate.py"
    script.write_text("""import json, os
from pathlib import Path
from kernel_envolve.validation import validate_candidate
state=Path(os.environ['KERNEL_ENVOLVE_STATE'])
before=json.loads((state/'jobs'/os.environ['KERNEL_ENVOLVE_GPU_JOB']/'research-source-before.json').read_text())
def expensive():
    raise AssertionError('expensive gate must never run')
validate_candidate(lambda: dict(scope='full call', mode='wall',
    baseline_revision=before['baseline_revision'], source_sha256=before['fingerprint'],
    cases=[dict(name='one',correctness=True,baseline=[100]*7,candidate=[110]*7)]),
    [('native', expensive)], cases=['one'], scope='full call', mode='wall')
""")
    key = prepare(
        store,
        parser().parse_args(
            [
                "--research",
                str(work),
                "--device",
                "cpu",
                "--seconds",
                "60",
                "--",
                sys.executable,
                "-c",
                "pass",
            ]
        ),
    )
    round_id = store.schedule()[0]
    assert store.claim(round_id, os.getpid(), identity(os.getpid()))
    # Same-state receipts are discovered without a watch or report operation.
    assert run(store.root, key, round_id, [sys.executable, str(script)], expected_seconds=10) == 0
    refresh_sources(store)
    view = task_view(store, key)
    assert view["latest_validation"]["status"] == "no_benefit"
    assert view["best"] is None and view["verified_assessment"] is None
    assert store.task(key)["desired"] == "active"
    assert len(store.rows("SELECT * FROM jobs WHERE kind='gpu'")) == 1
    assert not store.rows("SELECT * FROM promotions")
    store.finish(round_id, "succeeded", None, 0)
