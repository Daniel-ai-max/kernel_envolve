"""Real CLI lifecycle on synthetic CPU tasks, also runnable against an installed wheel.

No model or GPU calls. Deliberately slower control adapters make this an
engineering acceptance fixture, never evidence of RSI efficacy.
"""

import json
import os
import subprocess
import sys
import time
from pathlib import Path


def exercise(root, python, repo, *, installed=False):
    root.mkdir()
    env = {**os.environ, "KERNEL_ENVOLVE_AUTO_UPDATE": "0"}
    env.pop("PYTHONPATH", None)
    prefix = [str(python)]
    if installed:
        prefix.append("-I")
    else:
        env["PYTHONPATH"] = str(repo / "src")
    commands = []

    def command(args):
        result = subprocess.run(
            [*prefix, *map(str, args)],
            cwd=root,
            env=env,
            text=True,
            capture_output=True,
            timeout=150,
        )
        assert result.returncode == 0, result.stderr + result.stdout
        return result.stdout

    demo = root / "demo"
    config = Path(command([repo / "examples/rsi/demo.py", demo]).strip())
    adapter = demo / "adapters.py"
    adapter.write_text(
        adapter.read_text().replace(
            "else:\n    if packet['strategy']:",
            "else:\n    time.sleep(.02 if packet['strategy'] else 1)\n    if packet['strategy']:",
        )
    )
    production = demo / "production"
    original = {p.name: p.read_bytes() for p in production.iterdir()}
    state = root / "state"
    cli_args = ["-m", "kernel_envolve.cli", "rsi", "--state", str(state)]

    def cli(*args):
        value = json.loads(command([*cli_args, *args]))
        commands.append(str(args[0]))
        return value

    def checkpoint():
        return command(
            [
                "-c",
                "import sys; from kernel_envolve.manager.store import Store; "
                "from kernel_envolve.manager.worker import checkpoint; "
                "s=Store(sys.argv[1], readonly=True); "
                "print(checkpoint(s,s.task('demo'),{'id':'acceptance'})); s.close()",
                demo / "manager",
            ]
        )

    cli("init", str(config))
    base_prompt = checkpoint()
    result = cli("cycle")
    assert result["validation"]["accepted"], result
    adopted = result["adopted"]
    status = cli("status")
    assert status["active"] == {"demo": adopted}
    assert "Remove artificial sleep" in checkpoint()
    cli("disable")
    assert checkpoint() == base_prompt
    budgets = cli("status")["budgets"]
    assert {b["kind"] for b in budgets} == {"learn", "validation"}
    assert all(b["charged_seconds"] > 0 for b in budgets)

    last_cycle = status["last_cycle"]["at"] if status["last_cycle"] else 0
    for restart in range(2):
        with (root / f"daemon-{restart}.log").open("w") as log:
            process = subprocess.Popen(
                [*prefix, *cli_args, "run", "--interval", "1"],
                cwd=root,
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
            )
            try:
                deadline = time.monotonic() + 15
                while time.monotonic() < deadline:
                    assert process.poll() is None, "Sidecar exited unexpectedly"
                    current = cli("status")
                    observed = current["last_cycle"]
                    if observed and observed["at"] > last_cycle:
                        assert observed["status"] == "disabled"
                        assert current["budgets"] == budgets
                        last_cycle = observed["at"]
                        break
                    time.sleep(0.1)
                else:
                    raise AssertionError("Disabled sidecar did not become responsive")
                process.terminate()
                assert process.wait(timeout=5) == 0
            finally:
                if process.poll() is None:
                    process.kill()
                    process.wait(timeout=5)

    cli("enable")
    assert "Remove artificial sleep" in checkpoint()
    assert cli("rollback", "demo", "--reason", "Synthetic operator rollback")["active"] == "base"
    assert checkpoint() == base_prompt
    advice = demo / "manager/rsi-context/demo.json"
    advice.write_text("null")
    assert checkpoint() == base_prompt
    cli("disable")
    assert cli("status")["budgets"] == budgets
    assert {p.name: p.read_bytes() for p in production.iterdir()} == original
    return {
        "status": "passed",
        "commands": commands,
        "installed_wheel": installed,
        "daemon_starts": 2,
        "clean_sigterm_exits": 2,
        "production_source_unchanged": True,
        "corrupt_advice_preserves_checkpoint": True,
        "disabled_restart_preserves_budget": True,
        "external_model_calls": 0,
        "gpu_jobs": 0,
        "authority": "Synthetic CPU mechanics only; control delay is not an RSI benefit claim",
    }


def test_cli_lifecycle(tmp_path):
    exercise(tmp_path / "acceptance", sys.executable, Path(__file__).resolve().parents[1])


if __name__ == "__main__":
    result = exercise(Path(sys.argv[1]), sys.executable, Path(sys.argv[2]), installed=True)
    print(json.dumps(result, indent=2))
