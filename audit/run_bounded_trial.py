"""Bound one isolated comparison trial and freeze its declared submission.

Audit-only runner, not part of the kernel_envolve runtime. The image must already
exist locally and be pinned by digest or image ID. Model credentials/connectivity
are intentionally not supplied by this generic runner.
"""

import argparse
import hashlib
import json
import subprocess
import time
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--seconds", type=float, required=True)
    parser.add_argument("--network", choices=["none", "bridge"], default="none")
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    workspace, output = args.workspace.resolve(), args.output.resolve()
    if not workspace.is_dir() or output == workspace or output.is_relative_to(workspace):
        raise ValueError("Output evidence must be outside the trial workspace")
    if not 0 < args.seconds <= 480:
        raise ValueError("Trial deadline must be at most eight minutes")
    if not (args.image.startswith("sha256:") or "@sha256:" in args.image):
        raise ValueError("Pin the container image identity")
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command:
        raise ValueError("Supply the trial command")
    output.mkdir(parents=True, exist_ok=False)
    cidfile = output / "container-id"
    report = {
        "image": args.image,
        "command": command,
        "deadline_seconds": args.seconds,
        "network": args.network,
        "timed_out": False,
        "submission": None,
    }
    invocation = [
        "docker",
        "run",
        "--pull=never",
        "--cidfile",
        str(cidfile),
        "--network",
        args.network,
        "--read-only",
        "--cap-drop=ALL",
        "--security-opt=no-new-privileges",
        "--pids-limit=128",
        "--cpus=2",
        "--memory=2g",
        "--tmpfs",
        "/tmp:rw,nosuid,size=256m",
        "--mount",
        f"type=bind,src={workspace},dst=/work",
        "--workdir",
        "/work",
        args.image,
        *command,
    ]
    started = time.monotonic()
    process = None
    try:
        with (
            (output / "stdout.log").open("wb") as stdout,
            (output / "stderr.log").open("wb") as stderr,
        ):
            process = subprocess.Popen(invocation, stdout=stdout, stderr=stderr)
            try:
                process.wait(timeout=args.seconds)
            except subprocess.TimeoutExpired:
                report["timed_out"] = True
            finally:
                if cidfile.exists():
                    cid = cidfile.read_text().strip()
                    # Removing this exact owned container also kills detached children.
                    removed = subprocess.run(
                        ["docker", "rm", "--force", cid], capture_output=True, text=True, timeout=30
                    )
                    report["container_removed"] = removed.returncode == 0
                    if removed.returncode:
                        report["cleanup_error"] = removed.stderr
                if process.poll() is None:
                    process.terminate()
                    try:
                        process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait(timeout=5)
        report["returncode"] = process.returncode
        report["wall_seconds_including_cleanup"] = time.monotonic() - started
        submission = workspace / "submission.py"
        if submission.is_symlink():
            raise ValueError("Declared submission must not be a symlink")
        if submission.is_file():
            data = submission.read_bytes()
            if len(data) > 1024 * 1024:
                raise ValueError("Single-file pilot submission exceeds 1 MiB")
            (output / "submission.py").write_bytes(data)
            report["submission"] = {"sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)}
        report["outcome"] = (
            "declared_submission_frozen" if report["submission"] else "no_submission"
        )
    finally:
        (output / "trial.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report))
    if not report.get("container_removed"):
        raise SystemExit("Container cleanup was not confirmed")
    if report["submission"] is None:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
