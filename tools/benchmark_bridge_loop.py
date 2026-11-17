"""Real CPU CLI/bridge traces for protocol usage on one fixed runtime, not old binaries.

Example: python tools/benchmark_bridge_loop.py --runtime-ref <commit> \
  --output audit/bridge_loop_20261115.json --evidence-dir /tmp/bridge-loop-evidence
The source snapshot, complete subprocess logs and manager states remain in evidence-dir.
No GPU commands, model agents, failure injection or recovery benchmark are performed.
"""

import argparse
import hashlib
import io
import json
import os
import platform
import statistics
import subprocess
import sys
import tarfile
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
OUTCOME = "Verified the deterministic CPU result and complete raw logs."
NEXT = "Run the next bounded CPU experiment, or finish the sequence."
GROUPS = (
    ("legacy_separate_vs_combined", "separate", "consume_file"),
    ("optional_analysis_omitted", "consume_file", "no_analysis"),
    ("same_analysis_combined_vs_payload", "consume_file", "payload_analysis"),
)
WORKLOAD = """import json,pathlib,sys,time
root=pathlib.Path(sys.argv[1]);index=int(sys.argv[2]);delay=float(sys.argv[3])
with (root/(str(index)+'.executed')).open('x') as f:f.write('exactly once\\n')
print('starting CPU experiment '+str(index),flush=True)
time.sleep(delay)
print('x'*5000)
print(json.dumps({'index':index,'sum':sum((3*x+7) for x in range(4096))}),flush=True)
print('CPU-only stderr complete',file=sys.stderr,flush=True)
"""


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    Path(path).write_text(json.dumps(value, indent=2) + "\n")


def snapshot(ref, evidence):
    revision = subprocess.check_output(
        ["git", "rev-parse", ref + "^{commit}"], cwd=REPO, text=True
    ).strip()
    raw = subprocess.check_output(["git", "archive", revision, "src"], cwd=REPO)
    archive = evidence / "runtime.tar"
    archive.write_bytes(raw)
    runtime = evidence / "runtime"
    runtime.mkdir()
    with tarfile.open(fileobj=io.BytesIO(raw)) as package:
        package.extractall(runtime, filter="data")
    files = {str(p.relative_to(runtime)): sha(p) for p in runtime.rglob("*.py")}
    return runtime, dict(revision=revision, archive_sha256=sha(archive), files=files)


def sequence(root, env, runtime, mode, delay):
    # Import the same immutable source used by the real subprocess CLIs.
    from kernel_envolve.manager.process import alive
    from kernel_envolve.manager.store import Store
    from kernel_envolve.storage import config, contract

    root.mkdir()
    work = root / "work"
    work.mkdir()
    save(
        work / "task.json",
        dict(
            name="loop",
            objective="CPU feedback protocol only",
            device="cpu",
            cases=[dict(id="one", args={}, weight=1)],
        ),
    )
    (work / "candidate.py").write_text("def run(x): return x\n")
    (work / "task.py").write_text("def reference(x): return x\n")
    (work / "experiment.py").write_text(WORKLOAD)
    state, bridge = root / "state", root / "bridge"
    cli = [sys.executable, "-m", "kernel_envolve.cli", "manage", "--state", str(state)]
    calls = []

    def invoke(action, args, phase, inputs=()):
        argv = [*cli, action, *map(str, args)]
        started = time.perf_counter()
        process = subprocess.run(argv, env=env, capture_output=True, timeout=30, cwd=work)
        elapsed = time.perf_counter() - started
        index = len(calls)
        (root / f"cli-{index:02d}.stdout").write_bytes(process.stdout)
        (root / f"cli-{index:02d}.stderr").write_bytes(process.stderr)
        record = dict(
            index=index,
            action=action,
            phase=phase,
            seconds=elapsed,
            argv=argv,
            argv_utf8_bytes=sum(len(x.encode()) + 1 for x in argv),
            file_input_bytes=sum(Path(p).stat().st_size for p in inputs),
            stdout_bytes=len(process.stdout),
            stderr_bytes=len(process.stderr),
            stdout_sha256=hashlib.sha256(process.stdout).hexdigest(),
            stderr_sha256=hashlib.sha256(process.stderr).hexdigest(),
            returncode=process.returncode,
            requested_wait_seconds=(
                float(argv[argv.index("--wait-seconds") + 1]) if "--wait-seconds" in argv else None
            ),
        )
        calls.append(record)
        save(root / "calls.json", calls)
        if process.returncode:
            raise RuntimeError(f"CLI {action} failed: {process.stderr.decode()}")
        return json.loads(process.stdout) if process.stdout.strip() else None

    deployment = dict(
        version=1,
        resources=[dict(id="cpu", device="cpu", sessions=1)],
        tasks=[
            dict(
                id="loop",
                name="CPU protocol trace",
                workspace=str(work),
                task_dir=str(work),
                kind="generic",
                resource="cpu",
                host_driven=True,
                command=[
                    sys.executable,
                    "-m",
                    "kernel_envolve.cli",
                    "manage",
                    "bridge",
                    "--root",
                    str(bridge),
                ],
                round_timeout_seconds=120,
                wall_budget_seconds=180,
                gpu_budget_seconds=60,
                baseline=dict(
                    repository="local CPU diagnostic",
                    revision="fixed-runtime",
                    entrypoint="experiment.py",
                    hardware="CPU",
                    introduced_on=None,
                    introduction_url=None,
                    frozen_on="2026-11-15",
                    contract=contract(work, config(work)),
                ),
            )
        ],
    )
    save(root / "deployment.json", deployment)
    setup = time.perf_counter()
    invoke("init", [root / "deployment.json"], "setup", [root / "deployment.json"])
    invoke("bridge-open", ["loop", "--owner", "benchmark", "--claim-id", "single"], "setup")
    store = Store(state)
    worker = None
    receipts = []
    try:
        scheduled = store.schedule()
        assert len(scheduled) == 1, scheduled
        job = scheduled[0]
        with (root / "worker.stdout").open("wb") as out, (root / "worker.stderr").open("wb") as err:
            worker = subprocess.Popen(
                [sys.executable, "-m", "kernel_envolve.manager.worker", str(state), job],
                env=env,
                stdout=out,
                stderr=err,
                start_new_session=True,
            )
        directory = bridge / "loop" / job
        ready_start = time.perf_counter()
        while not (directory / "context.json").exists():
            assert worker.poll() is None and time.perf_counter() - ready_start < 15
            time.sleep(0.01)
        ready_wait = time.perf_counter() - ready_start
        setup_seconds = time.perf_counter() - setup
        common = ["--root", bridge, "--task", "loop", "--round", job, "--owner", "benchmark"]
        previous = None
        payloads = []
        start = time.perf_counter()
        for index in range(4):
            request_id = f"experiment-{index}" if index < 3 else "finish"
            payload = (
                dict(
                    kind="cpu",
                    argv=[
                        sys.executable,
                        str(work / "experiment.py"),
                        str(work),
                        str(index),
                        str(delay),
                    ],
                    expected_seconds=5,
                )
                if index < 3
                else dict(kind="finish")
            )
            extra, inputs = [], []
            if previous and mode != "no_analysis":
                analysis = dict(
                    request=previous["id"],
                    receipt_sha256=previous["receipt_sha256"],
                    outcome=OUTCOME,
                    next_action=NEXT,
                )
                if mode == "separate":
                    invoke(
                        "bridge-consume",
                        [
                            *common,
                            "--id",
                            analysis["request"],
                            "--sha256",
                            analysis["receipt_sha256"],
                            "--outcome",
                            OUTCOME,
                            "--next-action",
                            NEXT,
                        ],
                        "loop",
                    )
                elif mode == "consume_file":
                    path = root / f"analysis-{index}.json"
                    save(path, analysis)
                    extra = ["--consume-file", path]
                    inputs.append(path)
                else:
                    payload["analysis"] = analysis
            path = root / f"request-{index}.json"
            save(path, payload)
            inputs.append(path)
            arguments = [
                *common,
                "--id",
                request_id,
                "--file",
                path,
                *extra,
                "--wait-seconds",
                "10",
            ]
            receipt = invoke("bridge-request", arguments, "loop", inputs)
            payloads.append((arguments, inputs))
            while receipt["status"] == "pending":
                receipt = invoke(
                    "bridge-request", [*common, "--id", request_id, "--wait-seconds", "10"], "poll"
                )
            assert receipt["status"] == "succeeded" and receipt["returncode"] == 0, receipt
            assert sha(directory / "results" / (request_id + ".json")) == receipt["receipt_sha256"]
            if index < 3:
                stdout = Path(receipt["log_files"]["stdout"]["path"]).read_bytes()
                stderr = Path(receipt["log_files"]["stderr"]["path"]).read_bytes()
                expected = dict(index=index, sum=3 * 4096 * 4095 // 2 + 7 * 4096)
                assert json.loads(stdout.splitlines()[-1]) == expected
                assert (
                    stdout
                    == (
                        f"starting CPU experiment {index}\n"
                        + "x" * 5000
                        + "\n"
                        + json.dumps(expected)
                        + "\n"
                    ).encode()
                )
                assert stderr == b"CPU-only stderr complete\n"
                assert receipt["log_files"]["stdout"]["truncated"]
                assert len(receipt["stdout"].encode()) == 4096
                receipts.append(
                    dict(
                        id=request_id,
                        receipt_sha256=receipt["receipt_sha256"],
                        stdout_sha256=hashlib.sha256(stdout).hexdigest(),
                        stderr_sha256=hashlib.sha256(stderr).hexdigest(),
                        result=expected,
                        full_output_verified=True,
                        execution_seconds=receipt["finished_at"] - receipt["started_at"],
                    )
                )
            previous = receipt
        loop_seconds = time.perf_counter() - start
        cleanup_start = time.perf_counter()
        assert worker.wait(timeout=15) == 0
        # Actual same-ID submit replay and metadata-only status, outside timed sequence.
        replay = invoke("bridge-request", payloads[2][0], "validation_replay", payloads[2][1])
        assert replay["receipt_sha256"] == receipts[2]["receipt_sha256"]
        metadata = invoke(
            "bridge-request",
            [*common, "--id", "experiment-2", "--log-bytes", "0", "--wait-seconds", "0"],
            "validation_query",
        )
        assert not metadata["stdout"] and not metadata["stderr"]
        assert metadata["log_files"]["stdout"]["bytes"] > 4096
        assert sorted(p.name for p in work.glob("*.executed")) == [
            f"{i}.executed" for i in range(3)
        ]
        assert len(list((directory / "claims").glob("*.json"))) == 4
        assert len(list((directory / "results").glob("*.json"))) == 4
        jobs = store.rows("SELECT id,kind,status,exit_code FROM jobs")
        assert len(jobs) == 1 and jobs[0]["status"] == "succeeded" and jobs[0]["exit_code"] == 0
        assert not store.rows("SELECT id FROM jobs WHERE kind='gpu'")
        assert not store.rows("SELECT * FROM runs")
        assert not store.rows("SELECT id FROM resources WHERE problem IS NOT NULL")
        children = store.rows("SELECT pid,identity FROM processes")
        assert children and all(p["pid"] > 0 and p["identity"] for p in children)
        assert all(not alive(p["pid"], p["identity"]) for p in children)
        analyses = store.rows("SELECT data FROM events WHERE kind='host_consumed_result'")
        assert len(analyses) == (0 if mode == "no_analysis" else 3)
        assert all(json.loads(a["data"])["outcome"] == OUTCOME for a in analyses)
        assert all(json.loads(a["data"])["next_action"] == NEXT for a in analyses)
        report = dict(
            mode=mode,
            sequence_seconds=loop_seconds,
            cli_loop_seconds=sum(c["seconds"] for c in calls if c["phase"] == "loop"),
            extra_poll_seconds=sum(c["seconds"] for c in calls if c["phase"] == "poll"),
            cpu_receipt_execution_seconds=sum(r["execution_seconds"] for r in receipts),
            host_non_cli_loop_seconds=loop_seconds
            - sum(c["seconds"] for c in calls if c["phase"] in ("loop", "poll")),
            setup_seconds=setup_seconds,
            external_readiness_wait_seconds=ready_wait,
            cleanup_and_validation_seconds=time.perf_counter() - cleanup_start,
            calls=calls,
            results=receipts,
            analysis_count=len(analyses),
            jobs=jobs,
            no_duplicate_execution=True,
            terminal_cleanup=True,
            registered_processes_checked=len(children),
            gpu_jobs=0,
        )
        save(root / "sequence.json", report)
        return report
    finally:
        if worker and worker.poll() is None:
            store.control("loop", "pause")
            worker.wait(timeout=20)
        store.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime-ref", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--evidence-dir", type=Path, required=True)
    parser.add_argument("--pairs", type=int, default=3)
    parser.add_argument("--delay", type=float, default=0.03)
    args = parser.parse_args()
    if args.pairs < 1 or not 0 <= args.delay <= 1:
        parser.error("Require positive pairs and delay in [0,1]")
    evidence = args.evidence_dir.resolve()
    evidence.mkdir(parents=True, exist_ok=False)
    runtime, binding = snapshot(args.runtime_ref, evidence)
    sys.path.insert(0, str(runtime / "src"))
    sys.dont_write_bytecode = True
    env = {k: v for k, v in os.environ.items() if not k.startswith("KERNEL_ENVOLVE_")}
    env |= {
        "PYTHONPATH": str(runtime / "src"),
        "KERNEL_ENVOLVE_AUTO_UPDATE": "0",
        "CUDA_VISIBLE_DEVICES": "",
        "PYTHONDONTWRITEBYTECODE": "1",
    }
    os.environ.update(env)
    records = []
    for group, left, right in GROUPS:
        for pair in range(args.pairs):
            order = (left, right) if pair % 2 == 0 else (right, left)
            for position, mode in enumerate(order):
                record = sequence(
                    evidence / f"{group}-{pair}-{mode}", env, runtime, mode, args.delay
                )
                records.append(record | dict(group=group, pair=pair, position=position))
                print(f"{group} pair {pair} {mode}: {record['sequence_seconds']:.3f}s", flush=True)
    summaries = []
    for group, left, right in GROUPS:
        subset = [r for r in records if r["group"] == group]
        pairs = []
        for pair in range(args.pairs):
            a, b = [
                next(r for r in subset if r["pair"] == pair and r["mode"] == mode)
                for mode in (left, right)
            ]
            pairs.append(a["sequence_seconds"] - b["sequence_seconds"])
        arms = {}
        for mode in (left, right):
            rows = [r for r in subset if r["mode"] == mode]
            phases = {}
            for phase in ("setup", "loop", "poll", "validation_replay", "validation_query"):
                calls = [c for r in rows for c in r["calls"] if c["phase"] == phase]
                phases[phase] = dict(
                    calls=len(calls),
                    seconds=sum(c["seconds"] for c in calls),
                    input_bytes=sum(c["argv_utf8_bytes"] + c["file_input_bytes"] for c in calls),
                    output_bytes=sum(c["stdout_bytes"] + c["stderr_bytes"] for c in calls),
                )
            arms[mode] = dict(
                median_sequence_seconds=statistics.median(r["sequence_seconds"] for r in rows),
                analysis_count=sum(r["analysis_count"] for r in rows),
                phases=phases,
            )
        summaries.append(
            dict(
                group=group,
                arms=arms,
                paired_right_saved_seconds=pairs,
                median_right_saved_seconds=statistics.median(pairs),
                equal_analysis_work=group != "optional_analysis_omitted",
            )
        )
    assert all(sha(runtime / name) == digest for name, digest in binding["files"].items())
    result = dict(
        scope="Protocol usage on the same fixed new runtime; not old binary comparison, GPU performance or agent superiority",
        runtime=binding,
        benchmark_sha256=sha(__file__),
        platform=platform.platform(),
        python=sys.version,
        pairs=args.pairs,
        experiments_per_sequence=3,
        workload_sha256=hashlib.sha256(WORKLOAD.encode()).hexdigest(),
        outcome=OUTCOME,
        next_action=NEXT,
        delay_seconds=args.delay,
        gpu_used=False,
        recovery_scope="No failure injection/recovery rerun; existing recovery tests are separate evidence",
        byte_scope="UTF-8 argv including NUL separators plus JSON input files; stdout+stderr bytes, not network bytes or tokens",
        timing_scope="Loop includes host payload preparation, CLI startup, subprocess wait and verification; setup, extra polls and validation calls separately listed. Blocking wait is included in each CLI wall time, not claimed as pure transport overhead; receipt execution intervals are also reported. Extra polls, if any, remain included in loop wall time.",
        evidence_directory=str(evidence),
        summary=summaries,
        records=records,
    )
    save(args.output, result)
    save(evidence / "result.json", result)
    print(json.dumps(summaries, indent=2))


if __name__ == "__main__":
    main()
