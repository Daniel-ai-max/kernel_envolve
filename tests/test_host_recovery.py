import json
import subprocess
import sys
import time

import pytest

from kernel_envolve.manager.execution.allocation_guard import release, release_reason
from kernel_envolve.manager.execution.host_recovery import HostRecovery, transient_failure

NETWORK = [{"type": "turn.failed", "error": {"message": "stream disconnected before completion"}}]


def ledger(tmp_path, **changes):
    return HostRecovery(
        tmp_path / "recovery.json",
        **(dict(model="model", session="original", round_id="round", deadline=1000) | changes),
    )


def admit(h, **changes):
    return h.admit(
        **(dict(now=20, active=True, reconciled=True, session="original", model="model") | changes)
    )


def test_resumes_exact_identity_after_reconciliation_and_survives_restart(tmp_path):
    h = ledger(tmp_path)
    h.failed(NETWORK, 1, trace_sha256="raw-trace", now=10)
    assert admit(h, now=11) == "wait"
    assert admit(h, reconciled=False) == "wait"
    h = ledger(tmp_path)
    assert admit(h) == "resume"
    assert h.data["retries"] == 1
    assert h.data["attempts"][0]["trace_sha256"] == "raw-trace"
    with pytest.raises(ValueError):
        admit(ledger(tmp_path))  # admission cannot replay after a crash


def test_exhaustion_stops_instead_of_starting_fresh_session(tmp_path):
    h = ledger(tmp_path)
    for now in (10, 60):
        h.failed(NETWORK, 1, trace_sha256=str(now), now=now)
        assert admit(h, now=now + 40) == "resume"
    h.failed(NETWORK, 1, trace_sha256="last", now=110)
    assert admit(h, now=150) == "interrupt"
    assert ledger(tmp_path).data["reason"] == "same_model_recovery_exhausted"


@pytest.mark.parametrize(
    "changes", [{"session": "new"}, {"model": "other"}, {"deadline": 2000}, {"max_retries": 4}]
)
def test_recovery_cannot_reset_identity_or_budget(tmp_path, changes):
    ledger(tmp_path)
    with pytest.raises(ValueError):
        ledger(tmp_path, **changes)


@pytest.mark.parametrize("changes", [{"now": 910}, {"active": False}])
def test_stop_and_cleanup_reserve_preempt_resume(tmp_path, changes):
    h = ledger(tmp_path)
    h.failed(NETWORK, 1, trace_sha256="trace", now=10)
    assert admit(h, **changes) == "interrupt"


def test_only_trusted_final_transport_failure_allows_resume(tmp_path):
    assert not transient_failure([{"type": "item.completed", "item": {"text": str(NETWORK)}}])
    assert not transient_failure(NETWORK + [{"type": "turn.completed"}])
    h = ledger(tmp_path)
    h.failed(
        [{"type": "turn.failed", "error": {"code": "invalid_api_key"}}],
        1,
        trace_sha256="trace",
        now=10,
    )
    assert admit(h) == "interrupt"


def test_unknown_provider_failure_is_bounded_without_misclassifying_as_network(tmp_path):
    h = ledger(tmp_path)
    h.failed(
        [{"type": "turn.failed", "error": {"message": "HTTP 404"}}], 1, trace_sha256="trace", now=10
    )
    assert h.data["reason"] == "model_execution_failure"
    assert admit(h) == "resume"


@pytest.mark.parametrize("returncode", [-9, 137, 1, 0])
def test_missing_terminal_requires_independent_exit_evidence(tmp_path, returncode):
    h = ledger(tmp_path)
    h.failed([], returncode, trace_sha256="partial-trace", now=10, process_exited=True)
    assert h.data["reason"] == "model_process_exited"
    assert admit(h) == "resume"


@pytest.mark.parametrize("returncode", [None, 78])
def test_uncertain_or_explicitly_stopped_host_never_resumes(tmp_path, returncode):
    h = ledger(tmp_path)
    h.failed([], returncode, trace_sha256="trace", now=10, process_exited=True)
    assert admit(h) == "interrupt"


def test_silence_is_not_exit_and_restart_does_not_extend_failure_window(tmp_path):
    h = ledger(tmp_path)
    h.failed([], -9, trace_sha256="trace", now=10)
    assert admit(h) == "interrupt"


def test_real_model_progress_ends_outage_without_refunding_retry_budget(tmp_path):
    h = ledger(tmp_path)
    h.failed(NETWORK, 1, trace_sha256="first-failure", now=10)
    assert admit(h) == "resume"
    assert not h.progress([{"type": "thread.started"}], now=21, trace_sha256="startup")
    assert h.progress(
        [
            {
                "type": "item.completed",
                "item": {"type": "command_execution", "status": "completed", "exit_code": 0},
            }
        ],
        now=22,
        trace_sha256="new-tool",
    )
    h = ledger(tmp_path)
    assert h.data["retries"] == 1
    assert "recovery_started_at" not in h.data
    h.failed(NETWORK, 1, trace_sha256="later-failure", now=500)
    assert admit(h, now=520) == "resume"
    h.failed(NETWORK, 1, trace_sha256="last-failure", now=550)
    assert admit(h, now=590) == "interrupt"


@pytest.mark.parametrize(
    "lease",
    [
        {"expires_at": 10**400, "state": "running"},
        {"expires_at": 1e20, "state": "unknown"},
        {"expires_at": "Infinity", "state": "running"},
    ],
)
def test_real_watchdog_releases_malformed_lease(tmp_path, lease):
    (tmp_path / "lease.json").write_text(json.dumps(lease))
    config = dict(
        deadline=time.time() + 3600,
        lease_path=str(tmp_path / "lease.json"),
        release_marker=str(tmp_path / "RELEASE"),
        receipt=str(tmp_path / "receipt.json"),
    )
    (tmp_path / "config.json").write_text(json.dumps(config))
    subprocess.run(
        [
            sys.executable,
            "-m",
            "kernel_envolve.manager.execution.allocation_guard",
            str(tmp_path / "config.json"),
        ],
        check=True,
        timeout=5,
    )
    assert (tmp_path / "RELEASE").exists()


@pytest.mark.parametrize("value", [None, "1000", True, 10**400, float("inf")])
def test_malformed_allocation_controls_fail_closed(value):
    assert release_reason(now=1, deadline=value, lease_expires=500) == "invalid_allocation_control"


@pytest.mark.parametrize("timeout", [None, "30", float("nan"), float("inf"), -1])
def test_bad_cleanup_timeout_still_releases(tmp_path, timeout):
    release(tmp_path / "RELEASE", tmp_path / "receipt.json", reason="expired", timeout=timeout)
    assert (tmp_path / "RELEASE").exists()


def test_unresolved_gpu_receipt_cannot_renew_recovery_forever(tmp_path):
    h = ledger(tmp_path)
    h.failed(NETWORK, 1, trace_sha256="trace", now=10)
    assert admit(h, now=189, reconciled=False) == "wait"
    assert admit(ledger(tmp_path), now=190, reconciled=False) == "interrupt"
    assert ledger(tmp_path).data["reason"] == "recovery_window_exhausted"


def test_release_lease_and_deadline_are_independent_of_inflight_gpu():
    args = dict(now=100, deadline=1000, lease_expires=200)
    assert release_reason(**args) is None
    assert release_reason(**(args | {"lease_expires": 99})) == "model_host_lease_expired"
    assert release_reason(**(args | {"now": 820})) == "allocation_deadline"
    assert release_reason(**args, interrupted=True) == "autonomous_workflow_interrupted"
    assert (
        release_reason(**(args | {"lease_expires": float("nan")})) == "invalid_allocation_control"
    )


def test_cleanup_timeout_or_failure_cannot_prevent_release(tmp_path):
    marker, receipt = tmp_path / "RELEASE", tmp_path / "receipt.json"
    release(
        marker,
        receipt,
        reason="model_host_lease_expired",
        timeout=0.05,
        cleanup_commands=[
            [sys.executable, "-c", "import time; time.sleep(10)"],
            [sys.executable, "-c", "raise SystemExit(7)"],
        ],
    )
    result = json.loads(receipt.read_text())
    assert marker.read_text().strip() == "model_host_lease_expired"
    assert result["cleanup"][0]["error"] == "TimeoutExpired"
    assert result["cleanup"][1]["error"] == "cleanup_budget_exhausted"


def test_node_watchdog_exits_without_a_desktop_or_model_host(tmp_path):
    config = dict(
        deadline=time.time() + 3600,
        lease_path=str(tmp_path / "missing-lease"),
        release_marker=str(tmp_path / "RELEASE"),
        receipt=str(tmp_path / "receipt.json"),
    )
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config))
    subprocess.run(
        [sys.executable, "-m", "kernel_envolve.manager.execution.allocation_guard", str(path)],
        check=True,
        timeout=10,
    )
    receipt = json.loads((tmp_path / "receipt.json").read_text())
    assert receipt["reason"] == "model_host_lease_expired"
    assert (tmp_path / "RELEASE").is_file()
