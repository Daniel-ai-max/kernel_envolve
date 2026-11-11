"""Exercise real group admission parsing with controlled NVIDIA command responses."""

import subprocess

import pytest

from kernel_envolve.manager.process import gpu_group_idle


def driver(monkeypatch, reply):
    calls = []

    def run(argv, **kwargs):
        assert argv[0] == "nvidia-smi"
        assert kwargs["check"] and kwargs["timeout"] == 10
        calls.append(argv)
        return subprocess.CompletedProcess(argv, 0, reply(argv), "")

    monkeypatch.setattr("kernel_envolve.manager.process.subprocess.run", run)
    return calls


@pytest.mark.parametrize("count", [1, 2, 8])
def test_idle_group_uses_two_calls_and_preserves_rank_order(monkeypatch, count):
    devices = [f"GPU-{i}" for i in range(count)]
    original = devices[:]
    calls = driver(
        monkeypatch, lambda argv: "\n".join(devices[::-1]) if "--query-gpu=uuid" in argv else ""
    )
    gpu_group_idle(devices)
    assert len(calls) == 2
    assert all(argv[-2:] == ["-i", ",".join(devices)] for argv in calls)
    assert devices == original
    # Live occupancy must be checked again; no cached admission decision.
    gpu_group_idle(devices)
    assert len(calls) == 4


@pytest.mark.parametrize(
    "observed", ["", "GPU-0", "GPU-0\nGPU-0", "GPU-0\nGPU-X", "GPU-0\nGPU-1\nGPU-X"]
)
def test_incomplete_or_wrong_group_never_passes_after_successful_diagnosis(monkeypatch, observed):
    calls = driver(
        monkeypatch,
        lambda argv: (
            (observed if "," in argv[-1] else argv[-1]) if "--query-gpu=uuid" in argv else ""
        ),
    )
    with pytest.raises(ValueError, match="identity"):
        gpu_group_idle(["GPU-0", "GPU-1"])
    # Healthy single-card diagnostic responses cannot override a bad group response.
    assert len(calls) == 5
    assert [argv[-1] for argv in calls[1::2]] == ["GPU-0", "GPU-1"]


def test_busy_group_checks_all_cards_and_reports_both_failures(monkeypatch):
    def reply(argv):
        if "--query-gpu=uuid" in argv:
            return argv[-1].replace(",", "\n")
        return "123\n456" if "," in argv[-1] else ("123" if argv[-1] == "GPU-0" else "456")

    calls = driver(monkeypatch, reply)
    with pytest.raises(ValueError) as error:
        gpu_group_idle(["GPU-0", "GPU-1"])
    assert all(value in str(error.value) for value in ["GPU-0", "GPU-1", "123", "456"])
    assert len(calls) == 6


@pytest.mark.parametrize(
    "failure",
    [
        OSError("missing driver"),
        subprocess.TimeoutExpired("nvidia-smi", 10),
        subprocess.CalledProcessError(1, "nvidia-smi"),
    ],
)
def test_driver_errors_never_admit_work(monkeypatch, failure):
    attempted = []

    def fail(argv, **kwargs):
        attempted.append(argv[-1])
        raise failure

    monkeypatch.setattr("kernel_envolve.manager.process.subprocess.run", fail)
    with pytest.raises(ValueError, match="GPU group"):
        gpu_group_idle(["GPU-0", "GPU-1"])
    assert attempted == ["GPU-0,GPU-1", "GPU-0", "GPU-1"]


def test_cpu_never_queries_nvidia(monkeypatch):
    def unexpected(argv):
        raise AssertionError("CPU task attempted a GPU probe")

    calls = driver(monkeypatch, unexpected)
    gpu_group_idle(["cpu"])
    assert not calls
