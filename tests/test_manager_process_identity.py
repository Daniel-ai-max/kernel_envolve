"""Unreaped child identity is evidence binding, never proof of liveness."""

import subprocess
import sys
import time

from kernel_envolve.manager import process


def test_zombie_birth_can_bind_logs_without_becoming_alive():
    child = subprocess.Popen([sys.executable, "-c", "pass"], start_new_session=True)
    try:
        birth = process.identity(child.pid, include_zombie=True)
        assert birth
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            state = subprocess.check_output(
                ["ps", "-p", str(child.pid), "-o", "stat="], text=True
            ).strip()
            if state.startswith("Z"):
                break
            time.sleep(0.01)
        else:
            raise AssertionError("Child did not become an unreaped zombie")
        assert process.identity(child.pid) is None
        assert not process.alive(child.pid, birth)
        assert process.identity(child.pid, include_zombie=True) == birth
        assert child.wait(timeout=5) == 0
        assert process.identity(child.pid, include_zombie=True) is None
    finally:
        if child.poll() is None:
            child.kill()
        child.wait(timeout=5)


def test_linux_proc_birth_preserves_default_alive_semantics(tmp_path, monkeypatch):
    original_path = process.Path
    proc = tmp_path / "proc"
    stat = proc / "123" / "stat"
    stat.parent.mkdir(parents=True)
    boot = proc / "sys/kernel/random/boot_id"
    boot.parent.mkdir(parents=True)
    boot.write_text("test-boot\n")

    def mapped_path(value):
        text = str(value)
        if text == "/proc" or text.startswith("/proc/"):
            return proc / text.removeprefix("/proc").lstrip("/")
        return original_path(value)

    monkeypatch.setattr(process, "Path", mapped_path)
    fields = ["Z"] + ["0"] * 18 + ["12345"]
    stat.write_text("123 (cpu child) " + " ".join(fields))
    birth = "test-boot:12345"
    assert process.identity(123) is None
    assert process.identity(123, include_zombie=True) == birth
    assert not process.alive(123, birth)
    fields[0] = "S"
    stat.write_text("123 (cpu child) " + " ".join(fields))
    assert process.identity(123) == process.identity(123, include_zombie=True) == birth
    assert process.alive(123, birth)
    stat.unlink()
    assert process.identity(123, include_zombie=True) is None
