from __future__ import annotations

import ctypes
import os
import sys
import time
from dataclasses import replace

import pytest

from th06ai.config import Config, SafetyConfig
from th06ai.supervisor import run_guard
from test_safety import healthy

pytestmark = pytest.mark.skipif(os.name != "nt", reason="Windows Job Object integration")


class SimulatedGpu:
    driver = "simulated-no-GPU-work"
    capabilities = {}

    def __init__(self, fault: str):
        self.count = 0
        self.fault = fault

    def sample(self):
        self.count += 1
        if self.count >= 6:
            if self.fault == "lost":
                raise OSError("simulated driver lost")
            if self.fault == "hang":
                time.sleep(2)
            if self.fault == "temperature":
                return healthy(temperature_c=70)
        return healthy()

    def close(self):
        pass


@pytest.mark.parametrize("fault", ["temperature", "lost", "hang"])
def test_real_worker_is_terminated_on_simulated_gpu_fault(tmp_path, fault):
    released = []
    actual_pid = tmp_path / "actual-pid.txt"
    child = [sys.executable, "-c", f"import os,time; from pathlib import Path; Path({str(actual_pid)!r}).write_text(str(os.getpid())); time.sleep(120)"]
    config = Config(game_exe="unused", minutes=1,
                    safety=SafetyConfig(telemetry_timeout_seconds=.75))
    result = run_guard(config, "self-test", run_dir=tmp_path,
                       reader_factory=lambda: SimulatedGpu(fault), child_command=child,
                       release_fn=lambda: released.append(True), hotkey_fn=lambda: False, verbose=False)
    assert result.code == 90
    assert result.child_pid
    assert released
    from th06ai.win32 import bind, close_handle, k32
    from ctypes import wintypes as wt
    open_process = bind(k32, "OpenProcess", wt.HANDLE, wt.DWORD, wt.BOOL, wt.DWORD)
    get_exit = bind(k32, "GetExitCodeProcess", wt.BOOL, wt.HANDLE, ctypes.POINTER(wt.DWORD))
    assert actual_pid.is_file(), "the real Python worker must have started"
    for pid in (result.child_pid, int(actual_pid.read_text())):
        handle = open_process(0x1000, False, pid)
        if handle:
            try:
                code = wt.DWORD()
                assert get_exit(handle, ctypes.byref(code))
                assert code.value != 259  # STILL_ACTIVE
            finally:
                close_handle(handle)


def test_bad_preflight_never_starts_worker(tmp_path):
    from th06ai.config import ROOT
    pointer = ROOT / "runs" / "current.json"
    before = pointer.read_bytes() if pointer.exists() else None
    class UnsafeGpu(SimulatedGpu):
        def sample(self):
            return healthy(temperature_c=70)
    result = run_guard(Config(game_exe="unused"), "self-test", run_dir=tmp_path,
                       reader_factory=lambda: UnsafeGpu("temperature"),
                       child_command=[sys.executable, "-c", "raise AssertionError('must not start')"],
                       release_fn=lambda: None, hotkey_fn=lambda: False, verbose=False)
    assert result.code == 90
    assert result.child_pid is None
    assert (pointer.read_bytes() if pointer.exists() else None) == before, "simulated faults must not replace live dashboard data"


def test_gpu_fault_terminates_even_when_worker_reports_user_pause(tmp_path):
    script = tmp_path / "paused_worker.py"
    script.write_text(
        "import json,time\nfrom pathlib import Path\n"
        f"p=Path({str(tmp_path / 'worker.json')!r})\n"
        "while True:\n p.write_text(json.dumps({'monotonic':time.monotonic(),'user_paused':True}))\n time.sleep(.05)\n",
        encoding="utf-8")
    result = run_guard(Config(game_exe="unused", minutes=1), "train", run_dir=tmp_path,
                       reader_factory=lambda: SimulatedGpu("temperature"),
                       child_command=[sys.executable, str(script)], release_fn=lambda: None,
                       hotkey_fn=lambda: False, verbose=False)
    assert result.child_pid and result.code == 90
    assert "温度" in result.reason


def test_worker_exit_displays_original_failure_even_without_heartbeat(tmp_path):
    error = "float32 status serialization failed"
    script = ("from pathlib import Path\nfrom th06ai.worker import record_failure\n"
              f"record_failure(Path({str(tmp_path)!r}), RuntimeError({error!r}))\n"
              "raise SystemExit(1)\n")
    result = run_guard(Config(game_exe="unused"), "self-test", run_dir=tmp_path,
                       reader_factory=lambda: SimulatedGpu("healthy"),
                       child_command=[sys.executable, "-c", script],
                       release_fn=lambda: None, hotkey_fn=lambda: False, verbose=False)
    assert result.code == 90
    assert error in result.reason and "详见 worker.log" not in result.reason
