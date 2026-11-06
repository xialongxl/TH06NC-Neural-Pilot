from __future__ import annotations

import ctypes
import os
import sys
import time
from dataclasses import replace
from types import SimpleNamespace

import pytest

from th06ai.config import Config, SafetyConfig
from th06ai.control import SCENE_WAIT_LIMIT_SECONDS, STAGE_OVERTIME_LIMIT_SECONDS
from th06ai.supervisor import WORKER_EXIT_GRACE_SECONDS, run_guard
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


@pytest.mark.parametrize('emergency_stop, expected_code', [(False, 0), (True, 90)])
def test_finish_before_the_worker_starts_cancels_launch(tmp_path, emergency_stop, expected_code):
    (tmp_path / 'FINISH').touch()
    if emergency_stop:
        (tmp_path / 'STOP').touch()
    result = run_guard(Config(game_exe='unused'), 'train', run_dir=tmp_path,
                       reader_factory=lambda: SimulatedGpu('healthy'),
                       child_command=[sys.executable, '-c', "raise AssertionError('must not start')"],
                       release_fn=lambda: None, verbose=False)
    assert result.code == expected_code and result.child_pid is None


@pytest.mark.parametrize('worker_state', [{'user_paused': True}, {'scene_wait_step': 0}, {'exit_ready': True}],
                         ids=['user_pause', 'scene_wait', 'worker_exit'])
def test_gpu_fault_terminates_even_when_worker_is_waiting(tmp_path, worker_state):
    script = tmp_path / "paused_worker.py"
    script.write_text(
        "import json,time\nfrom pathlib import Path\n"
        f"p=Path({str(tmp_path / 'worker.json')!r})\n"
        f"state={worker_state!r}\n"
        "while True:\n p.write_text(json.dumps(dict(state,monotonic=time.monotonic())))\n time.sleep(.05)\n",
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


def _run_guard_on_clock(tmp_path, monkeypatch, worker_state, *, duration, tick, finish_at=None):
    from th06ai import supervisor, win32

    now = [0.]
    child_started = []
    terminated = []
    published = {}

    def elapsed():
        return now[0] - child_started[0]

    def poll():
        return 0 if terminated or (finish_at is not None and elapsed() >= finish_at) else None

    child = SimpleNamespace(pid=1, poll=poll,
                            wait=lambda **_: None, close=lambda: None)

    def start(*_):
        child_started.append(now[0])
        return child

    monkeypatch.setattr(supervisor, 'time', SimpleNamespace(
        monotonic=lambda: now[0], sleep=lambda _: now.__setitem__(0, now[0] + tick)))
    monkeypatch.setattr(win32, 'GuardLock', lambda _: SimpleNamespace(close=lambda: None))
    monkeypatch.setattr(win32, 'OwnedJob', lambda: SimpleNamespace(
        start=start, terminate=lambda: terminated.append(True), close=lambda: None))
    monkeypatch.setattr(supervisor, 'SharedLease', lambda **_: SimpleNamespace(
        name='test', publish=lambda *_: None, close=lambda: None))
    monkeypatch.setattr(supervisor, 'AsyncTelemetry', lambda _: SimpleNamespace(
        poll=lambda _: (healthy(monotonic=now[0]), {}), next=lambda: None, close=lambda: None))
    monkeypatch.setattr(supervisor, 'SafetyPolicy', lambda _: SimpleNamespace(check=lambda *_, **__: None))
    monkeypatch.setattr(supervisor, 'read_json', lambda _: {'monotonic': now[0], **worker_state(elapsed())})
    monkeypatch.setattr(supervisor, 'atomic_json', lambda path, state: published.__setitem__(path.name, state))

    result = run_guard(Config(game_exe='unused'), 'train', minutes=duration/60, run_dir=tmp_path,
                       reader_factory=lambda: None, child_command=['test-worker'],
                       release_fn=lambda: None, verbose=False)
    return result, elapsed(), bool(terminated), published['result.json']


@pytest.mark.parametrize('stage,allowance', [(None, 0), (True, 0), (5, STAGE_OVERTIME_LIMIT_SECONDS)])
def test_guard_bounds_stage_overtime_with_its_own_clock(tmp_path, monkeypatch, stage, allowance):
    result, elapsed, terminated, _ = _run_guard_on_clock(tmp_path, monkeypatch,
        lambda _: {'clock_started': True, 'overtime_stage': stage, 'budget_seconds': 99999}, duration=60, tick=30)
    assert result.code == 90 and terminated
    assert 60 + allowance + 120 < elapsed <= 60 + allowance + 150


def test_guard_counts_scene_waits_in_a_training_run_past_the_old_deadline(tmp_path, monkeypatch):
    def scene_wait(elapsed):
        for step, start in enumerate(range(60, 461, 100), start=1):
            if start <= elapsed < start + 30:
                return {'clock_started': True, 'scene_wait_step': step}
        return {'clock_started': True}

    result, elapsed, terminated, status = _run_guard_on_clock(tmp_path, monkeypatch, scene_wait,
                                                               duration=600, tick=1, finish_at=750)
    assert result.code == 0 and elapsed == 750 and not terminated
    assert status['guarded_scene_wait_seconds'] == 5 * 30


def test_guard_caps_stuck_scene_wait_and_ignores_replayed_steps(tmp_path, monkeypatch):
    def scene_wait(elapsed):
        return {'clock_started': True, 'scene_wait_step': 7 if elapsed < 100 else 6, 'paused_seconds': 99999}

    result, elapsed, terminated, status = _run_guard_on_clock(tmp_path, monkeypatch, scene_wait,
                                                               duration=60, tick=10)
    assert result.code == 90 and terminated
    assert status['guarded_scene_wait_seconds'] == SCENE_WAIT_LIMIT_SECONDS
    assert 180 + SCENE_WAIT_LIMIT_SECONDS < elapsed <= 190 + SCENE_WAIT_LIMIT_SECONDS


@pytest.mark.parametrize('finish_at, expected_active', [(190, 60), (None, 181)])
def test_guard_publishes_one_clock_after_startup_and_pause(tmp_path, monkeypatch, finish_at, expected_active):
    def worker(elapsed):
        paused_in_scene = 120 <= elapsed < 140
        return {'clock_started': elapsed >= 90, 'clock_paused': 100 <= elapsed < 125,
                'user_paused': paused_in_scene, 'scene_wait_step': 3 if paused_in_scene else None,
                'training_seconds': 99999, 'paused_seconds': 99999, 'budget_seconds': 99999}

    result, elapsed, terminated, status = _run_guard_on_clock(tmp_path, monkeypatch, worker,
                                                               duration=60, tick=1, finish_at=finish_at)
    assert result.code == (0 if finish_at else 90) and terminated == (finish_at is None)
    assert elapsed == 90 + 40 + expected_active
    assert status['session_time'] == {'training_seconds': expected_active, 'paused_seconds': 40, 'budget_seconds': 60}
    assert {key: status['worker'][key] for key in status['session_time']} == status['session_time']
    assert status['guarded_scene_wait_seconds'] == 0


def test_late_readiness_cannot_extend_startup_deadline(tmp_path, monkeypatch):
    result, elapsed, terminated, status = _run_guard_on_clock(tmp_path, monkeypatch,
        lambda t: {'clock_started': t >= 190, 'overtime_stage': 5}, duration=60, tick=10)
    assert result.code == 90 and elapsed == 190 and terminated
    assert status['session_time']['training_seconds'] is None


@pytest.mark.parametrize('exit_ready', [False, True])
def test_only_final_handoff_allows_the_worker_to_exit_after_heartbeat_stops(tmp_path, monkeypatch, exit_ready):
    from th06ai import supervisor
    from th06ai.i18n import msg

    final_beat = []

    def worker_state(elapsed):
        if elapsed < 1:
            return {'clock_started': True}
        if not final_beat:
            final_beat.append(supervisor.time.monotonic())
        return {'clock_started': True, 'finishing': True, 'phase': msg('backend.complete_2'),
                'exit_ready': exit_ready, 'monotonic': final_beat[0]}

    result, elapsed, terminated, status = _run_guard_on_clock(tmp_path, monkeypatch, worker_state,
                                                               duration=60, tick=.25, finish_at=4)
    if exit_ready:
        assert result.code == 0 and elapsed == 4 and not terminated
        assert status['session_time']['training_seconds'] == 1
    else:
        assert result.code == 90 and elapsed == 3.25 and terminated
        assert result.reason == msg('backend.the_ai_worker_heartbeat_is_stale_or')


def test_exit_deadline_cannot_be_extended_by_repeated_handoffs(tmp_path, monkeypatch):
    from th06ai.i18n import msg

    result, elapsed, terminated, _ = _run_guard_on_clock(tmp_path, monkeypatch,
        lambda elapsed: {'clock_started': True, 'exit_ready': elapsed >= 1}, duration=60, tick=.25)
    assert result.code == 90 and terminated
    assert elapsed == 1 + WORKER_EXIT_GRACE_SECONDS + .25
    assert result.reason == msg('backend.the_ai_worker_did_not_exit_after_cleanup')
