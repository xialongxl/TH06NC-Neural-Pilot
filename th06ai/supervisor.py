from __future__ import annotations
from .i18n import localized, msg, ui_text

import json
import os
import subprocess
import sys
import time
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from .config import Config, PROFILE_KINDS, ROOT
from .control import STAGE_OVERTIME_LIMIT_SECONDS, SessionStopwatch
from .files import atomic_json, new_run, read_json, timestamp
from .gpu import AsyncTelemetry, NvmlReader, SafetyPolicy, SafetyTrip
from .lease import SharedLease


GAME_MODES = ('train', 'play', 'evaluate')
FINISH_RESERVE_SECONDS = 120
WORKER_EXIT_GRACE_SECONDS = 5.0


class GuardedSessionClock:
    """Own the startup bound and the subsequent active-time deadlines."""

    def __init__(self, now, budget_seconds, mode):
        self.budget_seconds = budget_seconds
        self.mode = mode
        self.ready = False
        self.overtime_granted = False
        self.stopwatch = SessionStopwatch(now)

    def observe(self, now, worker):
        self.stopwatch.advance(now)
        if not worker or self.mode not in GAME_MODES:
            return
        if not self.ready and worker.get('clock_started') is True and not self.expired:
            # The startup timeout is independent of the planned game session.
            self.stopwatch = SessionStopwatch(now)
            self.ready = True
        paused = worker.get('clock_paused') is True or worker.get('user_paused') is True
        self.stopwatch.set_wait(now, paused=paused, scene_step=worker.get('scene_wait_step') if self.ready else None)
        stage = worker.get('overtime_stage')
        if (self.mode == 'train' and self.ready and self.target_reached and not self.expired
                and type(stage) is int and 1 <= stage <= 7):
            self.overtime_granted = True

    @property
    def target_reached(self):
        return self.ready and self.stopwatch.active_seconds >= self.budget_seconds

    @property
    def expired(self):
        allowance = STAGE_OVERTIME_LIMIT_SECONDS if self.overtime_granted else 0
        return self.stopwatch.active_seconds > self.budget_seconds + allowance + FINISH_RESERVE_SECONDS

    def snapshot(self):
        return {'training_seconds': self.stopwatch.active_seconds if self.ready else None,
                'paused_seconds': self.stopwatch.paused_seconds if self.ready else None,
                'budget_seconds': self.budget_seconds}


@dataclass
class GuardResult:
    code: int
    reason: str
    run_dir: Path
    child_pid: int | None


def run_guard(config: Config, mode: str, minutes: float | None = None,
              model: str | None = None, *, run_dir: Path | None = None,
              goal: str | None = None,
              game_setup=None,
              profile_kind: str | None = None,
              profile_wait_combat: bool = False,
              reader_factory: Callable | None = None, child_command: list[str] | None = None,
              release_fn: Callable | None = None, hotkey_fn: Callable | None = None,
              pause_hotkey_fn: Callable | None = None,
              hotkeys_factory: Callable | None = None,
              verbose: bool = True) -> GuardResult:
    """The default path cannot launch an AI worker before healthy telemetry.

    Factory/command injection is solely for low-load failure-injection tests;
    the CLI exposes neither. A fresh supervisor is required after every stop.
    """
    from .win32 import GuardLock, OwnedJob, get_async_key, release_all, allow_foreground
    import hashlib
    if goal is not None and goal not in ('survival','nmnb'):
        raise ValueError(msg('backend.unsupported_training_goal'))
    from .game_settings import parse_setup
    game_setup = parse_setup(game_setup if game_setup is not None else config.game_setup)
    if profile_kind is not None and profile_kind not in PROFILE_KINDS:
        raise ValueError(msg('backend.unsupported_profile_type'))

    suffix = hashlib.sha256(str(ROOT).encode()).hexdigest()[:16]
    if child_command is not None:
        suffix += "-test-" + str(os.getpid())
    guard_lock = GuardLock("Local\\TH06AI-" + suffix)

    if release_fn is None:
        release_fn = release_all if mode in ("calibrate", "train", "play", "evaluate") else lambda: None
    real_events = child_command is None and reader_factory is None and hotkey_fn is None and pause_hotkey_fn is None
    hotkey_fn = hotkey_fn or ((lambda: bool(get_async_key(0x77) & 0x8000)) if real_events else lambda: False)
    pause_hotkey_fn = pause_hotkey_fn or (lambda: False)  # Native listener is the sole F7 owner.
    pause_key_down = False
    run_dir = run_dir or new_run(ROOT, mode)
    run_dir.mkdir(parents=True, exist_ok=True)
    policy = SafetyPolicy(config.safety)
    telemetry = AsyncTelemetry(reader_factory or (lambda: NvmlReader(config.gpu_index)))
    current_file = run_dir / "status.json"
    lease = SharedLease(create=True)
    heartbeat_file = run_dir / "worker.json"
    child = None
    job = None
    log = None
    child_pid = None
    healthy = 0
    last_sample = None
    metadata = {}
    next_sample = time.monotonic()
    last_console = 0.0
    last_publish = 0.0
    pointer_published = False
    publish_public_pointer = reader_factory is None and child_command is None and mode != 'profile'
    started = time.monotonic()
    child_started = None
    session_clock = None
    worker_exit_deadline = None
    duration = 60 * (config.minutes if minutes is None else minutes)
    if not 0 < duration <= 3600:
        raise ValueError(msg('backend.monitoring_or_training_must_run_for_more'))
    code, reason = 0, msg('backend.monitoring_ended')
    state = "starting"
    worker = {}
    hotkeys = None
    stop_requested = threading.Event()

    def emergency_stop():
        # This dispatcher never waits for the telemetry/file-writing loop.
        stop_requested.set()
        try:
            owned_job = job
            if owned_job is not None:
                owned_job.terminate()
        finally:
            release_fn()

    def toggle_pause():
        if mode not in GAME_MODES or stop_requested.is_set():
            return
        from .control import set_paused
        paused = (run_dir / 'PAUSE').exists() or worker.get('focus_pending') is True
        if paused and isinstance(worker.get('pid'), int):
            allow_foreground(worker['pid'])
        set_paused(run_dir, not paused)

    def stopping():
        return stop_requested.is_set() or bool(hotkeys and hotkeys.stop_requested.is_set())

    def timing_status(heartbeat):
        snapshot = session_clock.snapshot() if session_clock and mode in GAME_MODES else None
        return {'worker': {**heartbeat, **snapshot} if snapshot is not None and mode == 'train' else heartbeat,
                'session_time': snapshot,
                'guarded_scene_wait_seconds': round(session_clock.stopwatch.scene_wait_seconds, 3) if session_clock else 0.0}

    try:
        if real_events or hotkeys_factory is not None:
            from .hotkeys import GlobalHotkeys
            factory = hotkeys_factory or GlobalHotkeys
            hotkeys = factory(emergency_stop, toggle_pause, enable_pause=mode in GAME_MODES)
        with (run_dir / "gpu.jsonl").open("a", encoding="utf-8", buffering=1) as gpu_log:
            while True:
                now = time.monotonic()
                if stopping() or (run_dir / "STOP").exists() or hotkey_fn():
                    raise SafetyTrip(msg('backend.emergency_stop_requested_by_the_user_button'))
                if child is None and mode in GAME_MODES and (run_dir / 'FINISH').exists():
                    reason = msg('backend.startup_cancelled')
                    break
                pause_key = pause_hotkey_fn()
                if mode in GAME_MODES and pause_key and not pause_key_down:
                    toggle_pause()
                pause_key_down = pause_key
                update = telemetry.poll(config.safety.telemetry_timeout_seconds)
                if update:
                    last_sample, metadata = update
                    policy.check(last_sample)
                    healthy += 1
                    record = {"time": timestamp(), **last_sample.to_dict()}
                    gpu_log.write(json.dumps(record, ensure_ascii=False) + "\n")
                    next_sample = last_sample.monotonic + config.safety.poll_seconds
                if last_sample is not None:
                    policy.check(last_sample, now=time.monotonic())
                    if time.monotonic() >= next_sample:
                        telemetry.next()
                    if healthy >= 3 and mode != "monitor" and child is None:
                        job = OwnedJob()
                        if stopping():
                            raise SafetyTrip(msg('backend.emergency_stop_requested_by_the_user_f8'))
                        command = child_command or [sys.executable, "-m", "th06ai.worker", "--mode", mode,
                                                    "--run-dir", str(run_dir), "--minutes", str(duration / 60),
                                                    "--lease-name", lease.name]
                        if model and child_command is None:
                            command += ["--model", model]
                        if goal is not None and child_command is None:
                            command += ['--goal',goal]
                        if game_setup is not None and child_command is None:
                            command += ['--game-setup',json.dumps(game_setup.to_dict(),separators=(',',':'))]
                        if profile_kind is not None and child_command is None:
                            command += ['--profile-kind',profile_kind]
                        if profile_wait_combat and child_command is None:
                            command += ['--profile-wait-combat']
                        env = os.environ.copy()
                        env["PYTHONUNBUFFERED"] = "1"
                        env["CUDA_VISIBLE_DEVICES"] = last_sample.uuid if config.device == "cuda" else "-1"
                        log = (run_dir / "worker.log").open("w", encoding="utf-8")
                        child = job.start(command, ROOT, env, log)
                        if stopping():
                            raise SafetyTrip(msg('backend.emergency_stop_requested_by_the_user_f8'))
                        child_pid = child.pid
                        # Worker cannot touch game/CUDA until the shared lease is live.
                        child_started = time.monotonic()
                        session_clock = GuardedSessionClock(child_started, duration, mode)
                        state = "running"
                    elif healthy >= 3 and mode == "monitor":
                        state = "monitoring"
                    worker = read_json(heartbeat_file) if child else {}
                    clock_changed = False
                    if child is not None:
                        if worker_exit_deadline is None:
                            old_clock_state = session_clock.ready, session_clock.target_reached
                            session_clock.observe(time.monotonic(), worker)
                            clock_changed = old_clock_state != (session_clock.ready, session_clock.target_reached)
                        exit_code = child.poll()
                        if exit_code is not None:
                            if stopping() or (run_dir / 'STOP').exists():
                                raise SafetyTrip(msg('backend.emergency_stop_requested_by_the_user_button'))
                            code = int(exit_code)
                            if code:
                                failure = read_json(run_dir / "worker_error.json")
                                detail = failure.get("error") or worker.get("error")
                                if isinstance(detail, str) and detail:
                                    raise SafetyTrip(msg('backend.ai_error_0', detail))
                                raise SafetyTrip(msg('backend.the_ai_process_exited_with_code_0', code))
                            reason = msg('backend.ai_run_complete')
                            break
                        if worker.get("error"):
                            raise SafetyTrip(msg('backend.ai_error_0', worker['error']))
                        heartbeat_now = time.monotonic()
                        if worker_exit_deadline is None:
                            beat = worker.get("monotonic", child_started)
                            if not isinstance(beat, (int, float)) or not 0 <= heartbeat_now - beat <= config.safety.worker_heartbeat_seconds:
                                raise SafetyTrip(msg('backend.the_ai_worker_heartbeat_is_stale_or'))
                            if heartbeat_now - child_started > config.safety.worker_heartbeat_seconds and not worker:
                                raise SafetyTrip(msg('backend.the_ai_worker_did_not_start_its'))
                            if worker.get('exit_ready') is True and 'monotonic' in worker:
                                worker_exit_deadline = beat + WORKER_EXIT_GRACE_SECONDS
                                clock_changed = True
                        elif heartbeat_now > worker_exit_deadline:
                            raise SafetyTrip(msg('backend.the_ai_worker_did_not_exit_after_cleanup'))
                        if session_clock.expired:
                            raise SafetyTrip(msg('backend.the_ai_exceeded_its_time_limit_and'))
                        state = "paused" if worker.get("user_paused") else "finishing" if worker.get("finishing") else "running"
                        if worker_exit_deadline is not None:
                            state = 'finishing'
                    if update or clock_changed or now - last_publish >= config.safety.poll_seconds:
                        status = {"time": timestamp(), "monotonic": time.monotonic(), "state": state,
                                  "reason": msg('backend.gpu_monitoring_healthy'), "mode": mode, "pid": os.getpid(),
                                  "child_pid": child_pid, "gpu": last_sample.to_dict(),
                                  "sensors": metadata, **timing_status(worker),
                                  "hotkeys": hotkeys.snapshot() if hotkeys else {},
                                  "elapsed_seconds": time.monotonic() - started,
                                  "limits": {"temperature_c": config.safety.temperature_limit_c,
                                             "vram_fraction": config.safety.vram_fraction_limit}}
                        atomic_json(current_file, status)
                        last_publish = now
                    if not pointer_published:
                        if publish_public_pointer:
                            atomic_json(ROOT / "runs" / "current.json", {"run_dir": str(run_dir)})
                        pointer_published = True
                    # A worker may only act while this independent lease is fresh.
                    if child is not None and update and worker_exit_deadline is None and not stopping():
                        lease.publish(last_sample.monotonic, last_sample.uuid)
                    if verbose and now - last_console >= 1:
                        s = last_sample
                        print(ui_text('backend.telemetry_summary', state, f'{s.temperature_c:.0f}', f'{config.safety.temperature_limit_c:.0f}', ','.join((f'{x:.0f}%' for x in s.fan_percent)), f'{s.power_w:.1f}', f'{s.memory_used / 2 ** 30:.2f}', f'{s.memory_total / 2 ** 30:.2f}'),
                              flush=True)
                        last_console = now
                if mode == "monitor" and now - started >= duration:
                    reason = msg('backend.read_only_monitoring_time_limit_reached')
                    break
                time.sleep(0.01)
    except Exception as exc:
        code, reason = 90, str(exc)
    except KeyboardInterrupt:
        code, reason = 90, msg('backend.stopped_by_the_user_ctrl_c')
    finally:
        # Terminate compute first; never wait for a potentially broken CUDA
        # checkpoint/save. Release keyboard regardless of logging failures.
        try:
            if child is not None and child.poll() is None:
                if job:
                    job.terminate()
                else:
                    child.kill()
                try:
                    child.wait(timeout=1)
                except subprocess.TimeoutExpired:
                    child.kill()
            release_fn()
        finally:
            if job:
                if hotkeys:
                    hotkeys.close()
                job.close()
            elif hotkeys:
                hotkeys.close()
            if child:
                child.close()
            telemetry.close()
            lease.close()
            if log:
                log.close()
        final = {"time": timestamp(), "monotonic": time.monotonic(), "state": "stopped" if code else "complete",
                 "mode": mode, "reason": reason, "code": code, "pid": os.getpid(),
                 "child_pid": child_pid, "gpu": last_sample.to_dict() if last_sample else None,
                 "sensors": metadata, **timing_status(read_json(heartbeat_file))}
        final['hotkeys'] = hotkeys.snapshot() if hotkeys else {}
        # Diagnostic writes must never prevent worker termination/input release.
        for target in (current_file, run_dir / "result.json"):
            try:
                atomic_json(target, final)
            except (OSError, RuntimeError):
                pass
        if not pointer_published and publish_public_pointer:
            try:
                atomic_json(ROOT / "runs" / "current.json", {"run_dir": str(run_dir)})
            except (OSError, RuntimeError):
                pass
        if verbose:
            print(ui_text('backend.run_summary', ui_text('backend.stopped_2') if code else ui_text('backend.complete_2'), localized(reason), run_dir), flush=True)
        guard_lock.close()
    return GuardResult(code, reason, run_dir, child_pid)
