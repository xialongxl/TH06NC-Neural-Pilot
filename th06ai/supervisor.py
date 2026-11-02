from __future__ import annotations

import json
import os
import subprocess
import sys
import time
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from .config import Config, ROOT
from .files import atomic_json, new_run, read_json, timestamp
from .gpu import AsyncTelemetry, NvmlReader, SafetyPolicy, SafetyTrip
from .lease import SharedLease


@dataclass
class GuardResult:
    code: int
    reason: str
    run_dir: Path
    child_pid: int | None


def run_guard(config: Config, mode: str, minutes: float | None = None,
              model: str | None = None, *, run_dir: Path | None = None,
              goal: str | None = None,
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
        raise ValueError('不支持的训练目标')
    if profile_kind is not None and profile_kind not in ('inference','latency'):
        raise ValueError('不支持的剖析类型')

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
    guarded_pause_seconds = 0.0
    last_guard_tick = time.monotonic()
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
    duration = 60 * (config.minutes if minutes is None else minutes)
    if not 0 < duration <= 3600:
        raise ValueError("监控/训练时长必须在 0 到 60 分钟之间")
    code, reason = 0, "监控结束"
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
        if mode not in ('train', 'play', 'evaluate') or stop_requested.is_set():
            return
        from .control import set_paused
        paused = (run_dir / 'PAUSE').exists() or worker.get('focus_pending') is True
        if paused and isinstance(worker.get('pid'), int):
            allow_foreground(worker['pid'])
        set_paused(run_dir, not paused)

    def stopping():
        return stop_requested.is_set() or bool(hotkeys and hotkeys.stop_requested.is_set())

    try:
        if real_events or hotkeys_factory is not None:
            from .hotkeys import GlobalHotkeys
            factory = hotkeys_factory or GlobalHotkeys
            hotkeys = factory(emergency_stop, toggle_pause, enable_pause=mode in ('train', 'play', 'evaluate'))
        with (run_dir / "gpu.jsonl").open("a", encoding="utf-8", buffering=1) as gpu_log:
            while True:
                now = time.monotonic()
                if stopping() or (run_dir / "STOP").exists() or hotkey_fn():
                    raise SafetyTrip("用户紧急停止（停止按钮 / F8）")
                pause_key = pause_hotkey_fn()
                if mode in ('train', 'play', 'evaluate') and pause_key and not pause_key_down:
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
                            raise SafetyTrip('用户紧急停止（F8）')
                        command = child_command or [sys.executable, "-m", "th06ai.worker", "--mode", mode,
                                                    "--run-dir", str(run_dir), "--minutes", str(duration / 60),
                                                    "--lease-name", lease.name]
                        if model and child_command is None:
                            command += ["--model", model]
                        if goal is not None and child_command is None:
                            command += ['--goal',goal]
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
                            raise SafetyTrip('用户紧急停止（F8）')
                        child_pid = child.pid
                        # Worker cannot touch game/CUDA until the shared lease is live.
                        child_started = time.monotonic()
                        state = "running"
                    elif healthy >= 3 and mode == "monitor":
                        state = "monitoring"
                    worker = read_json(heartbeat_file) if child else {}
                    if child is not None:
                        exit_code = child.poll()
                        if exit_code is not None:
                            if stopping() or (run_dir / 'STOP').exists():
                                raise SafetyTrip('用户紧急停止（停止按钮 / F8）')
                            code = int(exit_code)
                            if code:
                                failure = read_json(run_dir / "worker_error.json")
                                detail = failure.get("error") or worker.get("error")
                                if isinstance(detail, str) and detail:
                                    raise SafetyTrip(f"AI 报错：{detail}")
                                raise SafetyTrip(f"AI 进程异常退出（代码 {code}），详见 worker.log")
                            reason = "AI 运行完成"
                            break
                        beat = worker.get("monotonic", child_started)
                        if not isinstance(beat, (int, float)) or not 0 <= time.monotonic() - beat <= config.safety.worker_heartbeat_seconds:
                            raise SafetyTrip("AI 进程心跳超时或异常")
                        if time.monotonic() - child_started > config.safety.worker_heartbeat_seconds and not worker:
                            raise SafetyTrip("AI 进程未启动心跳")
                        if worker.get("error"):
                            raise SafetyTrip(f"AI 报错：{worker['error']}")
                        # Count pause time ourselves; never trust a worker-provided
                        # number to extend the independent execution deadline.
                        if mode in ('train', 'play', 'evaluate') and worker.get("user_paused") is True:
                            guarded_pause_seconds += max(0, now - last_guard_tick)
                        state = "paused" if worker.get("user_paused") else "finishing" if worker.get("finishing") else "running"
                    status = {"time": timestamp(), "monotonic": time.monotonic(), "state": state,
                              "reason": "显卡监控正常", "mode": mode, "pid": os.getpid(),
                              "child_pid": child_pid, "gpu": last_sample.to_dict(),
                              "sensors": metadata, "worker": worker,
                              "hotkeys": hotkeys.snapshot() if hotkeys else {},
                              "elapsed_seconds": time.monotonic() - started,
                              "limits": {"temperature_c": config.safety.temperature_limit_c,
                                         "vram_fraction": config.safety.vram_fraction_limit}}
                    if update or now - last_publish >= config.safety.poll_seconds:
                        atomic_json(current_file, status)
                        last_publish = now
                    if not pointer_published:
                        if publish_public_pointer:
                            atomic_json(ROOT / "runs" / "current.json", {"run_dir": str(run_dir)})
                        pointer_published = True
                    # A worker may only act while this independent lease is fresh.
                    if child is not None and update and not stopping():
                        lease.publish(last_sample.monotonic, last_sample.uuid)
                    if verbose and now - last_console >= 1:
                        s = last_sample
                        print(f"[{state}] {s.temperature_c:.0f}°C / 停机 {config.safety.temperature_limit_c:.0f}°C | "
                              f"风扇 {','.join(f'{x:.0f}%' for x in s.fan_percent)} | "
                              f"{s.power_w:.1f} W | 显存 {s.memory_used / 2**30:.2f}/{s.memory_total / 2**30:.2f} GB",
                              flush=True)
                        last_console = now
                if mode == "monitor" and now - started >= duration:
                    reason = "只读监控计时完成"
                    break
                # Worker controls the training budget; this hard cap also covers startup hangs.
                if child_started and now - child_started - guarded_pause_seconds > duration + 120:
                    raise SafetyTrip("AI 运行超过限定时间，强制停止")
                last_guard_tick = now
                time.sleep(0.01)
    except Exception as exc:
        code, reason = 90, str(exc)
    except KeyboardInterrupt:
        code, reason = 90, "用户停止（Ctrl+C）"
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
                 "sensors": metadata, "worker": read_json(heartbeat_file)}
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
            print(f"{'停止' if code else '完成'}：{reason}\n记录：{run_dir}", flush=True)
        guard_lock.close()
    return GuardResult(code, reason, run_dir, child_pid)
