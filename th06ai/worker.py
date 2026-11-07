from __future__ import annotations
from .i18n import msg

import argparse
import json
import os
import threading
from .input_timing import clock_ns
import time
import traceback
from pathlib import Path
from dataclasses import replace

from .config import PROFILE_KINDS, ROOT, SafetyConfig, load_config
from .files import atomic_json, read_json, timestamp
from .lease import SharedLease, LeaseBusy
from .control import FinishTraining, SamplingInterrupted


def record_failure(run_dir: Path, exc: BaseException):
    # Separate from the heartbeat payload: report the cause even if that
    # payload itself could not be serialized or its writer thread stopped.
    atomic_json(run_dir / "worker_error.json", {"time": timestamp(), "error": str(exc),
                "exception_type": type(exc).__name__, "pid": os.getpid()})


class Heartbeat:
    def __init__(self, run_dir: Path, shared_lease: SharedLease | None = None,
                 safety: SafetyConfig | None = None, *, track_session=False):
        self.run_dir = run_dir
        self.safety = safety or SafetyConfig()
        self.data = {"phase": msg('backend.loading_training_modules'), "steps": 0, "episodes": 0, "pid": os.getpid()}
        self.lock = threading.Lock()
        self.error = None
        self.done = threading.Event()
        self._wake = threading.Event()
        self._track_session = track_session
        self._session_time = {}
        self.last_lease_check = 0.0
        self.last_valid_permit = None
        self.lease_identity = None
        self.lease_busy_reads = 0
        self.shared_lease = shared_lease
        self.thread = threading.Thread(target=self._run, daemon=True, name="worker-heartbeat")
        self.thread.start()

    def update(self, **values):
        with self.lock:
            self.data.update(values)
        if self._track_session and ('clock_started' in values or 'clock_paused' in values
                                    or 'overtime_stage' in values):
            self._wake.set()

    def session_time(self):
        return self._session_time

    def _run(self):
        while not self.done.is_set():
            self._wake.clear()
            try:
                with self.lock:
                    snapshot = {**self.data, "monotonic": time.monotonic(), "time": timestamp()}
                atomic_json(self.run_dir / "worker.json", snapshot)
                if self._track_session:
                    status = read_json(self.run_dir / 'status.json')
                    timing = status.get('session_time') if isinstance(status, dict) else None
                    if isinstance(timing, dict):
                        self._session_time = timing
            except Exception as exc:
                self.error = exc
                return
            self._wake.wait(.20)

    def lease(self):
        now = time.monotonic()
        if self.error:
            raise RuntimeError(msg('backend.could_not_write_the_ai_heartbeat_0', self.error))
        if (self.run_dir / "STOP").exists():
            raise RuntimeError(msg('backend.stop_requested_by_the_user'))
        if 0 <= now - self.last_lease_check < .025:
            return
        read_state = "valid"
        try:
            permit = self.shared_lease.read_checked() if self.shared_lease else {}
        except LeaseBusy:
            # Reuse only the last fully checked permit, with its ORIGINAL
            # telemetry timestamp. A busy read never renews authorization.
            permit = self.last_valid_permit or {}
            self.lease_busy_reads += 1
            read_state = "concurrent_publish"
        # A publish may have happened after the clock read at function entry.
        # Taking 'now' after the snapshot prevents a false negative age.
        now = time.monotonic()
        age = now - permit["monotonic"] if permit else None
        identity = (permit.get("supervisor_pid"), permit.get("uuid")) if permit else None
        reason = ("missing" if not permit else "time_invalid" if not age >= 0 else
                  "expired" if age > self.safety.telemetry_timeout_seconds else
                  "identity_changed" if self.lease_identity is not None and identity != self.lease_identity else None)
        diagnostics = {"read_state": read_state, "sample_age_seconds": age,
                       "busy_reads": self.lease_busy_reads,
                       "timeout_seconds": self.safety.telemetry_timeout_seconds,
                       "supervisor_pid": permit.get("supervisor_pid") if permit else None}
        self.update(lease_diagnostics={**diagnostics, "error": reason})
        if reason:
            raise RuntimeError(msg('backend.watchdog_authorization_failed', {
                "missing": msg('backend.complete_authorization_was_not_received'), "time_invalid": msg('backend.invalid_authorization_timestamp'),
                "expired": msg('backend.telemetry_exceeded_the_configured_timeout'), "identity_changed": msg('backend.watchdog_process_or_gpu_identity_changed')}[reason]))
        self.last_valid_permit = permit
        self.lease_identity = identity
        self.last_lease_check = now

    def close(self, *, exit_ready=False):
        self.done.set()
        self._wake.set()
        self.thread.join(timeout=.5)
        if not self.error and not self.thread.is_alive():
            with self.lock:
                snapshot = {**self.data, "monotonic": time.monotonic(), "time": timestamp()}
            if exit_ready:
                snapshot['exit_ready'] = True
            atomic_json(self.run_dir / "worker.json", snapshot)


def measure_movement(env, focused=False):
    from .motion import held_speed, measured_speed
    first = env.reader.read()
    direction = 'right' if first.x < first.bounds[0]+first.bounds[2]/2 else 'left'
    keys = {direction, 'shift'} if focused else {direction}
    samples, frames = [], []
    previous = None
    deadline = time.monotonic()+1.5
    env.keys.set(keys)
    try:
        while len(samples) < 9 and time.monotonic() < deadline:
            env.lease()
            current = env.reader.read()
            if current.paused or current.state == 2 or not current.stage_active:
                raise RuntimeError(msg('backend.movement_calibration_requires_an_active_stage'))
            frames.append({'tick':current.tick, 'x':current.x, 'y':current.y,
                           'input_bits':current.input_bits})
            if previous is not None:
                value = held_speed(previous, current, keys)
                if value is not None:
                    samples.append(value)
            previous = current
            env._wait(.002)
    finally:
        env.keys.release()
    return measured_speed(samples), {'kind':'focus' if focused else 'normal',
            'keys':sorted(keys), 'samples_px_per_tick':samples, 'frames':frames,
            'measurement_version':2}


def calibrate(env, run_dir: Path, heartbeat: Heartbeat):
    import numpy as np
    from .native import SHA256
    env.ensure_game_started()
    heartbeat.update(phase=msg('backend.calibrating_live_game_state'), message=msg('backend.keep_the_game_focused_while_coordinates_and'))
    first = env.reader.read()
    if first.paused or first.state == 2:
        raise RuntimeError(msg('backend.calibration_requires_a_playable_stage_re_enter'))
    # Verify that coordinate changes follow real key input, including focus.
    fast, fast_probe = measure_movement(env)
    slow, slow_probe = measure_movement(env, focused=True)
    probes = [fast_probe, slow_probe]
    atomic_json(run_dir / "movement_probes.json", {"probes": probes})
    if not .1 < slow < fast < 20:
        raise RuntimeError(msg('backend.movement_validation_failed_normal_displacement_per_frame', f'{fast:.2f}', f'{slow:.2f}'))
    seen = 0
    samples = []
    deadline = time.monotonic() + 18
    while time.monotonic() < deadline:
        heartbeat.lease()
        f = env.reader.read()
        seen = max(seen, len(f.bullets))
        samples.append({"tick": f.tick, "x": f.x, "y": f.y, "state": f.state,
                        "bullets": f.bullets.tolist()})
        heartbeat.update(bullets=len(f.bullets), message=msg('backend.detected_0_visible_bullets_calibrating', seen))
        if seen >= 3 and len(samples) >= 30:
            break
        if f.state == 2:
            break
        env._wait(.05)
    atomic_json(run_dir / "calibration_samples.json", {"samples": samples})
    env.pause()
    if seen < 3:
        raise RuntimeError(msg('backend.calibration_failed_because_too_few_bullets_were'))
    atomic_json(run_dir / "calibration_progress.json", {
        "position_control_verified": True, "focus_speed_verified": True,
        "visible_bullets": seen, "fast_px_per_tick": fast, "slow_px_per_tick": slow,
        "game_sha256": SHA256, "restart_verified": False})
    heartbeat.update(phase=msg('backend.calibrating_the_r_restart_shortcut'), message=msg('backend.verifying_the_timer_reset_and_subsequent_frame'))
    env._restart()
    env.pause()
    report = {"game_sha256": SHA256, "time": timestamp(), "position_control_verified": True,
              "focus_speed_verified": True, "restart_verified": True,
              "visible_bullets": seen, "fast_px_per_tick": fast,
              "slow_px_per_tick": slow, "source": str(run_dir), "format": 1,
              "movement_measurement_version": 2}
    atomic_json(ROOT / "native_calibration.json", report)
    atomic_json(run_dir / "calibration.json", report)
    heartbeat.update(phase=msg('backend.calibration_passed'), message=msg('backend.game_state_and_key_input_verified_training'))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["check", "calibrate", "train", "play", "evaluate", "profile"], required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--minutes", type=float, required=True)
    parser.add_argument("--model")
    parser.add_argument('--goal',choices=['survival','nmnb'])
    parser.add_argument('--game-setup')
    parser.add_argument('--profile-kind',choices=PROFILE_KINDS,default='inference')
    parser.add_argument('--profile-wait-combat',action='store_true')
    parser.add_argument("--lease-name", required=True)
    args = parser.parse_args()
    run_dir = args.run_dir.resolve()
    if not run_dir.is_relative_to((ROOT / "runs").resolve()):
        raise RuntimeError(msg('backend.the_run_directory_must_be_inside_this'))
    config = load_config()
    config = replace(config,goal=args.goal or ('nmnb' if args.mode=='evaluate' else config.goal))
    if args.game_setup is not None:
        from .game_settings import parse_setup
        config = replace(config, game_setup=parse_setup(json.loads(args.game_setup)))
    lease = SharedLease(args.lease_name)
    game_session = args.mode in ('train', 'play', 'evaluate')
    heartbeat = Heartbeat(run_dir, lease, config.safety, track_session=game_session)
    env = None
    completed = False
    try:
        # Wait for the supervisor to attach the kill-on-close Job Object.
        deadline = time.monotonic() + 3
        while not (initial_permit := lease.read()):
            if time.monotonic() > deadline:
                raise RuntimeError(msg('backend.the_independent_watchdog_did_not_grant_authorization'))
            time.sleep(.02)
        heartbeat.last_valid_permit = initial_permit
        heartbeat.lease()
        if args.mode == "check":
            from .backend import check_cuda
            check_cuda(heartbeat.lease, heartbeat.update)
            completed = True
            return 0
        import torch
        torch.set_num_threads(1)
        if config.device=='cuda':
            if not torch.cuda.is_available():raise RuntimeError(msg('backend.cuda_is_disabled_the_training_device_was'))
            torch.cuda.set_per_process_memory_fraction(.15)
        if args.mode=='profile':
            if args.profile_kind=='latency':
                from .profile_latency import profile_latency
                profile_latency(config,run_dir,args.model,heartbeat,wait_for_combat=args.profile_wait_combat)
            elif args.profile_kind=='readback':
                from .profile_inference import profile_readback
                profile_readback(config, run_dir, args.model, heartbeat.update, heartbeat.lease)
            else:
                from .profile_inference import profile_inference
                profile_inference(config,run_dir,args.model,heartbeat.update,heartbeat.lease)
            completed = True
            return 0
        from .native import SHA256
        if args.mode != "calibrate":
            calibration = read_json(ROOT / "native_calibration.json")
            if calibration.get("game_sha256") != SHA256 or calibration.get("format") != 1:
                raise RuntimeError(msg('backend.calibrate_the_current_game_build_before_continuing'))
            resources = read_json(ROOT / "native_resources_calibration.json")
            if resources.get("game_sha256") != SHA256 or not resources.get("operator_confirmed"):
                raise RuntimeError(msg('backend.verify_the_displayed_power_bombs_and_lives'))
        from .environment import TouhouEnv
        env = TouhouEnv(config, heartbeat.lease, heartbeat.update)
        env.run_dir = run_dir
        if game_session:
            env.enable_diagnostics(run_dir,args.mode)
        heartbeat.update(game_pid=env.reader.window.pid)
        if args.mode == "calibrate":
            calibrate(env, run_dir, heartbeat)
        else:
            from .learning import train, evaluate, load_model
            if args.mode == "train":
                train(config, env, run_dir, args.minutes, heartbeat.update, heartbeat.lease, args.model,
                      session_time=heartbeat.session_time)
            else:
                from .control import TrainingControl
                env.training_control = TrainingControl(run_dir, heartbeat.lease, heartbeat.update,
                                                       session_time=heartbeat.session_time)
                model = load_model(Path(args.model or ROOT / "models" / "latest.zip"), env, config.device)
                from .inference import PolicyRunner
                model.decision_runner = PolicyRunner(model.policy, config.inference_backend,
                                                     readback=config.inference_readback)
                env.pause_for_user()
                model.decision_runner.warmup(lease=heartbeat.lease)
                if args.mode == "evaluate":
                    report = evaluate(model, env, run_dir, heartbeat.update, heartbeat.lease)
                    heartbeat.update(phase=msg('backend.evaluation_complete'), message=msg('backend.mean_survival_0_seconds', f"{report['mean_survival_seconds']:.1f}"))
                else:
                    obs, _ = env.reset(options={"restart": False})
                    env.training_control.start()
                    while not env.training_control.target_reached:
                        if env.training_control.gate(env):
                            obs = env.builder.build(env.current)
                        policy_started = env.begin_policy_timing()
                        decision = model.decision_runner.predict(obs,deterministic=True,lease=heartbeat.lease)
                        action = decision.actions[0]
                        env.record_policy_timing(policy_started, clock_ns())
                        try:
                            env.refresh_decision()
                            env.record_decision(decision)
                            obs, _, died, timeout, _ = env.step(int(action))
                        except SamplingInterrupted as interruption:
                            if interruption.episode_end is not None:
                                obs, _ = env.reset()
                                continue
                            env.training_control.gate(env)
                            obs = env.builder.build(env.current)
                            continue
                        heartbeat.update(phase=msg('backend.live_policy_playback'))
                        if died or timeout:
                            obs, _ = env.reset()
        heartbeat.update(phase=msg('backend.complete_2'), message=msg('backend.run_complete_game_keys_released'))
        completed = True
        return 0
    except FinishTraining:
        heartbeat.update(phase=msg('backend.complete_2'), message=msg('backend.session_ended_releasing_game_keys'))
        completed = True
        return 0
    except BaseException as exc:
        heartbeat.update(error=str(exc), phase=msg('backend.stopped_2'), message=str(exc),
                         state_reader=getattr(getattr(env, "reader", None), "snapshot_diagnostics", {}))
        traceback.print_exc()
        try:
            record_failure(run_dir, exc)
        except Exception:
            traceback.print_exc()
        time.sleep(.25)
        return 1
    finally:
        if env is not None:
            try:
                env.close()
            except Exception as error:
                from .win32 import release_all
                release_all()
                heartbeat.update(cleanup_error=str(error))
        try:
            lease.close()
        except BaseException:
            heartbeat.close()
            raise
        # Only the final record after saving and cleanup can start the exit deadline.
        heartbeat.close(exit_ready=completed)


if __name__ == "__main__":
    raise SystemExit(main())
