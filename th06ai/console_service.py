"""UI-independent session controls. Reading state never starts game/GPU work."""
from __future__ import annotations
from .i18n import msg

from dataclasses import asdict, replace
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import threading
import time

from .config import Config, DEFAULT_CONSOLE_PORT, ROOT
from .control import set_paused
from .files import atomic_json, new_run, read_json
from .model_store import inspect_checkpoint, CheckpointCatalog
from .observations import OBSERVATION_VERSION
from .planner_spec import PLANNER_VERSION
from .rewards import REWARD_VERSION
from .game_settings import GameSetup, parse_setup, observed_settings

ACTIVE_STATES = {"starting", "running", "paused", "finishing", "monitoring"}
MODES = {"resume", "train", "play", "evaluate", "check", "monitor", "calibrate"}


class ConsoleError(Exception):
    def __init__(self, message, status_code=409):
        super().__init__(message)
        self.status_code = status_code


def build_command(mode, minutes, model=None, goal=None, game_setup=None):
    if mode not in MODES:
        raise ValueError(msg('backend.unsupported_operation'))
    if not math.isfinite(float(minutes)) or not 0 < float(minutes) <= 60:
        raise ValueError(msg('backend.the_session_duration_must_be_greater_than_2'))
    if mode in {"resume", "play", "evaluate"} and model is None:
        raise ValueError(msg('backend.select_a_model_first'))
    command = [sys.executable, "-m", "th06ai", "train" if mode == "resume" else mode,
               "--minutes", str(minutes)]
    if mode in {"resume", "play", "evaluate"}:
        command += ["--model", str(model)]
    if goal is not None:
        if goal not in ('survival','nmnb'):raise ValueError(msg('backend.unsupported_training_goal'))
        command += ['--goal',goal]
    setup = parse_setup(game_setup)
    if setup is not None:
        command += ['--game-setup', json.dumps(setup.to_dict(), separators=(',', ':'))]
    return command


def current_session(root=ROOT, *, read=None):
    read = read or read_json
    pointer = read(root / "runs" / "current.json")
    if not isinstance(pointer.get("run_dir"), str):
        return None, {}
    path = Path(pointer["run_dir"]).resolve()
    if not path.is_relative_to((root / "runs").resolve()):
        return None, {}
    return path, read(path / "status.json")


def session_active(status):
    stamp = status.get("monotonic")
    return (status.get("state") in ACTIVE_STATES and isinstance(stamp, (int, float))
            and 0 <= time.monotonic()-stamp < 2)


def tail_text(path, *, limit=32768):
    try:
        with path.open("rb") as stream:
            size = stream.seek(0, 2)
            stream.seek(max(0, size-limit))
            data = stream.read(limit)
        if size > limit:
            data = data.partition(b"\n")[2]
        return data.decode("utf-8", errors="replace")
    except OSError:
        return ""


def json_safe(value):
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    return value


class ConsoleService:
    def __init__(self, config: Config, *, root=ROOT, preview=False, popen=subprocess.Popen, focus_fn=None):
        self.config, self.root, self.preview = config, Path(root).resolve(), preview
        self._popen, self.process = popen, None
        self._focus_fn = focus_fn
        self.last_focus = {}
        self.last_game_probe = None
        self.shutdown = {"state": "idle", "message": ""}
        self._closing = False
        self._shutdown_run = None
        from .acceleration import backend_status
        self._native_backend = backend_status()
        self.pending_run, self.pending_mode = None, None
        # The game and catalog locks may acquire the session lock for publication.
        self._lock = threading.RLock()
        self._game_lock = threading.Lock()
        self._cache_lock = threading.Lock()
        self._catalog, self._catalog_at = [], -math.inf
        self._catalog_cache = CheckpointCatalog(self.root / 'models')
        self._catalog_scan_lock = threading.Lock()
        self._catalog_refreshing = False
        self._catalog_thread = None
        self._catalog_revision = None
        self._catalog_error = None
        self._gpu_histories = {}
        self._json_cache, self._text_cache = {}, {}
        self._model_view = []
        self._safety = asdict(config.safety)
        saved = read_json(self.root / "ui_state.json").get("selected_model")
        self.selected_path = Path(saved).resolve() if isinstance(saved, str) else None
        self.refresh_models(force=True)

    @staticmethod
    def model_id(path):
        return hashlib.sha256(str(path).encode("utf-8")).hexdigest()[:20]

    def refresh_models(self, *, force=False):
        if force:
            return self._scan_models()
        revision = self._file_stamp(self.root / 'models/catalog_revision.json')
        with self._lock:
            if not self._catalog_refreshing and (revision != self._catalog_revision or time.monotonic()-self._catalog_at > 8):
                self._catalog_refreshing = True
                self._catalog_thread = threading.Thread(target=self._background_scan, name='model-catalog', daemon=True)
                self._catalog_thread.start()
            return self._catalog

    def _scan_models(self):
        with self._catalog_scan_lock:
            return self._publish_scan()

    def _publish_scan(self):
        revision = self._file_stamp(self.root / 'models/catalog_revision.json')
        entries = self._catalog_cache.scan()
        rows = self._model_rows(entries)
        view = self._model_view if rows == self._model_view else rows
        paths = {entry.path for entry in entries}
        fallback = next((entry.path for entry in entries if entry.usable), None)
        with self._lock:
            self._catalog = entries
            self._model_view = view
            self._catalog_at = time.monotonic()
            self._catalog_revision = revision
            self._catalog_error = None
            if self.selected_path not in paths:
                self.selected_path = fallback
            return self._catalog

    def _background_scan(self):
        try:
            self._scan_models()
        except Exception as error:
            revision = self._file_stamp(self.root / 'models/catalog_revision.json')
            with self._lock:
                self._catalog_error = str(error)
                self._catalog_at = time.monotonic()
                self._catalog_revision = revision
        finally:
            with self._lock:
                self._catalog_refreshing = False

    def models(self):
        self.refresh_models()
        with self._lock:
            return self._model_view

    def _model_rows(self, catalog):
        rows = [{"id": self.model_id(c.path), "name": msg('backend.latest_checkpoint') if c.path == self.root / "models/latest.zip"
                 else msg('backend.checkpoint_0_steps', f"{c.metadata.get('steps', 0):,}"),
                 "path": str(c.path.relative_to(self.root)), "steps": c.metadata.get("steps", 0),
                 "updates": c.metadata.get("gradient_updates", 0), "saved_at": c.metadata.get("time"),
                 "usable": c.usable, "compatibility": c.compatibility,
                 "format": c.metadata.get("format"), "planner_version": c.metadata.get("planner_version")}
                for c in catalog]
        return json_safe(rows)

    @staticmethod
    def _file_stamp(path):
        try:
            stat = path.stat()
            return stat.st_mtime_ns, stat.st_size, stat.st_ino
        except OSError:
            return None

    def _cached_file(self, path, cache, key, capacity, load):
        stamp = self._file_stamp(path)
        with self._cache_lock:
            cached = cache.get(key)
            if cached is not None and cached[0] == stamp:
                return cached[1]
        value = load()
        if stamp is not None and value:
            with self._cache_lock:
                current = cache.get(key)
                if current is not None and current[0] == stamp:
                    return current[1]
                if current is cached:
                    cache[key] = stamp, value
                    if len(cache) > capacity:
                        cache.pop(next(iter(cache)))
        return value

    def _read_json(self, path):
        return self._cached_file(path, self._json_cache, path, 64, lambda: json_safe(read_json(path)))

    def _tail(self, path, limit=32768):
        return self._cached_file(path, self._text_cache, (path, limit), 32, lambda: tail_text(path, limit=limit))

    def select_model(self, model_id):
        catalog = self.refresh_models(force=True)
        entry = next((c for c in catalog if self.model_id(c.path) == model_id), None)
        if entry is None:
            raise ConsoleError(msg('backend.model_not_found_refresh_the_model_library'), 404)
        if not entry.usable:
            raise ConsoleError(entry.compatibility)
        with self._catalog_scan_lock:
            checked = inspect_checkpoint(entry.path)
            if checked.path != entry.path or not checked.path.is_relative_to((self.root / 'models').resolve()) or not checked.usable:
                raise ConsoleError(msg('backend.the_model_path_or_metadata_changed_refresh'))
            if not self.preview:
                atomic_json(self.root / "ui_state.json", {"selected_model": str(entry.path)})
            with self._lock:
                self.selected_path = entry.path
        return model_id

    def readiness(self):
        backend = self._read_json(self.root / "training_backend.json")
        checks = {"gpu": bool(backend.get("forward_check")),
                  "calibration": bool(self._read_json(self.root / "native_calibration.json").get("restart_verified")),
                  "resources": bool(self._read_json(self.root / "native_resources_calibration.json").get("operator_confirmed"))}
        return {"ready": all(checks.values()), "checks": checks, "backend": backend}

    def _gpu_history(self, path, gpu):
        with self._cache_lock:
            history = self._gpu_histories.get(path)
        if history is None:
            history = []
            for line in tail_text(path / "gpu.jsonl", limit=180000).splitlines() if path else []:
                try:
                    point = self._gpu_point(json.loads(line))
                except (ValueError, AttributeError):
                    continue
                if point is not None:
                    history.append(point)
            history = history[-180:]
        point = self._gpu_point(gpu)
        with self._cache_lock:
            history = self._gpu_histories.get(path, history)
            if point is not None:
                stamp = point['stamp']
                last = history[-1]['stamp'] if history else None
                # A snapshot may finish after a newer one has already appended.
                if not history or (stamp is not None and (last is None or stamp > last)):
                    history = [*history, point][-180:]
            self._gpu_histories[path] = history
            if len(self._gpu_histories) > 4:
                self._gpu_histories.pop(next(iter(self._gpu_histories)))
            return history

    @staticmethod
    def _gpu_point(gpu):
        temperature = gpu.get('temperature_c')
        if not isinstance(temperature, (int, float)) or not math.isfinite(temperature):
            return None
        stamp = gpu.get('monotonic')
        return json_safe({'stamp': stamp if isinstance(stamp, (int, float)) else None,
                          'temperature_c': temperature, 'power_w': gpu.get('power_w'),
                          'utilization': gpu.get('utilization')})

    def snapshot(self):
        self.refresh_models()
        with self._lock:
            launch = self.process, self.pending_run, self.pending_mode
            config, native_backend = self.config, self._native_backend
            models, selected_path = self._model_view, self.selected_path
            shutdown, probe, focus = dict(self.shutdown), self.last_game_probe, self.last_focus
        path, status = self._session(launch)
        active = session_active(status)
        process = launch[0]
        launching = bool(process and process.poll() is None)
        gpu = status.get("gpu") or {}
        stamp = gpu.get("monotonic")
        gpu_age = max(0, time.monotonic()-stamp) if isinstance(stamp, (int, float)) and math.isfinite(stamp) else None
        return {"preview": self.preview, "session": status, "active": active, "shutdown": shutdown,
                'game_probe': probe,
                'acceleration': {**native_backend, 'collision': config.planning.collision_backend,
                                 'input': config.input_backend},
                "busy": active or launching, "run_dir": str(path) if path else None,
                'default_game_setup': (config.game_setup or GameSetup()).to_dict(),
                "hotkeys": status.get('hotkeys') or {}, "window_focus": focus,
                "pause_requested": bool(path and (path / "PAUSE").exists()),
                "finish_requested": active and (status.get('state') == 'finishing'
                        or bool(path and (path / "FINISH").exists())),
                "selected_model_id": self.model_id(selected_path) if selected_path else None,
                "models": models, "readiness": self.readiness(),
                "safety": self._safety, "default_minutes": config.minutes,
                "versions": {"observation": OBSERVATION_VERSION, "reward": REWARD_VERSION, "planner": PLANNER_VERSION},
                "gpu_history": self._gpu_history(path, gpu), "gpu_age_seconds": gpu_age,
                "gpu_live": active and gpu_age is not None and gpu_age < config.safety.telemetry_timeout_seconds
                    and isinstance(gpu.get("temperature_c"), (int, float)) and math.isfinite(gpu["temperature_c"])}

    def battle_view_session(self):
        with self._lock:
            if self.preview or self.shutdown.get('state') in ('waiting', 'stopping', 'closed'):
                return None, False
            launch = self.process, self.pending_run, self.pending_mode
        path, status = self._session(launch)
        if path is None or status.get('mode') not in ('train', 'play', 'evaluate'):
            return None, False
        live = (status.get('state') == 'running' and session_active(status)
                and not (path / 'PAUSE').exists() and not (path / 'STOP').exists())
        with self._lock:
            if not self._closing and self.pending_run == launch[1]:
                return path, live
        return None, False

    def _session(self, launch=None):
        if launch is None:
            with self._lock:
                launch = self.process, self.pending_run, self.pending_mode
        process, pending_run, pending_mode = launch
        path, status = current_session(self.root, read=self._read_json)
        if pending_run is not None and process is not None:
            code = process.poll()
            if path != pending_run:
                pending = self._read_json(pending_run / "status.json")
                return pending_run, pending or {"state": "starting" if code is None else "stopped",
                    "mode": pending_mode,
                    "reason": msg('backend.starting_the_independent_watchdog') if code is None else msg('backend.the_launcher_exited_with_code_0_check', code)}
            if code is None and not status:
                return pending_run, {"state": "starting", "mode": pending_mode,
                                     "reason": msg('backend.starting_the_independent_watchdog')}
        return path, status

    def _writable(self):
        if self.preview:
            raise ConsoleError(msg('backend.read_only_preview_cannot_start_training_or'), 403)

    def _activate_game(self):
        try:
            if self._focus_fn is not None:
                result = self._focus_fn(self.config.game_path)
            else:
                from .focus import activate_game
                result = activate_game(self.config.game_path)
        except Exception as error:
            result = {'activated': False, 'message': msg('backend.could_not_focus_the_game_window_0', error)}
        with self._lock:
            self.last_focus = result
        return result

    def focus_game(self):
        self._writable()
        with self._game_lock:
            with self._lock:
                path, status = self._session()
                if session_active(status):
                    return self._request_focus(path)
            result = self._activate_game()
            if not result['activated']:
                raise ConsoleError(result['message'])
            return result

    def probe_game(self):
        """Explicit one-shot inspection: no focus, input, worker or GPU."""
        from datetime import datetime
        from .game_phase import PHASE_NAMES
        from .stage import stage_label
        if self.preview:
            raise ConsoleError(msg('backend.read_only_preview_does_not_connect_to'), 403)
        with self._game_lock:
            with self._lock:
                _, status = self._session()
                if session_active(status) or self.process and self.process.poll() is None:
                    raise ConsoleError(msg('backend.a_session_is_running_check_the_worker'), 409)
                config = self.config
            reader = None
            try:
                from .native import GameReader
                reader = GameReader(config.game_path)
                if config.capture_backend != 'auto':
                    reader.process.configure_reads(config.capture_backend)
                f = reader.read()
                probe = {'available':True,'phase':f.phase,'label':PHASE_NAMES.get(f.phase,msg('backend.waiting_for_detection')),
                    'game_settings':observed_settings(f),
                    'stage_id':getattr(f,'stage_id',None),'stage_label':stage_label(getattr(f,'stage_id',None)),
                    'paused':f.paused,'tick':f.tick,'menu_selection':f.menu_selection,'menu_ready':f.menu_ready,
                    'evidence':f.phase_evidence,'player_shots':f.player_shots,'visible_lasers':len(f.lasers),
                    'observed_at':datetime.now().astimezone().isoformat(),'message':msg('backend.read_only_probe_complete')}
            except (OSError,RuntimeError,ValueError) as error:
                probe = {'available':False,'label':msg('backend.game_not_connected'),'message':str(error),
                    'observed_at':datetime.now().astimezone().isoformat()}
            finally:
                if reader is not None:
                    reader.close()
            with self._lock:
                self.last_game_probe = probe
            return probe

    def set_acceleration(self, collision, input_backend):
        from .acceleration import backend_status
        self._writable()
        with self._game_lock:
            with self._lock:
                _, status = self._session()
                if session_active(status) or self.process and self.process.poll() is None:
                    raise ConsoleError(msg('backend.end_the_current_session_before_switching_compute'), 409)
                current = self.config
            available = backend_status()
            if 'rust' in (collision, input_backend) and not available['available']:
                raise ConsoleError(msg('backend.the_rust_extension_has_not_been_built'), 422)
            if collision not in ('python', 'rust', 'auto') or input_backend not in ('python', 'rust', 'auto'):
                raise ConsoleError(msg('backend.unknown_compute_backend'), 422)
            config = replace(current, input_backend=input_backend,
                             planning=replace(current.planning, collision_backend=collision))
            atomic_json(self.root / 'config.json', asdict(config))
            with self._lock:
                self.config = config
                self._native_backend = available
            return {'message': msg('backend.backend_settings_saved_for_the_next_session')}

    def _check_start_allowed(self):
        if self._closing:
            raise ConsoleError(msg('backend.the_service_is_shutting_down_wait_for_2'))
        _, status = current_session(self.root)
        if session_active(status) or (self.process and self.process.poll() is None):
            raise ConsoleError(msg('backend.a_session_is_already_running_pause_resume'))

    def _checkpoint_stamp(self, path):
        return self._file_stamp(path), self._file_stamp(path.with_suffix('.json'))

    def start(self, mode, minutes, model_id=None, goal=None, game_setup=None):
        self._writable()
        with self._lock:
            self._check_start_allowed()
        model = None
        model_stamp = None
        if mode in {'resume', 'play', 'evaluate'}:
            catalog = self.refresh_models(force=True)
            entry = next((c for c in catalog if self.model_id(c.path) == model_id), None)
            if entry is None:
                raise ConsoleError(msg('backend.select_a_compatible_model_first'), 422)
            model = entry.path
            model_stamp = self._checkpoint_stamp(model)
            try:
                checked = inspect_checkpoint(model, verify=True)
            except OSError as error:
                raise ConsoleError(msg('backend.the_model_path_or_metadata_changed_refresh')) from error
            if not checked.path.is_relative_to((self.root / 'models').resolve()):
                raise ConsoleError(msg('backend.the_model_path_is_outside_the_project'))
            if checked.path != model:
                raise ConsoleError(msg('backend.the_model_path_or_metadata_changed_refresh'))
            if not checked.usable:
                raise ConsoleError(checked.compatibility)
        if mode in {'resume', 'train', 'play', 'evaluate'} and not self.readiness()['ready']:
            raise ConsoleError(msg('backend.complete_the_checks_on_the_safety_diagnostics'))
        try:
            setup = parse_setup(game_setup)
        except ValueError as error:
            raise ConsoleError(str(error), 422) from error
        if setup is not None and mode not in ('resume', 'train', 'play', 'evaluate'):
            raise ConsoleError(msg('backend.game_settings_apply_to_training_playback_and'), 422)
        command = build_command(mode, minutes, model, goal, setup)
        with self._game_lock:
            with self._lock:
                self._check_start_allowed()
                if model is not None and self._checkpoint_stamp(model) != model_stamp:
                    raise ConsoleError(msg('backend.the_model_path_or_metadata_changed_refresh'))
                run_path = self._launch_session(command, 'train' if mode == 'resume' else mode)
            focus = None
            if mode in ('resume', 'train', 'play', 'evaluate', 'calibrate'):
                with self._lock:
                    can_focus = not self._closing and not (run_path / 'STOP').exists()
                if can_focus:
                    # Activate on the explicit start while the worker loads.
                    focus = self._activate_game()
        return {"message": msg('backend.request_accepted_the_session_will_start_after'), "mode": mode, 'foreground': focus}

    def _launch_session(self, command, mode):
        """Publish the new run and launcher together under the session lock."""
        run_path = new_run(self.root, mode)
        command += ['--run-dir', str(run_path)]
        log = self.root / 'runs/launcher.log'
        try:
            with log.open('a', encoding='utf-8') as stream:
                process = self._popen(command, cwd=self.root,
                    env=dict(os.environ, PYTHONUTF8='1', PYTHONUNBUFFERED='1'), stdout=stream,
                    stderr=subprocess.STDOUT, creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
        except OSError as error:
            raise ConsoleError(msg('backend.launch_failed_0', error), 500) from error
        self.process, self.pending_run, self.pending_mode = process, run_path, mode
        return run_path

    @staticmethod
    def _request_focus(path):
        (path / 'FOCUS').touch()
        return {'message': msg('backend.game_focus_requested_training_and_pause_state')}

    def control(self, action, *, paused=None):
        with self._lock:
            self._writable()
            path, status = self._session()
            # STOP remains available for a known session even if UI telemetry
            # is stale; the independent guard and F8 do not rely on this API.
            if not path or (not session_active(status) and not (action == "stop" and status.get("state") in ACTIVE_STATES)):
                raise ConsoleError(msg('backend.there_is_no_active_session_to_control'))
            if action == "stop":
                (path / "STOP").touch()
                return {"message": msg('backend.emergency_stop_requested_the_last_saved_checkpoint')}
            if action == 'focus':
                return self._request_focus(path)
            if action == "pause":
                if status.get('mode') not in ('train', 'play', 'evaluate'):
                    raise ConsoleError(msg('backend.this_session_does_not_support_pausing_the'))
                set_paused(path, bool(paused))
                return {"message": msg('backend.pausing_with_progress_retained') if paused else msg('backend.resuming_the_current_round')}
            if action == "finish":
                if status.get('mode') != 'train':
                    raise ConsoleError(msg('backend.the_current_session_is_not_a_training'))
                (path / "FINISH").touch()
                return {"message": msg('backend.waiting_for_a_safe_point_to_save')}
            raise ConsoleError(msg('backend.unknown_control_action'), 422)

    def request_shutdown(self, *, port=DEFAULT_CONSOLE_PORT):
        with self._lock:
            if self._closing:
                return dict(self.shutdown)
            atomic_json(self.root / f'runs/console-stop-{port}.json', {'requested_ns': time.time_ns()})
            path, status = self._session()
            pending = bool(self.process and self.process.poll() is None)
            self._shutdown_run = None
            if not self.preview and path and (pending or status.get('state') in ACTIVE_STATES):
                mode = status.get('mode') or self.pending_mode
                marker = 'FINISH' if mode in ('train', 'play', 'evaluate') else 'STOP'
                (path / marker).touch()
                self._shutdown_run = path
            self._closing = True
            self.shutdown = {"state": "waiting", "message": msg('backend.finishing_the_session_and_shutting_down_training')}
            return dict(self.shutdown)

    def shutdown_ready(self):
        if self.preview:
            return True
        with self._lock:
            process, path = self.process, self._shutdown_run
        if process and process.poll() is None:
            return False
        if path is None:
            return True
        status = self._read_json(path / 'status.json')
        from .console_lifecycle import process_running
        for pid in (status.get('pid'), status.get('child_pid')):
            if isinstance(pid, int) and pid > 0 and process_running(pid):
                return False
        return not (status.get('pid') is None and status.get('state') in ACTIVE_STATES)

    def shutdown_status(self, state, message):
        with self._lock:
            self.shutdown = {"state": state, "message": message}
            if state == 'error':
                self._closing = False

    def logs(self):
        path, status = self._session()
        return {"run_dir": str(path) if path else None, "state": status.get("state"),
                "worker": self._tail(path / "worker.log") if path else "",
                "launcher": self._tail(self.root / "runs/launcher.log", limit=12000)}
