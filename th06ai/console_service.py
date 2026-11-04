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


def current_session(root=ROOT):
    pointer = read_json(root / "runs" / "current.json")
    if not isinstance(pointer.get("run_dir"), str):
        return None, {}
    path = Path(pointer["run_dir"]).resolve()
    if not path.is_relative_to((root / "runs").resolve()):
        return None, {}
    return path, read_json(path / "status.json")


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
        self._lock = threading.RLock()
        self._catalog, self._catalog_at = [], -math.inf
        self._catalog_cache = CheckpointCatalog(self.root / 'models')
        self._catalog_scan_lock = threading.Lock()
        self._catalog_refreshing = False
        self._catalog_thread = None
        self._catalog_revision = None
        self._catalog_error = None
        self._history, self._history_key = [], None
        self._json_cache, self._text_cache = {}, {}
        self._model_view = []
        self._model_catalog_ref = None
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
        with self._lock:
            self._catalog = entries
            self._catalog_at = time.monotonic()
            self._catalog_revision = revision
            self._catalog_error = None
            if not any(c.path == self.selected_path for c in entries):
                self.selected_path = next((c.path for c in entries if c.usable), None)
            return self._catalog

    def _background_scan(self):
        try:
            self._scan_models()
        except Exception as error:
            with self._lock:
                self._catalog_error = str(error)
                self._catalog_at = time.monotonic()
                self._catalog_revision = self._file_stamp(self.root / 'models/catalog_revision.json')
        finally:
            with self._lock:
                self._catalog_refreshing = False

    def models(self):
        catalog = self.refresh_models()
        if catalog is self._model_catalog_ref:
            return self._model_view
        rows = [{"id": self.model_id(c.path), "name": msg('backend.latest_checkpoint') if c.path == self.root / "models/latest.zip"
                 else msg('backend.checkpoint_0_steps', f"{c.metadata.get('steps', 0):,}"),
                 "path": str(c.path.relative_to(self.root)), "steps": c.metadata.get("steps", 0),
                 "updates": c.metadata.get("gradient_updates", 0), "saved_at": c.metadata.get("time"),
                 "usable": c.usable, "compatibility": c.compatibility,
                 "format": c.metadata.get("format"), "planner_version": c.metadata.get("planner_version")}
                for c in catalog]
        if rows != self._model_view:
            self._model_view = json_safe(rows)
        self._model_catalog_ref = catalog
        return self._model_view

    @staticmethod
    def _file_stamp(path):
        try:
            stat = path.stat()
            return stat.st_mtime_ns, stat.st_size, stat.st_ino
        except OSError:
            return None

    def _read_json(self, path):
        stamp = self._file_stamp(path)
        cached = self._json_cache.get(path)
        if stamp is not None and cached is not None and cached[0] == stamp:
            return cached[1]
        value = json_safe(read_json(path))
        if value and stamp is not None:
            self._json_cache[path] = stamp, value
            if len(self._json_cache) > 64:
                self._json_cache.pop(next(iter(self._json_cache)))
        return value

    def _tail(self, path, limit=32768):
        stamp = self._file_stamp(path)
        cached = self._text_cache.get((path, limit))
        if cached is not None and cached[0] == stamp:
            return cached[1]
        value = tail_text(path, limit=limit)
        self._text_cache[path, limit] = stamp, value
        if len(self._text_cache) > 32:
            self._text_cache.pop(next(iter(self._text_cache)))
        return value

    def select_model(self, model_id):
        catalog = self.refresh_models(force=True)
        with self._lock:
            entry = next((c for c in catalog if self.model_id(c.path) == model_id), None)
            if entry is None:
                raise ConsoleError(msg('backend.model_not_found_refresh_the_model_library'), 404)
            if not entry.usable:
                raise ConsoleError(entry.compatibility)
            checked = inspect_checkpoint(entry.path)
            if not checked.path.is_relative_to((self.root / 'models').resolve()) or not checked.usable:
                raise ConsoleError(msg('backend.the_model_path_or_metadata_changed_refresh'))
            self.selected_path = entry.path
            if not self.preview:
                atomic_json(self.root / "ui_state.json", {"selected_model": str(entry.path)})
            return self.model_id(entry.path)

    def readiness(self):
        backend = self._read_json(self.root / "training_backend.json")
        checks = {"gpu": bool(backend.get("forward_check")),
                  "calibration": bool(self._read_json(self.root / "native_calibration.json").get("restart_verified")),
                  "resources": bool(self._read_json(self.root / "native_resources_calibration.json").get("operator_confirmed"))}
        return {"ready": all(checks.values()), "checks": checks, "backend": backend}

    def _gpu_history(self, path, gpu):
        key = str(path) if path else None
        if key != self._history_key:
            records = []
            for line in tail_text(path / "gpu.jsonl", limit=180000).splitlines() if path else []:
                try:
                    row = json.loads(line)
                    if isinstance(row.get("temperature_c"), (int, float)) and math.isfinite(row["temperature_c"]):
                        records.append(json_safe({"stamp": row.get("monotonic"), "temperature_c": row["temperature_c"],
                                        "power_w": row.get("power_w"), "utilization": row.get("utilization")}))
                except (ValueError, AttributeError):
                    continue
            self._history, self._history_key = records[-180:], key
        if isinstance(gpu.get("temperature_c"), (int, float)) and math.isfinite(gpu["temperature_c"]) and (
                not self._history or self._history[-1]["stamp"] != gpu.get("monotonic")):
            self._history = [*self._history, json_safe({"stamp": gpu.get("monotonic"), "temperature_c": gpu["temperature_c"],
                              "power_w": gpu.get("power_w"), "utilization": gpu.get("utilization")} )][-180:]
        return self._history

    def snapshot(self):
        with self._lock:
            path, status = self._session()
            active = session_active(status)
            launching = bool(self.process and self.process.poll() is None)
            gpu = status.get("gpu") or {}
            stamp = gpu.get("monotonic")
            gpu_age = max(0, time.monotonic()-stamp) if isinstance(stamp, (int, float)) and math.isfinite(stamp) else None
            return {"preview": self.preview, "session": status, "active": active,
                    "shutdown": dict(self.shutdown),
                    'game_probe':self.last_game_probe,
                    'acceleration':{**self._native_backend,'collision':self.config.planning.collision_backend,
                                    'input':self.config.input_backend},
                    "busy": active or launching, "run_dir": str(path) if path else None,
                    'default_game_setup':(self.config.game_setup or GameSetup()).to_dict(),
                    "hotkeys": status.get('hotkeys') or {},
                    "window_focus": self.last_focus,
                    "pause_requested": bool(path and (path / "PAUSE").exists()),
                    "finish_requested": bool(path and (path / "FINISH").exists() and active),
                    "selected_model_id": self.model_id(self.selected_path) if self.selected_path else None,
                    "models": self.models(), "readiness": self.readiness(),
                    "safety": self._safety, "default_minutes": self.config.minutes,
                    "versions": {"observation": OBSERVATION_VERSION, "reward": REWARD_VERSION, "planner": PLANNER_VERSION},
                    "gpu_history": self._gpu_history(path, gpu), "gpu_age_seconds": gpu_age,
                    "gpu_live": active and gpu_age is not None and gpu_age < self.config.safety.telemetry_timeout_seconds
                        and isinstance(gpu.get("temperature_c"), (int, float)) and math.isfinite(gpu["temperature_c"])}

    def battle_view_run(self):
        with self._lock:
            if self.preview or self.shutdown.get('state') in ('waiting', 'stopping', 'closed'):
                return None
            path, status = self._session()
            if (path and status.get('mode') in ('train', 'play', 'evaluate') and status.get('state') == 'running'
                    and session_active(status) and not (path / 'PAUSE').exists()
                    and not (path / 'STOP').exists()):
                return path
        return None

    def _session(self):
        pointer = self._read_json(self.root / "runs/current.json")
        candidate = pointer.get("run_dir")
        path = Path(candidate).resolve() if isinstance(candidate, str) else None
        if path is not None and not path.is_relative_to((self.root / "runs").resolve()):
            path = None
        status = self._read_json(path / "status.json") if path else {}
        if self.pending_run is not None and self.process is not None:
            code = self.process.poll()
            if code is None or path != self.pending_run:
                pending = self._read_json(self.pending_run / "status.json")
                return self.pending_run, pending or {"state": "starting" if code is None else "stopped",
                    "mode": self.pending_mode,
                    "reason": msg('backend.starting_the_independent_watchdog') if code is None else msg('backend.the_launcher_exited_with_code_0_check', code)}
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
        self.last_focus = result
        return result

    def focus_game(self):
        with self._lock:
            self._writable()
            _, status = self._session()
            if session_active(status):
                return self.control('focus')
            result = self._activate_game()
            if not result['activated']:
                raise ConsoleError(result['message'])
            return result

    def probe_game(self):
        """Explicit one-shot inspection: no focus, input, worker or GPU."""
        from datetime import datetime
        from .game_phase import PHASE_NAMES
        from .stage import stage_label
        with self._lock:
            if self.preview:
                raise ConsoleError(msg('backend.read_only_preview_does_not_connect_to'),403)
            _,status=self._session()
            if session_active(status) or self.process and self.process.poll() is None:
                raise ConsoleError(msg('backend.a_session_is_running_check_the_worker'),409)
            reader=None
            try:
                from .native import GameReader
                reader=GameReader(self.config.game_path)
                if self.config.capture_backend != 'auto':
                    reader.process.configure_reads(self.config.capture_backend)
                f=reader.read()
                self.last_game_probe={'available':True,'phase':f.phase,'label':PHASE_NAMES.get(f.phase,msg('backend.waiting_for_detection')),
                    'game_settings':observed_settings(f),
                    'stage_id':getattr(f,'stage_id',None),'stage_label':stage_label(getattr(f,'stage_id',None)),
                    'paused':f.paused,'tick':f.tick,'menu_selection':f.menu_selection,'menu_ready':f.menu_ready,
                    'evidence':f.phase_evidence,'player_shots':f.player_shots,'visible_lasers':len(f.lasers),
                    'observed_at':datetime.now().astimezone().isoformat(),'message':msg('backend.read_only_probe_complete')}
            except (OSError,RuntimeError,ValueError) as error:
                self.last_game_probe={'available':False,'label':msg('backend.game_not_connected'),'message':str(error),
                    'observed_at':datetime.now().astimezone().isoformat()}
            finally:
                if reader is not None:reader.close()
            return self.last_game_probe

    def set_acceleration(self,collision,input_backend):
        from .acceleration import backend_status
        with self._lock:
            self._writable()
            _,status=self._session()
            if session_active(status) or self.process and self.process.poll() is None:
                raise ConsoleError(msg('backend.end_the_current_session_before_switching_compute'),409)
            available=backend_status()
            if 'rust' in (collision,input_backend) and not available['available']:
                raise ConsoleError(msg('backend.the_rust_extension_has_not_been_built'),422)
            if collision not in ('python','rust','auto') or input_backend not in ('python','rust','auto'):
                raise ConsoleError(msg('backend.unknown_compute_backend'),422)
            config=replace(self.config,input_backend=input_backend,
                           planning=replace(self.config.planning,collision_backend=collision))
            atomic_json(self.root/'config.json',asdict(config))
            self.config=config
            self._native_backend=available
            return {'message':msg('backend.backend_settings_saved_for_the_next_session')}

    def start(self, mode, minutes, model_id=None, goal=None, game_setup=None):
        catalog = self.refresh_models(force=True) if mode in {'resume', 'play', 'evaluate'} else []
        with self._lock:
            self._writable()
            if self._closing:
                raise ConsoleError(msg('backend.the_service_is_shutting_down_wait_for_2'))
            path, status = current_session(self.root)
            if session_active(status) or (self.process and self.process.poll() is None):
                raise ConsoleError(msg('backend.a_session_is_already_running_pause_resume'))
            model = None
            if mode in {"resume", "play", "evaluate"}:
                entry = next((c for c in catalog
                              if self.model_id(c.path) == model_id), None)
                if entry is None:
                    raise ConsoleError(msg('backend.select_a_compatible_model_first'), 422)
                checked = inspect_checkpoint(entry.path, verify=True)
                if not checked.path.is_relative_to((self.root / 'models').resolve()):
                    raise ConsoleError(msg('backend.the_model_path_is_outside_the_project'))
                if not checked.usable:
                    raise ConsoleError(checked.compatibility)
                model = entry.path
            if mode in {"resume", "train", "play", "evaluate"} and not self.readiness()["ready"]:
                raise ConsoleError(msg('backend.complete_the_checks_on_the_safety_diagnostics'))
            try:
                setup = parse_setup(game_setup)
            except ValueError as error:
                raise ConsoleError(str(error), 422) from error
            if setup is not None and mode not in ('resume','train','play','evaluate'):
                raise ConsoleError(msg('backend.game_settings_apply_to_training_playback_and'),422)
            command = build_command(mode, minutes, model, goal, setup)
            run_path = new_run(self.root, "train" if mode == "resume" else mode)
            command += ["--run-dir", str(run_path)]
            log = self.root / "runs/launcher.log"
            log.parent.mkdir(parents=True, exist_ok=True)
            try:
                with log.open("a", encoding="utf-8") as stream:
                    self.process = self._popen(command, cwd=self.root,
                        env=dict(os.environ, PYTHONUTF8="1", PYTHONUNBUFFERED="1"), stdout=stream,
                        stderr=subprocess.STDOUT, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
            except OSError as error:
                raise ConsoleError(msg('backend.launch_failed_0', error), 500) from error
            self.pending_run, self.pending_mode = run_path, "train" if mode == "resume" else mode
            # Web's foreground process is the browser, not this API. Activate
            # on the explicit start request now, rather than waiting through
            # CUDA/model loading before the worker first asks for focus.
            focus = self._activate_game() if mode in ('resume', 'train', 'play', 'evaluate', 'calibrate') else None
            return {"message": msg('backend.request_accepted_the_session_will_start_after'), "mode": mode, 'foreground': focus}

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
                (path / 'FOCUS').touch()
                return {'message': msg('backend.game_focus_requested_training_and_pause_state')}
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
            if not self.preview and path and (pending or status.get('state') in ACTIVE_STATES):
                mode = status.get('mode') or self.pending_mode
                marker = 'FINISH' if mode in ('train', 'play', 'evaluate') else 'STOP'
                (path / marker).touch()
                self._shutdown_run = path
            self._closing = True
            self.shutdown = {"state": "waiting", "message": msg('backend.finishing_the_session_and_shutting_down_training')}
            return dict(self.shutdown)

    def shutdown_ready(self):
        with self._lock:
            if self.preview:
                return True
            if self.process and self.process.poll() is None:
                return False
            if self._shutdown_run is None:
                return True
            status = self._read_json(self._shutdown_run / 'status.json')
            from .console_lifecycle import process_running
            for pid in (status.get('pid'), status.get('child_pid')):
                if isinstance(pid, int) and pid > 0 and process_running(pid):
                    return False
            if status.get('pid') is None and status.get('state') in ACTIVE_STATES:
                return False
            return True

    def shutdown_status(self, state, message):
        with self._lock:
            self.shutdown = {"state": state, "message": message}
            if state == 'error':
                self._closing = False

    def logs(self):
        with self._lock:
            path, status = self._session()
            return {"run_dir": str(path) if path else None, "state": status.get("state"),
                    "worker": self._tail(path / "worker.log") if path else "",
                    "launcher": self._tail(self.root / "runs/launcher.log", limit=12000)}
