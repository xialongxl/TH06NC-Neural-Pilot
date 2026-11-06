"""HTTP contract and session controls against isolated files and fake workers."""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import threading
import time

import pytest
from fastapi.testclient import TestClient

from th06ai.config import Config
from th06ai import console_service
from th06ai.console_service import ConsoleError, ConsoleService, build_command
from th06ai.files import atomic_json
from th06ai.native import SHA256
from th06ai.web_console import create_app


def model(root, name="latest", steps=1024):
    path = root / "models" / f"{name}.zip"
    path.parent.mkdir(exist_ok=True)
    path.write_bytes(b"fixture-checkpoint-never-loaded")
    atomic_json(path.with_suffix(".json"), {"steps": steps, "gradient_updates": 8,
        "time": "2026-11-01T01:32:11+08:00", "format": 6, "policy_version": 2,
        "game_sha256": SHA256, "model_sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
    return path


def ready(root):
    atomic_json(root / "training_backend.json", {"forward_check": True, "device": "fixture"})
    atomic_json(root / "native_calibration.json", {"restart_verified": True})
    atomic_json(root / "native_resources_calibration.json", {"operator_confirmed": True})


def session(root, *, state="running", mode="train", age=0, **extra):
    path = root / "runs/fixture"
    path.mkdir(parents=True, exist_ok=True)
    atomic_json(root / "runs/current.json", {"run_dir": str(path)})
    atomic_json(path / "status.json", {"state": state, "mode": mode, "monotonic": time.monotonic()-age, **extra})
    return path


@pytest.fixture
def setup(tmp_path):
    calls = []
    class Process:
        def poll(self):
            return None
    def popen(command, **kwargs):
        calls.append((command, kwargs))
        return Process()
    ready(tmp_path)
    checkpoint = model(tmp_path)
    frontend = tmp_path / "frontend/dist"
    (frontend / "assets").mkdir(parents=True)
    (frontend / "index.html").write_text("<!doctype html><title>fixture</title>", encoding="utf-8")
    (frontend / "assets/app.js").write_text("console.log('fixture')", encoding="utf-8")
    service = ConsoleService(Config(game_exe="never-opened"), root=tmp_path, popen=popen,
        focus_fn=lambda path: {'activated': True, 'message': 'fixture-focused'})
    with TestClient(create_app(service.config, root=tmp_path, service=service)) as client:
        bootstrap = client.get("/api/bootstrap").json()
        client.headers["X-Console-Token"] = bootstrap["token"]
        yield tmp_path, checkpoint, service, client, calls, bootstrap["state"]


def test_explicit_web_focus_does_not_create_a_session_or_load_a_model(setup):
    root, checkpoint, service, client, calls, initial = setup
    response = client.post('/api/game/focus')
    assert response.status_code == 200 and response.json()['activated']
    assert not calls and not (root/'runs/current.json').exists()


def test_battle_view_toggle_is_read_only_for_the_worker_and_uses_console_token(setup):
    root, checkpoint, service, client, calls, initial = setup
    assert initial['battle_view_enabled'] is False
    assert client.post('/api/battle-view', json={'enabled': True}, headers={'X-Console-Token': ''}).status_code == 403
    assert client.post('/api/battle-view', json={'enabled': 'true'}).status_code == 422
    assert client.post('/api/battle-view', json={'enabled': True}).json()['enabled'] is True
    assert client.get('/api/state').json()['battle_view_enabled'] is True
    assert client.get('/battle-view').status_code == 200
    assert not calls and not (root / 'runs').exists()
    assert client.post('/api/battle-view', json={'enabled': False}).json()['enabled'] is False

def test_acceleration_changes_only_idle_config_without_starting_a_worker(setup):
    root,checkpoint,service,client,calls,_=setup
    before=checkpoint.read_bytes()
    response=client.post('/api/settings/acceleration',json={'collision':'auto','input':'python'})
    assert response.status_code==200
    config=json.loads((root/'config.json').read_text(encoding='utf-8'))
    assert config['planning']['collision_backend']=='auto' and config['input_backend']=='python'
    assert config['safety']['temperature_limit_c']==70
    assert not calls and checkpoint.read_bytes()==before
    session(root)
    assert client.post('/api/settings/acceleration',json={'collision':'python','input':'python'}).status_code==409

def test_explicit_game_probe_is_read_only_and_separate_from_history(setup,monkeypatch):
    from types import SimpleNamespace
    from th06ai import native
    root,checkpoint,service,client,calls,_=setup
    closed=[]
    f=SimpleNamespace(phase='pause',paused=True,tick=576,menu_selection=0,menu_ready=True,
                      phase_evidence={'nc_pause':2},player_shots=0,lasers=[])
    class Reader:
        def __init__(self,path):assert str(path)=='never-opened'
        def read(self):return f
        def close(self):closed.append(True)
    monkeypatch.setattr(native,'GameReader',Reader)
    response=client.post('/api/game/probe')
    assert response.status_code==200 and response.json()['phase']=='pause'
    assert client.get('/api/state').json()['game_probe']['label']=='游戏已暂停'
    assert closed==[True] and not calls and not (root/'runs/current.json').exists()

def test_state_reads_do_not_probe_game_or_send_input(setup,monkeypatch):
    from th06ai import native
    *_,client,calls,initial=setup
    monkeypatch.setattr(native,'GameReader',lambda *_:(_ for _ in ()).throw(AssertionError('unexpected game access')))
    assert client.get('/api/state').status_code==200
    assert not calls


def test_web_start_activates_the_game_without_waiting_for_worker_loading(setup):
    root, checkpoint, service, client, calls, initial = setup
    order = []
    service._focus_fn = lambda path: order.append(path) or {'activated': True, 'message': 'focused'}
    response = client.post('/api/sessions', json={'mode': 'train', 'minutes': 1})
    assert response.status_code == 202 and response.json()['foreground']['activated']
    assert len(calls) == len(order) == 1 and service.pending_run is not None


@pytest.mark.parametrize(('read', 'filename'), [('logs', 'worker.log'), ('snapshot', 'gpu.jsonl')])
def test_slow_display_reads_do_not_hold_the_stop_request(setup, monkeypatch, read, filename):
    root, _, service, _, calls, _ = setup
    run = session(root)
    entered, release = threading.Event(), threading.Event()
    tail = console_service.tail_text

    def delayed_tail(path, *, limit):
        if path.name == filename:
            entered.set()
            assert release.wait(5)
        return tail(path, limit=limit)

    monkeypatch.setattr(console_service, 'tail_text', delayed_tail)
    with ThreadPoolExecutor(max_workers=2) as pool:
        reading = pool.submit(getattr(service, read))
        try:
            assert entered.wait(2)
            pool.submit(service.control, 'stop').result(timeout=2)
            assert (run / 'STOP').is_file() and not calls
        finally:
            release.set()
        assert reading.result(timeout=2)['run_dir'] == str(run)


def test_stop_can_reach_the_pending_run_while_window_activation_waits(setup):
    _, _, service, _, calls, _ = setup
    entered, release = threading.Event(), threading.Event()

    def focus(_):
        entered.set()
        assert release.wait(5)
        return {'activated': True}

    service._focus_fn = focus
    with ThreadPoolExecutor(max_workers=2) as pool:
        starting = pool.submit(service.start, 'train', 1)
        try:
            assert entered.wait(2)
            pool.submit(service.control, 'stop').result(timeout=2)
            assert (service.pending_run / 'STOP').is_file()
            assert len(calls) == 1
        finally:
            release.set()
        assert starting.result(timeout=2)['foreground']['activated']


@pytest.mark.parametrize('change', ['another_start', 'shutdown', 'model_replaced'])
def test_start_revalidates_after_off_lock_checkpoint_verification(setup, monkeypatch, change):
    _, checkpoint, service, _, calls, initial = setup
    verified, release = threading.Event(), threading.Event()
    inspect = console_service.inspect_checkpoint

    def delayed_verify(path, *, verify=False):
        result = inspect(path, verify=verify)
        if verify:
            verified.set()
            assert release.wait(5)
        return result

    monkeypatch.setattr(console_service, 'inspect_checkpoint', delayed_verify)
    with ThreadPoolExecutor(max_workers=2) as pool:
        resuming = pool.submit(service.start, 'resume', 1, initial['selected_model_id'])
        try:
            assert verified.wait(2)
            if change == 'another_start':
                assert pool.submit(service.start, 'monitor', 1).result(timeout=2)['mode'] == 'monitor'
            elif change == 'shutdown':
                assert pool.submit(service.request_shutdown).result(timeout=2)['state'] == 'waiting'
            else:
                checkpoint.write_bytes(b'different-checkpoint-after-hash-validation')
        finally:
            release.set()
        with pytest.raises(ConsoleError) as rejected:
            resuming.result(timeout=2)
        assert rejected.value.status_code == 409
    assert len(calls) == int(change == 'another_start')


def test_game_probe_allows_state_reads_and_closes_before_launch(setup, monkeypatch):
    from types import SimpleNamespace
    from th06ai import native

    _, _, service, _, calls, _ = setup
    entered, release, closed = threading.Event(), threading.Event(), threading.Event()

    class Reader:
        def __init__(self, path):
            pass

        def read(self):
            entered.set()
            assert release.wait(5)
            return SimpleNamespace(phase='pause', paused=True, tick=10, menu_selection=0, menu_ready=True,
                                   phase_evidence={}, player_shots=0, lasers=[])

        def close(self):
            closed.set()

    popen = service._popen

    def launch(*args, **kwargs):
        assert closed.is_set()
        return popen(*args, **kwargs)

    monkeypatch.setattr(native, 'GameReader', Reader)
    monkeypatch.setattr(service, '_popen', launch)
    with ThreadPoolExecutor(max_workers=3) as pool:
        probing = pool.submit(service.probe_game)
        try:
            assert entered.wait(2)
            starting = pool.submit(service.start, 'train', 1)
            assert pool.submit(service.snapshot).result(timeout=2)['game_probe'] is None
        finally:
            release.set()
        assert probing.result(timeout=2)['available']
        assert starting.result(timeout=2)['mode'] == 'train'
    assert len(calls) == 1


def test_explicit_goal_is_validated_and_forwarded_without_new_or_hidden_session(setup):
    root,checkpoint,service,client,calls,initial=setup
    rejected=client.post('/api/sessions',json={'mode':'resume','minutes':1,
        'model_id':initial['selected_model_id'],'goal':'unsupported'})
    assert rejected.status_code==422 and not calls
    response=client.post('/api/sessions',json={'mode':'resume','minutes':1,
        'model_id':initial['selected_model_id'],'goal':'nmnb'})
    assert response.status_code==202 and len(calls)==1
    command=calls[0][0]
    assert command[command.index('--goal')+1]=='nmnb'
    assert command[command.index('--model')+1]==str(checkpoint.resolve())


def test_a_denied_activation_preserves_the_accepted_session_for_worker_waiting(setup):
    root, checkpoint, service, client, calls, initial = setup
    def denied(path):
        raise OSError('activation denied')
    service._focus_fn = denied
    response = client.post('/api/sessions', json={'mode': 'train', 'minutes': 1})
    assert response.status_code == 202 and not response.json()['foreground']['activated']
    assert len(calls) == 1


@pytest.mark.parametrize('mode', ['train', 'play', 'evaluate'])
def test_pause_resume_and_focus_requests_are_consistent_in_all_game_modes(setup, mode):
    root, checkpoint, service, client, calls, initial = setup
    run = session(root, mode=mode)
    assert client.post('/api/session/pause', json={'paused': True}).status_code == 200
    assert (run/'PAUSE').exists()
    assert client.post('/api/session/pause', json={'paused': False}).status_code == 200
    assert not (run/'PAUSE').exists() and (run/'FOCUS').exists()
    assert client.post('/api/game/focus').status_code == 200
    assert not calls


def test_reading_and_page_reload_never_start_or_touch_training(setup):
    root, checkpoint, service, client, calls, initial = setup
    before = {str(p.relative_to(root)): p.read_bytes() for p in root.rglob("*") if p.is_file()}
    for url in ("/", "/#models", "/api/health", "/api/bootstrap", "/api/state", "/api/logs", "/assets/app.js"):
        assert client.get(url).status_code == 200
    after = {str(p.relative_to(root)): p.read_bytes() for p in root.rglob("*") if p.is_file()}
    assert before == after and not calls
    assert not initial["gpu_live"] and initial["readiness"]["ready"]
    assert not initial["active"]


@pytest.mark.parametrize('marker', ['STOP', 'FINISH'])
def test_resume_passes_exact_checkpoint_and_startup_can_be_cancelled(setup, monkeypatch, marker):
    root, checkpoint, service, client, calls, initial = setup
    result = client.post("/api/sessions", json={"mode": "resume", "minutes": 10, "model_id": initial["selected_model_id"]})
    assert result.status_code == 202
    command, options = calls[0]
    assert command[command.index("--model")+1] == str(checkpoint.resolve())
    assert "train" in command and options["cwd"] == root.resolve()
    run = Path(command[command.index("--run-dir")+1])
    assert run.is_relative_to(root / "runs")
    assert client.get("/api/state").json()["busy"]
    assert client.post("/api/sessions", json={"mode": "train", "minutes": 1}).status_code == 409
    assert len(calls) == 1
    if marker == 'STOP':
        assert client.post('/api/session/stop').status_code == 200
    else:
        assert service.request_shutdown()['state'] == 'waiting'
    assert (run / marker).exists(), "startup must accept cancellation before the guard publishes its pointer"
    # The real CLI must preserve cancellation that raced its initialization,
    # rather than rejecting the marker-only directory as an already used run.
    from types import SimpleNamespace
    from th06ai import __main__, supervisor
    delegated = []
    monkeypatch.setattr(__main__, "ROOT", root)
    monkeypatch.setattr(__main__, "load_config", lambda: service.config)
    monkeypatch.setattr(supervisor, "run_guard", lambda *args, **kwargs: delegated.append(kwargs) or SimpleNamespace(code=0))
    monkeypatch.setattr(sys, "argv", ["th06ai", *command[3:]])
    with pytest.raises(SystemExit) as result:
        __main__.main()
    assert result.value.code == 0 and delegated[0]["run_dir"] == run


def test_pause_resume_finish_and_stop_use_distinct_cooperative_markers(setup):
    root, checkpoint, service, client, calls, initial = setup
    run = session(root)
    before = checkpoint.read_bytes()
    assert client.post("/api/session/pause", json={"paused": True}).status_code == 200
    assert (run / "PAUSE").exists() and not (run / "STOP").exists()
    assert client.get("/api/state").json()["pause_requested"]
    assert client.post("/api/session/pause", json={"paused": False}).status_code == 200
    assert not (run / "PAUSE").exists()
    assert client.post("/api/session/finish").status_code == 200
    assert (run / "FINISH").exists() and not (run / "STOP").exists()
    assert client.post("/api/session/stop").status_code == 200 and (run / "STOP").exists()
    assert checkpoint.read_bytes() == before and not calls


@pytest.mark.parametrize("mode", ["resume", "train", "play", "evaluate", "check", "monitor", "calibrate"])
def test_preview_rejects_all_game_and_gpu_commands_without_writes(tmp_path, mode):
    ready(tmp_path)
    model(tmp_path)
    run = session(tmp_path)
    def forbidden(*args, **kwargs):
        raise AssertionError("preview must never spawn a worker")
    service = ConsoleService(Config(game_exe="unused"), root=tmp_path, preview=True, popen=forbidden)
    before = {str(p): p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()}
    with TestClient(create_app(service.config, root=tmp_path, service=service)) as client:
        data = client.get("/api/bootstrap").json()
        client.headers["X-Console-Token"] = data["token"]
        assert client.post("/api/sessions", json={"mode": mode, "minutes": 1, "model_id": data["state"]["selected_model_id"]}).status_code == 403
        for path, body in (("pause", {"paused": True}), ("finish", None), ("stop", None)):
            assert client.post(f"/api/session/{path}", json=body).status_code == 403
        assert client.post("/api/models/select", json={"model_id": data["state"]["selected_model_id"]}).status_code == 200
    assert before == {str(p): p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()}
    assert not (run / "STOP").exists()


def test_controls_require_same_origin_token_and_known_mode(setup):
    root, checkpoint, service, client, calls, initial = setup
    assert client.post("/api/sessions", headers={"X-Console-Token": ""}, json={"mode": "train", "minutes": 1}).status_code == 403
    assert client.post("/api/sessions", headers={"Origin": "https://foreign.example"}, json={"mode": "train", "minutes": 1}).status_code == 403
    assert client.post("/api/sessions", json={"mode": "shell", "minutes": 1}).status_code == 422
    assert client.post("/api/sessions", json={"mode": "train", "minutes": 61}).status_code == 422
    assert client.post("/api/sessions", json={"mode": "resume", "minutes": 1, "model": "../../arbitrary.zip"}).status_code == 422
    assert client.post("/api/sessions", json={"mode": "resume", "minutes": 1, "model_id": "../outside.zip"}).status_code == 422
    assert client.post("/api/session/pause", json={"paused": "false"}).status_code == 422
    assert client.get("/api/state", headers={"Host": "untrusted.example"}).status_code == 400
    assert not calls


def test_selection_persists_and_history_model_remains_unchanged(setup):
    root, checkpoint, service, client, calls, initial = setup
    older = model(root, "older", 512)
    service.refresh_models(force=True)
    before = older.read_bytes()
    chosen = service.model_id(older.resolve())
    assert client.post("/api/models/select", json={"model_id": chosen}).status_code == 200
    recreated = ConsoleService(service.config, root=root)
    assert recreated.snapshot()["selected_model_id"] == chosen
    assert client.post("/api/models/select", json={"model_id": "missing"}).status_code == 404
    assert older.read_bytes() == before and not calls


def test_stale_gpu_is_labeled_historical_and_stop_remains_available(setup):
    root, checkpoint, service, client, calls, initial = setup
    run = session(root, age=20, gpu={"monotonic": time.monotonic()-20, "temperature_c": 49})
    data = client.get("/api/state").json()
    assert not data["active"] and not data["gpu_live"]
    assert data["session"]["gpu"]["temperature_c"] == 49
    assert client.post("/api/session/stop").status_code == 200 and (run / "STOP").exists()


def test_invalid_pointer_cannot_read_or_control_outside_runs(setup):
    root, checkpoint, service, client, calls, initial = setup
    atomic_json(root / "status.json", {"state": "running", "monotonic": time.monotonic()})
    atomic_json(root / "runs/current.json", {"run_dir": str(root)})
    assert client.get("/api/state").json()["run_dir"] is None
    assert client.post("/api/session/stop").status_code == 409
    assert not (root / "STOP").exists()
    assert client.get("/api/not-a-route").status_code == 404
    assert "fixture" in client.get("/models/latest.zip").text


def test_nonfinite_history_metrics_are_unknown_not_a_stream_failure(setup):
    root, checkpoint, service, client, calls, initial = setup
    session(root, worker={"policy_diagnostics": {"loss": float("nan")}}, gpu={"temperature_c": float("inf")})
    response = client.get("/api/state")
    assert response.status_code == 200
    assert response.json()["session"]["worker"]["policy_diagnostics"]["loss"] is None
    assert response.json()["session"]["gpu"]["temperature_c"] is None


def test_build_command_never_accepts_shell_commands_or_invalid_time():
    for mode, minutes in (("shell", 1), ("train", 0), ("train", float("nan")), ("train", 61)):
        with pytest.raises(ValueError):
            build_command(mode, minutes)
