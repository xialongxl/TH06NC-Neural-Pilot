"""Shutdown must save cooperatively, drain workers and identify the right project."""
import io
import json
import subprocess
import threading
import time
from contextlib import contextmanager, nullcontext
from urllib.error import HTTPError, URLError

import pytest
from fastapi.testclient import TestClient

from th06ai.config import Config
from th06ai.console_lifecycle import open_console, project_identity, shutdown_console
from th06ai.console_service import ConsoleError, ConsoleService
from th06ai.files import atomic_json
from th06ai.web_console import create_app


def running(root, *, mode='train', age=0, pid=None):
    path = root / 'runs/session'
    path.mkdir(parents=True, exist_ok=True)
    atomic_json(root / 'runs/current.json', {'run_dir': str(path)})
    atomic_json(path / 'status.json', {'state': 'paused', 'mode': mode,
                'monotonic': time.monotonic()-age, 'pid': pid})
    return path


def test_idle_shutdown_blocks_new_start_without_creating_a_worker(tmp_path):
    service = ConsoleService(Config(game_exe='never-opened'), root=tmp_path)
    service.request_shutdown()
    assert service.shutdown_ready()
    assert not (tmp_path / 'runs/current.json').exists()
    with pytest.raises(ConsoleError, match='正在关闭'):
        service.start('monitor', 1)


@pytest.mark.parametrize('mode,marker', [('train', 'FINISH'), ('play', 'FINISH'),
                                      ('evaluate', 'FINISH'), ('monitor', 'STOP'), ('check', 'STOP')])
def test_running_session_is_ended_before_shutdown(tmp_path, monkeypatch, mode, marker):
    path = running(tmp_path, mode=mode, pid=1234, age=30)
    service = ConsoleService(Config(game_exe='never-opened'), root=tmp_path)
    alive = {1234}
    monkeypatch.setattr('th06ai.console_lifecycle.process_running', lambda pid: pid in alive)
    service.request_shutdown()
    assert (path / marker).exists()
    assert not (path / ('STOP' if marker == 'FINISH' else 'FINISH')).exists()
    assert not service.shutdown_ready(), 'stale heartbeat must not look like an exited process'
    alive.clear()
    assert service.shutdown_ready()


def test_pending_start_and_child_process_must_both_exit(tmp_path, monkeypatch):
    service = ConsoleService(Config(game_exe='never-opened'), root=tmp_path)
    class Process:
        code = None
        def poll(self): return self.code
    service.process = Process()
    path = tmp_path / 'runs/pending'
    path.mkdir(parents=True)
    service.pending_run, service.pending_mode = path, 'train'
    service.request_shutdown()
    assert (path / 'FINISH').exists() and not service.shutdown_ready()
    service.process.code = 0
    atomic_json(path / 'status.json', {'state': 'complete', 'pid': 123, 'child_pid': 456})
    monkeypatch.setattr('th06ai.console_lifecycle.process_running', lambda pid: pid == 456)
    assert not service.shutdown_ready()
    monkeypatch.setattr('th06ai.console_lifecycle.process_running', lambda pid: False)
    assert service.shutdown_ready()


def test_preview_shutdown_does_not_touch_real_training(tmp_path):
    path = running(tmp_path)
    service = ConsoleService(Config(game_exe='never-opened'), root=tmp_path, preview=True)
    service.request_shutdown()
    assert service.shutdown_ready()
    assert not (path / 'STOP').exists() and not (path / 'FINISH').exists()


def test_completed_historical_pid_is_not_treated_as_a_live_session(tmp_path, monkeypatch):
    path = running(tmp_path, pid=1234)
    atomic_json(path / 'status.json', {'state': 'complete', 'pid': 1234})
    service = ConsoleService(Config(game_exe='never-opened'), root=tmp_path)
    monkeypatch.setattr('th06ai.console_lifecycle.process_running', lambda pid: pytest.fail('historical PID reuse'))
    service.request_shutdown()
    assert service.shutdown_ready() and not (path / 'FINISH').exists()


def test_http_shutdown_is_authenticated_and_idempotent(tmp_path):
    service = ConsoleService(Config(game_exe='never-opened'), root=tmp_path)
    exited = threading.Event()
    with TestClient(create_app(service.config, root=tmp_path, service=service, shutdown=exited.set)) as client:
        assert client.post('/api/server/shutdown').status_code == 403
        token = client.get('/api/bootstrap?state=false').json()['token']
        client.headers['X-Console-Token'] = token
        assert client.post('/api/server/shutdown', headers={'Origin': 'https://foreign.example'}).status_code == 403
        for _ in range(2):
            assert client.post('/api/server/shutdown').status_code == 202
        assert exited.wait(2)
        assert client.get('/api/state').json()['shutdown']['state'] == 'stopping'


def test_shutdown_timeout_keeps_backend_and_emergency_stop_available(tmp_path):
    path = running(tmp_path)
    service = ConsoleService(Config(game_exe='never-opened'), root=tmp_path)
    exited = threading.Event()
    with TestClient(create_app(service.config, root=tmp_path, service=service,
                              shutdown=exited.set, shutdown_timeout=.01)) as client:
        client.headers['X-Console-Token'] = client.get('/api/bootstrap?state=false').json()['token']
        assert client.post('/api/server/shutdown').status_code == 202
        deadline = time.monotonic()+2
        while service.shutdown['state'] != 'error' and time.monotonic() < deadline:
            time.sleep(.01)
        assert service.shutdown['state'] == 'error' and not exited.is_set()
        assert client.get('/api/health').status_code == 200
        assert client.post('/api/session/stop').status_code == 200
        assert (path / 'STOP').exists()
        atomic_json(path / 'status.json', {'state': 'complete'})
        assert client.post('/api/server/shutdown').status_code == 202
        assert exited.wait(2)


class Opener:
    def __init__(self, responses):
        self.responses, self.requests = list(responses), []
    def open(self, request, **kwargs):
        self.requests.append(request)
        value = self.responses.pop(0)
        if isinstance(value, Exception): raise value
        return io.BytesIO(json.dumps(value).encode())


def test_open_reuses_a_matching_service_without_spawning(tmp_path):
    health = {'service': 'th06nc-console', 'project': project_identity(tmp_path), 'preview': False}
    shown = []
    result = open_console(root=tmp_path, opener=Opener([health, {}]), listener=lambda port: 1234,
                          lock=lambda *_: nullcontext(), show=shown.append,
                          popen=lambda *a, **kw: pytest.fail('must not spawn a duplicate'))
    assert '已就绪' in result and shown == ['http://127.0.0.1:18765']


def test_open_acknowledges_backend_readiness_before_returning(tmp_path):
    executable = tmp_path / '.venv/Scripts/pythonw.exe'
    executable.parent.mkdir(parents=True)
    executable.touch()
    calls = []
    class Process:
        def poll(self): return None
    def popen(command, **kwargs):
        calls.append((command, kwargs))
        return Process()
    health = {'service': 'th06nc-console', 'project': project_identity(tmp_path), 'preview': True}
    open_console(root=tmp_path, port=18767, preview=True, open_browser=False,
                 opener=Opener([health, {}]), listener=lambda port: 1234 if calls else None,
                 lock=lambda *_: nullcontext(), popen=popen,
                 show=lambda *_: pytest.fail('no-browser must be honored'))
    assert len(calls) == 1 and calls[0][1]['cwd'] == tmp_path
    assert calls[0][0][1:] == ['-m', 'th06ai', 'web', '--no-browser', '--port', '18767', '--preview']
    assert calls[0][1]['creationflags'] == getattr(subprocess, 'CREATE_NO_WINDOW', 0)


def test_browser_launch_failure_reports_the_ready_url(tmp_path):
    health = {'service': 'th06nc-console', 'project': project_identity(tmp_path), 'preview': False}
    with pytest.raises(RuntimeError, match='http://127.0.0.1:18765'):
        open_console(root=tmp_path, opener=Opener([health, {}]), listener=lambda port: 1234,
                     lock=lambda *_: nullcontext(), show=lambda url: False)


def test_a_queued_launch_is_cancelled_if_shutdown_won(tmp_path):
    @contextmanager
    def competing_shutdown(root, port):
        atomic_json(root / f'runs/console-stop-{port}.json', {'requested_ns': 1})
        yield
    result = open_console(root=tmp_path, lock=competing_shutdown,
                          listener=lambda port: pytest.fail('cancelled launch must not connect or spawn'))
    assert '已取消' in result


def test_open_refuses_an_unrelated_or_closing_service(tmp_path):
    for health, state in (({'service': 'other'}, None),
            ({'service': 'th06nc-console', 'project': project_identity(tmp_path), 'preview': False},
             {'shutdown': {'state': 'waiting'}})):
        opener = Opener([health] if state is None else [health, state])
        with pytest.raises(RuntimeError):
            open_console(root=tmp_path, opener=opener, listener=lambda port: 1234,
                         lock=lambda *_: nullcontext(), popen=lambda *a, **kw: pytest.fail('no spawn'))


def test_shutdown_client_uses_project_identity_and_control_token(tmp_path):
    health = {'service': 'th06nc-console', 'project': project_identity(tmp_path), 'preview': False}
    opener = Opener([health, {'token': 'fixture'}, {}])
    assert '已关闭' in shutdown_console(root=tmp_path, opener=opener, sleep=lambda _: None,
                                     listener=lambda port: 1234, alive=lambda pid: False)
    request = opener.requests[2]
    assert request.get_method() == 'POST' and request.full_url.endswith('/api/server/shutdown')
    assert request.get_header('X-console-token') == 'fixture'
    assert request.get_header('Origin') == 'http://127.0.0.1:18765'


def test_shutdown_client_refuses_other_project_and_unresponsive_ports(tmp_path):
    for value in ({'service': 'other-project'}, HTTPError('http://localhost', 404, '', {}, io.BytesIO()),
                  URLError(TimeoutError())):
        opener = Opener([value])
        with pytest.raises(RuntimeError):
            shutdown_console(root=tmp_path, opener=opener, listener=lambda port: 1234)
        assert len(opener.requests) == 1 and opener.requests[0].get_method() == 'GET'
    assert '已经关闭' in shutdown_console(root=tmp_path, opener=Opener([]), listener=lambda port: None)


def test_client_waits_for_actual_exit_when_http_is_unresponsive(tmp_path):
    health = {'service': 'th06nc-console', 'project': project_identity(tmp_path), 'preview': False, 'pid': 1234}
    opener = Opener([health, {'token': 'fixture'}, {}, URLError(TimeoutError())])
    checks = iter([True, False])
    assert '已关闭' in shutdown_console(root=tmp_path, opener=opener, listener=lambda port: 1234,
                                     alive=lambda pid: next(checks), sleep=lambda _: None)


def test_client_refuses_a_changed_listener_owner(tmp_path):
    health = {'service': 'th06nc-console', 'project': project_identity(tmp_path), 'preview': False, 'pid': 4321}
    opener = Opener([health])
    with pytest.raises(RuntimeError, match='身份不一致'):
        shutdown_console(root=tmp_path, opener=opener, listener=lambda port: 1234)
    assert len(opener.requests) == 1
