"""Local console launch/exit controls; never kill a PID or touch another project."""
from __future__ import annotations
from .i18n import msg

import hashlib
import json
from contextlib import contextmanager
import os
from pathlib import Path
import socket
import subprocess
import sys
import time
from urllib.error import HTTPError, URLError
from urllib.request import ProxyHandler, Request, build_opener

from .config import ROOT, DEFAULT_CONSOLE_PORT
from .files import read_json


def project_identity(root=ROOT):
    return hashlib.sha256(str(Path(root).resolve()).encode()).hexdigest()[:16]


def process_running(pid):
    """Read-only liveness check. An inaccessible process must not look exited."""
    import ctypes as ct
    from ctypes import wintypes as wt
    kernel = ct.WinDLL('kernel32', use_last_error=True)
    kernel.OpenProcess.argtypes = [wt.DWORD, wt.BOOL, wt.DWORD]
    kernel.OpenProcess.restype = wt.HANDLE
    kernel.WaitForSingleObject.argtypes = [wt.HANDLE, wt.DWORD]
    kernel.WaitForSingleObject.restype = wt.DWORD
    kernel.CloseHandle.argtypes = [wt.HANDLE]
    kernel.CloseHandle.restype = wt.BOOL
    handle = kernel.OpenProcess(0x100000, False, pid)  # SYNCHRONIZE only
    if not handle:
        if ct.get_last_error() == 87:  # PID no longer exists
            return False
        raise OSError(msg('backend.could_not_confirm_that_the_session_processes'))
    try:
        result = kernel.WaitForSingleObject(handle, 0)
        if result not in (0, 258):
            raise OSError(msg('backend.could_not_confirm_that_the_session_processes'))
        return result == 258
    finally:
        kernel.CloseHandle(handle)


def listener_pid(port):
    """Inspect the local IPv4 listener table without opening a TCP connection."""
    import ctypes as ct
    from ctypes import wintypes as wt
    class Row(ct.Structure):
        _fields_ = [(name, wt.DWORD) for name in ('state', 'address', 'port', 'remote_address', 'remote_port', 'pid')]
    query = ct.WinDLL('iphlpapi').GetExtendedTcpTable
    query.argtypes = [ct.c_void_p, ct.POINTER(wt.DWORD), wt.BOOL, wt.ULONG, ct.c_int, wt.ULONG]
    query.restype = wt.DWORD
    size = wt.DWORD()
    for _ in range(4):
        data = ct.create_string_buffer(size.value) if size.value else None
        result = query(data, ct.byref(size), False, socket.AF_INET, 3, 0)
        if result == 122:  # ERROR_INSUFFICIENT_BUFFER; table may grow between calls
            continue
        if result:
            raise OSError(msg('backend.could_not_read_local_listening_ports_0', result))
        count = wt.DWORD.from_buffer_copy(data.raw).value
        rows = (Row*count).from_buffer_copy(data.raw[ct.sizeof(wt.DWORD):])
        for row in rows:
            address = socket.inet_ntoa(int(row.address).to_bytes(4, 'little'))
            if socket.ntohs(row.port & 0xffff) == port and address in ('127.0.0.1', '0.0.0.0'):
                return row.pid
        return None
    raise OSError(msg('backend.local_port_ownership_is_changing_try_again'))


@contextmanager
def launch_lock(root, port):
    import msvcrt
    path = Path(root) / f'runs/console-launch-{port}.lock'
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a+b') as stream:
        if stream.tell() == 0:
            stream.write(b'0')
            stream.flush()
        deadline = time.monotonic()+30
        while True:
            stream.seek(0)
            try:
                msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
                break
            except OSError:
                if time.monotonic() >= deadline:
                    raise RuntimeError(msg('backend.another_startup_request_is_in_progress_try')) from None
                time.sleep(.05)
        try:
            yield
        finally:
            stream.seek(0)
            msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)


def open_console(*, port=DEFAULT_CONSOLE_PORT, root=ROOT, preview=False, open_browser=True,
                 timeout=20, opener=None, popen=subprocess.Popen, listener=listener_pid,
                 lock=launch_lock, show=None):
    """One foreground launcher acknowledges readiness; the API remains headless."""
    if not 1024 <= port <= 65535:
        raise ValueError(msg('backend.the_console_port_must_be_between_1024'))
    root = Path(root).resolve()
    url = f'http://127.0.0.1:{port}'
    opener = opener or build_opener(ProxyHandler({}))
    generation_file = root / f'runs/console-stop-{port}.json'
    generation = read_json(generation_file)
    if show is None:
        import webbrowser
        show = webbrowser.open

    def existing():
        if listener(port) is None:
            return False
        try:
            with opener.open(url+'/api/health', timeout=1) as response:
                data = json.load(response)
        except (OSError, ValueError) as error:
            raise RuntimeError(msg('backend.port_0_is_in_use_and_console', port)) from error
        if not isinstance(data, dict) or (data.get('service'), data.get('project'), data.get('preview')) != (
                'th06nc-console', project_identity(root), preview):
            raise RuntimeError(msg('backend.port_0_belongs_to_another_service_startup', port))
        with opener.open(url+'/api/state', timeout=2) as response:
            state = json.load(response)
        if (state.get('shutdown') or {}).get('state') in ('waiting', 'stopping'):
            raise RuntimeError(msg('backend.the_service_is_shutting_down_wait_for'))
        return True

    with lock(root, port):
        if read_json(generation_file) != generation:
            return msg('backend.startup_cancelled_because_a_shutdown_was_just')
        if not existing():
            executable = root / '.venv/Scripts/pythonw.exe'
            if not executable.is_file():
                raise RuntimeError(msg('backend.run_install_ps1_to_install_the_training'))
            command = [str(executable), '-m', 'th06ai', 'web', '--no-browser', '--port', str(port)]
            if preview:
                command.append('--preview')
            log_path = root/'runs/web-launcher.log'
            log_path.parent.mkdir(parents=True, exist_ok=True)
            with log_path.open('a', encoding='utf-8') as log:
                child = popen(command, cwd=root, env=dict(os.environ, PYTHONUTF8='1'), stdout=log,
                              stderr=subprocess.STDOUT, creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
            deadline = time.monotonic()+timeout
            while not existing():
                if child.poll() is not None:
                    raise RuntimeError(msg('backend.the_backend_failed_to_start_see_0', log_path))
                if time.monotonic() >= deadline:
                    raise RuntimeError(msg('backend.the_backend_is_still_unavailable_see_0', log_path))
                time.sleep(.1)
        if open_browser:
            show(url)
    return msg('backend.console_ready_at_0_use_shut_down', url)


def shutdown_console(*, port=DEFAULT_CONSOLE_PORT, root=ROOT, preview=False, timeout=60,
                     opener=None, clock=time.monotonic, sleep=time.sleep,
                     listener=listener_pid, alive=process_running):
    if not 1024 <= port <= 65535:
        raise ValueError(msg('backend.the_console_port_must_be_between_1024'))
    url = f'http://127.0.0.1:{port}'
    opener = opener or build_opener(ProxyHandler({}))
    owner = listener(port)
    if owner is None:
        return msg('backend.the_backend_has_already_shut_down')

    def request(path, *, token=None):
        headers = {'X-Console-Token': token, 'Origin': url} if token else {}
        req = Request(url + path, data=b'' if token else None, headers=headers)
        with opener.open(req, timeout=2) as response:
            return json.load(response)

    def health():
        try:
            data = request('/api/health')
        except HTTPError as error:
            raise RuntimeError(msg('backend.shutdown_cancelled_because_this_port_does_not')) from error
        except URLError as error:
            if isinstance(error.reason, ConnectionRefusedError):
                return None
            raise RuntimeError(msg('backend.could_not_reach_the_backend_to_verify')) from error
        if not isinstance(data, dict) or (data.get('service'), data.get('project'), data.get('preview')) != (
                'th06nc-console', project_identity(root), preview):
            raise RuntimeError(msg('backend.shutdown_cancelled_because_this_port_belongs_to'))
        return data

    existing = health()
    if existing is None:
        return msg('backend.the_backend_has_already_shut_down')
    if existing.get('pid', owner) != owner:
        raise RuntimeError(msg('backend.shutdown_cancelled_because_the_port_owner_does'))
    token = request('/api/bootstrap?state=false').get('token')
    if not isinstance(token, str) or not token:
        raise RuntimeError(msg('backend.shutdown_cancelled_because_the_backend_did_not'))
    try:
        request('/api/server/shutdown', token=token)
    except HTTPError as error:
        detail = json.load(error).get('detail', msg('backend.the_shutdown_request_failed'))
        raise RuntimeError(detail) from error
    deadline = clock() + timeout
    while clock() < deadline:
        if not alive(owner):
            return msg('backend.the_ai_session_has_ended_and_the')
        try:
            state = request('/api/state').get('shutdown') or {}
        except (OSError, ValueError):
            # Windows may time out instead of refusing a recently closed port.
            # Only the verified server's actual exit establishes completion.
            sleep(.2)
            continue
        if state.get('state') == 'error':
            raise RuntimeError(state.get('message', msg('backend.shutdown_is_incomplete_you_can_press_f8')))
        sleep(.2)
    raise RuntimeError(msg('backend.shutdown_is_still_pending_check_the_console'))
