import struct
import time
import subprocess
import sys
import json
import threading
from types import SimpleNamespace

import pytest

from th06ai.config import SafetyConfig
from th06ai.lease import SharedLease, LeaseBusy
from th06ai.worker import Heartbeat


def test_new_lease_does_not_authorize_work_until_guard_publishes():
    owner = SharedLease(create=True)
    reader = SharedLease(owner.name)
    try:
        assert reader.read() == {}
        before = time.monotonic()
        owner.publish(before, "GPU-test")
        assert reader.read()["monotonic"] == before
        assert reader.read()["uuid"] == "GPU-test"
    finally:
        reader.close()
        owner.close()


def test_incomplete_publish_is_never_a_valid_lease():
    owner = SharedLease(create=True)
    try:
        owner.publish(time.monotonic(), "GPU-test")
        version = struct.unpack_from("<Q", owner.memory.buf, 0)[0]
        struct.pack_into("<Q", owner.memory.buf, 0, version + 1)
        assert owner.read() == {}
    finally:
        owner.close()


@pytest.mark.parametrize("timeout, age, expired", [
    (60.0, 10.0, False),
    (60.0, 60.0, False),
    (60.0, 60.01, True),
    (.75, 1.0, True),
])
def test_worker_obeys_configured_monitor_timeout(tmp_path, monkeypatch, timeout, age, expired):
    monkeypatch.setattr("th06ai.worker.time", SimpleNamespace(monotonic=lambda: 100.0))
    owner = SharedLease(create=True)
    heartbeat = Heartbeat(tmp_path, owner, SafetyConfig(telemetry_timeout_seconds=timeout))
    try:
        owner.publish(100.0 - age, "GPU-test")
        if expired:
            with pytest.raises(RuntimeError, match="保护授权已失效"):
                heartbeat.lease()
        else:
            heartbeat.lease()
    finally:
        heartbeat.close()
        owner.close()


def test_concurrent_publish_reuses_last_valid_permit_without_renewing_its_time(tmp_path, monkeypatch):
    clock = [100.0]
    monkeypatch.setattr("th06ai.worker.time", SimpleNamespace(monotonic=lambda: clock[0]))
    owner = SharedLease(create=True)
    heartbeat = Heartbeat(tmp_path, owner)
    try:
        owner.publish(clock[0], "GPU-test")
        heartbeat.lease()
        sequence = struct.unpack_from("<Q", owner.memory.buf, 0)[0]
        struct.pack_into("<Q", owner.memory.buf, 0, sequence + 1)
        assert owner.read() == {}
        with pytest.raises(LeaseBusy):
            owner.read_checked()
        clock[0] = 100.05
        heartbeat.lease()  # Previously stopped immediately on this busy read.
        assert heartbeat.last_valid_permit["monotonic"] == 100.0
        assert heartbeat.data["lease_diagnostics"]["read_state"] == "concurrent_publish"
        clock[0] = 160.0
        heartbeat.lease()
        assert heartbeat.last_valid_permit["monotonic"] == 100.0
        clock[0] = 160.05
        with pytest.raises(RuntimeError, match="配置时限"):
            heartbeat.lease()
    finally:
        heartbeat.close()
        owner.close()


def test_age_is_measured_after_read_when_publish_happens_during_read(tmp_path, monkeypatch):
    clock = [100.0]
    monkeypatch.setattr("th06ai.worker.time", SimpleNamespace(monotonic=lambda: clock[0]))
    owner = SharedLease(create=True)
    class PublishDuringRead:
        def read_checked(self):
            clock[0] += .03125
            owner.publish(clock[0], "GPU-test")
            return owner.read_checked()
    heartbeat = Heartbeat(tmp_path, PublishDuringRead())
    try:
        heartbeat.lease()  # Entry time is older than the new permit timestamp.
        assert heartbeat.data["lease_diagnostics"]["sample_age_seconds"] == 0
    finally:
        heartbeat.close()
        owner.close()


@pytest.mark.parametrize("fault", ["empty", "invalid", "future", "identity"])
def test_cached_permit_never_hides_a_stable_invalid_record(tmp_path, monkeypatch, fault):
    clock = [100.0]
    monkeypatch.setattr("th06ai.worker.time", SimpleNamespace(monotonic=lambda: clock[0]))
    owner = SharedLease(create=True)
    heartbeat = Heartbeat(tmp_path, owner)
    try:
        owner.publish(100.0, "GPU-test")
        heartbeat.lease()
        clock[0] += .05
        if fault == "empty":
            owner.memory.buf[:] = b"\0" * owner.SIZE
        elif fault == "invalid":
            struct.pack_into("<d", owner.memory.buf, 8, float("nan"))
        elif fault == "future":
            owner.publish(110.0, "GPU-test")
        else:
            owner.publish(100.05, "GPU-other")
        with pytest.raises(RuntimeError, match="保护授权已失效"):
            heartbeat.lease()
    finally:
        heartbeat.close()
        owner.close()


def test_real_child_keeps_working_during_concurrent_publications(tmp_path):
    """Actual shared memory across processes, with publication deliberately held open."""
    script = tmp_path / "lease_reader.py"
    ready, finish, result = (tmp_path / name for name in ("READY", "FINISH", "reader-result.json"))
    script.write_text(
        "import json,sys,time\nfrom pathlib import Path\n"
        "from th06ai.lease import SharedLease\nfrom th06ai.worker import Heartbeat\n"
        f"directory=Path({str(tmp_path)!r})\n"
        "lease=SharedLease(sys.argv[1])\nheartbeat=Heartbeat(directory,lease)\n"
        "try:\n"
        " deadline=time.monotonic()+5\n"
        " while not (permit:=lease.read()):\n"
        "  if time.monotonic()>deadline: raise RuntimeError('no initial permit')\n"
        " heartbeat.last_valid_permit=permit\n heartbeat.lease()\n"
        f" Path({str(ready)!r}).touch()\n count=0\n"
        f" while time.monotonic()<deadline and not Path({str(finish)!r}).exists():\n"
        "  heartbeat.last_lease_check=0\n  heartbeat.lease()\n  count+=1\n"
        f" Path({str(result)!r}).write_text(json.dumps({{'checks':count,'busy_reads':heartbeat.lease_busy_reads}}))\n"
        "finally:\n heartbeat.close()\n lease.close()\n",
        encoding="utf-8")
    owner = SharedLease(create=True)
    child = None
    try:
        owner.publish(time.monotonic(), "GPU-test")
        import os
        from pathlib import Path
        child_env = dict(os.environ, PYTHONPATH=str(Path(__file__).resolve().parent.parent))
        child = subprocess.Popen([sys.executable, str(script), owner.name], env=child_env,
                                 stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        deadline = time.monotonic() + 5
        while not ready.exists() and time.monotonic() < deadline and child.poll() is None:
            time.sleep(.01)
        assert ready.exists()
        for _ in range(100):
            # Model the real publisher's mutex transaction. Mutating the
            # header outside the mutex fabricated stable corruption that the
            # production reader correctly rejects, making this test flaky.
            acquire_deadline = time.monotonic() + 1
            while True:
                try:
                    owner._acquire()
                    break
                except LeaseBusy:
                    assert time.monotonic() < acquire_deadline
                    time.sleep(.001)
            try:
                owner.memory.buf[:8] = b"\0" * 8
                time.sleep(.001)
                owner._publish(time.monotonic(), b"GPU-test")
            finally:
                owner._release()
            time.sleep(.001)
        finish.touch()
        out, err = child.communicate(timeout=5)
        assert child.returncode == 0, (out, err)
        data = json.loads(result.read_text())
        assert data["checks"] > 0 and data["busy_reads"] > 0
    finally:
        finish.touch()
        if child and child.poll() is None:
            child.communicate(timeout=6)
        owner.close()


@pytest.mark.skipif(sys.platform != "win32", reason="Windows named mutex transaction")
def test_writer_header_clear_is_hidden_inside_the_mutex_transaction(tmp_path, monkeypatch):
    clock = [100.0]
    monkeypatch.setattr("th06ai.worker.time", SimpleNamespace(monotonic=lambda: clock[0]))
    owner = SharedLease(create=True)
    reader = SharedLease(owner.name)
    heartbeat = Heartbeat(tmp_path, reader)
    entered, finish = threading.Event(), threading.Event()
    errors = []
    def writer():
        owner._acquire()
        try:
            # Reproduce struct.pack_into's intermediate header memset exactly.
            owner.memory.buf[:8] = b"\0" * 8
            entered.set()
            assert finish.wait(2)
            owner._publish(100.05, b"GPU-test")
        except BaseException as exc:
            errors.append(exc)
        finally:
            owner._release()
    thread = None
    try:
        owner.publish(100.0, "GPU-test")
        heartbeat.lease()
        thread = threading.Thread(target=writer)
        thread.start()
        assert entered.wait(2)
        clock[0] = 100.05
        heartbeat.lease()
        assert heartbeat.last_valid_permit["monotonic"] == 100.0
        assert heartbeat.lease_busy_reads == 1
        finish.set()
        thread.join(2)
        assert not errors and not thread.is_alive()
        clock[0] = 100.1
        heartbeat.lease()
        assert heartbeat.last_valid_permit["monotonic"] == 100.05
    finally:
        finish.set()
        if thread:
            thread.join(3)
        heartbeat.close()
        reader.close()
        owner.close()


@pytest.mark.skipif(sys.platform != "win32", reason="Windows abandoned mutex semantics")
def test_abandoned_authorization_mutex_is_never_hidden_by_cached_permit(tmp_path, monkeypatch):
    import ctypes as ct
    clock = [100.0]
    monkeypatch.setattr("th06ai.worker.time", SimpleNamespace(monotonic=lambda: clock[0]))
    owner = SharedLease(create=True)
    reader = SharedLease(owner.name)
    heartbeat = Heartbeat(tmp_path, reader)
    try:
        owner.publish(100.0, "GPU-test")
        heartbeat.lease()
        acquired,exit_owner=threading.Event(),threading.Event()
        def abandon():
            owner._acquire()
            acquired.set()
            assert exit_owner.wait(2)
        thread = threading.Thread(target=abandon)
        thread.start()
        assert acquired.wait(2)
        # CPython 3.10's join sentinel can be released before ExitThread has
        # completed. Wait for the native handle, so this tests an abandoned
        # mutex rather than an ordinary brief WAIT_TIMEOUT during teardown.
        kernel=ct.WinDLL('kernel32',use_last_error=True)
        open_thread=kernel.OpenThread
        open_thread.argtypes=(ct.c_uint32,ct.c_int,ct.c_uint32)
        open_thread.restype=ct.c_void_p
        wait=kernel.WaitForSingleObject
        wait.argtypes=(ct.c_void_p,ct.c_uint32);wait.restype=ct.c_uint32
        close=kernel.CloseHandle
        close.argtypes=(ct.c_void_p,);close.restype=ct.c_int
        handle=open_thread(0x100000,False,thread.native_id)
        assert handle,ct.WinError(ct.get_last_error())
        try:
            exit_owner.set()
            thread.join(2)  # Owner exits without releasing the mutex.
            assert wait(handle,2000)==0
        finally:
            close(handle)
        assert not thread.is_alive()
        clock[0] = 100.05
        with pytest.raises(RuntimeError, match="异常退出"):
            heartbeat.lease()
    finally:
        heartbeat.close()
        reader.close()
        owner.close()
