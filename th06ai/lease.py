"""A fail-closed shared-memory lease; no periodically replaced JSON files."""
from __future__ import annotations

import math
import os
import struct
import ctypes as ct
from multiprocessing.shared_memory import SharedMemory


class LeaseBusy(RuntimeError):
    """A concurrent publish, distinct from an empty or invalid authorization."""


if os.name == "nt":
    _kernel = ct.WinDLL("kernel32", use_last_error=True)
    _create_mutex = _kernel.CreateMutexW
    _create_mutex.argtypes = (ct.c_void_p, ct.c_int, ct.c_wchar_p)
    _create_mutex.restype = ct.c_void_p
    _wait_mutex = _kernel.WaitForSingleObject
    _wait_mutex.argtypes = (ct.c_void_p, ct.c_uint32)
    _wait_mutex.restype = ct.c_uint32
    _release_mutex = _kernel.ReleaseMutex
    _release_mutex.argtypes = (ct.c_void_p,)
    _release_mutex.restype = ct.c_int
    _close_mutex = _kernel.CloseHandle
    _close_mutex.argtypes = (ct.c_void_p,)
    _close_mutex.restype = ct.c_int


class SharedLease:
    SIZE = 128

    def __init__(self, name=None, *, create=False):
        self.memory = SharedMemory(name=name, create=create, size=self.SIZE if create else 0)
        self.owner = create
        self.mutex = None
        if create:
            self.memory.buf[:] = b"\0" * self.SIZE
        if os.name == "nt":
            self.mutex = _create_mutex(None, False, "Local\\TH06AI-Lease-" + self.memory.name)
            if not self.mutex:
                self.memory.close()
                raise ct.WinError(ct.get_last_error())

    def _acquire(self):
        if self.mutex is None:
            return
        result = _wait_mutex(self.mutex, 0)
        if result == 0x102:
            raise LeaseBusy("显卡监控授权正在更新")
        if result == 0x80:
            _release_mutex(self.mutex)
            raise RuntimeError("显卡保护通信持有进程异常退出")
        if result != 0:
            raise ct.WinError(ct.get_last_error())

    def _release(self):
        if self.mutex is not None and not _release_mutex(self.mutex):
            raise ct.WinError(ct.get_last_error())

    @property
    def name(self):
        return self.memory.name

    def publish(self, monotonic: float, uuid: str):
        if not self.owner:
            raise RuntimeError("只允许独立监控进程更新授权")
        encoded = uuid.encode("ascii")
        if not 0 < len(encoded) < 96:
            raise ValueError("GPU 身份长度异常")
        try:
            self._acquire()
        except LeaseBusy:
            # Never block the hardware/F8 watchdog behind a reader. Keep the
            # prior timestamp and retry only on a later healthy GPU sample.
            return False
        try:
            self._publish(monotonic, encoded)
        finally:
            self._release()
        return True

    def _publish(self, monotonic, encoded):
        sequence = struct.unpack_from("<Q", self.memory.buf, 0)[0]
        sequence += 2 if sequence % 2 == 0 else 1
        # struct.pack_into can briefly clear the header before writing it;
        # the named mutex makes this whole record an atomic transaction.
        struct.pack_into("<Q", self.memory.buf, 0, sequence - 1)
        struct.pack_into("<dI", self.memory.buf, 8, monotonic, os.getpid())
        self.memory.buf[24:120] = encoded.ljust(96, b"\0")
        struct.pack_into("<Q", self.memory.buf, 0, sequence)

    def read_checked(self):
        self._acquire()
        try:
            return self._snapshot_checked()
        finally:
            self._release()

    def _snapshot_checked(self):
        for _ in range(3):
            first = struct.unpack_from("<Q", self.memory.buf, 0)[0]
            if not first:
                return {}
            if first % 2:
                continue
            data = bytes(self.memory.buf[8:120])
            second = struct.unpack_from("<Q", self.memory.buf, 0)[0]
            if first == second:
                timestamp, owner_pid = struct.unpack_from("<dI", data, 0)
                try:
                    uuid = data[16:112].split(b"\0", 1)[0].decode("ascii")
                except UnicodeDecodeError:
                    return {}
                if math.isfinite(timestamp) and owner_pid and uuid:
                    return {"monotonic": timestamp, "uuid": uuid, "supervisor_pid": owner_pid}
                return {}
        raise LeaseBusy("显卡监控授权正在更新")

    def read(self):
        """Conservative discovery API used before a worker starts any work."""
        try:
            return self.read_checked()
        except LeaseBusy:
            return {}

    def close(self):
        if self.mutex is not None:
            _close_mutex(self.mutex)
            self.mutex = None
        self.memory.close()
        if self.owner:
            self.memory.unlink()
