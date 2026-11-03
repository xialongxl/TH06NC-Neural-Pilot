"""Small Windows API boundary: process READ only, ordinary input, owned jobs."""
from __future__ import annotations

import ctypes as ct
import os
import time
import threading
from ctypes import wintypes as wt
from dataclasses import dataclass

if os.name != "nt":
    raise RuntimeError("实机控制需要 Windows 10/11")

k32 = ct.WinDLL("kernel32", use_last_error=True)
u32 = ct.WinDLL("user32", use_last_error=True)
INVALID_HANDLE = ct.c_void_p(-1).value


def bind(dll, name, result, *args):
    fn = getattr(dll, name)
    fn.restype = result
    fn.argtypes = list(args)
    return fn


close_handle = bind(k32, "CloseHandle", wt.BOOL, wt.HANDLE)
snapshot = bind(k32, "CreateToolhelp32Snapshot", wt.HANDLE, wt.DWORD, wt.DWORD)
open_process = bind(k32, "OpenProcess", wt.HANDLE, wt.DWORD, wt.BOOL, wt.DWORD)
read_memory = bind(k32, "ReadProcessMemory", wt.BOOL, wt.HANDLE, ct.c_void_p, ct.c_void_p,
                   ct.c_size_t, ct.POINTER(ct.c_size_t))


class PROCESSENTRY32W(ct.Structure):
    _fields_ = [("dwSize", wt.DWORD), ("cntUsage", wt.DWORD), ("th32ProcessID", wt.DWORD),
                ("th32DefaultHeapID", ct.c_size_t), ("th32ModuleID", wt.DWORD),
                ("cntThreads", wt.DWORD), ("th32ParentProcessID", wt.DWORD),
                ("pcPriClassBase", wt.LONG), ("dwFlags", wt.DWORD), ("szExeFile", wt.WCHAR * 260)]


class MODULEENTRY32W(ct.Structure):
    _fields_ = [("dwSize", wt.DWORD), ("th32ModuleID", wt.DWORD), ("th32ProcessID", wt.DWORD),
                ("GlblcntUsage", wt.DWORD), ("ProccntUsage", wt.DWORD),
                ("modBaseAddr", ct.c_void_p), ("modBaseSize", wt.DWORD), ("hModule", wt.HMODULE),
                ("szModule", wt.WCHAR * 256), ("szExePath", wt.WCHAR * 260)]


def processes() -> list[tuple[int, str]]:
    handle = snapshot(0x2, 0)
    if handle == INVALID_HANDLE:
        raise ct.WinError(ct.get_last_error())
    first = bind(k32, "Process32FirstW", wt.BOOL, wt.HANDLE, ct.POINTER(PROCESSENTRY32W))
    next_ = bind(k32, "Process32NextW", wt.BOOL, wt.HANDLE, ct.POINTER(PROCESSENTRY32W))
    entry = PROCESSENTRY32W(dwSize=ct.sizeof(PROCESSENTRY32W))
    result = []
    try:
        ok = first(handle, ct.byref(entry))
        while ok:
            result.append((int(entry.th32ProcessID), entry.szExeFile))
            ok = next_(handle, ct.byref(entry))
    finally:
        close_handle(handle)
    return result


def process_path(pid):
    """Identity check for window activation; no game memory/input handle."""
    handle = open_process(0x1000, False, pid)
    if not handle:
        raise ct.WinError(ct.get_last_error())
    try:
        size = wt.DWORD(32768)
        value = ct.create_unicode_buffer(size.value)
        query = bind(k32, 'QueryFullProcessImageNameW', wt.BOOL, wt.HANDLE, wt.DWORD,
                     wt.LPWSTR, ct.POINTER(wt.DWORD))
        if not query(handle, 0, value, ct.byref(size)):
            raise ct.WinError(ct.get_last_error())
        return value.value
    finally:
        close_handle(handle)


def module_info(pid: int, name: str) -> tuple[int, int, str]:
    handle = snapshot(0x8 | 0x10, pid)
    if handle == INVALID_HANDLE:
        raise ct.WinError(ct.get_last_error())
    first = bind(k32, "Module32FirstW", wt.BOOL, wt.HANDLE, ct.POINTER(MODULEENTRY32W))
    next_ = bind(k32, "Module32NextW", wt.BOOL, wt.HANDLE, ct.POINTER(MODULEENTRY32W))
    entry = MODULEENTRY32W(dwSize=ct.sizeof(MODULEENTRY32W))
    try:
        ok = first(handle, ct.byref(entry))
        while ok:
            if entry.szModule.casefold() == name.casefold():
                return int(entry.modBaseAddr), int(entry.modBaseSize), entry.szExePath
            ok = next_(handle, ct.byref(entry))
    finally:
        close_handle(handle)
    raise RuntimeError(f"进程中没有找到 {name}")


class ReadOnlyProcess:
    def __init__(self, pid: int, module: str):
        self.pid = pid
        self.base, self.size, self.path = module_info(pid, module)
        # Deliberately no PROCESS_VM_WRITE/OPERATION, no injection/debug rights.
        self.handle = open_process(0x10 | 0x1000, False, pid)
        if not self.handle:
            raise ct.WinError(ct.get_last_error())
        self.configure_reads('auto')

    def configure_reads(self, backend):
        if backend not in ('auto', 'python', 'rust'):
            raise ValueError('capture_backend 必须是 auto、python 或 rust')
        from .acceleration import extension
        self._native_reader = None if backend == 'python' else extension()
        if backend == 'rust' and self._native_reader is None:
            raise RuntimeError('Rust 只读采集扩展不可用，请构建 ABI v3 或使用 auto/python')
        # Large owned-result copies regress the warm bulk read benchmark.
        # Auto keeps read_into buffers and batches only small phase/control reads.
        self.native_batch = backend == 'rust'
        self.last_batch_timing = {}

    def read(self, rva: int, size: int) -> bytes:
        buf = bytearray(size)
        self.read_into(rva, buf)
        return bytes(buf)

    def read_many(self, requests):
        requests = tuple(requests)
        for rva, size in requests:
            if rva < 0 or size <= 0 or rva+size > self.size:
                raise ValueError('读取范围不在已验证的游戏主模块内')
        module = getattr(self, '_native_reader', None)
        if module is None:
            return tuple(self.read(rva, size) for rva, size in requests)
        blocks, compute, handoff = module.read_many(int(self.handle), [(self.base+rva,size) for rva,size in requests])
        self.last_batch_timing = {'native_compute_ms': compute/1e6, 'native_handoff_ms': handoff/1e6}
        return blocks

    def read_indirect(self, pointer_rva: int, offset: int, size: int) -> bytes | None:
        """Read a pinned game's heap object; never grant write/operation rights."""
        if not 0 <= offset <= 0x20000 or not 0 < size <= 0x20000-offset:
            raise ValueError('间接读取范围异常')
        if pointer_rva < 0 or pointer_rva+8 > self.size:
            raise ValueError('读取范围不在已验证的游戏主模块内')
        module = getattr(self, '_native_reader', None)
        if module is not None:
            return module.read_indirect_bytes(int(self.handle), self.base+pointer_rva, offset, size)
        address=int.from_bytes(self.read(pointer_rva,8),'little')
        if not address:
            return None
        if not 0x10000 <= address <= 0x7fffffffffff-offset-size:
            raise RuntimeError('游戏对象指针异常')
        buf=ct.create_string_buffer(size)
        count=ct.c_size_t()
        if not read_memory(self.handle,address+offset,buf,size,ct.byref(count)) or count.value!=size:
            raise RuntimeError('只读游戏菜单对象失败')
        return buf.raw

    @property
    def alive(self):
        code = wt.DWORD()
        get_code = bind(k32, 'GetExitCodeProcess', wt.BOOL, wt.HANDLE, ct.POINTER(wt.DWORD))
        return bool(self.handle and get_code(self.handle, ct.byref(code)) and code.value == 259)

    def read_into(self, rva: int, target) -> None:
        """Fill a reusable writable buffer without allocating/copying a new blob."""
        size = len(target)
        if rva < 0 or size <= 0 or rva + size > self.size:
            raise ValueError("读取范围不在已验证的游戏主模块内")
        buf = (ct.c_ubyte * size).from_buffer(target)
        count = ct.c_size_t()
        if not read_memory(self.handle, self.base + rva, buf, size, ct.byref(count)) or count.value != size:
            raise RuntimeError(f"只读游戏状态失败 RVA=0x{rva:x} size={size}")

    def close(self):
        if self.handle:
            close_handle(self.handle)
            self.handle = None


try:
    bind(u32, "SetProcessDpiAwarenessContext", wt.BOOL, ct.c_void_p)(ct.c_void_p(-4))
except AttributeError:
    pass

foreground = bind(u32, "GetForegroundWindow", wt.HWND)
get_async_key = bind(u32, "GetAsyncKeyState", wt.SHORT, ct.c_int)
get_window_pid = bind(u32, "GetWindowThreadProcessId", wt.DWORD, wt.HWND, ct.POINTER(wt.DWORD))
is_visible = bind(u32, "IsWindowVisible", wt.BOOL, wt.HWND)
is_iconic = bind(u32, "IsIconic", wt.BOOL, wt.HWND)
get_client_rect = bind(u32, "GetClientRect", wt.BOOL, wt.HWND, ct.POINTER(wt.RECT))
client_to_screen = bind(u32, "ClientToScreen", wt.BOOL, wt.HWND, ct.POINTER(wt.POINT))
set_foreground = bind(u32, "SetForegroundWindow", wt.BOOL, wt.HWND)
show_window_async = bind(u32, "ShowWindowAsync", wt.BOOL, wt.HWND, ct.c_int)
window_exists = bind(u32, "IsWindow", wt.BOOL, wt.HWND)
allow_foreground = bind(u32, "AllowSetForegroundWindow", wt.BOOL, wt.DWORD)
send_message_timeout = bind(u32, "SendMessageTimeoutW", ct.c_size_t, wt.HWND, wt.UINT,
                            wt.WPARAM, wt.LPARAM, wt.UINT, wt.UINT, ct.POINTER(ct.c_size_t))


@dataclass
class GameWindow:
    pid: int
    hwnd: int

    @classmethod
    def find(cls, pid: int):
        candidates = []
        callback_type = ct.WINFUNCTYPE(wt.BOOL, wt.HWND, wt.LPARAM)

        @callback_type
        def collect(hwnd, _):
            owner = wt.DWORD()
            get_window_pid(hwnd, ct.byref(owner))
            if owner.value == pid and is_visible(hwnd):
                rect = wt.RECT()
                get_client_rect(hwnd, ct.byref(rect))
                if rect.right - rect.left >= 320 and rect.bottom - rect.top >= 240:
                    candidates.append(hwnd)
            return True

        enum = bind(u32, "EnumWindows", wt.BOOL, callback_type, wt.LPARAM)
        enum(collect, 0)
        if len(candidates) != 1:
            raise RuntimeError(f"需要一个可见游戏窗口，目前找到 {len(candidates)} 个")
        return cls(pid, candidates[0])

    @property
    def focused(self) -> bool:
        return foreground() == self.hwnd and not is_iconic(self.hwnd)

    def rect(self) -> dict[str, int]:
        rect, point = wt.RECT(), wt.POINT(0, 0)
        if not get_client_rect(self.hwnd, ct.byref(rect)) or not client_to_screen(self.hwnd, ct.byref(point)):
            raise RuntimeError("无法读取游戏窗口位置")
        return {"left": point.x, "top": point.y, "width": rect.right, "height": rect.bottom}

    def focus_for_start(self) -> bool:
        """Bounded activation on an explicit start/resume/pause transaction.

        Do not attach input queues: an unresponsive window could then hang
        the caller. GetForegroundWindow is checked after the target has had
        time to process activation, not immediately after the request.
        """
        if self.focused:
            return True
        started = time.perf_counter()
        self.focus_diagnostics = {'target_pid': self.pid, 'target_hwnd': int(self.hwnd), 'alt_assist': False}
        if not window_exists(self.hwnd):
            try:
                self.hwnd = self.find(self.pid).hwnd
            except RuntimeError:
                return False
        if is_iconic(self.hwnd):
            show_window_async(self.hwnd, 9)
        self.focus_diagnostics['target_hwnd'] = int(self.hwnd)
        for attempt in range(2):
            if attempt and not (get_async_key(0x12) & 0x8000):
                # A neutral, paired Alt gesture permits normal activation on
                # Windows. Never release a physically held modifier, alter a
                # foreground lock setting, or send a game action to another app.
                events = (INPUT*2)(
                    INPUT(type=1, data=INPUTUNION(ki=KEYBDINPUT(0x12, 0x38, 0, 0, 0))),
                    INPUT(type=1, data=INPUTUNION(ki=KEYBDINPUT(0x12, 0x38, 2, 0, 0))))
                self.focus_diagnostics['alt_assist'] = send_input(2, events, ct.sizeof(INPUT)) == 2
                if not self.focus_diagnostics['alt_assist']:
                    send_input(1, ct.byref(events[1]), ct.sizeof(INPUT))
            accepted = bool(set_foreground(self.hwnd))
            result = ct.c_size_t()
            send_message_timeout(self.hwnd, 0, 0, 0, 2, 50, ct.byref(result))  # WM_NULL, ABORTIFHUNG
            deadline = time.perf_counter()+.2
            while not self.focused and time.perf_counter() < deadline:
                checkpoint = getattr(self, 'checkpoint', None)
                if checkpoint is not None:
                    checkpoint()
                time.sleep(.01)
            self.focus_diagnostics.update(request_accepted=accepted, foreground_hwnd=int(foreground() or 0),
                                          elapsed_ms=round((time.perf_counter()-started)*1000, 2))
            if self.focused:
                return True
        return False


class KEYBDINPUT(ct.Structure):
    _fields_ = [("wVk", wt.WORD), ("wScan", wt.WORD), ("dwFlags", wt.DWORD),
                ("time", wt.DWORD), ("dwExtraInfo", ct.c_size_t)]


class MOUSEINPUT(ct.Structure):
    _fields_ = [("dx", wt.LONG), ("dy", wt.LONG), ("mouseData", wt.DWORD),
                ("dwFlags", wt.DWORD), ("time", wt.DWORD), ("dwExtraInfo", ct.c_size_t)]


class INPUTUNION(ct.Union):
    _fields_ = [("ki", KEYBDINPUT), ("mi", MOUSEINPUT)]


class INPUT(ct.Structure):
    _fields_ = [("type", wt.DWORD), ("data", INPUTUNION)]


send_input = bind(u32, "SendInput", wt.UINT, wt.UINT, ct.POINTER(INPUT), ct.c_int)
KEYS = {"z": (0x2C, False), "x": (0x2D, False), "shift": (0x2A, False), "up": (0x48, True),
        "down": (0x50, True), "left": (0x4B, True), "right": (0x4D, True),
        "esc": (0x01, False), "enter": (0x1C, False), "r": (0x13, False), "q": (0x10, False)}
VIRTUAL_KEYS = {"z": 0x5A, "x": 0x58, "shift": 0x10, "up": 0x26, "down": 0x28,
                "left": 0x25, "right": 0x27, "esc": 0x1B, "enter": 0x0D, "r": 0x52, "q": 0x51}


def _key_input(name: str, down: bool):
    scan, extended = KEYS[name]
    # NC's keyboard path also consumes virtual-key state. Supply VK and scan
    # together instead of a zero-VK, scancode-only packet.
    flags = (0x1 if extended else 0) | (0 if down else 0x2)
    vk = VIRTUAL_KEYS[name]
    if extended:  # Preserve real extended-arrow scan codes for DirectInput.
        flags |= 0x8
        vk = 0
    return INPUT(type=1, data=INPUTUNION(ki=KEYBDINPUT(vk, scan, flags, 0, 0)))


def key_events(changes):
    """One ordered OS input transaction, with a best-effort failure release."""
    changes = tuple(changes)
    events = (INPUT * len(changes))(*(_key_input(name, down) for name, down in changes))
    started = time.perf_counter_ns()
    sent = send_input(len(events), events, ct.sizeof(INPUT)) if events else 0
    completed = time.perf_counter_ns()
    if sent != len(events):
        # A partial batch may already have pressed some keys. Never retain
        # those as an apparently successful held-key state.
        releases = (INPUT * len(changes))(*(_key_input(name, False) for name, _ in changes))
        if releases:
            send_input(len(releases), releases, ct.sizeof(INPUT))
        raise RuntimeError("键盘输入失败；请确保游戏与 AI 使用相同权限等级")
    return {"started_ns": started, "completed_ns": completed, "events": len(events),
            "send_ms": (completed-started)/1e6}


def key_event(name: str, down: bool):
    return key_events(((name, down),))


def release_all():
    try:
        key_events((name, False) for name in KEYS)
    except Exception:
        pass


class KeyController:
    def __init__(self, window: GameWindow):
        self.window = window
        self.held: set[str] = set()
        self._lock = threading.RLock()

    def reconcile(self,name):
        # SendInput does not reset keyboard state. A real key-up (e.g. during
        # dialogue) can invalidate our logical cache without changing desired Z.
        with self._lock:
            if name in self.held and not get_async_key(VIRTUAL_KEYS[name]) & 0x8000:
                self.held.discard(name)

    def set(self, keys: set[str]):
        with self._lock:
            return self._set_locked(keys)

    def _set_locked(self, keys):
        keys = set(keys)
        if not keys <= KEYS.keys():
            raise ValueError("不支持的游戏按键")
        if not self.window.focused:
            self.release()
            raise GameFocusLost("游戏失去焦点，已经停止发送按键")
        # Deterministic order: release removed keys, set Shift, then movement
        # and buttons. The entire sequence is one SendInput call.
        removed, added = self.held-keys, keys-self.held
        changes = [(name, False) for name in KEYS if name in removed]
        changes += [(name, True) for name in ("shift", *[n for n in KEYS if n != "shift"])
                    if name in added]
        try:
            timing = key_events(changes)
        except Exception:
            self.release()
            raise
        self.held = set(keys)
        return timing

    def release(self):
        with self._lock:
            return self._release_locked()

    def _release_locked(self):
        try:
            return key_events((name, False) for name in KEYS if name in self.held)
        finally:
            self.held.clear()


class GameFocusLost(RuntimeError):
    pass


def restore_foreground(hwnd):
    if hwnd:
        bind(u32, "SetForegroundWindow", wt.BOOL, wt.HWND)(hwnd)


class IOCOUNTERS(ct.Structure):
    _fields_ = [(n, ct.c_uint64) for n in ("ReadOperationCount", "WriteOperationCount", "OtherOperationCount",
                                        "ReadTransferCount", "WriteTransferCount", "OtherTransferCount")]


class BASICLIMIT(ct.Structure):
    _fields_ = [("PerProcessUserTimeLimit", ct.c_int64), ("PerJobUserTimeLimit", ct.c_int64),
                ("LimitFlags", wt.DWORD), ("MinimumWorkingSetSize", ct.c_size_t),
                ("MaximumWorkingSetSize", ct.c_size_t), ("ActiveProcessLimit", wt.DWORD),
                ("Affinity", ct.c_size_t), ("PriorityClass", wt.DWORD), ("SchedulingClass", wt.DWORD)]


class EXTENDEDLIMIT(ct.Structure):
    _fields_ = [("BasicLimitInformation", BASICLIMIT), ("IoInfo", IOCOUNTERS),
                ("ProcessMemoryLimit", ct.c_size_t), ("JobMemoryLimit", ct.c_size_t),
                ("PeakProcessMemoryUsed", ct.c_size_t), ("PeakJobMemoryUsed", ct.c_size_t)]


class OwnedJob:
    """Closing the supervisor (including a crash) kills only its AI workers."""
    def __init__(self):
        create = bind(k32, "CreateJobObjectW", wt.HANDLE, ct.c_void_p, wt.LPCWSTR)
        set_info = bind(k32, "SetInformationJobObject", wt.BOOL, wt.HANDLE, ct.c_int, ct.c_void_p, wt.DWORD)
        self.handle = create(None, None)
        if not self.handle:
            raise ct.WinError(ct.get_last_error())
        info = EXTENDEDLIMIT()
        info.BasicLimitInformation.LimitFlags = 0x2000
        if not set_info(self.handle, 9, ct.byref(info), ct.sizeof(info)):
            self.close()
            raise ct.WinError(ct.get_last_error())

    def assign(self, process):
        fn = bind(k32, "AssignProcessToJobObject", wt.BOOL, wt.HANDLE, wt.HANDLE)
        if not fn(self.handle, int(process._handle)):
            raise ct.WinError(ct.get_last_error())

    def start(self, command, cwd, env, log):
        """Assign a SUSPENDED child before any launcher can create descendants.

        Windows venv python.exe is a redirector. Assigning a running Popen
        process has a race: its real Python child might escape the Job Object.
        """
        import _winapi
        import msvcrt
        import subprocess
        startup = subprocess.STARTUPINFO()
        startup.dwFlags = subprocess.STARTF_USESTDHANDLES
        output = msvcrt.get_osfhandle(log.fileno())
        with open(os.devnull, "rb") as null:
            input_handle = msvcrt.get_osfhandle(null.fileno())
            output_inherit = os.get_handle_inheritable(output)
            os.set_handle_inheritable(output, True)
            os.set_handle_inheritable(input_handle, True)
            startup.hStdInput = input_handle
            startup.hStdOutput = startup.hStdError = output
            try:
                hp, ht, pid, _ = _winapi.CreateProcess(
                    command[0], subprocess.list2cmdline(command), None, None, True,
                    0x4 | 0x400 | 0x08000000, env, str(cwd), startup)
                child = ManagedProcess(hp, pid)
                try:
                    self.assign(child)
                    resume = bind(k32, "ResumeThread", wt.DWORD, wt.HANDLE)
                    if resume(ht) == 0xFFFFFFFF:
                        raise ct.WinError(ct.get_last_error())
                except BaseException:
                    child.kill()
                    child.close()
                    raise
                finally:
                    close_handle(ht)
                return child
            finally:
                os.set_handle_inheritable(output, output_inherit)

    def terminate(self):
        fn = bind(k32, "TerminateJobObject", wt.BOOL, wt.HANDLE, wt.UINT)
        if self.handle and not fn(self.handle, 90):
            raise ct.WinError(ct.get_last_error())

    def close(self):
        if self.handle:
            close_handle(self.handle)
            self.handle = None


class ManagedProcess:
    def __init__(self, handle, pid):
        self._handle = handle
        self.pid = pid

    def poll(self):
        wait = bind(k32, "WaitForSingleObject", wt.DWORD, wt.HANDLE, wt.DWORD)
        if wait(self._handle, 0) == 0x102:
            return None
        value = wt.DWORD()
        get_code = bind(k32, "GetExitCodeProcess", wt.BOOL, wt.HANDLE, ct.POINTER(wt.DWORD))
        if not get_code(self._handle, ct.byref(value)):
            raise ct.WinError(ct.get_last_error())
        return int(value.value)

    def wait(self, timeout=None):
        import subprocess
        wait = bind(k32, "WaitForSingleObject", wt.DWORD, wt.HANDLE, wt.DWORD)
        milliseconds = 0xFFFFFFFF if timeout is None else int(timeout * 1000)
        if wait(self._handle, milliseconds) == 0x102:
            raise subprocess.TimeoutExpired(str(self.pid), timeout)
        return self.poll()

    def kill(self):
        fn = bind(k32, "TerminateProcess", wt.BOOL, wt.HANDLE, wt.UINT)
        if self.poll() is None:
            fn(self._handle, 90)

    def close(self):
        if self._handle:
            close_handle(self._handle)
            self._handle = None


class GuardLock:
    def __init__(self, name: str):
        create = bind(k32, "CreateMutexW", wt.HANDLE, ct.c_void_p, wt.BOOL, wt.LPCWSTR)
        self.handle = create(None, True, name)
        error = ct.get_last_error()
        if not self.handle:
            raise ct.WinError(error)
        if error == 183:  # ERROR_ALREADY_EXISTS
            close_handle(self.handle)
            self.handle = None
            raise RuntimeError("本项目已有监控或 AI 进程运行，请先停止当前进程")

    def close(self):
        if self.handle:
            release = bind(k32, "ReleaseMutex", wt.BOOL, wt.HANDLE)
            release(self.handle)
            close_handle(self.handle)
            self.handle = None
