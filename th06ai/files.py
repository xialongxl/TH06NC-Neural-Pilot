from __future__ import annotations

import json
import os
import sys
import threading
import time
from datetime import datetime
from pathlib import Path


def _json_default(value):
    # NumPy scalars can arrive from reward/metric calculations. Preserve their
    # numeric/boolean JSON types; never stringify the whole status on failure.
    # Do not import NumPy into the independent hardware watchdog.
    numpy = sys.modules.get("numpy")
    if numpy is not None:
        if isinstance(value, numpy.generic):
            return value.item()
        if isinstance(value, numpy.ndarray):
            return value.tolist()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".{os.getpid()}.{threading.get_ident()}.tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=_json_default), encoding="utf-8")
    # A viewer may briefly hold the destination open on Windows. Do not leave
    # the previous sample looking fresh when replacement fails.
    try:
        for attempt in range(24):
            try:
                os.replace(tmp, path)
                break
            except PermissionError:
                if attempt == 23:
                    raise RuntimeError(f"无法更新 {path.name}：文件被占用或权限不足") from None
                time.sleep(0.005)
    finally:
        tmp.unlink(missing_ok=True)


def read_json(path: Path) -> dict:
    try:
        if os.name == "nt":
            # Python's default Windows open() omits FILE_SHARE_DELETE. A GUI
            # reader can then prevent an atomic replacement by its writer.
            import ctypes as ct
            import msvcrt
            from ctypes import wintypes as wt
            kernel = ct.WinDLL("kernel32", use_last_error=True)
            create = kernel.CreateFileW
            create.argtypes = [wt.LPCWSTR, wt.DWORD, wt.DWORD, ct.c_void_p, wt.DWORD, wt.DWORD, wt.HANDLE]
            create.restype = wt.HANDLE
            handle = create(str(path), 0x80000000, 0x7, None, 3, 0x80, None)
            if handle == ct.c_void_p(-1).value:
                return {}
            fd = msvcrt.open_osfhandle(handle, os.O_RDONLY | os.O_BINARY)
            with os.fdopen(fd, "r", encoding="utf-8") as stream:
                return json.load(stream)
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def timestamp() -> str:
    return datetime.now().astimezone().isoformat(timespec="milliseconds")


def new_run(root: Path, mode: str) -> Path:
    path = root / "runs" / (datetime.now().strftime("%Y%m%d-%H%M%S-%f") + "-" + mode)
    path.mkdir(parents=True, exist_ok=False)
    return path
