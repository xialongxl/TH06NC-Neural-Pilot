"""On-demand CPU telemetry for the local console."""
from __future__ import annotations

from collections import deque
import ctypes as ct
from ctypes import wintypes as wt
from dataclasses import dataclass
from functools import cache
import os
from threading import Lock
import time

SAMPLE_SECONDS = 1.0
MAX_SAMPLE_GAP = 4.0
HISTORY_LIMIT = 90
SYSTEM_PROCESSOR_PERFORMANCE_INFORMATION = 8
ALL_PROCESSOR_GROUPS = 0xffff


class _ProcessorTimes(ct.Structure):
    _fields_ = [('idle', ct.c_longlong), ('kernel', ct.c_longlong), ('user', ct.c_longlong),
                ('reserved1', ct.c_longlong * 2), ('reserved2', wt.ULONG)]


@dataclass(frozen=True, slots=True)
class CpuTimes:
    processors: int
    total: tuple[int, int]  # Idle ticks, all ticks (the kernel count already includes idle).
    cores: tuple[tuple[int, int], ...] = ()


@cache
def _windows_queries():
    from .win32 import bind, k32

    active_count = bind(k32, 'GetActiveProcessorCount', wt.DWORD, wt.WORD)
    group_count = bind(k32, 'GetActiveProcessorGroupCount', wt.WORD)
    system = bind(k32, 'GetSystemTimes', wt.BOOL, *[ct.POINTER(wt.FILETIME)] * 3)
    try:
        cores = bind(ct.WinDLL('ntdll'), 'NtQuerySystemInformation', wt.LONG, wt.ULONG,
                     ct.c_void_p, wt.ULONG, ct.POINTER(wt.ULONG))
    except (AttributeError, OSError):
        cores = None
    return active_count, group_count, system, cores


def _filetime_ticks(value):
    return (value.dwHighDateTime << 32) | value.dwLowDateTime


def read_cpu_times():
    if os.name != 'nt':
        return None
    active_count, group_count, system, read_cores = _windows_queries()
    processors = active_count(ALL_PROCESSOR_GROUPS)
    if not 0 < processors <= 4096:
        return None
    if read_cores is not None:
        # Validate the returned layout and fall back if this optional NT counter is unavailable.
        rows = (_ProcessorTimes * processors)()
        size = wt.ULONG()
        status = read_cores(SYSTEM_PROCESSOR_PERFORMANCE_INFORMATION, rows, ct.sizeof(rows), ct.byref(size))
        if status == 0 and size.value == ct.sizeof(rows):
            cores = tuple((row.idle, row.kernel + row.user) for row in rows)
            if all(0 <= idle <= total for idle, total in cores):
                return CpuTimes(processors, (sum(row[0] for row in cores), sum(row[1] for row in cores)), cores)
    # GetSystemTimes can be group-local. A partial system reading would mislabel the total.
    if group_count() != 1:
        return None
    idle, kernel, user = wt.FILETIME(), wt.FILETIME(), wt.FILETIME()
    if not system(ct.byref(idle), ct.byref(kernel), ct.byref(user)):
        return None
    return CpuTimes(processors, (_filetime_ticks(idle), _filetime_ticks(kernel) + _filetime_ticks(user)))


def _busy_percent(previous, current):
    idle = current[0] - previous[0]
    total = current[1] - previous[1]
    if total > 0 and 0 <= idle <= total:
        return round(100 * (total - idle) / total, 1)
    return None


class CpuMonitor:
    def __init__(self, *, read=None, clock=time.monotonic):
        self._read, self._clock = read or read_cpu_times, clock
        self._lock = Lock()
        self._previous = self._sampled_at = self._snapshot = None
        self._history = deque(maxlen=HISTORY_LIMIT)

    def snapshot(self):
        with self._lock:
            now = self._clock()
            if self._sampled_at is not None:
                elapsed = now - self._sampled_at
                if 0 <= elapsed < SAMPLE_SECONDS:
                    return self._snapshot
                if not 0 <= elapsed <= MAX_SAMPLE_GAP:
                    self._previous = None
            try:
                times = self._read()
            except (OSError, AttributeError):
                times = None
            previous = self._previous
            self._previous, self._sampled_at = times, now
            usage = busiest = None
            if (times is not None and previous is not None and times.processors == previous.processors
                    and len(times.cores) == len(previous.cores)):
                usage = _busy_percent(previous.total, times.total)
                core_loads = [_busy_percent(old, new) for old, new in zip(previous.cores, times.cores)]
                if core_loads and all(value is not None for value in core_loads):
                    busiest = max(core_loads)
            if usage is None:
                self._history.clear()
            else:
                self._history.append(usage)
            status = 'unavailable' if times is None else 'warming' if usage is None else 'live'
            self._snapshot = {'status': status, 'usage_percent': usage, 'busiest_core_percent': busiest,
                              'logical_processors': times.processors if times is not None else None,
                              'history': tuple(self._history)}
            return self._snapshot
