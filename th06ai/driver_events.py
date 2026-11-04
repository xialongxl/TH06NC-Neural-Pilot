"""Read new Windows display-driver error events; never change event logs."""
from __future__ import annotations
from .i18n import msg

import ctypes as ct
from ctypes import wintypes as wt
from xml.etree import ElementTree

from .win32 import bind, k32, close_handle


class DriverEvents:
    def __init__(self):
        dll = ct.WinDLL("wevtapi", use_last_error=True)
        subscribe = bind(dll, "EvtSubscribe", wt.HANDLE, wt.HANDLE, wt.HANDLE, wt.LPCWSTR,
                         wt.LPCWSTR, wt.HANDLE, ct.c_void_p, ct.c_void_p, wt.DWORD)
        self.next = bind(dll, "EvtNext", wt.BOOL, wt.HANDLE, wt.DWORD,
                         ct.POINTER(wt.HANDLE), wt.DWORD, wt.DWORD, ct.POINTER(wt.DWORD))
        self.render = bind(dll, "EvtRender", wt.BOOL, wt.HANDLE, wt.HANDLE, wt.DWORD,
                           wt.DWORD, ct.c_void_p, ct.POINTER(wt.DWORD), ct.POINTER(wt.DWORD))
        self.evt_close = bind(dll, "EvtClose", wt.BOOL, wt.HANDLE)
        self.wait = bind(k32, "WaitForSingleObject", wt.DWORD, wt.HANDLE, wt.DWORD)
        self.reset_event = bind(k32, "ResetEvent", wt.BOOL, wt.HANDLE)
        create_event = bind(k32, "CreateEventW", wt.HANDLE, ct.c_void_p, wt.BOOL, wt.BOOL, wt.LPCWSTR)
        self.signal = create_event(None, True, False, None)
        if not self.signal:
            raise ct.WinError(ct.get_last_error())
        query = ("*[System[(Provider[@Name='Display'] and EventID=4101) or "
                 "(Provider[@Name='nvlddmkm'] and (Level=1 or Level=2 or Level=3))]]")
        # Let the Windows Event Log service enqueue FUTURE matching events.
        # Re-querying the entire System channel every 250 ms took 0.3–0.5 s.
        self.subscription = subscribe(None, self.signal, "System", query, None, None, None, 1)
        if not self.subscription:
            error = ct.get_last_error()
            close_handle(self.signal)
            self.signal = None
            raise ct.WinError(error)
        self.seen: set[str] = set()

    def poll(self) -> list[dict]:
        wait_result = self.wait(self.signal, 0)
        if wait_result == 0x102:
            return []
        if wait_result != 0:
            raise ct.WinError(ct.get_last_error())
        result = []
        # Reset before draining; a concurrently arriving event stays signalled.
        if not self.reset_event(self.signal):
            raise ct.WinError(ct.get_last_error())
        while True:
            events = (wt.HANDLE * 32)()
            count = wt.DWORD()
            if not self.next(self.subscription, 32, events, 0, 0, ct.byref(count)):
                if ct.get_last_error() == 259:  # ERROR_NO_MORE_ITEMS
                    return result
                raise ct.WinError(ct.get_last_error())
            handles = list(events)[:count.value]
            try:
                for event in handles:
                    used, properties = wt.DWORD(), wt.DWORD()
                    self.render(None, event, 1, 0, None, ct.byref(used), ct.byref(properties))
                    if ct.get_last_error() != 122 or not 0 < used.value < 1024 * 1024:
                        raise RuntimeError(msg('backend.could_not_read_windows_gpu_driver_events'))
                    buf = ct.create_unicode_buffer(used.value // 2 + 1)
                    if not self.render(None, event, 1, ct.sizeof(buf), buf,
                                       ct.byref(used), ct.byref(properties)):
                        raise ct.WinError(ct.get_last_error())
                    doc = ElementTree.fromstring(buf.value)
                    ns = {"e": "http://schemas.microsoft.com/win/2004/08/events/event"}
                    system = doc.find("e:System", ns)
                    record = system.findtext("e:EventRecordID", "", ns)
                    if record not in self.seen:
                        self.seen.add(record)
                        result.append({"record": record,
                                       "provider": system.find("e:Provider", ns).attrib.get("Name"),
                                       "id": system.findtext("e:EventID", "", ns)})
            finally:
                for event in handles:
                    self.evt_close(event)

    def close(self):
        if self.subscription:
            self.evt_close(self.subscription)
            self.subscription = None
        if self.signal:
            close_handle(self.signal)
            self.signal = None
