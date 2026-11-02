"""F7/F8 only. Native key events are queued; hook never does I/O or compute."""
from __future__ import annotations

import ctypes as ct
from ctypes import wintypes as wt
import queue
import threading
import time


class Win32KeyEvents:
    """Low-level events include key-up and work while the AI holds Shift."""
    def __init__(self):
        from . import win32 as w
        self.w = w
        self.get_thread = w.bind(w.k32, 'GetCurrentThreadId', wt.DWORD)
        self.peek = w.bind(w.u32, 'PeekMessageW', wt.BOOL, ct.POINTER(wt.MSG), wt.HWND, wt.UINT, wt.UINT, wt.UINT)
        self.get_message = w.bind(w.u32, 'GetMessageW', wt.BOOL, ct.POINTER(wt.MSG), wt.HWND, wt.UINT, wt.UINT)
        self.translate = w.bind(w.u32, 'TranslateMessage', wt.BOOL, ct.POINTER(wt.MSG))
        self.dispatch = w.bind(w.u32, 'DispatchMessageW', ct.c_ssize_t, ct.POINTER(wt.MSG))
        self.post = w.bind(w.u32, 'PostThreadMessageW', wt.BOOL, wt.DWORD, wt.UINT, wt.WPARAM, wt.LPARAM)
        self.remove = w.bind(w.u32, 'UnhookWindowsHookEx', wt.BOOL, wt.HANDLE)
        self.next_hook = w.bind(w.u32, 'CallNextHookEx', ct.c_ssize_t, wt.HANDLE, ct.c_int, wt.WPARAM, wt.LPARAM)
        self.callback_type = ct.WINFUNCTYPE(ct.c_ssize_t, ct.c_int, wt.WPARAM, wt.LPARAM)
        self.install = w.bind(w.u32, 'SetWindowsHookExW', wt.HANDLE, ct.c_int,
                              self.callback_type, wt.HMODULE, wt.DWORD)
        self.module = w.bind(w.k32, 'GetModuleHandleW', wt.HMODULE, wt.LPCWSTR)
        self.hook = None

    def start(self, on_key, keys):
        message = wt.MSG()
        self.peek(ct.byref(message), None, 0, 0, 0)  # Create thread queue before close can post WM_QUIT.
        self.thread_id = self.get_thread()
        def handle(code, message_id, data):
            if code == 0 and message_id in (0x100, 0x101, 0x104, 0x105):
                vk = ct.cast(data, ct.POINTER(wt.DWORD))[0]
                if vk in keys:
                    on_key(vk, message_id in (0x100, 0x104))
                    return 1  # Reserved only while this AI session owns the listener.
            return self.next_hook(None, code, message_id, data)
        self.callback = self.callback_type(handle)
        self.hook = self.install(13, self.callback, self.module(None), 0)  # WH_KEYBOARD_LL, no DLL injection.
        return bool(self.hook)

    def pump(self):
        message = wt.MSG()
        while True:
            result = self.get_message(ct.byref(message), None, 0, 0)
            if result == 0:
                return
            if result == -1:
                raise OSError(ct.get_last_error(), '快捷键消息循环失败')
            self.translate(ct.byref(message))
            self.dispatch(ct.byref(message))

    def down(self, vk):
        return bool(self.w.get_async_key(vk) & 0x8000)

    def wake(self):
        if getattr(self, 'thread_id', None):
            self.post(self.thread_id, 0x12, 0, 0)  # WM_QUIT

    def close(self):
        if self.hook:
            self.remove(self.hook)
            self.hook = None


class GlobalHotkeys:
    """Independent event pump plus separate F8 and F7 dispatchers.

    F8 must never queue behind a pause callback or a file/GPU operation.
    Only fallback operation polls; the normal listener sleeps in GetMessage.
    """
    def __init__(self, stop, pause, *, enable_pause=True, source=None):
        self.on_stop, self.on_pause = stop, pause
        self.keys = {0x77: 'F8', **({0x76: 'F7'} if enable_pause else {})}
        self.source = source or Win32KeyEvents()
        self.done, self.ready, self.stop_requested = threading.Event(), threading.Event(), threading.Event()
        self.pause_queue = queue.Queue()
        self.lock = threading.Lock()
        self.held = set()
        self.data = {'method': 'starting', 'F7': 'starting' if enable_pause else 'disabled', 'F8': 'starting',
                     'F7_count': 0, 'F8_count': 0}
        self.stop_thread = threading.Thread(target=self._stop_dispatch, daemon=True, name='F8-emergency')
        self.pause_thread = threading.Thread(target=self._pause_dispatch, daemon=True, name='F7-control')
        self.thread = threading.Thread(target=self._run, daemon=True, name='AI-hotkey-events')
        self.stop_thread.start(); self.pause_thread.start(); self.thread.start()
        if not self.ready.wait(1):
            self.close()
            raise RuntimeError('快捷键监听未能启动，未启动训练')

    def _feed(self, vk, down):
        if vk not in self.keys or self.done.is_set():
            return
        with self.lock:
            if not down:
                self.held.discard(vk)
                return
            if vk in self.held:
                return
            self.held.add(vk)
            name = self.keys[vk]
            stamp = time.perf_counter_ns()
            self.data[name+'_count'] += 1
            self.data['last_key'] = name
            self.data[name+'_received_ns'] = stamp
            if vk == 0x77:
                self.stop_requested.set()
            else:
                self.pause_queue.put(stamp)

    def _invoke(self, name, callback, stamp):
        try:
            callback()
        except Exception as error:
            with self.lock:
                self.data[name+'_error'] = str(error)
        finally:
            with self.lock:
                self.data[name+'_dispatch_ms'] = (time.perf_counter_ns()-stamp)/1e6

    def _stop_dispatch(self):
        self.stop_requested.wait()
        if not self.done.is_set():
            with self.lock:
                stamp = self.data.get('F8_received_ns', time.perf_counter_ns())
            self._invoke('F8', self.on_stop, stamp)

    def _pause_dispatch(self):
        while True:
            stamp = self.pause_queue.get()
            if stamp is None or self.done.is_set():
                return
            if not self.stop_requested.is_set():
                self._invoke('F7', self.on_pause, stamp)

    def _run(self):
        try:
            hooked = self.source.start(self._feed, set(self.keys))
            with self.lock:
                method = 'key_events' if hooked else 'poll_5ms'
                self.data.update(method=method, F8=method)
                if 0x76 in self.keys:
                    self.data['F7'] = method
            self.ready.set()
            if hooked:
                self.source.pump()
                if self.done.is_set():
                    return
        except Exception as error:
            with self.lock:
                self.data['listener_error'] = str(error)
        finally:
            self.source.close()
        if not self.done.is_set():
            with self.lock:
                self.data.update(method='poll_5ms', F8='poll_5ms')
                if 0x76 in self.keys:
                    self.data['F7'] = 'poll_5ms'
            self.ready.set()
            while not self.done.is_set():
                for vk in self.keys:
                    self._feed(vk, self.source.down(vk))
                self.done.wait(.005)

    def snapshot(self):
        with self.lock:
            return {key: value for key, value in self.data.items() if not key.endswith('_ns')}

    def close(self):
        self.done.set()
        self.stop_requested.set()
        self.pause_queue.put(None)
        if self.thread.is_alive():
            self.source.wake()
        for thread in (self.thread, self.stop_thread, self.pause_thread):
            if thread is not threading.current_thread():
                thread.join(.3)
