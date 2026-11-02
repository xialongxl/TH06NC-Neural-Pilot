"""Mock Windows activation and event sources; no real input or game/GPU work."""
from dataclasses import replace
from types import SimpleNamespace
import queue
import threading
import time

import pytest

from th06ai import environment, win32
from th06ai.control import FinishTraining, SamplingInterrupted, TrainingControl, set_paused
from th06ai.environment import TouhouEnv
from th06ai.hotkeys import GlobalHotkeys
from test_planning import frame


class Clock:
    now = 100.
    def sleep(self, seconds):
        self.now += seconds


def native_focus(monkeypatch):
    clock = Clock()
    state = {'foreground': 1, 'activate_at': float('inf'), 'alt': False}
    monkeypatch.setattr(win32, 'time', SimpleNamespace(perf_counter=lambda: clock.now, sleep=clock.sleep))
    monkeypatch.setattr(win32, 'foreground', lambda: 10 if clock.now >= state['activate_at'] else state['foreground'])
    monkeypatch.setattr(win32, 'window_exists', lambda _: True)
    monkeypatch.setattr(win32, 'is_iconic', lambda _: False)
    monkeypatch.setattr(win32, 'get_async_key', lambda _: 0x8000 if state['alt'] else 0)
    monkeypatch.setattr(win32, 'send_message_timeout', lambda *args: 1)
    window = win32.GameWindow(111, 10)
    return window, state, clock


def test_activation_waits_for_the_foreground_change_instead_of_failing_immediately(monkeypatch):
    window, state, clock = native_focus(monkeypatch)
    def request(_):
        state['activate_at'] = clock.now+.06
        return 1
    monkeypatch.setattr(win32, 'set_foreground', request)
    monkeypatch.setattr(win32, 'send_input', lambda *args: pytest.fail('Alt should not be needed'))
    assert window.focus_for_start()
    assert .05 < clock.now-100 < .2
    assert not window.focus_diagnostics['alt_assist']


def test_explicit_activation_assist_is_a_paired_alt_batch(monkeypatch):
    window, state, clock = native_focus(monkeypatch)
    packets = []
    def send(count, events, size):
        packets.append([(item.data.ki.wVk, item.data.ki.dwFlags) for item in events])
        state['can_activate'] = True
        return count
    def request(_):
        if state.get('can_activate'):
            state['foreground'] = 10
            return 1
        return 0
    monkeypatch.setattr(win32, 'send_input', send)
    monkeypatch.setattr(win32, 'set_foreground', request)
    assert window.focus_for_start()
    assert packets == [[(0x12, 0), (0x12, 2)]]


def test_activation_does_not_release_a_physical_alt_or_change_system_settings(monkeypatch):
    window, state, clock = native_focus(monkeypatch)
    state['alt'] = True
    monkeypatch.setattr(win32, 'set_foreground', lambda _: 0)
    monkeypatch.setattr(win32, 'send_input', lambda *args: pytest.fail('physical Alt must be preserved'))
    assert not window.focus_for_start()
    assert clock.now-100 < .5


def focus_env(tmp_path, monkeypatch):
    clock = Clock()
    records, releases = [], []
    window = SimpleNamespace(focused=False, pid=111, focus_for_start=lambda: False)
    env = TouhouEnv.__new__(TouhouEnv)
    env.reader = SimpleNamespace(window=window, process=SimpleNamespace(alive=True))
    env.keys = SimpleNamespace(release=lambda: releases.append(True))
    env.status = lambda **row: records.append(row)
    env.lease = lambda: None
    env._sampling = False
    env.training_control = TrainingControl(tmp_path, env.lease, env.status,
        clock=lambda: clock.now, sleep=clock.sleep)
    monkeypatch.setattr(environment, 'time', SimpleNamespace(monotonic=lambda: clock.now, sleep=clock.sleep))
    return env, window, clock, records, releases


def test_initial_transaction_focus_failure_waits_and_accepts_an_explicit_retry(tmp_path, monkeypatch):
    env, window, clock, records, releases = focus_env(tmp_path, monkeypatch)
    attempts = []
    def activate():
        attempts.append(True)
        if len(attempts) == 2:
            window.focused = True
        return window.focused
    window.focus_for_start = activate
    def status(**row):
        records.append(row)
        if row.get('focus_pending'):
            set_paused(tmp_path, False)
    env.status = status
    env._ensure_transaction_focus()
    assert len(attempts) == 2 and window.focused
    assert releases and not (tmp_path/'PAUSE').exists() and not (tmp_path/'FOCUS').exists()
    assert records[-1]['focus_pending'] is False


def test_focus_wait_excludes_its_time_from_training_budget_and_still_checks_stop(tmp_path, monkeypatch):
    env, window, clock, records, _ = focus_env(tmp_path, monkeypatch)
    env.training_control.start()
    def sleep(seconds):
        clock.sleep(seconds)
        if clock.now >= 100.2:
            window.focused = True
    environment.time.sleep = sleep
    env._acquire_focus()
    assert env.training_control.paused_total >= .2
    assert env.training_control.active_seconds == pytest.approx(0)
    window.focused = False
    env.lease = lambda: (_ for _ in ()).throw(RuntimeError('hardware stop'))
    with pytest.raises(RuntimeError, match='hardware stop'):
        env._acquire_focus()


def test_f7_preempts_an_unsent_action_and_finish_has_priority(tmp_path):
    from test_sampling_timing import game
    env, commands, releases, _ = game(frame(), frame(tick=104))
    env.training_control = TrainingControl(tmp_path, lambda: None, lambda **row: None)
    set_paused(tmp_path, True)
    with pytest.raises(SamplingInterrupted):
        env.step(4)
    assert not commands and releases and env.total_steps == 0
    (tmp_path/'FINISH').touch()
    with pytest.raises(FinishTraining):
        env.step(4)
    assert not commands


def test_pause_reacts_to_the_native_paused_bit_without_a_350ms_pre_wait(monkeypatch):
    clock = Clock()
    held, changes, waits = set(), [], []
    env = TouhouEnv.__new__(TouhouEnv)
    captures = iter((frame(), replace(frame(), paused=True)))
    env.reader = SimpleNamespace(read=lambda: next(captures))
    def send(keys):
        held.update(keys); changes.append(keys)
    env.keys = SimpleNamespace(held=held, set=send, release=held.clear)
    env.status = lambda **row: None
    env._ensure_transaction_focus = lambda: None
    env._wait = lambda seconds: waits.append(seconds) or clock.sleep(seconds)
    monkeypatch.setattr(environment, 'time', SimpleNamespace(monotonic=lambda: clock.now))
    env.pause()
    assert changes == [{'esc'}] and waits == [.005]
    assert env.current.paused and env.managed_pause and not held


def test_confirmed_pause_does_not_activate_or_send_keys_to_an_unfocused_window(monkeypatch):
    env = TouhouEnv.__new__(TouhouEnv)
    releases = []
    env.reader = SimpleNamespace(read=lambda: replace(frame(), paused=True))
    env.keys = SimpleNamespace(release=lambda: releases.append(True))
    env.status = lambda **row: None
    monkeypatch.setattr(environment, 'foreground', lambda: pytest.fail('No focus change is needed'))
    env.pause_for_user()
    assert releases and env.current.paused


class KeySource:
    def __init__(self, hooked=True):
        self.hooked = hooked
        self.events = queue.Queue()
        self.states = {}
        self.closed = False
    def start(self, callback, keys):
        self.callback = callback
        return self.hooked
    def pump(self):
        while True:
            event = self.events.get()
            if event is None:
                return
            self.callback(*event)
    def emit(self, vk, down):
        self.events.put((vk, down))
    def down(self, vk):
        return self.states.get(vk, False)
    def wake(self):
        self.events.put(None)
    def close(self):
        self.closed = True


def wait_until(predicate):
    deadline = time.monotonic()+1
    while not predicate() and time.monotonic() < deadline:
        time.sleep(.001)
    assert predicate()


def test_short_taps_are_queued_and_held_shift_and_autorepeat_do_not_break_f7():
    source, calls = KeySource(), []
    keys = GlobalHotkeys(lambda: None, lambda: calls.append(True), source=source)
    try:
        source.emit(0x10, True)  # AI low-speed Shift is irrelevant to F7/F8 matching.
        for event in ((0x76, True), (0x76, True), (0x76, False), (0x76, True), (0x76, False)):
            source.emit(*event)
        wait_until(lambda: len(calls) == 2)
        assert keys.snapshot()['F7_count'] == 2 and keys.snapshot()['method'] == 'key_events'
    finally:
        keys.close()
    assert source.closed and not keys.thread.is_alive()


def test_f8_dispatch_does_not_queue_behind_a_blocked_f7_callback():
    source, pause_started, release_pause, stopped = KeySource(), threading.Event(), threading.Event(), threading.Event()
    def pause():
        pause_started.set()
        release_pause.wait(2)
    keys = GlobalHotkeys(stopped.set, pause, source=source)
    try:
        source.emit(0x76, True)
        assert pause_started.wait(1)
        source.emit(0x77, True)
        assert stopped.wait(1) and not release_pause.is_set()
        assert keys.snapshot()['F8_count'] == 1
    finally:
        release_pause.set()
        keys.close()


def test_unavailable_native_source_uses_independent_five_ms_edge_polling():
    source, paused = KeySource(hooked=False), threading.Event()
    keys = GlobalHotkeys(lambda: None, paused.set, source=source)
    try:
        source.states[0x76] = True
        assert paused.wait(1)
        assert keys.snapshot()['F7_count'] == 1 and keys.snapshot()['method'] == 'poll_5ms'
    finally:
        keys.close()


def test_f8_wins_over_later_f7_requests():
    source, calls, stopped = KeySource(), [], threading.Event()
    keys = GlobalHotkeys(stopped.set, lambda: calls.append(True), source=source)
    try:
        source.emit(0x77, True); source.emit(0x76, True)
        assert stopped.wait(1)
        wait_until(lambda: keys.snapshot()['F7_count'] == 1)
        assert not calls
    finally:
        keys.close()


def test_f8_terminates_the_owned_job_while_the_guard_loop_is_blocked(tmp_path, monkeypatch):
    from th06ai import supervisor
    from th06ai.config import Config
    from test_safety import healthy
    blocked, killed = threading.Event(), threading.Event()
    class Child:
        pid = 123
        def poll(self): return 90 if killed.is_set() else None
        def wait(self, timeout=None): return 90
        def kill(self): killed.set()
        def close(self): pass
    class Job:
        def start(self, *args): return Child()
        def terminate(self):
            assert blocked.is_set()
            killed.set()
        def close(self): pass
    class Telemetry:
        def __init__(self, *args): self.n = 0
        def poll(self, timeout):
            self.n += 1
            if self.n == 4:
                blocked.set()
                assert killed.wait(1), 'F8 must not wait for the blocked guard loop'
                return None
            return healthy(), {}
        def next(self): pass
        def close(self): pass
    class Listener:
        def __init__(self, stop, pause, **kwargs):
            self.stop_requested = threading.Event()
            def fire():
                assert blocked.wait(1)
                self.stop_requested.set()
                stop()
            self.thread = threading.Thread(target=fire, daemon=True)
            self.thread.start()
        def snapshot(self): return {'F8_count': int(self.stop_requested.is_set())}
        def close(self): self.thread.join(1)
    monkeypatch.setattr(win32, 'OwnedJob', Job)
    monkeypatch.setattr(supervisor, 'AsyncTelemetry', Telemetry)
    result = supervisor.run_guard(Config(game_exe='unused'), 'train', run_dir=tmp_path,
        reader_factory=lambda: None, child_command=['never-launched'], release_fn=lambda: None,
        hotkeys_factory=Listener, verbose=False)
    assert killed.is_set() and result.code == 90 and 'F8' in result.reason
