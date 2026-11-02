from types import SimpleNamespace

import numpy as np
import pytest

from th06ai import environment
from th06ai.environment import TouhouEnv
from th06ai.native import Frame
from th06ai.observations import ObservationBuilder
from th06ai.rewards import REWARD_VERSION, RewardTracker


class Clock:
    now = 0.0


def make_fake(monkeypatch, resume_delay=.6):
    clock = Clock()
    env = object.__new__(TouhouEnv)
    env.status = lambda **_: None
    env.lease = lambda: None
    env.managed_pause = True
    env._wait = lambda seconds: setattr(clock, "now", clock.now + seconds)
    calls = []
    resume_at = [float("inf")]
    def read():
        tick = 100 + max(0, int((clock.now - resume_at[0]) * 60)) if clock.now >= resume_at[0] else 100
        return Frame(tick, 192, 384, 0, False, False, np.empty((0, 3), np.float32),
                     (8, 16, 368, 416), True, clock.now)
    env.reader = SimpleNamespace(read=read,window=SimpleNamespace(focused=True))
    held=set()
    def press(keys):
        if 'z' in keys and 'z' not in held:
            calls.append('z')
            resume_at[0]=clock.now+resume_delay
        held.clear()
        held.update(keys)
    env.keys = SimpleNamespace(set=press,release=lambda:held.clear(),held=held)
    def tap(key):
        calls.append(key)
        if key == "z":
            resume_at[0] = clock.now + resume_delay
        clock.now += .26
    env.tap = tap
    monkeypatch.setattr(environment, "time", SimpleNamespace(monotonic=lambda: clock.now))
    return env, clock, calls


def test_resume_waits_for_menu_animation_and_real_frames(monkeypatch):
    env, clock, calls = make_fake(monkeypatch)
    env.resume()
    assert calls == ["z"]
    assert clock.now >= .6
    assert env.current.tick == 101
    assert env.managed_pause is False
    assert not env.keys.held


def test_play_attach_can_resume_current_boss_without_restarting_level(monkeypatch):
    env, _, calls = make_fake(monkeypatch)
    env.wait_for_focus = lambda: None
    env.ensure_game_started(restart=False)
    assert calls == ["z"]
    assert env.current.tick == 101


def test_frozen_menu_never_claims_resume_success(monkeypatch):
    env, _, _ = make_fake(monkeypatch, resume_delay=float("inf"))
    with pytest.raises(RuntimeError, match="未能确认游戏恢复"):
        env.resume()
    assert env.managed_pause is True
    assert not env.keys.held


def test_resume_does_not_wait_extra_frames_after_game_is_already_advancing(monkeypatch):
    env,clock,calls=make_fake(monkeypatch,resume_delay=0)
    env.resume()
    assert calls==['z'] and env.current.tick==101
    assert 1/60 <= clock.now < 2/60
    assert not env.keys.held


def test_advancing_timer_during_pause_is_not_enough_to_confirm_resume(monkeypatch):
    from dataclasses import replace
    env,_,_=make_fake(monkeypatch,resume_delay=0)
    first=replace(env.reader.read(),paused=True)
    frames=iter([first,replace(first,tick=101),replace(first,tick=101,paused=False)])
    env.reader.read=lambda:next(frames)
    env.resume()
    assert env.current.tick==101 and not env.current.paused
    assert not env.keys.held


def test_initial_attach_does_not_blindly_confirm_unknown_menus(monkeypatch):
    env, _, calls = make_fake(monkeypatch)
    original = env.reader.read()
    original.stage_active = False
    env.reader.read = lambda: original
    env.wait_for_focus = lambda: None
    with pytest.raises(RuntimeError, match="未识别到已开始的关卡"):
        env.ensure_game_started()
    assert calls == []


def test_restart_uses_observed_nc_order_and_verifies_counter(monkeypatch):
    env, _, calls = make_fake(monkeypatch)
    old = env.reader.read()
    old.tick = 1000
    fresh = Frame(181, 192, 384, 3, False, False, np.empty((0, 3), np.float32),
                  (8, 16, 368, 416), True, 0)
    reset = [False]
    env.pause = lambda: None
    env._frozen = lambda: False
    env.reader.read = lambda: fresh if reset[0] else old
    def press(keys):
        assert keys == {"r"}
        calls.append("r")
        reset[0] = True
    env.keys = SimpleNamespace(set=press, release=lambda: None)
    env._restart()
    assert calls == ["r"]
    assert env.current.tick == 181


def test_real_step_wires_reward_diagnostics_and_keeps_time_limit_nonterminal():
    before = Frame(3598, 8, 384, 3, False, False, np.array([(1, 8, 380)], np.float32),
                   (8, 16, 368, 416), True, 0)
    after = Frame(3600, 8, 384, 3, False, False, before.bullets.copy(), before.bounds, True, 0)
    env = object.__new__(TouhouEnv)
    env.current = before
    captures = iter((before, after))
    env.reader = SimpleNamespace(read=lambda: next(captures), window=SimpleNamespace(pid=1))
    actions = []
    env.keys = SimpleNamespace(set=lambda keys: actions.append(keys), release=lambda: None)
    env.lease = lambda: None
    env._wait = lambda _: None
    status = []
    env.status = lambda **values: status.append(values)
    env.builder = ObservationBuilder()
    env.builder.build(before)
    env.reward_tracker = RewardTracker()
    env.reward_tracker.reset(before)
    env.start_tick = env.total_steps = env.episode_steps = 0
    env.episode_seconds = 60
    env.episode_return = 0.0
    _, reward, terminated, truncated, info = env.step(3)
    assert actions == [{"left", "z"}]
    assert truncated and not terminated
    assert info["reward_version"] == REWARD_VERSION
    assert info["reward_components"]["threat_shaping"] < .1  # no terminal refund at a time limit
    assert info["movement"] == {"dx": 0, "dy": 0, "blocked_fraction": 1}
    assert "wall_push_seconds" in info["episode_diagnostics"]
    assert reward == pytest.approx(sum(info["reward_components"].values()))
    assert status[-1]["reward_version"] == REWARD_VERSION
