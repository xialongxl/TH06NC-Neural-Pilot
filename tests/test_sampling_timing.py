"""Timing recovery with a fake game and CPU only; no OS input is sent."""
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from th06ai import environment, learning
from th06ai.control import SamplingInterrupted, TrainingControl
from th06ai.environment import TouhouEnv
from th06ai.observations import ObservationBuilder
from th06ai.rewards import RewardTracker, SURVIVAL_PER_SECOND, DEATH_COST
from test_learning import LearningFixture
from test_rewards import frame


def game(before, after):
    env = TouhouEnv.__new__(TouhouEnv)
    env.current = before
    captures = iter((before, after))
    env.reader = SimpleNamespace(read=lambda: next(captures, after), window=SimpleNamespace(pid=1))
    actions, releases, statuses = [], [], []
    env.keys = SimpleNamespace(set=lambda keys: actions.append(keys), release=lambda: releases.append(True))
    env.lease = lambda: None
    env._wait = lambda _: None
    env.status = lambda **values: statuses.append(values)
    env.builder = ObservationBuilder()
    env.builder.build(before)
    env.reward_tracker = RewardTracker()
    env.reward_tracker.reset(before)
    env.start_tick = before.tick
    env.episode_ticks = env.total_steps = env.episode_steps = 0
    env.episode_seconds = 180
    env.episode_return = 0.0
    env.training_control = None
    return env, actions, releases, statuses


@pytest.mark.parametrize("ticks", [16, 30, 120, 600])
def test_delayed_sample_uses_actual_elapsed_time_without_stopping_or_capping(ticks):
    before = frame()
    env, actions, _, statuses = game(before, replace(before, tick=before.tick+ticks))
    _, reward, died, truncated, info = env.step(0)
    assert actions == [{"z"}]
    assert not died and not truncated
    assert env.total_steps == 1 and env.episode_ticks == ticks
    assert info["survival_seconds"] == pytest.approx(ticks/60)
    assert info["reward_components"]["survival"] == pytest.approx(SURVIVAL_PER_SECOND*ticks/60)
    assert info["native_tick_before"] == before.tick
    assert info["native_tick_delta"] == ticks
    assert statuses[-1]["native_tick_delta"] == ticks
    assert np.isfinite(reward)


def test_death_during_a_delayed_sample_keeps_the_actual_death_penalty():
    before = frame()
    env, _, _, _ = game(before, replace(before, tick=before.tick+60, state=2))
    _, reward, died, truncated, info = env.step(0)
    assert died and not truncated and reward < 0
    assert info["reward_components"]["death"] == -DEATH_COST
    assert info["survival_seconds"] == 1


def test_long_sample_reaching_episode_time_limit_is_a_normal_truncation():
    before = frame()
    env, _, _, _ = game(before, replace(before, tick=before.tick+120))
    env.episode_seconds = 1
    _, _, died, truncated, info = env.step(0)
    assert truncated and not died
    assert info["episode_diagnostics"]["reward_survival"] == pytest.approx(2*SURVIVAL_PER_SECOND)


def test_clock_reset_reobserves_without_fake_reward_restart_or_lost_progress():
    before, fresh = frame(tick=1000), frame(tick=3, x=120)
    env, actions, releases, statuses = game(before, fresh)
    env.episode_ticks = 120
    env.total_steps = 40
    env.episode_return = .5
    env.reward_tracker.totals["survival"] = .5
    with pytest.raises(SamplingInterrupted):
        env.step(0)
    assert env.current is fresh and env.builder.previous is None
    assert env.reward_tracker.frame is fresh
    assert env.episode_ticks == 120 and env.total_steps == 40 and env.episode_return == .5
    assert env.reward_tracker.totals["survival"] == .5
    assert releases and actions == [{"z"}]  # No R/Esc/menu automation.
    assert statuses[-1]["sampling_timing"]["delta"] == -997
    captures = iter((fresh, replace(fresh, tick=5)))
    env.reader.read = lambda: next(captures)
    _, _, _, _, info = env.step(0)
    assert env.total_steps == 41
    assert info["survival_seconds"] == pytest.approx(122/60)


@pytest.mark.parametrize("when", ["before", "during"])
@pytest.mark.parametrize("state", ["paused", "outside_stage"])
def test_menu_or_scene_transition_requests_pause_instead_of_exiting(tmp_path, when, state):
    before = frame()
    changed = replace(before, paused=state == "paused", stage_active=state != "outside_stage")
    env, actions, releases, _ = game(changed if when == "before" else before, changed)
    env.training_control = TrainingControl(tmp_path, lambda: None, lambda **_: None)
    with pytest.raises(SamplingInterrupted):
        env.step(0)
    assert (tmp_path / "PAUSE").exists() and releases
    assert env.total_steps == 0 and env.episode_return == 0
    assert env.reward_tracker.total_ticks == 0
    if when == "before":
        assert not actions


def test_temporarily_frozen_game_pauses_without_fabricated_survival(tmp_path, monkeypatch):
    before = frame()
    env, _, releases, _ = game(before, before)
    clock = [0.0]
    env._wait = lambda _: clock.__setitem__(0, clock[0]+.1)
    monkeypatch.setattr(environment, "time", SimpleNamespace(monotonic=lambda: clock[0]))
    env.training_control = TrainingControl(tmp_path, lambda: None, lambda **_: None)
    with pytest.raises(SamplingInterrupted):
        env.step(0)
    assert (tmp_path / "PAUSE").exists() and releases
    assert env.reward_tracker.total_ticks == 0 and env.total_steps == 0


class ClockResetGame(LearningFixture):
    def __init__(self):
        super().__init__()
        self.calls = 0

    def step(self, action):
        self.calls += 1
        if self.calls == 8:
            self.current.tick = 0
            self.builder.reset()
            raise SamplingInterrupted()
        return super().step(action)


def test_real_cpu_ppo_preserves_partial_rollout_across_rebase(tmp_path, monkeypatch):
    torch.set_num_threads(1)
    env = ClockResetGame()
    model = learning.make_model(env, "cpu")
    control = TrainingControl(tmp_path, lambda: None, lambda **_: None)
    model.training_control, model.training_game = control, env
    monkeypatch.setattr(learning, "save_model", lambda *args: None)
    callback = learning.TrainingCallback(env, tmp_path, 1, lambda **_: None, lambda: None, control)
    model.learn(512, callback=callback)
    assert env.calls == 513 and model.num_timesteps == 512
    assert model._n_updates == 4 and model.rollout_buffer.full


def test_evaluation_also_recovers_from_clock_rebase(tmp_path):
    env = ClockResetGame()
    env.episode_seconds = 180
    model = SimpleNamespace(predict=lambda *args, **kwargs: (0, None))
    report = learning.evaluate(model, env, tmp_path, lambda **_: None, lambda: None, episodes=1)
    assert len(report["episodes"]) == 1
    assert env.calls == 33 and env.paused
