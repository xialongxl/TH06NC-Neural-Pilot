"""Exercise a real PPO parameter update without game input or GPU load."""
import time
import csv
import json

import gymnasium as gym
import numpy as np
import torch

from th06ai import learning
from th06ai.native import Frame
from th06ai.observations import ACTION_COUNT, ObservationBuilder, make_observation_space
from th06ai.observations import OBSERVATION_VERSION
from th06ai.rewards import REWARD_VERSION


class LearningFixture(gym.Env):
    def __init__(self):
        self.observation_space = make_observation_space()
        self.action_space = gym.spaces.Discrete(ACTION_COUNT)
        self.builder = ObservationBuilder()
        self.paused = False
        self.current = Frame(0, 192, 384, 0, False, False, np.empty((0, 3), np.float32),
                             (8, 16, 368, 416), True, time.monotonic())
        self.local_steps = 0

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.local_steps = 0
        self.current.tick = 0
        self.builder.reset()
        return self.builder.build(self.current), {}

    def pause(self):
        self.paused = True

    def resume(self):
        self.paused = False

    def step(self, action):
        assert not self.paused, "samples must not be collected during an update pause"
        self.local_steps += 1
        self.current.tick += 2
        return self.builder.build(self.current), float(action == 9), self.local_steps == 32, False, {
            "survival_seconds": self.local_steps / 30, "died": False,
            "reward_version": REWARD_VERSION, "episode_diagnostics": {"wall_push_seconds": .75, "mean_threat": .2}}


def test_real_ppo_update_pause_resume_and_checkpoint_are_wired(tmp_path, monkeypatch):
    torch.set_num_threads(1)
    env = LearningFixture()
    model = learning.make_model(env, device="cpu")
    before = [p.detach().clone() for p in model.policy.parameters()]
    checkpoints = []
    real_save = learning.save_model
    def save(learner, path, run_dir, steps):
        assert learner._n_updates > 0
        assert env.paused
        checkpoints.append(int(steps))
    monkeypatch.setattr(learning, "save_model", save)
    callback = learning.TrainingCallback(env, tmp_path, 1, lambda **_: None, lambda: None)
    model.learn(total_timesteps=512, callback=callback)
    assert model._n_updates == 4
    assert checkpoints == [512]
    assert any(not torch.equal(a, b) for a, b in zip(before, model.policy.parameters()))
    traces = list(tmp_path.glob("observations-*.npz"))
    assert len(traces) == 1
    with np.load(traces[0], allow_pickle=False) as trace:
        assert np.array_equal(trace["steps"], np.arange(1, 513))
        assert trace["obs_control"].shape == (512, 1, 28)
        assert trace["obs_bullets"].shape == (512, 1, 64, 5)
        assert trace["obs_full_bullets"].shape == (512, 1, 640, 10)
        assert trace["actions"].shape == (512, 1, 1)
        assert json.loads(str(trace["metadata"]))["observation_version"] == OBSERVATION_VERSION
        assert json.loads(str(trace["metadata"]))["gae_ready"] is True
        assert np.isfinite(trace["returns"]).all()
    assert (tmp_path / "training_summary.json").is_file()
    with (tmp_path / "episodes.csv").open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 16
    assert all(int(row["reward_version"]) == REWARD_VERSION for row in rows)
    assert all(float(row["wall_push_seconds"]) == .75 for row in rows)
    assert all(float(row["mean_threat"]) == .2 for row in rows)
    monkeypatch.setattr(learning, "ROOT", tmp_path)
    model_path = tmp_path / "models" / "latest.zip"
    real_save(model, model_path, tmp_path, 512)
    metadata = json.loads(model_path.with_suffix(".json").read_text(encoding="utf-8"))
    assert metadata["format"] == OBSERVATION_VERSION
    assert metadata["reward_version"] == REWARD_VERSION
    loaded = learning.load_model(model_path, env, device="cpu")
    assert loaded.action_space.n == ACTION_COUNT
    assert torch.equal(next(model.policy.parameters()), next(loaded.policy.parameters()))


def test_resume_saves_each_new_update_before_next_rollout_not_unchanged_input(tmp_path, monkeypatch):
    torch.set_num_threads(1)
    env = LearningFixture()
    model = learning.make_model(env, device="cpu")
    model.learn(512)
    checkpoints = []
    def save(learner, path, run_dir, steps):
        assert env.paused
        checkpoints.append((steps, learner._n_updates))
    monkeypatch.setattr(learning, "save_model", save)
    callback = learning.TrainingCallback(env, tmp_path, 1, lambda **_: None, lambda: None)
    model.learn(1024, callback=callback, reset_num_timesteps=False)
    assert checkpoints == [(1024, 8), (1536, 12)]
    assert callback.base_steps == 512
