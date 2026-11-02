import hashlib
import json

import gymnasium as gym
import numpy as np
import torch
import pytest

from th06ai import learning
from th06ai.dashboard import build_command
from th06ai.files import atomic_json
from th06ai.model_store import list_checkpoints
from th06ai.native import SHA256
from th06ai.observations import make_observation_space
from th06ai.pausable_ppo import PausablePPO
from th06ai.policy import BulletEncoder, BOMB_PRIOR
from test_learning import LearningFixture


class LegacyGame(gym.Env):
    def __init__(self):
        space = make_observation_space()
        self.observation_space = gym.spaces.Dict({name: space[name] for name in ("bullets", "player")})
        self.action_space = gym.spaces.Discrete(18)
        self.observation = self.observation_space.sample()
    def reset(self, **kwargs):
        return self.observation, {}
    def step(self, action):
        return self.observation, float(action == 4), False, False, {}


def test_legacy_migration_preserves_movement_value_steps_optimizer_and_source(tmp_path, monkeypatch):
    torch.set_num_threads(1)
    old_game, game = LegacyGame(), LearningFixture()
    old = PausablePPO("MultiInputPolicy", old_game, device="cpu", n_steps=32, batch_size=32, n_epochs=1,
                      policy_kwargs={"features_extractor_class": BulletEncoder, "net_arch": {"pi": [128, 128], "vf": [128, 128]}},
                      gamma=learning.DISCOUNT)
    old.learn(64)
    path = tmp_path / "models" / "archive" / "old.zip"
    path.parent.mkdir(parents=True)
    old.save(path)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    atomic_json(path.with_suffix(".json"), {"format": 1, "game_sha256": SHA256, "model_sha256": digest,
                                           "steps": old.num_timesteps, "gradient_updates": old._n_updates})
    monkeypatch.setattr(learning, "ROOT", tmp_path)
    new = learning.load_model(path, game, "cpu")
    obs = game.observation_space.sample()
    obs["planning"].fill(0)
    obs['decision'].fill(0)  # Neutral rescue/goal context when comparing the retained legacy distribution.
    obs.update(old_game.observation)
    obs["resources"][1] = 3 / 8
    old_obs = {name: obs[name] for name in ("bullets", "player")}
    with torch.no_grad():
        old_tensor, _ = old.policy.obs_to_tensor(old_obs)
        new_tensor, _ = new.policy.obs_to_tensor(obs)
        old_prob = old.policy.get_distribution(old_tensor).distribution.probs
        new_prob = new.policy.get_distribution(new_tensor).distribution.probs
        assert torch.allclose(old_prob, new_prob[:, :18] + new_prob[:, 18:], atol=1e-6)
        assert new_prob[:, 18:].sum().item() == pytest.approx(BOMB_PRIOR, abs=1e-8)
        assert torch.allclose(old.policy.predict_values(old_tensor), new.policy.predict_values(new_tensor), atol=1e-6)
    assert new.num_timesteps == 64 and new._n_updates == 2
    new_params = dict(new.policy.named_parameters())
    old_params = dict(old.policy.named_parameters())
    key = "mlp_extractor.policy_net.0.weight"
    assert torch.equal(new.policy.optimizer.state[new_params[key]]["exp_avg"], old.policy.optimizer.state[old_params[key]]["exp_avg"])
    assert new.source_checkpoint == str(path.resolve())
    assert hashlib.sha256(path.read_bytes()).hexdigest() == digest
    new.learn(32, reset_num_timesteps=False)
    assert new.num_timesteps >= 96 and new._n_updates > 2
    entries = list_checkpoints(tmp_path / "models")
    assert entries[0].usable and entries[0].compatibility == "旧版可迁移"


def test_snapshots_keep_old_files_and_checkpoint_resume_counters(tmp_path, monkeypatch):
    torch.set_num_threads(1)
    monkeypatch.setattr(learning, "ROOT", tmp_path)
    game = LearningFixture()
    model = learning.make_model(game, "cpu")
    # A small but actual parameter update, without collecting real gameplay.
    model.learn(512)
    path = tmp_path / "models" / "latest.zip"
    snapshot = learning.save_model(model, path, tmp_path, model.num_timesteps)
    before = snapshot.read_bytes()
    restored = learning.load_model(snapshot, game, "cpu")
    assert restored.num_timesteps == model.num_timesteps
    assert restored._n_updates == model._n_updates
    for name, parameter in model.policy.state_dict().items():
        assert torch.equal(parameter, restored.policy.state_dict()[name])
    restored.learn(32, reset_num_timesteps=False)
    learning.save_model(restored, path, tmp_path, restored.num_timesteps)
    assert snapshot.read_bytes() == before
    assert len(list_checkpoints(tmp_path / "models")) >= 3
    assert json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))["steps"] > model.num_timesteps
    cmd = build_command("resume", 30, snapshot)
    assert cmd[-2:] == ["--model", str(snapshot)]
    assert "train" in cmd and "--model" not in build_command("train", 30, snapshot)
