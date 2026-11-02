"""Resource exploration regressions with real CPU networks and fake games."""
import hashlib
import math

import numpy as np
import pytest
import torch

from th06ai import learning
from th06ai.files import atomic_json, read_json
from th06ai.native import SHA256
from th06ai.pausable_ppo import PausablePPO
from th06ai.policy import BOMB_PRIOR, POLICY_VERSION, BulletEncoder, ResourcePolicy, exploration_terms
from test_learning import LearningFixture


def tensor_observation(policy, game, *, bombs=3, batch=1):
    obs = game.builder.build(game.current)
    obs["resources"][1] = bombs / 8
    if batch > 1:
        obs = {key: np.repeat(value[None, ...], batch, axis=0) for key, value in obs.items()}
        obs["player"][:, 0] = np.linspace(.05, .95, batch)
        obs["player"][:, 1] = np.linspace(.3, .9, batch)
        obs["bullets"][:, :10, 0] = np.linspace(-.3, .3, batch)[:, None]
        obs["bullets"][:, :10, 1] = -.04
        obs["bullets"][:, :10, -1] = 1
    return policy.obs_to_tensor(obs)[0]


def test_new_policy_has_rare_bombs_preserves_movement_exploration_and_masks_empty_inventory():
    torch.set_num_threads(1)
    game = LearningFixture()
    model = learning.make_model(game, "cpu")
    obs = tensor_observation(model.policy, game, batch=32)
    with torch.no_grad():
        distribution = model.policy.get_distribution(obs)
        probs = distribution.distribution.probs
        assert torch.allclose(probs[:, 18:].sum(-1), torch.full((32,), BOMB_PRIOR), atol=1e-8)
        movement_entropy, bomb_kl = exploration_terms(probs, torch.ones(32, dtype=torch.bool))
        assert movement_entropy.min() > 2.8 and bomb_kl.abs().max() < 1e-6
        deterministic, values, log_prob = model.policy(obs, deterministic=True)
        assert (deterministic < 18).all()
        evaluated_values, evaluated_log_prob, true_entropy = model.policy.evaluate_actions(obs, deterministic)
        assert torch.equal(values, evaluated_values) and torch.allclose(log_prob, evaluated_log_prob)
        assert torch.allclose(true_entropy, distribution.entropy())
        empty = tensor_observation(model.policy, game, bombs=0)
        assert model.policy.get_distribution(empty).distribution.probs[..., 18:].sum() == 0


def test_regularizer_targets_rare_prior_instead_of_half_bombs_and_has_finite_gradients():
    movement = torch.full((18,), 1 / 18, dtype=torch.float64)
    losses = []
    for q_value in (BOMB_PRIOR / 10, BOMB_PRIOR, .001, .1, .5, .999999):
        bomb_logit = torch.tensor(math.log(q_value / (1 - q_value)), dtype=torch.float64, requires_grad=True)
        q = bomb_logit.sigmoid()
        probs = torch.cat([movement * (1 - q), movement * q])[None, :]
        entropy, kl = exploration_terms(probs, torch.tensor([True]))
        loss = (kl - entropy).sum()
        loss.backward()
        assert entropy.item() == pytest.approx(math.log(18))
        assert torch.isfinite(bomb_logit.grad)
        if q_value > BOMB_PRIOR:
            assert bomb_logit.grad > 0  # Gradient descent reduces excess bomb probability.
        if q_value < BOMB_PRIOR:
            assert bomb_logit.grad < 0
        losses.append(loss.item())
    assert int(np.argmin(losses)) == 1


def test_bomb_use_is_learnable_and_not_blocked_by_a_runtime_threshold_or_cooldown():
    torch.set_num_threads(1)
    game = LearningFixture()
    model = learning.make_model(game, "cpu")
    obs = tensor_observation(model.policy, game)
    before = model.policy.get_distribution(obs).distribution.probs[:, 18:].sum().item()
    model.policy.optimizer.zero_grad()
    _, log_prob, _ = model.policy.evaluate_actions(obs, torch.tensor([18]))
    (-log_prob.mean()).backward()
    assert model.policy.action_net.bias.grad[18] < 0
    model.policy.optimizer.step()
    after = model.policy.get_distribution(obs).distribution.probs[:, 18:].sum().item()
    assert after > before
    with torch.no_grad():
        model.policy.action_net.bias[18:] += 20
        action, _, _ = model.policy(obs, deterministic=True)
        assert action.item() >= 18  # A learned high bomb probability is respected.
        empty = tensor_observation(model.policy, game, bombs=0)
        assert model.policy(empty, deterministic=True)[0].item() < 18


def test_upgrade_existing_36_action_model_keeps_movement_weights_optimizer_and_training_counts(tmp_path, monkeypatch):
    torch.set_num_threads(1)
    game = LearningFixture()
    old = PausablePPO("MultiInputPolicy", game, device="cpu", n_steps=64, batch_size=32, n_epochs=2, seed=17,
                      policy_kwargs={"features_extractor_class": BulletEncoder, "net_arch": {"pi": [128, 128], "vf": [128, 128]}},
                      gamma=learning.DISCOUNT, ent_coef=.02)
    old.learn(128)
    source = tmp_path / "models" / "old36.zip"
    source.parent.mkdir(parents=True)
    old.save(source)
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    atomic_json(source.with_suffix(".json"), {"format": 2, "game_sha256": SHA256,
                "model_sha256": digest, "steps": old.num_timesteps, "gradient_updates": old._n_updates})
    monkeypatch.setattr(learning, "ROOT", tmp_path)
    fixed = learning.load_model(source, game, "cpu")
    assert isinstance(fixed.policy, ResourcePolicy) and fixed.bomb_policy_upgraded
    assert fixed.num_timesteps == 128 and fixed._n_updates == old._n_updates
    for name, before in old.policy.state_dict().items():
        after = fixed.policy.state_dict()[name]
        assert torch.equal(before[:18] if name.startswith("action_net.") else before,
                           after[:18] if name.startswith("action_net.") else after)
    old_parameters, new_parameters = dict(old.policy.named_parameters()), dict(fixed.policy.named_parameters())
    for name, old_parameter in old_parameters.items():
        if old_parameter not in old.policy.optimizer.state:
            continue
        for key, before in old.policy.optimizer.state[old_parameter].items():
            after = fixed.policy.optimizer.state[new_parameters[name]][key]
            if name.startswith("action_net.") and before.ndim:
                assert torch.equal(before[:18], after[:18]) and (after[18:] == 0).all()
            else:
                assert torch.equal(before, after)
    obs = tensor_observation(fixed.policy, game, batch=8)
    with torch.no_grad():
        old_probs = old.policy.get_distribution(obs).distribution.probs
        new_probs = fixed.policy.get_distribution(obs).distribution.probs
        assert torch.allclose(old_probs[:, :18] / old_probs[:, :18].sum(-1, keepdim=True),
                              new_probs[:, :18] + new_probs[:, 18:], atol=1e-6)
        assert torch.allclose(new_probs[:, 18:].sum(-1), torch.full((8,), BOMB_PRIOR), atol=1e-8)
        # Saving a corrected model must retain later learned bomb weights.
        fixed.policy.action_net.bias[18:] += 3
    repaired_path = tmp_path / "models" / "repaired.zip"
    learning.save_model(fixed, repaired_path, tmp_path, fixed.num_timesteps)
    reloaded = learning.load_model(repaired_path, game, "cpu")
    assert read_json(repaired_path.with_suffix(".json"))["policy_version"] == POLICY_VERSION
    assert not reloaded.bomb_policy_upgraded
    for name, value in fixed.policy.state_dict().items():
        assert torch.equal(value, reloaded.policy.state_dict()[name])
    assert hashlib.sha256(source.read_bytes()).hexdigest() == digest


class EmptyGame(LearningFixture):
    def reset(self, **kwargs):
        obs, info = super().reset(**kwargs)
        obs["resources"][1] = 3 / 8
        return obs, info
    def step(self, action):
        obs, _, _, _, _ = super().step(action)
        obs["resources"][1] = 3 / 8
        return obs, 0.0, False, False, {}


def test_actual_ppo_updates_log_resource_regularizer_and_do_not_restore_half_bombs():
    torch.set_num_threads(1)
    game = EmptyGame()
    model = PausablePPO(ResourcePolicy, game, device="cpu", n_steps=64, batch_size=32, n_epochs=2,
                        policy_kwargs={"features_extractor_class": BulletEncoder, "net_arch": {"pi": [128, 128], "vf": [128, 128]}},
                        ent_coef=.02, seed=17)
    model.learn(512)
    obs = tensor_observation(model.policy, game)
    with torch.no_grad():
        p = model.policy.get_distribution(obs).distribution.probs
        assert p[:, 18:].sum().item() < .001
        assert model.policy(obs, deterministic=True)[0].item() < 18
    assert model._n_updates == 16
    assert all(np.isfinite(value) for value in model.last_policy_diagnostics.values())
    assert "movement_entropy" in model.last_policy_diagnostics and "bomb_prior_kl" in model.last_policy_diagnostics
