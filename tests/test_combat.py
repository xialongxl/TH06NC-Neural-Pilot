from dataclasses import replace
import hashlib
import json

import numpy as np
import pytest
import torch

from th06ai import learning
from th06ai.combat import ENEMY_DTYPE, ENEMY_POOL, ENEMY_CAPACITY, ENEMY_STRIDE, decode_enemies, damage_feedback
from th06ai.files import atomic_json
from th06ai.native import SHA256
from th06ai.observations import ObservationBuilder, make_observation_space
from th06ai.pausable_ppo import PausablePPO
from th06ai.policy import ResourcePolicy, BulletEncoder
from th06ai.policy_spec import POLICY_VERSION
from th06ai.rewards import RewardTracker
from test_rewards import frame
from test_behavior_objective import trajectory
from test_learning import LearningFixture


def enemy_frame(tick=100, hp=2632, maximum=6000, phase=6000, flags=0x3f, x=192, boss=True):
    return replace(frame(tick=tick), enemies=np.array([(0, x, 80, hp, maximum, phase, flags, boss)], np.float32))


def test_decode_visible_enemies_and_hud_confirmed_boss_marker():
    data = np.zeros(ENEMY_CAPACITY, dtype=ENEMY_DTYPE)
    data[0] = (192, 80, -128, 0x3f, 548, 6000, 6000)
    data[1] = (32, 88, -128, 0x17, 32, 80, 80)
    data[2] = (10, -200, -128, 0x3f, 1000, 1000, 1000)
    data[3] = (200, 80, -128, 0x3f, 0, 6000, 6000)
    result = decode_enemies(data.tobytes(), 384, 448, [ENEMY_POOL])
    assert result.shape == (2, 8)
    assert result[:, 7].tolist() == [1, 0]
    assert result[0, 3] == 548
    assert decode_enemies(data.tobytes(), 384, 448)[:, 7].sum() == 0


def test_enemy_observation_contains_position_health_boss_and_damageable_state():
    builder = ObservationBuilder()
    observation = builder.build(enemy_frame())
    row = observation["enemies"][0]
    assert make_observation_space().contains(observation)
    assert row[0] == 0 and row[1] < 0
    assert row[4] == pytest.approx(2632/6000)
    assert row[5:].tolist() == [1, 1, 1]
    assert builder.build(enemy_frame(flags=8))["enemies"][0, 6] == 0


def test_plausible_same_phase_hp_loss_gives_boss_damage_credit():
    before = enemy_frame()
    after = enemy_frame(tick=102, hp=2560)
    small, boss, credit = damage_feedback(before, after)
    assert small == 0 and boss == 72 and credit > 0
    assert all(type(value) is float for value in (small, boss, credit))
    tracker = RewardTracker()
    tracker.reset(before)
    hit = tracker.step(after, 0, False)
    assert hit.components["damage"] > 0
    assert tracker.diagnostics()["boss_damage"] == 72
    # The previous implementation compared correctly but crashed the status
    # writer after the first real hit because damage credit was np.float32.
    assert all(type(value) is float for value in hit.components.values())
    json.dumps({"reward_components": hit.components, "diagnostics": tracker.diagnostics()})


@pytest.mark.parametrize("change", [dict(hp=0), dict(hp=2500, phase=3000), dict(hp=2500, maximum=3000),
                                       dict(hp=2500, flags=8), dict(hp=1000), dict(hp=2500, x=20)])
def test_phase_reset_inactivity_and_implausible_hp_drop_do_not_fake_damage(change):
    assert damage_feedback(enemy_frame(), enemy_frame(tick=102, **change)) == (0, 0, 0)


def test_enemy_disappearance_alone_does_not_mean_a_kill():
    assert damage_feedback(enemy_frame(), frame(tick=102)) == (0, 0, 0)


@pytest.mark.parametrize("position", [(8,224),(376,224),(192,16),(192,432)])
def test_prolonged_stationary_single_wall_exposure_cannot_farm_time_credit(position):
    total, discounted, diagnostics = trajectory(lambda _: position)
    assert total < 0 and discounted < 0
    assert diagnostics["edge_exposure_seconds"] > 5.9


class OldObservationGame(LearningFixture):
    def __init__(self):
        super().__init__()
        self.observation_space = make_observation_space(version=2)
    def reset(self, **kwargs):
        obs, info = super().reset(**kwargs)
        return {key: obs[key] for key in self.observation_space.spaces}, info
    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)
        return {key: obs[key] for key in self.observation_space.spaces}, reward, terminated, truncated, info


def test_real_format_two_checkpoint_keeps_decisions_values_optimizer_and_counts_after_enemy_extension(tmp_path, monkeypatch):
    torch.set_num_threads(1)
    old_game, new_game = OldObservationGame(), LearningFixture()
    old = PausablePPO(ResourcePolicy, old_game, device="cpu", n_steps=32, batch_size=32, n_epochs=1,
                      policy_kwargs={"features_extractor_class": BulletEncoder, "net_arch": {"pi":[128,128],"vf":[128,128]}})
    old.learn(64)
    path = tmp_path / "models" / "format2.zip"
    path.parent.mkdir(parents=True)
    old.save(path)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    atomic_json(path.with_suffix(".json"), {"format":2,"policy_version":POLICY_VERSION,"game_sha256":SHA256,
                                           "model_sha256":digest,"steps":64,"gradient_updates":2})
    monkeypatch.setattr(learning,"ROOT",tmp_path)
    new = learning.load_model(path,new_game,"cpu")
    observation = ObservationBuilder().build(enemy_frame())
    old_observation = {key:observation[key] for key in old.observation_space.spaces}
    old_tensor,_ = old.policy.obs_to_tensor(old_observation)
    new_tensor,_ = new.policy.obs_to_tensor(observation)
    with torch.no_grad():
        assert torch.allclose(old.policy.get_distribution(old_tensor).distribution.probs,
                              new.policy.get_distribution(new_tensor).distribution.probs,atol=1e-6)
        assert torch.allclose(old.policy.predict_values(old_tensor),new.policy.predict_values(new_tensor),atol=1e-6)
    fusion = new.policy.features_extractor.fusion[0].weight
    assert torch.equal(fusion[:,:201],old.policy.features_extractor.fusion[0].weight)
    assert (fusion[:,201:]==0).all()
    assert new.num_timesteps==64 and new._n_updates==2
    old_param=dict(old.policy.named_parameters())["features_extractor.fusion.0.weight"]
    new_param=dict(new.policy.named_parameters())["features_extractor.fusion.0.weight"]
    assert torch.equal(old.policy.optimizer.state[old_param]["exp_avg"],new.policy.optimizer.state[new_param]["exp_avg"][:,:201])
    assert hashlib.sha256(path.read_bytes()).hexdigest()==digest
