"""Regression cases for delayed controls and history-dependent feedback."""
from dataclasses import replace
import hashlib

import numpy as np
import pytest
import torch

from th06ai import learning
from th06ai.files import atomic_json
from th06ai.native import SHA256
from th06ai.observations import ObservationBuilder, make_observation_space
from th06ai.pausable_ppo import PausablePPO
from th06ai.policy import ResourcePolicy, BulletEncoder
from th06ai.policy_spec import POLICY_VERSION
from test_learning import LearningFixture
from test_sampling_timing import game
from test_rewards import frame


def test_inference_elapsed_time_cannot_satisfy_new_commands_hold_time():
    before = frame(tick=100, x=192, y=300)
    # The old down key continues during inference. New up has not yet moved.
    dispatch = replace(before, tick=106, y=324)
    intermediate = replace(dispatch, tick=107, y=320)
    completed = replace(dispatch, tick=108, y=316)
    env, commands, _, _ = game(before, completed)
    env.last_action = 2
    captures = iter((dispatch, intermediate, completed))
    env.reader.read = lambda: next(captures)
    observation, _, _, _, info = env.step(1)
    assert commands == [{"z", "up"}]
    assert info["native_tick_dispatch"] == 106
    assert info["pre_dispatch_ticks"] == 6
    assert info["action_hold_ticks"] == 2
    assert info["native_tick_delta"] == 8
    assert info["previous_action"] == 2
    # All actual elapsed time and net motion are retained, not fabricated as
    # an instantaneous, zero-delay transition in the new direction.
    assert info["movement"]["dy"] == 16
    assert info["survival_seconds"] == pytest.approx(8/60)
    assert observation["control"][1] == 1
    assert observation["control"][19] == 1


def test_death_during_inference_is_not_discarded_or_waited_past():
    before = frame(tick=100)
    dead = replace(before, tick=105, state=2)
    env, _, _, _ = game(before, dead)
    env.reader.read = lambda: dead
    _, _, died, _, info = env.step(4)
    assert died and info["reward_components"]["death"] == -20
    assert info["action_hold_ticks"] == 0
    assert info["pre_dispatch_ticks"] == 5


def test_control_observation_disambiguates_identical_positions_and_pending_keys():
    current = frame(x=8, y=432)
    builder = ObservationBuilder()
    fresh = builder.build(current)
    builder.set_control(3, blocked_ticks=(120, 0), residence=1., resource_saturation=.4)
    stuck = builder.build(current)
    for name in ("bullets", "player", "resources", "items", "enemies"):
        assert np.array_equal(fresh[name], stuck[name])
    assert stuck["control"][3] == 1
    assert stuck["control"][19] == 1
    assert stuck["control"][24] == pytest.approx(2/3)
    assert stuck["control"][26] == 1
    assert stuck["control"][27] == pytest.approx(.4)
    assert make_observation_space().contains(stuck)
    builder.reset()
    assert not builder.build(current)["control"].any()


def test_refreshing_same_frame_preserves_self_and_bullet_velocity():
    builder = ObservationBuilder()
    first = replace(frame(tick=100, x=190), bullets=np.array([[1, 190, 370]], np.float32))
    second = replace(first, tick=102, x=194, bullets=np.array([[1, 192, 374]], np.float32))
    builder.build(first)
    measured = builder.build(second)
    refreshed = builder.build(second)
    assert np.allclose(measured["control"][20:24], [2/16, 0, 2/60, 1])
    assert np.array_equal(measured["control"], refreshed["control"])
    assert np.array_equal(measured["bullets"], refreshed["bullets"])
    second.tick = 104
    second.x = 198
    assert builder.build(second)["control"][20] == pytest.approx(2/16)


def test_unknown_long_gap_keeps_duration_without_faking_known_motion():
    builder = ObservationBuilder()
    builder.build(frame(tick=100))
    observed = builder.build(frame(tick=160, x=250))
    assert observed["control"][22] == 1
    assert not observed["control"][[20, 21, 23]].any()


def test_corner_history_exposes_elapsed_grace_before_any_cost_and_resets_on_exit():
    before = frame(tick=100, x=8, y=432)
    env, _, _, _ = game(before, before)
    for ticks in (60, 120, 180):
        current = replace(before, tick=100+ticks)
        feedback = env.reward_tracker.step(current, 0, False)
        env._observe_control(current, 0)
        observation = env.builder.build(current)
        assert observation["control"][26] == pytest.approx(ticks/180)
        if ticks <= 120:
            assert feedback.components["edge_loiter"] == 0
    exited = replace(current, tick=282, x=100)
    env.reward_tracker.step(exited, 4, False)
    returned = replace(exited, tick=284, x=8)
    env.reward_tracker.step(returned, 3, False)
    env._observe_control(returned, 3)
    assert env.builder.build(returned)["control"][26] == 0


@pytest.mark.parametrize("inference_only", [False, True])
@pytest.mark.parametrize("version", [3, 4, 5])
def test_legacy_migration_preserves_policy_values_adam_counters_and_source(tmp_path, monkeypatch, inference_only, version):
    torch.set_num_threads(1)
    old_game = LearningFixture()
    old_game.observation_space = make_observation_space(version=version)
    old = PausablePPO(ResourcePolicy, old_game, device="cpu", n_steps=32, batch_size=32, n_epochs=1,
            policy_kwargs={"features_extractor_class": BulletEncoder, "net_arch": {"pi": [128, 128], "vf": [128, 128]}})
    # Populate Adam moments without collecting game data or labeling a random
    # model as trained. This is a numerical migration regression fixture.
    old.policy.optimizer.zero_grad()
    sum(p.square().sum() for p in old.policy.parameters()).backward()
    old.policy.optimizer.step()
    old.num_timesteps, old._n_updates = 3072, 24
    path = tmp_path / "models" / f"v{version}.zip"
    path.parent.mkdir(parents=True)
    old.save(path)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    atomic_json(path.with_suffix(".json"), {"format": version, "policy_version": POLICY_VERSION,
                "game_sha256": SHA256, "model_sha256": digest, "steps": 3072, "gradient_updates": 24})
    monkeypatch.setattr(learning, "ROOT", tmp_path)
    upgraded = learning.load_model(path, None if inference_only else LearningFixture(), "cpu")
    observation = make_observation_space().sample()
    observation["planning"].fill(0)  # Compare the retained neural prior; planning is a new explicit input.
    observation['decision'].fill(0)  # Neutral goal/rescue context for equivalence of legacy outputs.
    observation["resources"][1] = 3/8
    old_observation = {name: observation[name] for name in old.observation_space.spaces}
    with torch.no_grad():
        old_tensor, _ = old.policy.obs_to_tensor(old_observation)
        new_tensor, _ = upgraded.policy.obs_to_tensor(observation)
        assert torch.allclose(old.policy.get_distribution(old_tensor).distribution.probs,
                              upgraded.policy.get_distribution(new_tensor).distribution.probs, atol=1e-7)
        assert torch.allclose(old.policy.predict_values(old_tensor), upgraded.policy.predict_values(new_tensor), atol=1e-6)
    assert upgraded.num_timesteps == 3072 and upgraded._n_updates == 24
    old_parameters = dict(old.policy.named_parameters())
    for name, parameter in upgraded.policy.named_parameters():
        if name not in old_parameters:
            assert ".full_bullet." in name
            assert parameter not in upgraded.policy.optimizer.state
            original = old_parameters[name.replace(".full_bullet.", ".bullet.")]
            if name.endswith("full_bullet.0.weight"):
                assert torch.equal(parameter[:, :4], original[:, :4])
                assert torch.equal(parameter[:, -1], original[:, 4])
                assert not parameter[:, 4:-1].any()
            else:
                assert torch.equal(parameter, original)
            continue
        original = old_parameters[name]
        assert torch.equal(parameter[..., :original.shape[-1]], original)
        old_state = old.policy.optimizer.state[original]
        new_state = upgraded.policy.optimizer.state[parameter]
        for key in ("exp_avg", "exp_avg_sq"):
            assert torch.equal(new_state[key][..., :original.shape[-1]], old_state[key])
        if parameter.shape != original.shape:
            assert not parameter[..., original.shape[-1]:].any()
            assert not new_state["exp_avg"][..., original.shape[-1]:].any()
    assert hashlib.sha256(path.read_bytes()).hexdigest() == digest
    assert (upgraded.env is None) == inference_only
    # Zero-influence warm start must not leave the new branch frozen. One
    # update learns fusion weights; the next backward reaches the point MLP.
    valid_observation = ObservationBuilder().build(frame(bullets=[(1, 192, 360)]))
    valid_observation["planning"][:, :2] = 1  # An enabled, unrestricted forecast activates the new input columns.
    tensor, _ = upgraded.policy.obs_to_tensor(valid_observation)
    fusion = upgraded.policy.features_extractor.fusion[0].weight
    old_width = old.policy.features_extractor.fusion[0].weight.shape[1]
    for update in range(2):
        upgraded.policy.optimizer.zero_grad()
        distribution = upgraded.policy.get_distribution(tensor)
        (-distribution.log_prob(torch.tensor([4])).mean()).backward()
        assert fusion.grad[:, old_width:].abs().sum() > 0
        if update:
            assert upgraded.policy.features_extractor.full_bullet[0].weight.grad.abs().sum() > 0
        upgraded.policy.optimizer.step()
    assert hashlib.sha256(path.read_bytes()).hexdigest() == digest
