"""Stage identity, return isolation and zero-influence checkpoint migration."""
import hashlib
from dataclasses import replace
import numpy as np
import pytest
import torch
from stable_baselines3.common.callbacks import BaseCallback
from th06ai import learning
from th06ai.control import SamplingInterrupted
from th06ai.files import atomic_json
from th06ai.native import SHA256
from th06ai.observations import ObservationBuilder, make_observation_space
from th06ai.pausable_ppo import PausablePPO
from th06ai.policy import BulletEncoder, ResourcePolicy
from th06ai.stage import decode_stage, stage_label, stage_vector, context_changed
from test_learning import LearningFixture
from test_rewards import frame
from test_sampling_timing import game


def test_stage_ids_are_categories_not_elapsed_time_guesses():
    for stage in range(1, 8):
        vector = stage_vector(stage)
        assert decode_stage(stage) == stage and vector[stage-1] == 1 and vector[-1] == 1
        assert vector.sum() == 2
        assert stage_label(stage) == ('EX' if stage == 7 else f'Stage {stage}')
    for raw in (None, 0, -1, 8, 999, True, '1', 1.1):
        assert decode_stage(raw) is None and not stage_vector(raw).any()


def test_stage_change_clears_reused_bullet_slot_motion_even_with_forward_clock():
    first = replace(frame(tick=1000, bullets=[(1, 100, 80)]), stage_id=1)
    later = replace(first, tick=1002, bullets=np.array([(1, 102, 80)], np.float32))
    next_stage = replace(later, stage_id=2, tick=1004, bullets=np.array([(1, 300, 200)], np.float32))
    builder = ObservationBuilder()
    builder.build(first)
    assert builder.build(later)['full_bullets'][0, 4] == 1
    result = builder.build(next_stage)
    assert result['full_bullets'][0, 4] == 0 and not result['full_bullets'][0, 2:4].any()
    assert result['stage'][1] == 1 and make_observation_space().contains(result)


@pytest.mark.parametrize('next_tick', [2, 104])
def test_cross_stage_action_is_not_a_death_or_a_made_up_reward(next_tick):
    before = replace(frame(tick=100), stage_id=1)
    env, _, _, _ = game(before, replace(before, stage_id=2, tick=next_tick))
    env.stage_ticks, env.episode_ticks = 90, 300
    old_totals = dict(env.reward_tracker.totals)
    with pytest.raises(SamplingInterrupted) as caught: env.step(4)
    assert caught.value.stage_boundary and env.current.stage_id == 2 and env.stage_ticks == 0
    assert env.episode_ticks == 300 and env.reward_tracker.totals == old_totals
    assert env.last_stage_transition['before_stage'] == 1 and env.last_stage_transition['stage_id'] == 2


def test_unknown_stage_remains_unknown_without_stopping():
    before = frame(tick=100)
    env, _, _, _ = game(before, replace(before, tick=104))
    obs, _, died, _, info = env.step(4)
    assert info['stage_id'] is None and not obs['stage'].any() and not died
    assert not context_changed(before, env.current)


def test_disabled_episode_limit_allows_more_than_180_game_seconds():
    before = replace(frame(tick=11000), stage_id=2)
    env, _, _, _ = game(before, replace(before, tick=11004))
    env.episode_seconds, env.episode_ticks = 0, 11000
    _, _, died, truncated, info = env.step(4)
    assert info['survival_seconds'] > 180 and not died and not truncated


@pytest.mark.parametrize('via_unknown', [False, True])
def test_ppo_cuts_gae_and_bootstraps_old_stage_without_a_game_restart(via_unknown):
    torch.set_num_threads(1)
    class Stages(LearningFixture):
        resets = 0
        def reset(self, **kwargs):
            super().reset(**kwargs)
            self.resets += 1
            self.current.stage_id = 1
            self.transitions = [None, 2] if via_unknown else [2]
            self.actual_rewards = []
            self.builder.reset()
            return self.builder.build(self.current), {}
        def step(self, action):
            if self.local_steps == 3 and self.transitions:
                if self.current.stage_id == 1:
                    with torch.no_grad():
                        tensor = {k: torch.as_tensor(v, device=self.model.device) for k,v in self.model._last_obs.items()}
                        self.old_boundary_value = self.model.policy.predict_values(tensor).item()
                self.current = replace(self.current, stage_id=self.transitions.pop(0), tick=0)
                self.builder.reset()
                raise SamplingInterrupted(stage_boundary=True)
            result = super().step(action)
            self.actual_rewards.append(result[1])
            return result
    class Control:
        def gate(self, game): return False
    class Callback(BaseCallback):
        def _on_step(self): return True
    env = Stages()
    model = PausablePPO(ResourcePolicy, env, device='cpu', n_steps=8, batch_size=8, n_epochs=1,
        gamma=learning.DISCOUNT, policy_kwargs={'features_extractor_class': BulletEncoder})
    env.model = model
    model.training_control, model.training_game = Control(), env
    model.train = lambda: None
    model.learn(8, callback=Callback())
    buffer = model.rollout_buffer
    assert env.resets == 1 and model.num_timesteps == 8 and buffer.episode_starts[3,0] == 1
    assert buffer.observations['stage'][2,0,0] == 1 and buffer.observations['stage'][3,0,1] == 1
    assert buffer.rewards[2,0] == pytest.approx(env.actual_rewards[2]+model.gamma*env.old_boundary_value)
    old_advantages = buffer.advantages[:3].copy()
    buffer.rewards[3:] += 10000
    buffer.compute_returns_and_advantage(last_values=torch.zeros(1), dones=np.array([False]))
    assert np.array_equal(old_advantages, buffer.advantages[:3])


def test_v6_migration_keeps_all_parameters_adam_moments_and_bomb_policy(tmp_path, monkeypatch):
    torch.set_num_threads(1)
    class Version6(LearningFixture):
        def __init__(self):
            super().__init__()
            self.observation_space = make_observation_space(6)
        def reset(self, **kwargs):
            obs, info = super().reset(**kwargs)
            return {k:v for k,v in obs.items() if k in self.observation_space.spaces}, info
        def step(self, action):
            obs, reward, died, truncated, info = super().step(action)
            return {k:v for k,v in obs.items() if k in self.observation_space.spaces}, reward, died, truncated, info
    old = PausablePPO(ResourcePolicy, Version6(), device='cpu', n_steps=32, batch_size=32, n_epochs=1,
        gamma=learning.DISCOUNT, policy_kwargs={'features_extractor_class': BulletEncoder,
                                              'net_arch': {'pi':[128,128], 'vf':[128,128]}})
    old.learn(64)
    path = tmp_path/'models/v6.zip'
    path.parent.mkdir()
    old.save(path)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    atomic_json(path.with_suffix('.json'), {'format':6,'policy_version':2,'game_sha256':SHA256,
                'model_sha256':digest,'steps':64,'gradient_updates':old._n_updates})
    monkeypatch.setattr(learning,'ROOT',tmp_path)
    new = learning.load_model(path, LearningFixture(), 'cpu')
    assert new.num_timesteps == old.num_timesteps and new._n_updates == old._n_updates
    assert not new.bomb_policy_upgraded
    new_params = dict(new.policy.named_parameters())
    for name, parameter in old.policy.named_parameters():
        current = new_params[name]
        if name.endswith('fusion.0.weight'):
            assert torch.equal(current[:,:parameter.shape[1]],parameter)
            assert not current[:,parameter.shape[1]:].any()
        else: assert torch.equal(current,parameter)
        for key, value in old.policy.optimizer.state.get(parameter,{}).items():
            migrated = new.policy.optimizer.state[current][key]
            if isinstance(value,torch.Tensor) and value.ndim and name.endswith('fusion.0.weight'):
                assert torch.equal(migrated[:,:value.shape[1]],value) and not migrated[:,value.shape[1]:].any()
            elif isinstance(value,torch.Tensor): assert torch.equal(migrated,value)
    obs = new.observation_space.sample()
    obs['decision'][:]=0
    obs['resources'][1] = 3/8
    with torch.no_grad():
        old_tensors,_ = old.policy.obs_to_tensor({k:v for k,v in obs.items() if k in old.observation_space.spaces})
        old_probs = old.policy.get_distribution(old_tensors).distribution.probs
        old_value = old.policy.predict_values(old_tensors)
        for stage in (None,1,2,6,7):
            obs['stage'] = stage_vector(stage)
            new_tensors,_ = new.policy.obs_to_tensor(obs)
            assert torch.allclose(old_probs,new.policy.get_distribution(new_tensors).distribution.probs,atol=1e-6)
            assert torch.allclose(old_value,new.policy.predict_values(new_tensors),atol=1e-6)
    assert hashlib.sha256(path.read_bytes()).hexdigest() == digest
