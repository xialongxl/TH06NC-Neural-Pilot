"""Stage identity, return isolation and zero-influence checkpoint migration."""
import hashlib
from dataclasses import replace
from types import SimpleNamespace
import numpy as np
import pytest
import torch
from stable_baselines3.common.callbacks import BaseCallback
from th06ai import environment, learning
from th06ai.control import FinishTraining, SamplingInterrupted, TrainingControl
from th06ai.files import atomic_json
from th06ai.game_settings import GameSetup
from th06ai.native import ControlSample, SHA256
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


@pytest.fixture
def story_handoff(tmp_path):
    before = replace(frame(tick=9019), phase='boss', stage_id=1, game_mode='story',
                     difficulty='normal', character='reimu_a')
    loading = replace(before, tick=0, phase='unknown', stage_active=False, stage_id=None,
                      game_mode=None, difficulty=None, character=None)
    after = replace(before, tick=1, phase='combat', stage_id=2)
    env, actions, releases, statuses = game(before, loading)
    env.game_setup = GameSetup()
    env.stage_ticks = env.episode_ticks = 8658
    env.training_control = TrainingControl(tmp_path, env.lease, env.status)
    env.reader.window.focused = True
    return env, actions, releases, statuses, before, loading, after


@pytest.mark.parametrize('entry', ('script_wait', 'decision_probe', 'after_dispatch'))
def test_story_handoff_recovers_stage_one_to_two_without_pausing_or_sampling(story_handoff, entry):
    env, actions, releases, statuses, before, loading, after = story_handoff
    env.training_control.start()
    # The stage number can appear before the rest of the new scene identity.
    incomplete = replace(after, game_mode=None, difficulty=None)
    initial = replace(before, phase='script_wait') if entry == 'script_wait' else loading
    frames = iter((initial, loading, incomplete, after))
    env.reader.read = lambda: next(frames, after)
    env.reader.read_control = lambda: ControlSample(0, loading.x, loading.y, loading.state,
        False, False, None, 'unknown', None, 0, 0, 0)
    operation = {'script_wait': env._advance_dialogue, 'decision_probe': env.refresh_decision,
                 'after_dispatch': lambda: env.step(0)}[entry]
    with pytest.raises(SamplingInterrupted) as caught:
        operation()
    assert caught.value.stage_boundary and not (env.training_control.run_dir / 'PAUSE').exists()
    assert env.current is after and env.reward_tracker.frame is after
    assert env.stage_changes == 1 and env.stage_ticks == 0 and env.episode_ticks == 8658
    assert env.total_steps == 0 and env.episode_return == 0
    assert releases and actions == ([{'z'}] if entry == 'after_dispatch' else [])
    assert env.last_stage_transition['before_stage'] == 1
    assert env.last_stage_transition['after_tick'] == 1
    assert any(status.get('game_phase') == 'stage_transition' for status in statuses)
    assert [status['scene_wait_step'] for status in statuses if 'scene_wait_step' in status] == [0, None]
    scene = next(status for status in reversed(statuses) if 'game_phase' in status)
    assert scene['game_phase'] == 'combat' and scene['stage_id'] == 2


def test_pause_and_resume_during_story_loading_keep_the_confirmed_stage(story_handoff):
    env, actions, _, _, before, loading, after = story_handoff
    env.reader.read = lambda: loading
    env.pause_for_user()
    env.resume()
    assert env.current is before and actions == []
    env.reader.read = lambda: after
    env.reader.read_control = lambda: loading
    with pytest.raises(SamplingInterrupted) as caught:
        env.refresh_decision()
    assert caught.value.stage_boundary and env.last_stage_transition['before_stage'] == 1


def test_first_stage_two_pause_preserves_the_stage_one_rollout_boundary(story_handoff):
    env, actions, _, _, before, _, after = story_handoff
    paused = replace(after, paused=True, phase='pause', menu_selection=0)
    advancing = replace(after, tick=2)
    frames = iter((paused, paused, advancing))
    env.reader.read = lambda: next(frames, advancing)
    env.pause_for_user()
    assert env.current is before and actions == []
    env.resume()
    assert env.current is advancing and actions == [{'z'}]
    assert env.pending_stage_boundary and env.last_stage_transition['before_stage'] == 1
    assert env.stage_changes == 1


@pytest.mark.parametrize('entry', ('loading', 'already_paused', 'dialogue'))
def test_overtime_stops_at_confirmed_story_handoff_before_next_stage_input(story_handoff, entry):
    env, actions, _, statuses, before, loading, after = story_handoff
    now = [0.]
    control = env.training_control
    control.budget_seconds, control.clock = 60., lambda: now[0]
    control.start()
    now[0] = 60.
    control.check_budget(before)
    assert control.overtime_stage == 1

    if entry == 'already_paused':
        env.reader.read = lambda: replace(after, paused=True, phase='pause', menu_selection=0)
        operation = env.resume
    elif entry == 'dialogue':
        env.reader.read_control = env.reader.read = lambda: replace(after, phase='dialogue')
        env._advance_dialogue = lambda: pytest.fail('overtime must finish before the next stage dialogue')
        operation = env.refresh_decision
    else:
        env.reader.read_control = lambda: loading
        env.reader.read = lambda: after
        operation = env.refresh_decision
    with pytest.raises(FinishTraining) as caught:
        operation()
    assert caught.value.reason == 'stage_complete'
    assert actions == [] and env.total_steps == 0
    assert statuses[-1]['stage_transition']['before_stage'] == 1 and env.current.stage_id == 2
    assert not (control.run_dir / 'PAUSE').exists()


def test_extra_native_results_end_the_round_without_waiting_for_another_stage():
    before = replace(frame(tick=15000), phase='boss', stage_id=7, game_mode='story',
                     difficulty='extra', character='reimu_a')
    result = replace(before, stage_active=False, stage_id=None, game_mode=None, phase='result')
    env, actions, releases, _ = game(before, result)
    env.goal = 'survival'
    with pytest.raises(SamplingInterrupted) as caught:
        env._check_sampling_scene(result, before)
    info = caught.value.episode_end
    assert info['episode_end_reason'] == 'story_result' and info['stage_id'] == 7
    assert releases and not actions and env.total_steps == 0


@pytest.mark.parametrize('phase', ('title', 'pause'))
def test_story_loading_does_not_send_input_to_real_menus(story_handoff, phase):
    env, actions, releases, _, before, loading, after = story_handoff
    env.reader.read_control = lambda: loading
    env.reader.read = lambda: replace(after, phase=phase, paused=phase == 'pause',
                                      stage_active=phase == 'pause')
    with pytest.raises(SamplingInterrupted) as caught:
        env.refresh_decision()
    assert not caught.value.stage_boundary and env.current is before
    assert releases and actions == [] and (env.training_control.run_dir / 'PAUSE').exists()


def test_story_loading_that_stays_unconfirmed_keeps_keys_released(story_handoff, monkeypatch):
    env, actions, releases, _, before, loading, _ = story_handoff
    env.reader.read_control = env.reader.read = lambda: loading
    elapsed = [0.]
    monkeypatch.setattr(environment, 'time', SimpleNamespace(monotonic=lambda: elapsed[0]))
    env._wait = lambda seconds: elapsed.__setitem__(0, elapsed[0]+1)
    with pytest.raises(SamplingInterrupted) as caught:
        env.refresh_decision()
    assert not caught.value.stage_boundary and env.current is before
    assert releases and actions == [] and (env.training_control.run_dir / 'PAUSE').exists()


def test_disabled_episode_limit_allows_more_than_180_game_seconds():
    before = replace(frame(tick=11000), stage_id=2)
    env, _, _, _ = game(before, replace(before, tick=11004))
    env.episode_seconds, env.episode_ticks = 0, 11000
    _, _, died, truncated, info = env.step(4)
    assert info['survival_seconds'] > 180 and not died and not truncated


@pytest.mark.parametrize('handoff', ['direct', 'via_unknown', 'on_resume'])
def test_ppo_cuts_gae_and_bootstraps_old_stage_without_a_game_restart(handoff):
    torch.set_num_threads(1)
    class Stages(LearningFixture):
        resets = 0
        def reset(self, **kwargs):
            super().reset(**kwargs)
            self.resets += 1
            self.current.stage_id = 1
            self.transitions = [None, 2] if handoff == 'via_unknown' else [2]
            self.actual_rewards = []
            self.resume_requested = False
            self.builder.reset()
            return self.builder.build(self.current), {}
        def step(self, action):
            if self.local_steps == 3 and self.transitions:
                if self.current.stage_id == 1:
                    with torch.no_grad():
                        tensor = {k: torch.as_tensor(v, device=self.model.device) for k,v in self.model._last_obs.items()}
                        self.old_boundary_value = self.model.policy.predict_values(tensor).item()
                if handoff == 'on_resume':
                    self.resume_requested = True
                    raise SamplingInterrupted()
                self.current = replace(self.current, stage_id=self.transitions.pop(0), tick=0)
                self.builder.reset()
                raise SamplingInterrupted(stage_boundary=True)
            result = super().step(action)
            self.actual_rewards.append(result[1])
            return result
    class Control:
        def gate(self, game):
            if not game.resume_requested:
                return False
            game.resume_requested = False
            game.current = replace(game.current, stage_id=game.transitions.pop(0), tick=0)
            game.builder.reset()
            game.pending_stage_boundary = True
            return True
    class Callback(BaseCallback):
        def _on_step(self): return True
    env = Stages()
    model = PausablePPO(ResourcePolicy, env, device='cpu', n_steps=8, batch_size=8, n_epochs=1,
        gamma=learning.DISCOUNT, policy_kwargs={'features_extractor_class': BulletEncoder})
    env.model = model
    model.training_control, model.training_game = Control(), env
    if handoff == 'on_resume':
        # Distinct values make the observation used for bootstrap observable.
        def stage_values(observation):
            return (10 * observation['stage'][:, 0] + 20 * observation['stage'][:, 1])[:, None]
        model.policy.predict_values = stage_values
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
