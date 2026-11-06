"""Real CPU PPO + fake game; no game input, GPU work or public run pointer."""
from dataclasses import replace
import json
import threading
import time
from types import SimpleNamespace

import pytest
import torch

from th06ai import learning
from th06ai.control import (FinishTraining, SamplingInterrupted, SCENE_WAIT_LIMIT_SECONDS,
                            STAGE_OVERTIME_LIMIT_SECONDS, TrainingControl, set_paused)
from test_learning import LearningFixture


def test_user_pause_keeps_partial_rollout_parameters_optimizer_and_episode(tmp_path, monkeypatch):
    torch.set_num_threads(1)
    game = LearningFixture()
    model = learning.make_model(game, "cpu")
    # This checks pause/partial-buffer semantics, not CPU throughput for the
    # production 512-step, four-epoch GPU workload. Use a small real rollout
    # so reaching the pause does not depend on machine speed.
    model.n_steps, model.batch_size, model.n_epochs = 64, 16, 1
    model.rollout_buffer = model.rollout_buffer_class(model.n_steps, model.observation_space,
            model.action_space, device=model.device, gamma=model.gamma, gae_lambda=model.gae_lambda,
            n_envs=model.n_envs, **model.rollout_buffer_kwargs)
    paused = threading.Event()
    records, errors = [], []
    def status(**values):
        records.append(values)
        if values.get("steps") == 72:
            set_paused(tmp_path, True)
        if values.get("user_paused"):
            paused.set()
    control = TrainingControl(tmp_path, lambda: None, status)
    model.training_control, model.training_game = control, game
    monkeypatch.setattr(learning, "save_model", lambda *args: None)
    callback = learning.TrainingCallback(game, tmp_path, 1, status, lambda: None, control)
    def learn():
        try:
            model.learn(total_timesteps=128, callback=callback)
        except BaseException as exc:
            errors.append(exc)
    thread = threading.Thread(target=learn, daemon=True)
    thread.start()
    try:
        assert paused.wait(10), errors
        assert game.paused and model.num_timesteps == 72
        assert model.rollout_buffer.pos == 8 and model._n_updates == 1
        assert model.policy.optimizer.state
        parameters = [p.detach().clone() for p in model.policy.parameters()]
        optimizer_step = next(iter(model.policy.optimizer.state.values()))["step"].clone()
        local_steps, tick, active = game.local_steps, game.current.tick, control.active_seconds
        time.sleep(.2)
        assert model.num_timesteps == 72 and model.rollout_buffer.pos == 8
        assert game.local_steps == local_steps and game.current.tick == tick
        assert control.active_seconds == pytest.approx(active, abs=.01)
        assert all(torch.equal(a, b) for a, b in zip(parameters, model.policy.parameters()))
        assert torch.equal(next(iter(model.policy.optimizer.state.values()))["step"], optimizer_step)
    finally:
        set_paused(tmp_path, False)
        thread.join(10)
    assert not thread.is_alive() and not errors
    assert model.num_timesteps == 128 and model._n_updates == 2
    assert control.paused_seconds >= .2
    assert callback.base_steps == 0
    assert records[-1].get("checkpoint") or (tmp_path / "training_summary.json").is_file()


def test_save_finish_from_pause_is_distinct_from_emergency_stop(tmp_path):
    game = LearningFixture()
    control = TrainingControl(tmp_path, lambda: None, lambda **_: None)
    control.start()
    set_paused(tmp_path, True)
    (tmp_path / "FINISH").touch()
    with pytest.raises(FinishTraining):
        control.gate(game)
    def emergency():
        raise RuntimeError("safety lease revoked")
    control.lease = emergency
    with pytest.raises(RuntimeError, match="safety lease revoked"):
        control.gate(game)


def test_paused_wait_keeps_checking_safety_lease(tmp_path):
    game = LearningFixture()
    calls = []
    def lease():
        calls.append(True)
        if len(calls) == 4:
            raise RuntimeError("GPU fault")
    control = TrainingControl(tmp_path, lease, lambda **_: None, sleep=lambda _: None)
    control.start()
    set_paused(tmp_path, True)
    with pytest.raises(RuntimeError, match="GPU fault"):
        control.gate(game)
    assert game.paused and len(calls) == 4


@pytest.mark.parametrize('game_read_fails', (False, True))
def test_save_and_finish_preserves_real_updated_model_from_paused_training(tmp_path, monkeypatch, game_read_fails):
    from th06ai.config import Config
    torch.set_num_threads(1)
    game = LearningFixture()
    released = []
    game.release_input = lambda: released.append(True)
    pause = game.pause

    def pause_game():
        if game_read_fails and (tmp_path / 'FINISH').exists():
            raise RuntimeError('simulated read-only game state failure during finish')
        pause()

    game.pause = pause_game
    monkeypatch.setattr(learning, "ROOT", tmp_path)
    def status(**values):
        if values.get("steps") == 520:
            set_paused(tmp_path, True)
        if values.get("user_paused"):
            (tmp_path / "FINISH").touch()
    model = learning.train(Config(game_exe="unused", device="cpu"), game, tmp_path, 1,
                           status, lambda: None, resume=None)
    assert model.num_timesteps == 520 and model._n_updates == 4
    assert game.paused and released == [True]
    assert (tmp_path / 'training_summary.json').is_file()
    restored = learning.load_model(tmp_path / "models" / "latest.zip", game, "cpu")
    assert restored.num_timesteps == 520 and restored._n_updates == 4
    assert all(torch.equal(a, b) for a, b in zip(model.policy.parameters(), restored.policy.parameters()))
    step = next(iter(model.policy.optimizer.state.values()))['step']
    restored_step = next(iter(restored.policy.optimizer.state.values()))['step']
    assert torch.equal(step, restored_step)


def test_finish_releases_input_but_does_not_save_with_revoked_authorization(tmp_path, monkeypatch):
    released = []
    game = SimpleNamespace(release_input=lambda: released.append(True))

    def lease():
        raise RuntimeError('safety lease revoked')

    monkeypatch.setattr(learning, 'save_model', lambda *args: pytest.fail('unauthorized checkpoint'))
    callback = learning.TrainingCallback(game, tmp_path, 1, lambda **_: None, lease)
    callback.model = SimpleNamespace(_n_updates=1)
    with pytest.raises(RuntimeError, match='safety lease revoked'):
        callback.on_training_end()
    assert released == [True]


def test_stage_overtime_is_bounded_and_checks_the_lease_before_saving(tmp_path):
    now = [0.]
    game = LearningFixture()
    game.current.stage_id = 5
    control = TrainingControl(tmp_path, lambda: None, lambda **_: None,
                              budget_seconds=60, clock=lambda: now[0])
    control.start()
    now[0] = 59.
    with control.suspended_budget():
        now[0] += 120
        control.check_budget(game.current)
    assert control.overtime_stage is None
    now[0] += 1
    control.check_budget(game.current)
    assert control.overtime_stage == 5
    now[0] += STAGE_OVERTIME_LIMIT_SECONDS
    with pytest.raises(FinishTraining) as caught:
        control.check_budget(game.current)
    assert caught.value.reason == 'overtime_limit'
    (tmp_path / 'FINISH').touch()
    with pytest.raises(FinishTraining) as caught:
        control.check_budget(boundary='stage_complete')
    assert caught.value.reason == 'manual'

    def revoked():
        raise RuntimeError('lease revoked')
    control.lease = revoked
    with pytest.raises(RuntimeError, match='lease revoked'):
        control.check_budget(boundary='stage_complete')


def test_scene_wait_clock_caps_retries_and_clears_the_heartbeat_on_interrupt(tmp_path):
    now = [0.]
    statuses = []
    control = TrainingControl(tmp_path, lambda: None, lambda **fields: statuses.append(fields),
                              budget_seconds=600, clock=lambda: now[0])
    control.start()
    with control.suspended_budget(scene_step=17):
        now[0] += 40
        with control.suspended_budget(scene_step=17):
            now[0] += 10
        assert control.active_seconds == 0
    with pytest.raises(SamplingInterrupted):
        with control.suspended_budget(scene_step=17):
            now[0] += 20
            assert control.paused_seconds == SCENE_WAIT_LIMIT_SECONDS
            raise SamplingInterrupted()
    assert control.active_seconds == 10
    assert [fields['scene_wait_step'] for fields in statuses] == [17, None, 17, None]
    with control.suspended_budget(scene_step=18):
        now[0] += 25
    assert control.active_seconds == 10 and control.paused_seconds == SCENE_WAIT_LIMIT_SECONDS + 25


def test_live_training_uses_guard_time_for_stage_and_boundary_decisions(tmp_path):
    timing = {}
    statuses = []
    game = LearningFixture()
    game.current.stage_id = 2
    control = TrainingControl(tmp_path, lambda: None, lambda **row: statuses.append(row),
                              budget_seconds=1, session_time=lambda: timing,
                              clock=lambda: pytest.fail('live control started a local timer'))
    control.start()
    assert statuses == [{'clock_started': True}]
    assert control.active_seconds is None and not control.budget_reached
    timing.update(training_seconds=59.9, paused_seconds=25, budget_seconds=60)
    with control.suspended_budget(scene_step=3):
        control.check_budget(game.current)
        assert control.active_seconds == 59.9 and control.paused_seconds == 25
    assert statuses[-2:] == [{'clock_paused': False, 'scene_wait_step': 3},
                              {'clock_paused': False, 'scene_wait_step': None}]
    timing['training_seconds'] = 60
    assert control.budget_reached
    control.check_budget(game.current)
    assert control.overtime_stage == 2
    with pytest.raises(FinishTraining) as caught:
        control.check_budget(boundary='stage_complete')
    assert caught.value.reason == 'stage_complete'


def test_timed_practice_finish_saves_without_entering_another_round(tmp_path, monkeypatch):
    from th06ai.config import Config
    now = [0.]

    class Practice(LearningFixture):
        resets = 0

        def reset(self, **kwargs):
            self.resets += 1
            self.current.stage_id = 1
            self.current.game_mode = 'practice'
            return super().reset(**kwargs)

        def step(self, action):
            if self.local_steps == 10:
                raise SamplingInterrupted(episode_end={
                    'episode_end_reason': 'practice_result', 'stage_id': 1, 'died': False,
                    'episode': {'r': 0., 'l': 10, 't': 1.}, 'survival_seconds': 1.,
                })
            now[0] += 1
            return super().step(action)

    game = Practice()
    model = learning.PausablePPO(learning.ResourcePolicy, game, device='cpu', n_steps=8, batch_size=8, n_epochs=1,
                                 policy_kwargs={'features_extractor_class': learning.BulletEncoder})
    monkeypatch.setattr(learning, 'make_model', lambda *args: model)
    monkeypatch.setattr(learning, 'ROOT', tmp_path)
    monkeypatch.setattr(learning, 'TrainingControl',
                        lambda *args, **kwargs: TrainingControl(*args, **kwargs, clock=lambda: now[0]))
    statuses = []
    learning.train(Config(game_exe='unused', device='cpu'), game, tmp_path, 9/60,
                   lambda **values: statuses.append(values), lambda: None, resume=None)
    assert game.resets == 1 and model.num_timesteps == 10 and model._n_updates == 1
    assert any(status.get('overtime_stage') == 1 for status in statuses)
    assert (tmp_path / 'models' / 'latest.zip').is_file()
    summary = json.loads((tmp_path / 'training_summary.json').read_text(encoding='utf-8'))
    assert summary['budget_seconds'] == 9 and summary['overtime_stage'] == 1 and summary['episodes'] == 1
    assert (tmp_path / 'observations-000000009-000000010.npz').is_file()


@pytest.mark.parametrize('target_during_reset', [False, True])
def test_target_crossed_on_a_miss_delivers_terminal_observation_without_restarting(tmp_path, target_during_reset):
    from test_rewards import frame
    from test_sampling_timing import game

    before = replace(frame(tick=100), stage_id=1, game_mode='practice')
    env, actions, releases, _ = game(before, replace(before, tick=104, state=2))
    now = [0.]
    control = TrainingControl(tmp_path, env.lease, env.status, budget_seconds=1., clock=lambda: now[0])
    env.training_control = control
    control.start()
    read = env.reader.read

    def capture():
        sample = read()
        if sample.state == 2 and not target_during_reset:
            now[0] = 1.
        return sample

    env.reader.read = capture
    env.first_reset = False

    def restart():
        assert target_during_reset, 'auto-reset must not start a new attempt after the target'
        now[0] = 1.
        env._enter_from_menu()

    env._restart = restart
    terminal, _, died, _, info = env.step(0)
    assert died
    observation, _ = env.reset()
    assert observation is terminal and releases and actions == [{'z'}]
    callback = learning.TrainingCallback(env, tmp_path, 1/60, env.status, env.lease, control)
    with pytest.raises(FinishTraining) as caught:
        callback.record_episode_end(info)
    assert caught.value.reason == 'stage_ended' and callback.episode_rows[0]['died']
