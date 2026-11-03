"""Actual rollout-buffer traces, without a game or GPU."""
import json

import numpy as np
import torch
import pytest

from th06ai import learning
from test_learning import LearningFixture


def partial_callback(tmp_path):
    torch.set_num_threads(1)
    game = LearningFixture()
    model = learning.make_model(game, "cpu")
    callback = learning.TrainingCallback(game, tmp_path, 1, lambda **_: None, lambda: None)
    callback.init_callback(model)
    callback.rollout_first_step = 1
    # An incomplete/external buffer may lag a counter. Trace identifiers must
    # come from its actual contents, never from the counter alone.
    callback.num_timesteps = 3
    observations = []
    for action in (3, 4):
        game.builder.set_control(action)
        observation = game.builder.build(game.current)
        batch = {name: value[None].copy() for name, value in observation.items()}
        observations.append(batch)
        model.rollout_buffer.add(batch, np.array([[action]]), np.array([.25]), np.array([False]),
                                 torch.zeros(1), torch.zeros(1))
    return callback, observations


def test_partial_trace_uses_buffer_steps_and_exact_input_arrays(tmp_path):
    callback, observations = partial_callback(tmp_path)
    callback.flush_observations()
    path = tmp_path / "observations-000000001-000000002.npz"
    with np.load(path, allow_pickle=False) as trace:
        assert np.array_equal(trace["steps"], [1, 2])
        assert np.array_equal(trace["actions"].ravel(), [3, 4])
        for name in observations[0]:
            assert np.array_equal(trace["obs_"+name], np.stack([obs[name] for obs in observations]))
        assert json.loads(str(trace["metadata"]))["policy_checkpoint"] is None
        assert json.loads(str(trace["metadata"]))["gae_ready"] is False
        assert "returns" not in trace.files and "advantages" not in trace.files
    original = path.read_bytes()
    callback.flush_observations()
    assert path.read_bytes() == original


def test_optional_trace_io_failure_reports_without_stopping_training(tmp_path, monkeypatch):
    callback, _ = partial_callback(tmp_path)
    messages = []
    callback.status = lambda **values: messages.append(values)
    def unavailable(*args, **kwargs):
        raise PermissionError("diagnostic destination unavailable")
    monkeypatch.setattr(np, "savez_compressed", unavailable)
    callback.flush_observations()
    assert messages == [{"trajectory_error": "diagnostic destination unavailable"}]


def test_async_trace_owns_data_before_rollout_buffer_is_reused(tmp_path):
    from th06ai.diagnostics import SessionDiagnostics
    callback, observations=partial_callback(tmp_path)
    callback.game.diagnostics=SessionDiagnostics(tmp_path,'train')
    try:
        callback.flush_observations()
        for values in callback.model.rollout_buffer.observations.values():values.fill(-.123)
    finally:callback.game.diagnostics.close(timeout=5)
    with np.load(tmp_path/'observations-000000001-000000002.npz',allow_pickle=False) as trace:
        for name in observations[0]:
            np.testing.assert_array_equal(trace['obs_'+name],np.stack([obs[name] for obs in observations]))


@pytest.mark.parametrize('level',['summary','off'])
def test_reduced_trace_levels_preserve_explicit_semantics(tmp_path,level):
    from th06ai.config import DiagnosticsConfig
    callback,_=partial_callback(tmp_path)
    callback.diagnostics_config=DiagnosticsConfig(trajectory=level)
    callback.flush_observations()
    path=tmp_path/'observations-000000001-000000002.npz'
    if level=='off':
        assert not path.exists()
    else:
        with np.load(path,allow_pickle=False) as trace:
            assert not any(name.startswith('obs_') for name in trace.files)
            assert 'actions' in trace.files and json.loads(str(trace['metadata']))['trajectory_level']=='summary'


def test_unsaved_updates_do_not_mislabel_previous_checkpoint_as_exact_sampling_policy(tmp_path):
    callback,_=partial_callback(tmp_path)
    callback.rollout_checkpoint='older-saved-checkpoint.zip'
    callback.model._n_updates=4
    callback.flush_observations()
    with np.load(tmp_path/'observations-000000001-000000002.npz',allow_pickle=False) as trace:
        meta=json.loads(str(trace['metadata']))
        assert meta['policy_checkpoint'] is None
        assert meta['last_saved_checkpoint']=='older-saved-checkpoint.zip'


@pytest.mark.parametrize('stop,buffer_size',[(5,512),(4,4)])
def test_budget_finish_preserves_the_last_completed_environment_transition(tmp_path,stop,buffer_size):
    from th06ai.control import TrainingControl
    torch.set_num_threads(1)
    game=LearningFixture()
    model=learning.make_model(game,'cpu')
    model.n_steps=buffer_size
    model.rollout_buffer.buffer_size=buffer_size
    model.rollout_buffer.reset()
    control=TrainingControl(tmp_path,lambda:None,lambda **_:None)
    model.training_game=game
    model.training_control=control
    class FinishAfterBudget(learning.TrainingCallback):
        def _on_step(self):
            return super()._on_step() and self.num_timesteps<stop
    callback=FinishAfterBudget(game,tmp_path,1,lambda **_:None,lambda:None,control)
    model.learn(512,callback=callback)
    assert model.num_timesteps==stop and model.rollout_buffer.pos==stop
    assert model._n_updates==0
    with np.load(tmp_path/f'observations-000000001-{stop:09d}.npz',allow_pickle=False) as z:
        assert np.array_equal(z['steps'],np.arange(1,stop+1))
        assert z['actions'].shape[0]==stop
        assert not json.loads(str(z['metadata']))['gae_ready']
        assert 'returns' not in z.files
