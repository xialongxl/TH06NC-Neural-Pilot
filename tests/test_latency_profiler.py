"""Profiling is optional and never changes an action/value distribution."""
import importlib.util
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
import torch
from th06ai import learning
from th06ai.inference import PolicyRunner
from th06ai.profile_inference import _readback_observations
from th06ai.profile_latency import percentiles
from test_learning import LearningFixture


def test_cpu_profiling_keeps_outputs_and_timing_stages_consistent():
    torch.set_num_threads(1)
    game=LearningFixture();model=learning.make_model(game,'cpu')
    obs=game.builder.build(game.current)
    plain=PolicyRunner(model.policy,'eager')
    traced=PolicyRunner(model.policy,'eager',profile=True)
    a=plain.predict(obs,True);b=traced.predict(obs,True)
    np.testing.assert_array_equal(a.actions,b.actions)
    assert torch.equal(a.values,b.values) and torch.equal(a.log_probs,b.log_probs)
    assert a.bomb_probability==b.bomb_probability
    for decision in (a,b):
        timing=decision.timing
        assert timing['readback'] == 'synchronous'
        assert timing['gpu_stream_ms'] is None
        assert all(timing[key]>=0 for key in ('prepare_ms','submit_ms','result_wait_copy_ms'))
        assert timing['total_ms']==pytest.approx(timing['prepare_ms']+timing['submit_ms']+timing['result_wait_copy_ms'])


def test_profile_percentiles_keep_missing_device_measurements_unknown():
    assert percentiles([{'gpu':None}], 'gpu') is None
    assert percentiles([{'gpu':None},{'gpu':2},{'gpu':4}],'gpu')['p50_ms']==3


def test_result_readback_waits_for_copy_on_the_model_stream_before_exposing_host_view(monkeypatch):
    calls = []
    stream, result, tensor = object(), object(), object()

    class BufferOwner:
        device = torch.device('cuda:0')
        result_host = SimpleNamespace(copy_=lambda source, *, non_blocking: calls.append(('copy', source, non_blocking)))

        @property
        def result_view(self):
            calls.append(('read',))
            return result

    owner = BufferOwner()
    monkeypatch.setattr(torch.cuda, 'current_stream', lambda device: stream if device == owner.device else None)
    completion = SimpleNamespace(record=lambda value: calls.append(('record', value)),
                                 synchronize=lambda: calls.append(('wait',)))
    output = SimpleNamespace(detach=lambda: tensor)
    assert PolicyRunner._read_result(owner, output, completion) is result
    assert calls == [('copy', tensor, True), ('record', stream), ('wait',), ('read',)]
    calls.clear()
    assert PolicyRunner._read_result(owner, output, None) is result
    assert calls == [('copy', tensor, False), ('read',)]


@pytest.mark.parametrize('backend,readback,capture_fails,expected', [
    ('auto', 'auto', False, 'blocking_event'),
    ('auto', 'synchronous', False, 'synchronous'),
    ('auto', 'auto', True, 'synchronous'),
    ('eager', 'auto', False, 'synchronous'),
])
def test_graph_readback_default_and_explicit_sync(monkeypatch, backend, readback, capture_fails, expected):
    game = LearningFixture()
    model = learning.make_model(game, 'cpu')
    runner = PolicyRunner(model.policy, backend, readback=readback)
    observation = game.builder.build(game.current)
    # Exercise warmup and readback selection with CPU buffers and a fake CUDA stream.
    runner.device = torch.device('cuda:0')
    runner.engine = 'cuda_eager'
    output = torch.tensor([[0.0, 0.25, -0.5, 0.0]])
    monkeypatch.setattr(runner, '_submit', lambda _: output)
    stream = SimpleNamespace(wait_stream=lambda _: None)
    monkeypatch.setattr(torch.cuda, 'Stream', lambda **_: stream)
    monkeypatch.setattr(torch.cuda, 'current_stream', lambda _: stream)
    monkeypatch.setattr(torch.cuda, 'stream', lambda _: nullcontext())
    monkeypatch.setattr(torch.cuda, 'synchronize', lambda _: None)
    monkeypatch.setattr(torch.cuda, 'graph', lambda _: nullcontext())

    def create_graph():
        if capture_fails:
            raise RuntimeError('graph capture unavailable')
        return SimpleNamespace(replay=lambda: None)

    waits = []

    def create_event(*, blocking):
        assert blocking
        return SimpleNamespace(record=lambda _: None, synchronize=lambda: waits.append(True))

    monkeypatch.setattr(torch.cuda, 'CUDAGraph', create_graph)
    monkeypatch.setattr(torch.cuda, 'Event', create_event)
    assert runner.predict(observation, True).timing['readback'] == expected
    assert runner.diagnostics()['readback'] == expected
    assert runner.predict(observation, True, readback_event=None).timing['readback'] == 'synchronous'
    assert waits == ([True] if expected == 'blocking_event' else [])


def test_readback_profile_uses_only_complete_observations_with_the_model_shape(tmp_path):
    summary, invalid, full = (tmp_path / name for name in ('summary.npz', 'invalid.npz', 'full.npz'))
    steps = np.array([10, 11, 12])
    observations = np.arange(6, dtype=np.float32).reshape(3, 1, 2)
    np.savez(summary, steps=steps)
    np.savez(invalid, steps=steps, obs_player=observations.reshape(3, 2))
    np.savez(full, steps=steps, obs_player=observations)
    path, selected = _readback_observations({'player': SimpleNamespace(shape=(2,))}, [summary, invalid, full], limit=2)
    assert path == full and [step for step, _ in selected] == [10, 12]
    np.testing.assert_array_equal(selected[-1][1]['player'], observations[-1, 0])


def test_contiguous_runner_matches_policy_and_returns_independent_results():
    torch.set_num_threads(1)
    game=LearningFixture();model=learning.make_model(game,'cpu')
    obs=game.builder.build(game.current)
    runner=PolicyRunner(model.policy,'eager')
    decision=runner.predict(obs,True)
    with torch.no_grad():
        values=model.policy(model.policy.obs_to_tensor(obs)[0],deterministic=True)
    np.testing.assert_array_equal(decision.actions,values[0].numpy())
    torch.testing.assert_close(decision.values,values[1],rtol=0,atol=0)
    torch.testing.assert_close(decision.log_probs,values[2],rtol=0,atol=0)
    saved_value=decision.values.clone();saved_log=decision.log_probs.clone()
    changed={key:value.copy() for key,value in obs.items()};changed['player'][0]=.75
    runner.predict(changed,True)
    assert torch.equal(decision.values,saved_value) and torch.equal(decision.log_probs,saved_log)
    assert all(value.untyped_storage().data_ptr()==runner.host_flat.untyped_storage().data_ptr()
               for value in runner.host.values())


def test_diagnostic_native_entry_keeps_owned_geometry_and_checks_shapes():
    source=Path(__file__).resolve().parents[1]/'research/latency-cause-20261102/_native_backend.cp310-win_amd64.pyd'
    if not source.exists():pytest.skip('optional independent diagnostic extension not built')
    spec=importlib.util.spec_from_file_location('latency_tests._native_backend',source)
    probe=importlib.util.module_from_spec(spec);spec.loader.exec_module(probe)
    # n=1: start, velocity, duration, start tick, low and high bounds.
    data=np.array([100,100,2,0,2,0,8,16,376,432],dtype='<f4').tobytes()
    ordinary=probe.sweep_bytes(data,1,0,0,0)
    for release in (True,False):
        result,compute,handoff=probe.profile_sweep_bytes(data,1,0,0,0,release)
        assert result==ordinary and compute>=0 and handoff>=0
    with pytest.raises(ValueError):probe.profile_sweep_bytes(b'invalid',1,0,0,0)
