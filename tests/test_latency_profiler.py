"""Profiling is optional and never changes an action/value distribution."""
import importlib.util
from pathlib import Path
import numpy as np
import pytest
import torch
from th06ai import learning
from th06ai.inference import PolicyRunner
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
        assert timing['gpu_stream_ms'] is None
        assert all(timing[key]>=0 for key in ('prepare_ms','submit_ms','result_wait_copy_ms'))
        assert timing['total_ms']==pytest.approx(timing['prepare_ms']+timing['submit_ms']+timing['result_wait_copy_ms'])


def test_profile_percentiles_keep_missing_device_measurements_unknown():
    assert percentiles([{'gpu':None}], 'gpu') is None
    assert percentiles([{'gpu':None},{'gpu':2},{'gpu':4}],'gpu')['p50_ms']==3


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
