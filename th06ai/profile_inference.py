"""Guarded, inference-only benchmark. Never opens the game or saves a model."""
from __future__ import annotations
from .i18n import msg
import hashlib
from pathlib import Path
import time
import numpy as np
import torch
from .config import ROOT
from .files import atomic_json
from .inference import PolicyRunner
from .learning import load_model
from .thread_profile import thread_cycle_reader


def profile_inference(config, run_dir, source, status, lease, repeats=96):
    path=Path(source or ROOT/'models/latest.zip').resolve()
    digest=hashlib.sha256(path.read_bytes()).hexdigest()
    status(phase=msg('backend.offline_inference_benchmark'),message=msg('backend.replaying_recorded_observations_without_game_reads_key'))
    model=load_model(path,None,config.device)
    traces=sorted((ROOT/'runs').glob('*/observations-*.npz'),key=lambda p:p.stat().st_mtime,reverse=True)
    if not traces:raise RuntimeError(msg('backend.recorded_ppo_observations_are_required_for_an'))
    with np.load(traces[0],allow_pickle=False) as data:
        count=len(data['steps'])
        observations=[]
        for index in np.linspace(0,count-1,min(count,16),dtype=int):
            observation={}
            for name,space in model.observation_space.spaces.items():
                key='obs_'+name
                observation[name]=data[key][index,0].copy() if key in data else np.zeros(space.shape,np.float32)
            observations.append(observation)
    for goal in ('survival','nmnb'):
        observation={key:value.copy() for key,value in observations[0].items()}
        observation['decision'][:]=[1,1,0,1,float(goal!='nmnb'),.25,0,0]
        observation['resources'][1]=3/8
        observation['control'][18]=0
        observations.append(observation)
    eager=PolicyRunner(model.policy,'eager')
    fast=PolicyRunner(model.policy,config.inference_backend,readback=config.inference_readback)
    eager.warmup(observations[0],lease)
    fast.warmup(observations[0],lease)
    # PPO uses in-place optimizer updates. Verify captured inference reads the
    # live parameter storage rather than a frozen copy. Restore RAM exactly;
    # there is no optimizer step, checkpoint write or learned-state change.
    original_bias=model.policy.value_net.bias.detach().clone()
    before=fast.predict(observations[0],True,lease)
    try:
        with torch.no_grad():model.policy.value_net.bias.add_(1)
        after=fast.predict(observations[0],True,lease)
        if not torch.allclose(after.values,before.values+1,atol=1e-4):
            raise RuntimeError(msg('backend.captured_inference_did_not_read_the_current'))
    finally:
        with torch.no_grad():model.policy.value_net.bias.copy_(original_bias)
    restored=fast.predict(observations[0],True,lease)
    if not torch.equal(restored.values,before.values):raise RuntimeError(msg('backend.benchmark_parameter_restoration_failed_verification'))
    errors=[]
    for observation in observations:
        lease()
        a=eager.predict(observation,True,lease)
        b=fast.predict(observation,True,lease)
        if not np.array_equal(a.actions,b.actions):raise RuntimeError(msg('backend.optimized_and_reference_inference_selected_different_deterministic'))
        errors.append(max(float(torch.max(torch.abs(a.values-b.values))),
                          float(torch.max(torch.abs(a.log_probs-b.log_probs))),abs(a.bomb_probability-b.bomb_probability)))
        sample=fast.predict(observation,False,lease)
        tensor,_=model.policy.obs_to_tensor(observation)
        with torch.no_grad():
            values,logs,_=model.policy.evaluate_actions(tensor,torch.as_tensor(sample.actions,device=model.device))
        error=max(float(torch.max(torch.abs(values.cpu()-sample.values))),float(torch.max(torch.abs(logs.cpu()-sample.log_probs))))
        if error>1e-4:raise RuntimeError(msg('backend.sampled_action_and_training_probability_disagree_0', error))
        errors.append(error)
    measurements={'original_call':[],'buffered_eager':[],'optimized':[]}
    for index in range(repeats):
        observation=observations[index%len(observations)]
        # Alternate order so one path does not always receive the cold slot.
        names=list(measurements)
        names=names[index%3:]+names[:index%3]
        for name in names:
            lease()
            start=time.perf_counter_ns()
            if name=='original_call':
                with torch.no_grad():
                    tensor,_=model.policy.obs_to_tensor(observation)
                    actions,values,logs=model.policy(tensor,deterministic=False)
                    actions.cpu().numpy()
                    getattr(model.policy,'last_bomb_probability').cpu().numpy()
            else:(eager if name=='buffered_eager' else fast).predict(observation,False,lease)
            measurements[name].append((time.perf_counter_ns()-start)/1e6)
        status(profile_samples=index+1,inference=fast.diagnostics())
    summaries={name:{'samples':len(values),'p50_ms':float(np.percentile(values,50)),
                         'p95_ms':float(np.percentile(values,95)),'p99_ms':float(np.percentile(values,99))}
               for name,values in measurements.items()}
    unchanged=hashlib.sha256(path.read_bytes()).hexdigest()==digest
    report={'mode':'inference_only','game_opened':False,'keys_sent':False,'weights_updated':False,
            'source':str(path),'source_sha256':digest,'source_unchanged':unchanged,
            'steps':model.num_timesteps,'updates':model._n_updates,'trace':str(traces[0]),
            'device':str(model.device),'torch':torch.__version__,'engine':fast.engine,
            'live_parameter_storage_verified':True,'in_memory_probe_restored':True,
            'fallback_reason':fast.fallback_reason,'max_equivalence_error':max(errors),
            'latency':summaries,'raw_ms':measurements,
            'scope':'Inference microbenchmark on recorded inputs; not an end-to-end live latency or survival measurement.'}
    atomic_json(Path(run_dir)/'inference_profile.json',report)
    if not unchanged:raise RuntimeError(msg('backend.the_source_checkpoint_changed_during_the_benchmark'))
    status(phase=msg('backend.offline_inference_benchmark_complete'),message=msg('backend.the_benchmark_left_model_files_intact_and'),inference=summaries['optimized'])
    return report


def _readback_observations(spaces, paths, limit=64):
    fields = {name: 'obs_' + name for name in spaces}
    for path in paths:
        with np.load(path, allow_pickle=False) as trace:
            if 'steps' not in trace or any(key not in trace for key in fields.values()):
                continue
            steps = trace['steps']
            if steps.ndim != 1 or not len(steps):
                continue
            indices = np.linspace(0, len(steps) - 1, min(limit, len(steps)), dtype=int)
            columns = {}
            for name, key in fields.items():
                values = trace[key]
                if values.shape != (len(steps), 1, *spaces[name].shape):
                    break
                columns[name] = values[indices, 0]
            if len(columns) == len(fields):
                return path, [(int(steps[index]), {name: values[i] for name, values in columns.items()})
                              for i, index in enumerate(indices)]
    raise RuntimeError(msg('backend.readback_requires_matching_full_observations'))


def _verify_readback_outputs(first, second, step):
    if not (np.array_equal(first.actions, second.actions) and torch.equal(first.values, second.values)
            and torch.equal(first.log_probs, second.log_probs) and first.bomb_probability == second.bomb_probability):
        raise RuntimeError(msg('backend.readback_outputs_differ_at_step_0', step))


def _summarize_readback(rows):
    summary = {}
    for key in ('wall_ms', 'prepare_ms', 'submit_ms', 'result_wait_copy_ms', 'gpu_stream_ms',
                'thread_cycles', 'readback_thread_cycles'):
        values = [row[key] for row in rows if row[key] is not None]
        summary[key] = ({'mean': float(np.mean(values)),
                         **{f'p{q}': float(np.percentile(values, q)) for q in (50, 95, 99)}} if values else None)
    return summary


def profile_readback(config, run_dir, source, status, lease, repeats=512):
    """Compare two host readbacks on the same protected CUDA graph and inputs."""
    if config.device != 'cuda':
        raise RuntimeError(msg('backend.cuda_readback_requires_a_cuda_model'))
    path = Path(source or ROOT / 'models/latest.zip').resolve()
    if not path.is_relative_to((ROOT / 'models').resolve()):
        raise RuntimeError(msg('backend.only_project_generated_models_within_the_models'))
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    status(phase=msg('backend.cuda_readback_benchmark'), profile_pairs=0)
    lease()
    model = load_model(path, None, config.device)
    traces = sorted((ROOT / 'runs').glob('*/observations-*.npz'), key=lambda p: p.stat().st_mtime, reverse=True)
    trace, observations = _readback_observations(model.observation_space.spaces, traces)
    runner = PolicyRunner(model.policy, 'cuda_graph', profile=True, readback=config.inference_readback)
    runner.warmup(observations[0][1], lease)
    completion = torch.cuda.Event(blocking=True)
    methods = (('synchronous', None), ('blocking_event', completion))
    expected_readback = 'blocking_event' if config.inference_readback == 'auto' else 'synchronous'
    for step, observation in observations:
        first, second = (runner.predict(observation, True, lease, readback_event=event) for _, event in methods)
        _verify_readback_outputs(first, second, step)
        configured = runner.predict(observation, True, lease)
        if configured.timing['readback'] != expected_readback:
            raise RuntimeError(msg('backend.configured_readback_mismatch', expected_readback, configured.timing['readback']))
        _verify_readback_outputs(first, configured, step)

    read_cycles = thread_cycle_reader()
    trials = {name: [] for name, _ in methods}
    for index in range(repeats):
        step, observation = observations[index % len(observations)]
        lease()
        rng = torch.cuda.get_rng_state(model.device)
        pair = []
        for name, event in methods if index % 2 == 0 else reversed(methods):
            # Restore the same generator offset outside both measured intervals.
            torch.cuda.set_rng_state(rng, model.device)
            cpu_start = read_cycles() if read_cycles is not None else None
            start = time.perf_counter_ns()
            decision = runner.predict(observation, False, lease, readback_event=event)
            wall_ms = (time.perf_counter_ns() - start) / 1e6
            cpu_end = read_cycles() if read_cycles is not None else None
            cpu = cpu_end - cpu_start if cpu_start is not None and cpu_end is not None and cpu_end >= cpu_start else None
            trials[name].append({'step': step, 'wall_ms': wall_ms, 'thread_cycles': cpu, **decision.timing})
            pair.append(decision)
        _verify_readback_outputs(*pair, step)
        if (index + 1) % 32 == 0:
            status(profile_pairs=index + 1)

    unchanged = hashlib.sha256(path.read_bytes()).hexdigest() == digest
    report = {'mode': 'cuda_readback', 'game_opened': False, 'keys_sent': False, 'weights_updated': False,
              'device': str(model.device), 'torch': torch.__version__, 'engine': runner.engine,
              'source': str(path), 'source_sha256': digest, 'source_unchanged': unchanged,
              'steps': model.num_timesteps, 'trace': str(trace), 'observations': len(observations),
              'deterministic_pairs_verified': len(observations), 'sampled_pairs_verified': repeats,
              'configured_readback': runner.diagnostics()['readback'],
              'configured_readback_outputs_verified': len(observations),
              'summaries': {name: _summarize_readback(rows) for name, rows in trials.items()},
              'trials': trials}
    atomic_json(Path(run_dir) / 'readback_profile.json', report)
    if not unchanged:
        raise RuntimeError(msg('backend.the_source_checkpoint_changed_during_the_benchmark'))
    status(phase=msg('backend.offline_inference_benchmark_complete'), profile_pairs=repeats)
    return report
