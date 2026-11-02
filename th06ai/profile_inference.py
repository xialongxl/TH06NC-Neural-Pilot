"""Guarded, inference-only benchmark. Never opens the game or saves a model."""
from __future__ import annotations
import hashlib
from pathlib import Path
import time
import numpy as np
import torch
from .config import ROOT
from .files import atomic_json
from .inference import PolicyRunner
from .learning import load_model


def profile_inference(config, run_dir, source, status, lease, repeats=96):
    path=Path(source or ROOT/'models/latest.zip').resolve()
    digest=hashlib.sha256(path.read_bytes()).hexdigest()
    status(phase='离线推理基准',message='只重放已记录观测，不读取游戏、不发按键、不更新权重')
    model=load_model(path,None,config.device)
    traces=sorted((ROOT/'runs').glob('*/observations-*.npz'),key=lambda p:p.stat().st_mtime,reverse=True)
    if not traces:raise RuntimeError('没有已记录的 PPO 观测，无法进行真实输入基准')
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
    fast=PolicyRunner(model.policy,config.inference_backend)
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
            raise RuntimeError('捕获推理没有读取当前参数')
    finally:
        with torch.no_grad():model.policy.value_net.bias.copy_(original_bias)
    restored=fast.predict(observations[0],True,lease)
    if not torch.equal(restored.values,before.values):raise RuntimeError('基准参数恢复验证失败')
    errors=[]
    for observation in observations:
        lease()
        a=eager.predict(observation,True,lease)
        b=fast.predict(observation,True,lease)
        if not np.array_equal(a.actions,b.actions):raise RuntimeError('优化推理与常规推理的确定性动作不一致')
        errors.append(max(float(torch.max(torch.abs(a.values-b.values))),
                          float(torch.max(torch.abs(a.log_probs-b.log_probs))),abs(a.bomb_probability-b.bomb_probability)))
        sample=fast.predict(observation,False,lease)
        tensor,_=model.policy.obs_to_tensor(observation)
        with torch.no_grad():
            values,logs,_=model.policy.evaluate_actions(tensor,torch.as_tensor(sample.actions,device=model.device))
        error=max(float(torch.max(torch.abs(values.cpu()-sample.values))),float(torch.max(torch.abs(logs.cpu()-sample.log_probs))))
        if error>1e-4:raise RuntimeError(f'采样动作与训练概率不一致：{error}')
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
    if not unchanged:raise RuntimeError('源检查点在基准期间发生变化')
    status(phase='离线推理基准完成',message='模型文件保持不变；未启动游戏接管',inference=summaries['optimized'])
    return report
