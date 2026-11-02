"""Guarded read/plan/inference trace; no key controller or PPO update exists here."""
from __future__ import annotations
from dataclasses import replace
import hashlib
import ctypes as ct
import json
from pathlib import Path
import sys
import time
import numpy as np
from .config import ROOT
from .files import atomic_json,read_json
from .inference import PolicyRunner
from .learning import load_model
from .native import Frame,GameReader
from .observations import ObservationBuilder,action_keys
from .planning import RollingPlanner
from .input_timing import INPUT_BITS,MOVEMENT_MASK
from .diagnostics import SessionDiagnostics


def percentiles(rows,key):
    values=[r[key] for r in rows if r.get(key) is not None]
    return {f'p{p}_ms':float(np.percentile(values,p)) for p in (50,95,99)} if values else None


def profile_latency(config,run_dir,source,heartbeat,repeats=96,*,wait_for_combat=False):
    path=Path(source or ROOT/'models/latest.zip').resolve()
    digest=hashlib.sha256(path.read_bytes()).hexdigest()
    heartbeat.update(phase='只读延迟剖析',message='只读游戏和推理计时；不发键、不训练')
    model=load_model(path,None,config.device)
    runner=PolicyRunner(model.policy,config.inference_backend,profile=True)
    runner.warmup(lease=heartbeat.lease)
    reader=GameReader(config.game_path)
    reader.checkpoint=heartbeat.lease
    diagnostics=SessionDiagnostics(run_dir,'train')
    reader.observer=diagnostics.observe
    live_builder=ObservationBuilder(RollingPlanner(config.planning),goal=config.goal)
    query_cpu=ct.PyDLL('kernel32').GetCurrentProcessorNumber
    query_cpu.restype=ct.c_uint32
    # Use a real recorded scene for the paired replay controls. No recorded
    # scene is presented as a live observation or put in a training buffer.
    with np.load(ROOT/'tests/fixtures/stage5-switch-window.npz',allow_pickle=False) as data:
        def f(index,suffix):
            row=data['frames'][index];b=data['bullets_'+suffix]
            return Frame(int(row[0]),float(row[2]),float(row[3]),int(row[4]),bool(row[6]),False,b[:,:3],
                (8,16,368,416),True,0,bullet_sizes=b[:,3:5],player_radius=float(row[7]),
                enemy_bodies=data['enemy_bodies_'+suffix],lasers=data['lasers_'+suffix])
        previous,recorded=f(0,'previous'),f(1,'current')
    rows=[]
    interval=sys.getswitchinterval()
    historical=read_json(ROOT/'runs/20261102-183615-816926-train/result.json').get('worker',{})
    keep={key:value for key,value in heartbeat.data.items() if key in ('pid','lease_diagnostics')}
    modes=('live_full','replay_full','replay_minimal','replay_switch_1ms')
    try:
        if wait_for_combat:
            heartbeat.update(profile_ready=True,phase='只读诊断已就绪',message='等待手动解除游戏暂停；不发送按键')
            deadline=time.monotonic()+60
            while True:
                heartbeat.lease()
                probe=reader.read()
                if probe.phase in ('combat','boss') and not probe.paused:break
                if time.monotonic()>deadline:raise RuntimeError('未等到手动解除暂停；只读诊断结束，未发送按键')
                time.sleep(.05)
            modes=('live_full',)
            repeats=256
        live_start=time.monotonic()
        for index in range(repeats):
            if wait_for_combat and time.monotonic()-live_start>=12:break
            mode=modes[index%len(modes)]
            sys.setswitchinterval(.001 if mode=='replay_switch_1ms' else interval)
            # Mandatory heartbeat remains live in all modes; only diagnostic
            # payload size differs. No hardware authorization is disabled.
            if index%len(modes)==0 or mode=='replay_minimal' or mode=='replay_switch_1ms':
                with heartbeat.lock:
                    # Exercise equivalent serialization work without presenting
                    # an old session's game/training fields as current facts.
                    payload={'diagnostic_load_payload':historical} if mode!='replay_minimal' else {}
                    heartbeat.data={**payload,**keep,
                        'phase':'只读延迟剖析','user_paused':False,'error':None,'profile_mode':mode}
            heartbeat.lease()
            cpu_before=int(query_cpu())
            started=time.perf_counter_ns()
            actual=reader.read()
            if wait_for_combat and actual.paused:break
            captured=time.perf_counter_ns()
            if mode=='live_full':
                builder=live_builder
                current=actual
                bits=actual.input_bits
                action=next((a for a in range(18) if bits is not None and
                    sum(INPUT_BITS.get(key,0) for key in action_keys(a))==(bits&MOVEMENT_MASK)),None)
                builder.set_control(action)
            else:
                # Independent, fixed-age replay state avoids a paused/live
                # frame resetting another mode's motion or latency history.
                builder=ObservationBuilder(RollingPlanner(config.planning),goal=config.goal)
                builder.previous=previous
                builder._motion_previous=None
                builder.set_control(4)
                current=replace(recorded,capture_started_ns=0,observed_at=0)
            observation=builder.build(current)
            planned=time.perf_counter_ns()
            policy_start=time.perf_counter_ns()
            decision=runner.predict(observation,lease=heartbeat.lease)
            inferred=time.perf_counter_ns()
            reader.read_control()
            checked=time.perf_counter_ns()
            diag=builder.last_plan.diagnostics
            row={'index':index,'mode':mode,'phase':actual.phase,'tick':actual.tick,'bullets':len(current.bullets),
                'search_backend':diag.get('search_backend'),
                'cpu_before':cpu_before,'cpu_after':int(query_cpu()),
                'considered_bullets':diag.get('considered_bullets'),'capture_ms':(captured-started)/1e6,
                'build_ms':(planned-captured)/1e6,
                'planner_ms':diag.get('elapsed_ms') if diag.get('enabled') else None,
                'policy_ms':(inferred-policy_start)/1e6,'control_probe_ms':(checked-inferred)/1e6,
                'read_to_ready_ms':(checked-started)/1e6,**decision.timing}
            rows.append(row)
            heartbeat.update(profile_samples=index+1,profile_mode=mode,phase='只读延迟剖析',
                game_phase=actual.phase,paused=actual.paused,
                inference=decision.timing,message='没有发送按键，未更新模型')
            time.sleep(.033 if wait_for_combat else .01)
    finally:
        sys.setswitchinterval(interval)
        reader.close();diagnostics.close()
    report={'mode':'read_only_latency','keys_sent':False,'weights_updated':False,'source_sha256':digest,
        'source_unchanged':hashlib.sha256(path.read_bytes()).hexdigest()==digest,'engine':runner.engine,
        'waited_for_manual_combat':wait_for_combat,
        'summaries':{mode:{key:percentiles([r for r in rows if r['mode']==mode],key) for key in
             ('capture_ms','build_ms','planner_ms','policy_ms','prepare_ms','submit_ms','result_wait_copy_ms','gpu_stream_ms',
              'control_probe_ms','read_to_ready_ms')} for mode in modes},'rows':rows,
        'scope':'Live rows reflect the observed game phase. Replay rows use recorded geometry plus actual read I/O; not live transitions.',
        'gpu_event_scope':'Events inside the captured graph measure its device-stream span; host wait also includes queueing and scheduling.',
        'limits':'No SendInput call, PPO rollout callbacks or reward computation is run. This is not the full training loop; no end-to-end key-response, before/after speedup or survival claim can follow.'}
    atomic_json(Path(run_dir)/'latency_profile.json',report)
    if not report['source_unchanged']:raise RuntimeError('源模型在诊断期间发生变化')
    heartbeat.update(phase='只读延迟剖析完成',message='模型文件保持不变；没有启动游戏接管')
    return report
