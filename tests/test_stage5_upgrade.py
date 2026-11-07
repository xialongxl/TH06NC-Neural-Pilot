"""Stage 5 regressions: real CPU networks, recorded-shaped data, no game keys."""
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
import csv
import hashlib
import json
import struct
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from th06ai import learning, native
from th06ai.battle_view import BattleViewPublisher
from th06ai.config import Config
from th06ai.control import SamplingInterrupted, TrainingControl
from th06ai.diagnostics import SessionDiagnostics
from th06ai.files import atomic_json
from th06ai.inference import PolicyRunner
from th06ai.native import ControlSample, GameResources
from th06ai.navigation import RESCUE_BOMB_PRIOR, EXIT_PROBE_LIMIT
from th06ai.observations import ObservationBuilder, make_observation_space
from th06ai.pausable_ppo import PausablePPO
from th06ai.planning import RollingPlanner
from th06ai.planner_spec import PlanningConfig
from th06ai.policy import BulletEncoder, ResourcePolicy, BOMB_PRIOR
from test_learning import LearningFixture
from test_rewards import frame
from test_sampling_timing import game
from test_state_capture import UpdatingGameMemory, reader_for


def test_buffered_inference_matches_training_probabilities_and_uses_live_parameters():
    torch.set_num_threads(1)
    env=LearningFixture()
    model=learning.make_model(env)
    observation=env.builder.build(env.current)
    observation['resources'][1]=3/8
    runner=PolicyRunner(model.policy,'eager')
    addresses={key:value.data_ptr() for key,value in runner.host.items()}
    for _ in range(3):
        decision=runner.predict(observation)
        tensor,_=model.policy.obs_to_tensor(observation)
        with torch.no_grad():
            values,logs,_=model.policy.evaluate_actions(tensor,torch.as_tensor(decision.actions))
        assert torch.allclose(values,decision.values,atol=1e-6)
        assert torch.allclose(logs,decision.log_probs,atol=1e-6)
        assert not getattr(model.policy,'_fast_inference',False)
    old=runner.predict(observation,True)
    with torch.no_grad():model.policy.value_net.bias.add_(2)
    new=runner.predict(observation,True)
    assert torch.allclose(new.values,old.values+2,atol=1e-6)
    assert addresses=={key:value.data_ptr() for key,value in runner.host.items()}
    assert 'decision_runner' in model._excluded_save_params()
    invalid={**observation,'player':observation['player'].copy()}
    invalid['player'][0]=np.nan
    with pytest.raises(ValueError):runner.predict(invalid)


def test_shared_inference_buffers_do_not_cross_contaminate_concurrent_callers():
    torch.set_num_threads(1)
    env=LearningFixture()
    model=learning.make_model(env)
    a=env.builder.build(env.current)
    b={key:value.copy() for key,value in a.items()}
    b['player'][:]=[.9,.2,.1,.8]
    runner=PolicyRunner(model.policy,'eager')
    expected=[runner.predict(obs,True).values.clone() for obs in (a,b)]
    with ThreadPoolExecutor(max_workers=2) as pool:
        results=list(pool.map(lambda index:runner.predict((a,b)[index%2],True),range(12)))
    for index,result in enumerate(results):assert torch.equal(result.values,expected[index%2])


def test_rescue_exploration_is_conditional_and_nmnb_cannot_bomb():
    torch.set_num_threads(1)
    env=LearningFixture()
    policy=learning.make_model(env).policy
    obs=env.builder.build(env.current)
    obs['resources'][1]=3/8
    obs['decision'][:]=[1,0,0,0,1,1,0,1]
    def probability():
        tensor,_=policy.obs_to_tensor(obs)
        with torch.no_grad():return policy.get_distribution(tensor).distribution.probs[:,18:].sum().item()
    assert probability()==pytest.approx(BOMB_PRIOR,rel=1e-4)
    obs['decision'][1]=obs['decision'][3]=1
    assert probability()==pytest.approx(RESCUE_BOMB_PRIOR,rel=.03)
    obs['decision'][2]=1
    assert probability()==pytest.approx(BOMB_PRIOR,rel=.03)
    obs['decision'][2]=0
    tensor,_=policy.obs_to_tensor(obs)
    assert policy(tensor,deterministic=True)[0].item()<18  # exploration is not a forced bomb
    obs['control'][18]=1
    assert probability()==pytest.approx(BOMB_PRIOR,rel=.03)
    obs['control'][18]=0
    obs['decision'][4]=0
    with torch.no_grad():policy.action_net.bias[18:]+=30
    assert probability()==0
    obs['decision'][4]=1
    obs['resources'][1]=0
    assert probability()==0


def test_v7_migration_keeps_learned_parameters_and_all_adam_history(tmp_path,monkeypatch):
    torch.set_num_threads(1)
    class Version7(LearningFixture):
        def __init__(self):
            super().__init__();self.observation_space=make_observation_space(7)
        def reset(self,**kwargs):
            obs,info=super().reset(**kwargs)
            return {k:v for k,v in obs.items() if k in self.observation_space.spaces},info
        def step(self,action):
            obs,reward,died,truncated,info=super().step(action)
            return {k:v for k,v in obs.items() if k in self.observation_space.spaces},reward,died,truncated,info
    old=PausablePPO(ResourcePolicy,Version7(),device='cpu',n_steps=32,batch_size=32,n_epochs=1,
        policy_kwargs={'features_extractor_class':BulletEncoder,'net_arch':{'pi':[128,128],'vf':[128,128]}})
    old.learn(64)
    path=tmp_path/'models/v7.zip';path.parent.mkdir();old.save(path)
    digest=hashlib.sha256(path.read_bytes()).hexdigest()
    atomic_json(path.with_suffix('.json'),{'format':7,'policy_version':2,'game_sha256':native.SHA256,
        'model_sha256':digest,'steps':64,'gradient_updates':old._n_updates})
    monkeypatch.setattr(learning,'ROOT',tmp_path)
    new=learning.load_model(path,LearningFixture())
    assert new.num_timesteps==old.num_timesteps and new._n_updates==old._n_updates
    assert not new.bomb_policy_upgraded
    for name,parameter in old.policy.named_parameters():
        current=dict(new.policy.named_parameters())[name]
        if current.shape!=parameter.shape:
            assert torch.equal(current[:,:parameter.shape[1]],parameter)
            assert not current[:,parameter.shape[1]:].any()
        else:assert torch.equal(current,parameter)
        for key,value in old.policy.optimizer.state.get(parameter,{}).items():
            after=new.policy.optimizer.state[current][key]
            if value.ndim and value.shape!=after.shape:
                assert torch.equal(after[:,:value.shape[1]],value) and not after[:,value.shape[1]:].any()
            else:assert torch.equal(after,value)
    assert hashlib.sha256(path.read_bytes()).hexdigest()==digest


def control_sample(f):
    return ControlSample(f.tick,f.x,f.y,f.state,f.paused,f.stage_active,f.stage_id,f.phase,
                         f.input_bits,f.input_observed_ns,f.capture_started_ns,f.capture_finished_ns)


def test_action_hold_polls_small_samples_and_captures_full_pool_only_at_the_end():
    before=frame(tick=100)
    final=replace(before,tick=104)
    env,actions,_,_=game(before,final)
    calls=[]
    samples=iter([replace(before,tick=101),replace(before,tick=102),replace(before,tick=103)])
    env.reader.read_control=lambda:control_sample(next(samples))
    env.reader.read=lambda:(calls.append('full') or final)
    _,_,_,_,info=env.step(0)
    assert calls==['full'] and actions==[{'z'}]
    assert info['pre_dispatch_ticks']==1 and info['action_hold_ticks']==3


def test_lightweight_reader_preserves_native_pause_semantics_without_pool_reads():
    memory=UpdatingGameMemory(advancing=False)
    original=memory.read
    calls=[]
    def read(address,size):
        calls.append((address,size))
        if address in (native.BULLET_MANAGER,native.INPUT_WORD):return original(address,size)
        if address==native.ENGINE_STATE:return struct.pack('<i',2)
        if address==native.NC_PAUSE:return bytes([1,0,0,0,0,0,0,0,0,0])
        return bytes(memory.world[address-native.WORLD_START:address-native.WORLD_START+size])
    memory.read=read
    reader=reader_for(memory)
    sample=reader.read_control()
    assert sample.paused and sample.phase=='pause'
    assert memory.bulk_reads==0 and max(size for _,size in calls)<1024
    assert sample.input_bits==0x84


def test_miss_seen_in_control_sample_survives_a_full_respawn_snapshot(tmp_path):
    before=frame(tick=100)
    respawn=replace(before,tick=106,state=3)
    env,_,_,_=game(before,respawn)
    env.episode_count = 1
    battle_view = BattleViewPublisher(tmp_path)
    env.diagnostics = SimpleNamespace(observe_control=lambda _: None, record=lambda *_: None,
                                      status=lambda: {}, battle_view=battle_view)
    samples=iter([replace(before,tick=102,state=0),replace(before,tick=103,state=2)])
    env.reader.read_control=lambda:control_sample(next(samples))
    env.reader.read=lambda:respawn
    _,_,died,_,info=env.step(0)
    assert died and info['death_evidence']=='control_sample' and info['death_probe_tick']==103
    assert info['reward_components']['death']<0
    assert battle_view.miss_count == 1


def test_stale_decision_retries_once_without_fake_samples_or_repeated_stops():
    before=frame(tick=100)
    latest=replace(before,tick=130)
    env,actions,_,_=game(before,latest)
    env.reader.window.focused=True
    env.reader.read_control=lambda:control_sample(latest)
    env.reader.read=lambda:latest
    with pytest.raises(SamplingInterrupted) as exc:env.refresh_decision()
    assert exc.value.bootstrap_boundary and not actions
    assert env.current is latest and env.total_steps==0 and env.reward_tracker.total_ticks==0
    env.reader.read_control=lambda:control_sample(replace(latest,tick=160))
    env.refresh_decision()
    assert env.freshness_warning['retry_exhausted'] and not actions


def test_freshness_check_does_not_rebase_away_a_real_death():
    before=frame(tick=100)
    dead=replace(before,tick=130,state=2)
    env,_,_,_=game(before,dead)
    env.reader.window.focused=True
    env.reader.read_control=lambda:control_sample(dead)
    env.reader.read=lambda:dead
    env.refresh_decision()
    assert env.current is before
    _,_,died,_,info=env.step(0)
    assert died and info['reward_components']['death']<0


def test_exit_preference_preserves_robust_geometry_mask_and_budget():
    planner=RollingPlanner(PlanningConfig(beam_width=1))
    old=frame(tick=100,x=16,y=424,bullets=[(2,80,300)])
    current=replace(old,tick=102,bullets=np.array([(2,80,304)],np.float32))
    plan=planner.plan(current,old)
    assert plan.navigation.shape==(18,9) and plan.decision.shape==(8,)
    allowed=plan.features[:,1]>.5
    if not plan.diagnostics['all_routes_risky']:
        assert np.all(plan.navigation[allowed,0]==1)
    assert plan.diagnostics['mobility_probes']<=EXIT_PROBE_LIMIT
    assert not plan.diagnostics['rescue_opportunity']  # one avoidable bullet is not a bomb trigger
    assert np.isfinite(plan.navigation).all()


def test_confirmed_closing_ring_enables_exploration_but_unknown_motion_does_not():
    angle=np.arange(48)*2*np.pi/48
    radial=np.column_stack((np.cos(angle),np.sin(angle))).astype(np.float32)
    center=np.array([192,300])
    previous=replace(frame(tick=100,x=192,y=300),
        bullets=np.column_stack((np.arange(48),center+radial*36)).astype(np.float32),
        bullet_sizes=np.full((48,2),4,np.float32),player_radius=2)
    current=replace(previous,tick=102,bullets=np.column_stack((np.arange(48),center+radial*30)).astype(np.float32))
    plan=RollingPlanner(PlanningConfig()).plan(current,previous)
    assert plan.diagnostics['rescue_opportunity'] and not plan.diagnostics['late_control_risk']
    assert plan.diagnostics['route_mode']=='fallback' and not plan.diagnostics['nominal_escape_roots']
    uncertain=RollingPlanner(PlanningConfig()).plan(current)
    assert not uncertain.diagnostics['rescue_opportunity']


def test_actual_runner_rollout_discards_stale_actions_and_cuts_gae_once(tmp_path,monkeypatch):
    from th06ai.control import SamplingInterrupted
    class RefreshGame(LearningFixture):
        def __init__(self):
            super().__init__();self.executed=[];self.refreshes=0;self.observed=[]
        def refresh_decision(self):
            if len(self.executed)==3 and self.refreshes<2:
                self.refreshes+=1
                self.current=replace(self.current,tick=self.current.tick+2,x=self.current.x+20)
                raise SamplingInterrupted(bootstrap_boundary=True)
        def record_decision(self,*args):pass
        def step(self,action):
            self.executed.append(int(action));self.observed.append(self.builder.build(self.current)['player'].copy())
            return super().step(action)
    class Control:
        lease=staticmethod(lambda:None)
        gate=staticmethod(lambda *args,**kwargs:False)
    env=RefreshGame();model=learning.make_model(env)
    model.n_steps=16;model.batch_size=8;model.n_epochs=1
    model.rollout_buffer=model.rollout_buffer_class(16,model.observation_space,model.action_space,
        device=model.device,gamma=model.gamma,gae_lambda=model.gae_lambda,n_envs=1)
    model.decision_runner=PolicyRunner(model.policy,'eager')
    model.training_control=Control();model.training_game=env
    class Verify(learning.TrainingCallback):
        def _on_rollout_end(self):
            buffer=self.model.rollout_buffer
            obs={key:value[:,0] for key,value in buffer.observations.items()}
            tensor,_=self.model.policy.obs_to_tensor(obs)
            with torch.no_grad():_,logs,_=self.model.policy.evaluate_actions(tensor,torch.as_tensor(env.executed))
            assert np.allclose(logs.numpy(),buffer.log_probs[:,0],atol=1e-6)
            assert np.allclose(obs['player'],env.observed)
            assert buffer.episode_starts[3,0]==1 and buffer.episode_starts[4:,0].sum()==0
            old_advantages=buffer.advantages[:3].copy()
            buffer.rewards[3:]+=10000
            buffer.compute_returns_and_advantage(last_values=torch.zeros(1),dones=np.array([False]))
            assert np.array_equal(old_advantages,buffer.advantages[:3])
            super()._on_rollout_end()
    monkeypatch.setattr(learning,'save_model',lambda *args:None)
    model.learn(16,callback=Verify(env,tmp_path,1,lambda **_:None,lambda:None))
    assert env.refreshes==2 and len(env.executed)==16 and model.num_timesteps==16


def test_terminal_snapshot_includes_both_ends_pool_ids_and_stage_in_all_modes(tmp_path):
    for mode in ('train','play','evaluate'):
        directory=tmp_path/mode
        diagnostics=SessionDiagnostics(directory,mode)
        before=replace(frame(tick=600,bullets=[(333,192,370)]),stage_id=5,phase='combat',
            bullet_sizes=np.array([[4,8]],np.float32),bullet_states=np.array([1]),player_radius=2,
            resources=GameResources(0,128,3,2,0),capture_started_ns=1000,capture_finished_ns=2000)
        dead=replace(before,tick=606,state=2,phase='dead',bullets=np.array([(333,192,382)],np.float32))
        diagnostics.observe(before);diagnostics.observe_control(control_sample(dead));diagnostics.observe(dead)
        info={'action':10,'native_tick':606,'died':True,'survival_seconds':10.1,'stage_id':5,'stage_label':'Stage 5',
              'goal':'nmnb','episode_bombs_used':0,'episode_end_reason':'death','episode_diagnostics':{}}
        diagnostics.record(info,70735)
        diagnostics.close()
        summary=json.loads((directory/'diagnostics_summary.json').read_text(encoding='utf-8'))
        assert summary['complete'] and summary['written_deaths']==1 and not summary['errors']
        with np.load(directory/(info['death_id']+'.npz'),allow_pickle=False) as data:
            assert data['frames'][:,4].tolist()==[before.state,2]
            assert data['frames'][:,1].tolist()==[5,5]
            assert data['bullets'][:,0].tolist()==[333,333]
            assert data['bullets'][:,3:5].tolist()==[[4,8],[4,8]]
            assert data['bullets_offsets'].tolist()==[0,1,2]
            assert json.loads(str(data['metadata']))['native_damage_source_id'] is None
        if mode!='train':
            with (directory/'episodes.csv').open(encoding='utf-8',newline='') as stream:row=next(csv.DictReader(stream))
            assert row['stage_id']=='5' and row['death_id']==info['death_id'] and row['nmnb_clean_so_far']=='False'


def test_nmnb_is_explicit_even_when_planner_disabled():
    observation=ObservationBuilder(goal='nmnb').build(frame())
    assert observation['decision'][0]==1 and observation['decision'][4]==0


def test_goal_goes_through_cli_command_and_invalid_values_are_rejected():
    from th06ai.console_service import build_command
    command=build_command('resume',5,'model.zip','nmnb')
    assert command[-2:]==['--goal','nmnb']
    with pytest.raises(ValueError):build_command('resume',5,'model.zip','unknown')
