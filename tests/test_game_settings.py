"""Setup, Practice boundaries and v8 migration with fake game/CPU only."""
from dataclasses import replace
import hashlib
import json
import struct
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from th06ai import learning, native
from th06ai.battle_view import BattleViewPublisher
from th06ai.control import SamplingInterrupted, TrainingControl
from th06ai.game_settings import (GameSetup, parse_setup, decode_identity, context_vector,
                                  CONTEXT_FEATURES, menu_key, observed_settings)
from th06ai.game_phase import classify
from th06ai.observations import ObservationBuilder, make_observation_space
from th06ai.planning import RollingPlanner
from th06ai.pausable_ppo import PausablePPO
from th06ai.policy import BulletEncoder, ResourcePolicy
from th06ai.files import atomic_json
from test_environment import make_fake
from test_learning import LearningFixture
from test_rewards import frame
from test_sampling_timing import game
from test_state_capture import UpdatingGameMemory, reader_for
from test_web_console import setup


@pytest.mark.parametrize('difficulty',range(5))
@pytest.mark.parametrize('character,shot',[(0,0),(0,1),(1,0),(1,1)])
def test_identity_decodes_all_difficulties_and_shots_without_using_request(difficulty,character,shot):
    d,c,m=decode_identity(difficulty,character,shot,1,0,0)
    assert d==('easy','normal','hard','lunatic','extra')[difficulty]
    assert c==('reimu_a','reimu_b','marisa_a','marisa_b')[character*2+shot]
    assert m=='practice'
    assert decode_identity(99,9,9,2,0,0)==(None,None,None)


@pytest.mark.parametrize('values',[{'mode':'practice','difficulty':'extra'},
    {'mode':'practice','difficulty':'easy','stage':6},{'mode':'story','stage':4},
    {'stage':True},{'stage':2.5},{'difficulty':'bogus'},{'character':'bogus'}, {'unexpected':1}])
def test_invalid_or_unavailable_setup_is_never_silently_replaced(values):
    with pytest.raises(ValueError):parse_setup(values)


def test_practice_is_a_real_stage_menu_and_locked_stage_cannot_be_confirmed():
    assert classify(active=False,engine_state=1,nc_screen=22,selection=2,menu_ticks=20).phase=='practice_stage'
    setup=GameSetup(mode='practice',difficulty='hard',character='marisa_b',stage=5)
    f=replace(frame(),phase='practice_stage',menu_selection=4,phase_evidence={'practice_stage_count':3})
    with pytest.raises(ValueError,match='解锁'):menu_key(f,setup)
    f.phase_evidence['practice_stage_count']=6
    assert menu_key(f,setup)=='z'
    assert menu_key(replace(f,menu_selection=1),setup)=='down'
    assert menu_key(replace(f,phase='title',menu_selection=0),setup)=='down'
    assert menu_key(replace(f,phase='character',menu_selection=0),setup)=='right'


def context_memory(difficulty=2, character=1, shot=1, practice=1):
    memory=UpdatingGameMemory(advancing=False)
    memory.put(native.STAGE_RVA,'i',5)
    memory.put(native.CHARACTER_RVA,'BB',character,shot)
    memory.put(native.DIFFICULTY_RVA,'i',difficulty)
    memory.put(native.NC_PAUSE+4,'B',practice)
    memory.put(native.PLAYER+0x7710,'ff',1,1)
    memory.put(native.PLAYER+0x7860,'ff',5,2.5)
    original=memory.read
    calls=[]
    def read(address,size):
        calls.append((address,size))
        if address in (native.INPUT_WORD,native.BULLET_MANAGER):return original(address,size)
        if address==native.ENGINE_STATE:return struct.pack('<i',2)
        if address==native.NC_PAUSE_UI+0xB44:return bytes(0x254)
        return bytes(memory.world[address-native.WORLD_START:address-native.WORLD_START+size])
    def indirect(*_):
        data=bytearray(0x2A4);struct.pack_into('<i',data,0x48,-1);return bytes(data)
    memory.read, memory.read_indirect=read,indirect
    return memory,calls


def test_full_and_light_readers_agree_and_light_read_adds_no_pool_or_context_rpc():
    memory,calls=context_memory()
    reader=reader_for(memory)
    f=reader.read()
    assert (f.difficulty,f.character,f.game_mode,f.stage_id)==('hard','marisa_b','practice',5)
    assert f.movement_speeds==(5,2.5)
    calls.clear();bulk=memory.bulk_reads
    light=reader.read_control()
    assert (light.difficulty,light.character,light.game_mode,light.stage_id)==('hard','marisa_b','practice',5)
    assert memory.bulk_reads==bulk and len(calls)==6
    assert max(size for _,size in calls)<1024


def test_practice_menu_metadata_does_not_become_a_combat_observation():
    from th06ai.game_settings import PRACTICE_UNLOCK_RVA
    memory,_=context_memory()
    memory.put(PRACTICE_UNLOCK_RVA+3*24+2,'B',6)
    read=memory.read
    def menu_read(address,size):
        if address==native.ENGINE_STATE:return struct.pack('<i',1)
        if address==native.NC_MENU+0x168B0:return struct.pack('<i',22)
        if address==native.NC_MENU+0x28:return struct.pack('<i',4)
        if address==native.NC_MENU+0x9638:return struct.pack('<i',60)
        if address==native.NC_MENU+0x168D1:return bytes([1,0,0])
        return read(address,size)
    memory.read=menu_read
    f=reader_for(memory).read()
    selected=observed_settings(f)
    assert selected['source']=='menu' and selected['stage']==5 and selected['character']=='marisa_b'
    assert not f.stage_active and f.stage_id is None and f.difficulty is None
    assert not context_vector(f).any()


def test_unknown_identity_has_known_bits_zero_and_planner_uses_current_character_speed():
    f=replace(frame(),difficulty='lunatic',character='marisa_a',game_mode='practice',movement_speeds=(5.,2.5))
    planner=RollingPlanner()
    builder=ObservationBuilder(planner)
    obs=builder.build(f)
    assert obs['game_context'].shape==(CONTEXT_FEATURES,)
    assert obs['game_context'].sum()>0 and make_observation_space().contains(obs)
    assert planner.fast_speed==5 and planner.slow_speed==2.5
    assert np.linalg.norm(planner.velocities[4])==pytest.approx(5)
    assert not context_vector(frame()).any()


def test_wrong_observed_difficulty_pauses_without_sending_an_action(monkeypatch):
    env,_,calls=make_fake(monkeypatch)
    env.game_setup=GameSetup(difficulty='lunatic')
    with pytest.raises(RuntimeError,match='实际难度'):
        env._verify_setup(replace(frame(),difficulty='normal',character='reimu_a',game_mode='story'))
    assert calls==[]


@pytest.mark.parametrize('tick', range(6))
@pytest.mark.parametrize('restart', (False,True))
def test_immediate_opening_pause_is_an_entered_practice_stage(monkeypatch,tick,restart):
    env,_,calls=make_fake(monkeypatch)
    env.wait_for_focus=lambda:None
    env.game_setup=GameSetup(mode='practice',difficulty='lunatic')
    f=replace(frame(),tick=tick,paused=True,phase='pause',stage_active=True,stage_id=1,
              difficulty='lunatic',character='reimu_a',game_mode='practice')
    env.reader.read=lambda:f
    transactions=[]
    env._restart=lambda:transactions.append('restart')
    env.resume=lambda:transactions.append('resume')
    env._return_to_title=lambda:pytest.fail('A confirmed first-tick pause must not return to title')
    env.ensure_game_started(restart=restart)
    assert transactions==['restart' if restart else 'resume']
    assert env.current is f and calls==[]


@pytest.mark.parametrize('phase',('character','shot','run_mode','practice_stage','start_confirm'))
def test_deep_menu_rewinds_wrong_confirmed_difficulty_before_stage_unlock_check(phase):
    f=replace(frame(),phase=phase,menu_selection=0,phase_evidence={
        'practice_menu':True,'practice_stage_count':0,
        'menu_confirmed_settings':{'difficulty':'normal','character':'reimu','shot':'a'}})
    assert menu_key(f,GameSetup(mode='practice',difficulty='lunatic'))=='x'


def test_current_cursor_is_not_mistaken_for_an_already_confirmed_parent():
    evidence={'practice_menu':True,
              'menu_confirmed_settings':{'difficulty':'normal','character':'marisa','shot':'b'}}
    setup=GameSetup(mode='practice',difficulty='lunatic')
    f=replace(frame(),phase='difficulty',menu_selection=3,phase_evidence=evidence)
    assert menu_key(f,setup)=='z'
    evidence['menu_confirmed_settings']['difficulty']='lunatic'
    assert menu_key(replace(f,phase='character',menu_selection=0),setup)=='z'
    evidence['menu_confirmed_settings']['character']='reimu'
    assert menu_key(replace(f,phase='shot',menu_selection=0),setup)=='z'


def test_open_practice_menu_rewinds_and_applies_difficulty_character_and_shot(monkeypatch):
    env,clock,calls=make_fake(monkeypatch)
    env.game_setup=GameSetup(mode='practice',difficulty='lunatic')
    env.wait_for_focus=lambda:None
    screens=['title','difficulty','character','shot','run_mode','practice_stage','start_confirm','combat']
    state={'index':5,'selection':0,'difficulty':'normal','character':'marisa','shot':'b'}
    def cursor(index):
        return {'difficulty':('easy','normal','hard','lunatic').index(state['difficulty']),
                'character':int(state['character']=='marisa'),'shot':int(state['shot']=='b')}.get(screens[index],0)
    def read():
        phase=screens[state['index']]
        return replace(frame(),phase=phase,menu_selection=state['selection'],menu_ready=True,
            stage_active=phase=='combat',stage_id=1,difficulty=state['difficulty'],
            character=state['character']+'_'+state['shot'],game_mode='practice',
            phase_evidence={'practice_menu':True,'practice_stage_count':6,
                'menu_confirmed_settings':{k:state[k] for k in ('difficulty','character','shot')}})
    def tap(key,**_):
        calls.append(key);clock.now+=.1
        phase=screens[state['index']]
        if key=='x':
            state['index']-=1;state['selection']=cursor(state['index'])
        elif key=='z':
            if phase=='difficulty':state['difficulty']=('easy','normal','hard','lunatic')[state['selection']]
            elif phase=='character':state['character']=('reimu','marisa')[state['selection']]
            elif phase=='shot':state['shot']=('a','b')[state['selection']]
            state['index']+=1;state['selection']=cursor(state['index'])
        elif key in ('down','right'):state['selection']+=1
        elif key in ('up','left'):state['selection']-=1
        else:raise AssertionError(key)
    env.reader.read=read;env.tap=tap
    env.ensure_game_started()
    assert calls[:4]==['x']*4
    assert env.current.difficulty=='lunatic' and env.current.character=='reimu_a'
    assert state['index']==7 and env.current.stage_id==1


def test_setting_mismatch_at_context_boundary_keeps_bootstrap_cut_and_no_keys(monkeypatch):
    env,_,calls=make_fake(monkeypatch)
    env.game_setup=GameSetup(difficulty='lunatic')
    reasons=[]
    env.training_control=SimpleNamespace(request_pause=reasons.append)
    with pytest.raises(SamplingInterrupted) as error:
        env._verify_setup(replace(frame(),difficulty='normal',character='reimu_a',game_mode='story'),stage_boundary=True)
    assert error.value.stage_boundary and error.value.bootstrap_boundary
    assert reasons and not calls


def test_practice_entry_drives_all_requested_choices_and_verifies_actual_game(monkeypatch):
    env,clock,calls=make_fake(monkeypatch)
    env.wait_for_focus=lambda:None
    env.game_setup=GameSetup(mode='practice',difficulty='hard',character='marisa_b',stage=5)
    screens=['title','difficulty','character','shot','run_mode','practice_stage','start_confirm','combat']
    state={'index':0,'selection':0}
    chosen={}
    def read():
        phase=screens[state['index']]
        return replace(frame(),phase=phase,menu_selection=state['selection'],menu_ready=True,
            stage_active=phase=='combat',difficulty='hard' if phase=='combat' else None,
            character='marisa_b' if phase=='combat' else None,game_mode='practice' if phase=='combat' else None,
            stage_id=5,phase_evidence={'practice_menu':state['index']>0,'practice_stage_count':6})
    def tap(key,**_):
        calls.append(key);clock.now+=.1
        phase=screens[state['index']]
        if key=='z':
            chosen[phase]=state['selection'];state['index']+=1;state['selection']=0
        elif key in ('down','right'):state['selection']+=1
        elif key in ('up','left'):state['selection']-=1
        else:raise AssertionError(key)
    env.reader.read=read;env.tap=tap
    env.ensure_game_started()
    assert chosen==dict(title=2,difficulty=2,character=1,shot=1,run_mode=0,practice_stage=4,start_confirm=0)
    assert env.current.game_mode=='practice' and env.current.stage_id==5


def test_practice_terminal_boundary_cuts_gae_without_inventing_samples():
    from stable_baselines3.common.callbacks import BaseCallback
    torch.set_num_threads(1)
    class PracticeFixture(LearningFixture):
        actual_steps=0
        def step(self,action):
            if self.local_steps==2:
                raise SamplingInterrupted(episode_end={'episode':{'r':2.,'l':2,'t':1.},
                    'episode_end_reason':'practice_result','died':False})
            obs,_,_,_,info=super().step(action)
            self.actual_steps+=1
            return obs,1.,False,False,info
    class Callback(BaseCallback):
        endings=0
        def _on_step(self):return True
        def record_episode_end(self,info):self.endings+=1
    env=PracticeFixture()
    model=PausablePPO(ResourcePolicy,env,device='cpu',n_steps=8,batch_size=8,n_epochs=1,
        policy_kwargs={'features_extractor_class':BulletEncoder})
    model.training_control=SimpleNamespace(gate=lambda game:False,lease=lambda:None)
    model.training_game=env
    callback=Callback()
    model.learn(16,callback=callback)
    assert model.num_timesteps==env.actual_steps==16 and callback.endings==7
    np.testing.assert_allclose(model.rollout_buffer.returns[[1,3,5]],1.,atol=1e-5)


def test_practice_result_closes_episode_without_fake_combat_reward_or_action():
    before=replace(frame(),difficulty='hard',character='marisa_b',game_mode='practice',stage_id=5)
    result=replace(before,phase='result',stage_active=False)
    env,actions,releases,_=game(before,result)
    env.goal='survival';env.diagnostics=None
    env.episode_steps=8;env.episode_return=3.;env.episode_ticks=300
    with pytest.raises(SamplingInterrupted) as error:env._practice_result(result)
    info=error.value.episode_end
    assert info['episode_end_reason']=='practice_result' and not info['died']
    assert info['survival_seconds']==5 and 'action' not in info
    assert not actions and releases and env.total_steps==0


@pytest.fixture
def practice_return():
    before = replace(frame(tick=9135), phase='boss', difficulty='lunatic',
                     character='reimu_a', game_mode='practice', stage_id=1)
    menu = replace(before, tick=9347, phase='practice_stage', stage_active=False,
                   stage_id=None, difficulty=None, character=None, game_mode=None,
                   menu_selection=0, menu_ready=True, phase_evidence={
                       'practice_menu':True, 'practice_stage_count':1,
                       'menu_confirmed_settings':{'difficulty':'lunatic', 'character':'reimu', 'shot':'a'},
                   })
    env, actions, _, _ = game(before, menu)
    env.goal = 'survival'
    env.game_setup = GameSetup(mode='practice', difficulty='lunatic', character='reimu_a', stage=1)
    env._practice_origin = before
    return env, actions, before, menu


def test_practice_script_exit_restarts_selected_stage_without_a_combat_sample(practice_return, tmp_path):
    env, actions, before, menu = practice_return
    env.training_control = TrainingControl(tmp_path, env.lease, env.status)
    # The stage clears before the native engine has switched to the menu.
    incomplete = replace(before, tick=9346, stage_id=None, game_mode='story', phase='combat')
    loading = replace(menu, phase='menu_transition', phase_evidence={})
    frames = iter((replace(before, phase='script_wait'), incomplete, loading, menu))
    env.reader.read = lambda: next(frames, menu)
    with pytest.raises(SamplingInterrupted) as error:
        env._advance_dialogue()
    info = error.value.episode_end
    assert info['episode_end_reason'] == 'practice_return' and not info['died']
    assert info['stage_id'] == 1 and info['game_settings']['difficulty'] == 'lunatic'
    assert env.total_steps == 0 and actions == []
    assert not (tmp_path / 'PAUSE').exists() and not hasattr(env, 'last_stage_transition')

    entered = replace(before, tick=2, phase='combat')
    def tap(key, **kwargs):
        actions.append(key)
        env.reader.read = lambda: entered
    env.tap = tap
    env._restart()
    assert actions == ['z'] and env.current is entered


@pytest.mark.parametrize('resumed_at_menu', (False, True))
def test_practice_sampler_recognizes_menu_after_fast_probe_or_user_resume(practice_return, resumed_at_menu):
    env, actions, _, menu = practice_return
    env.reader.window.focused = True
    probe = replace(menu, phase='menu_transition', menu_ready=False, phase_evidence={})
    frames = iter((probe, menu))
    env.reader.read = lambda: next(frames, menu)
    env.reader.read_control = lambda: native.ControlSample(
        menu.tick, menu.x, menu.y, menu.state, False, False, None, 'menu_transition', None, 0, 0, 0)
    if resumed_at_menu:
        env.reader.read = lambda: menu
        env.resume()
        assert not getattr(env, 'pending_stage_boundary', False)
    with pytest.raises(SamplingInterrupted) as error:
        env.refresh_decision()
    assert error.value.episode_end['episode_end_reason'] == 'practice_return'
    assert actions == []


def test_practice_return_requires_menu_evidence_and_preserves_observed_miss(practice_return, tmp_path):
    env, actions, _, menu = practice_return
    env.episode_count = 1
    battle_view = BattleViewPublisher(tmp_path)
    env.diagnostics = SimpleNamespace(event=lambda **_: None, battle_view=battle_view)
    env._practice_result(replace(menu, phase_evidence={}))
    assert battle_view.miss_count == 0
    with pytest.raises(SamplingInterrupted) as error:
        env._practice_result(menu, death_probe_tick=menu.tick-1)
    assert error.value.episode_end['episode_end_reason'] == 'death'
    assert error.value.episode_end['died'] and actions == []
    assert battle_view.miss_count == 1


def test_session_passes_an_immutable_setup_and_rejects_bad_combinations(setup):
    _,_,service,client,calls,_=setup
    selected={'mode':'practice','difficulty':'hard','character':'marisa_b','stage':5}
    response=client.post('/api/sessions',json={'mode':'train','minutes':1,'game_setup':selected})
    assert response.status_code==202
    command=calls[0][0]
    assert json.loads(command[command.index('--game-setup')+1])==selected


def test_setup_api_rejects_unknown_fields_and_bool_stages_before_launch(setup):
    _,_,_,client,calls,_=setup
    for selected in ({'stage':True},{'mode':'practice','difficulty':'extra'}, {'stage':3},
                     {'mode':'practice','unknown':2}):
        assert client.post('/api/sessions',json={'mode':'train','minutes':1,'game_setup':selected}).status_code==422
    assert calls==[]


def test_v8_migration_preserves_outputs_and_adam_for_all_new_settings(tmp_path,monkeypatch):
    torch.set_num_threads(1)
    class Version8(LearningFixture):
        def __init__(self):
            super().__init__();self.observation_space=make_observation_space(8)
        def reset(self,**kwargs):
            obs,info=super().reset(**kwargs)
            return {k:v for k,v in obs.items() if k in self.observation_space.spaces},info
        def step(self,action):
            obs,*rest=super().step(action)
            return ({k:v for k,v in obs.items() if k in self.observation_space.spaces},*rest)
    old=PausablePPO(ResourcePolicy,Version8(),device='cpu',n_steps=32,batch_size=32,n_epochs=1,
        policy_kwargs={'features_extractor_class':BulletEncoder,'net_arch':{'pi':[128,128],'vf':[128,128]}})
    old.learn(64)
    path=tmp_path/'models/v8.zip';path.parent.mkdir();old.save(path)
    digest=hashlib.sha256(path.read_bytes()).hexdigest()
    atomic_json(path.with_suffix('.json'),{'format':8,'policy_version':2,'game_sha256':native.SHA256,
        'model_sha256':digest,'steps':64,'gradient_updates':old._n_updates})
    monkeypatch.setattr(learning,'ROOT',tmp_path)
    new=learning.load_model(path,LearningFixture(),'cpu')
    assert new.num_timesteps==old.num_timesteps and new._n_updates==old._n_updates
    assert not new.bomb_policy_upgraded
    new_params=dict(new.policy.named_parameters())
    for name,param in old.policy.named_parameters():
        current=new_params[name]
        if name.endswith('fusion.0.weight'):
            assert torch.equal(param,current[:,:param.shape[1]]) and not current[:,param.shape[1]:].any()
        else:assert torch.equal(param,current)
        for key,value in old.policy.optimizer.state.get(param,{}).items():
            mapped=new.policy.optimizer.state[current][key]
            if isinstance(value,torch.Tensor) and value.ndim and name.endswith('fusion.0.weight'):
                assert torch.equal(value,mapped[:,:value.shape[1]]) and not mapped[:,value.shape[1]:].any()
            elif isinstance(value,torch.Tensor):assert torch.equal(value,mapped)
    obs=LearningFixture().reset()[0]
    with torch.no_grad():
        tensor,_=old.policy.obs_to_tensor({k:v for k,v in obs.items() if k!='game_context'})
        probs=old.policy.get_distribution(tensor).distribution.probs
        values=old.policy.predict_values(tensor)
        for difficulty in ('easy','normal','hard','lunatic','extra'):
            obs['game_context']=context_vector(replace(frame(),difficulty=difficulty,character='marisa_b',game_mode='practice'))
            tensor,_=new.policy.obs_to_tensor(obs)
            assert torch.allclose(probs,new.policy.get_distribution(tensor).distribution.probs,atol=1e-6)
            assert torch.allclose(values,new.policy.predict_values(tensor),atol=1e-6)
    assert hashlib.sha256(path.read_bytes()).hexdigest()==digest
