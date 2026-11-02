from dataclasses import replace
from types import SimpleNamespace
import struct
import numpy as np
import pytest
from th06ai.game_phase import classify,phase_of
from th06ai.environment import TouhouEnv
from th06ai.control import SamplingInterrupted,TrainingControl
from th06ai.observations import ObservationBuilder
from th06ai.rewards import RewardTracker
from test_rewards import frame
from test_environment import make_fake
from test_state_capture import UpdatingGameMemory,reader_for
from th06ai import native

def test_nc_pause_is_not_the_ecl_script_stop_flag():
    state=classify(active=True,engine_state=2,pause_flag=2,pause_selection=0,pause_animation=1)
    assert state.phase=='pause' and state.paused and state.ready
    script=classify(active=True,engine_state=2,script_stop=True)
    assert script.phase=='script_wait' and not script.paused
    assert classify(active=True,engine_state=2,dialogue_index=0).phase=='dialogue'
    assert classify(active=True,engine_state=2).phase=='combat'

def test_blank_opening_is_combat_and_stale_bullet_node_does_not_hide_title_menu():
    assert classify(active=True,engine_state=2,player_state=0).phase=='combat'
    state=classify(active=True,engine_state=1,nc_screen=3,selection=0,menu_ticks=60)
    assert state.phase=='title' and not state.stage_active
    assert not classify(active=False,engine_state=1,nc_screen=2,menu_ticks=15).ready
    assert classify(active=False,engine_state=1,nc_screen=2,menu_ticks=30).ready

def test_current_real_nc_pause_layout_decodes_without_sending_keys():
    memory=UpdatingGameMemory(advancing=False)
    def read(address,size):
        if address==native.INPUT_WORD:return bytes(4)
        if address==native.ENGINE_STATE:return struct.pack('<i',2)
        if address==native.NC_PAUSE:return bytes([2,0,0,0,0,0,0,0,2,0])
        if address==native.NC_PAUSE_UI+0xB44:
            result=bytearray(0x254);struct.pack_into('<i',result,0x250,1);return bytes(result)
        assert address==native.BULLET_MANAGER
        return struct.pack('<ii',99,100)
    memory.read=read
    def indirect(*_):
        result=bytearray(0x2A4);struct.pack_into('<i',result,0x48,-1);return bytes(result)
    memory.read_indirect=indirect
    result=reader_for(memory).read()
    assert result.paused and result.phase=='pause' and result.menu_selection==0 and result.menu_ready

def test_already_paused_does_not_press_escape_again(monkeypatch):
    env,_,calls=make_fake(monkeypatch)
    f=replace(env.reader.read(),paused=True,phase='pause',menu_selection=0,menu_ready=True)
    env.reader.read=lambda:f
    env.pause()
    assert calls==[] and env.current is f and env.managed_pause

def test_positive_pause_menu_drives_resume_without_frozen_tick_heuristic(monkeypatch):
    env,clock,calls=make_fake(monkeypatch,resume_delay=0)
    original=env.reader.read
    env.reader.read=lambda: replace(original(),phase='pause' if clock.now==0 else 'combat',paused=clock.now==0,menu_selection=0)
    env._frozen=lambda:(_ for _ in ()).throw(AssertionError('frozen alone is not a menu'))
    env.resume()
    assert calls==['z'] and env.current.tick>100

def test_resume_waits_for_real_menu_readiness_before_sending_z(monkeypatch):
    env,clock,calls=make_fake(monkeypatch,resume_delay=0)
    original=env.reader.read
    def read():
        f=original()
        paused=f.tick==100
        ready=clock.now>=.3
        return replace(f,phase='pause' if paused else 'combat',paused=paused,menu_selection=0,
                       menu_ready=ready,phase_evidence={'pause_animation':1 if ready else 0})
    env.reader.read=read
    pressed=[];old_set=env.keys.set
    def press(keys):pressed.append(clock.now);old_set(keys)
    env.keys.set=press
    env.resume()
    assert calls==['z'] and pressed[0]>=.3 and env.current.tick>100

def test_main_menu_to_normal_character_shot_and_run_mode_uses_confirmed_states(monkeypatch):
    env,clock,calls=make_fake(monkeypatch)
    states=[('title_prompt',0),('title',2),('title',1),('title',0),('difficulty',0),('difficulty',1),('character',0),('shot',1),('run_mode',0),('start_confirm',0),('combat',0)]
    index=[0]
    def read():
        phase,selection=states[index[0]]
        return replace(frame(),phase=phase,menu_selection=selection,menu_ready=True,stage_active=phase=='combat')
    env.reader.read=read
    env.wait_for_focus=lambda:None
    def tap(key,**_):calls.append(key);index[0]+=1;clock.now+=.1
    env.tap=tap
    env.ensure_game_started()
    assert phase_of(env.current)=='combat'
    assert calls==['z','up','up','z','down','z','z','z','z','z']

def test_dialogue_never_carries_arrows_bomb_or_shift_and_forces_new_shoot_edge(monkeypatch):
    env,_,calls=make_fake(monkeypatch)
    phases=iter([replace(frame(tick=100),phase='dialogue'),replace(frame(tick=120),phase='combat')])
    env.reader.read=lambda:next(phases)
    env.builder=ObservationBuilder()
    env.reward_tracker=RewardTracker();env.reward_tracker.reset(frame())
    env._observe_control=lambda *a:None
    held=[]
    env.keys=SimpleNamespace(set=lambda keys:held.append(set(keys)),release=lambda:held.append(set()))
    with pytest.raises(SamplingInterrupted):env._advance_dialogue()
    assert all(keys<={'z'} for keys in held) and held[-1]==set()
    assert env.current.phase=='combat' and env.reward_tracker.frame.tick==120

def test_unsampled_pause_gap_counts_time_once_without_inventing_reward():
    env=TouhouEnv.__new__(TouhouEnv)
    env.current=replace(frame(tick=100),phase='combat')
    env.episode_count=1;env.episode_ticks=60
    paused=replace(env.current,tick=104,phase='pause',paused=True)
    env._account_menu_gap(paused);env.current=paused
    env._account_menu_gap(paused)
    assert env.episode_ticks==64 and env.unsampled_ticks==4
