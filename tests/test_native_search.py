"""Differential geometry/search tests against the retained Python reference."""
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path
import numpy as np
import pytest
from th06ai.acceleration import search_extension
from th06ai.native import Frame,GameResources
from th06ai.planning import RollingPlanner
from th06ai.planner_spec import PlanningConfig

pytestmark=pytest.mark.skipif(search_extension() is None,reason='optional Rust search not built')


def scene(seed):
    rng=np.random.default_rng(seed)
    count=(0,1,48,350,640)[seed%5]
    x,y=([8.,432.] if seed%6==0 else [192.,320.] if seed%6==1 else rng.uniform([8,16],[376,432]))
    b=np.column_stack((np.arange(count),rng.uniform([-30,-30],[420,470],(count,2)))).astype(np.float32)
    v=rng.uniform(-3,3,(count,2)).astype(np.float32)
    sizes=rng.uniform(1,10,(count,2)).astype(np.float32)
    items=np.column_stack((np.arange(9),rng.uniform([8,16],[376,432],(9,2)),np.arange(9)%7,
                           np.zeros(9),np.ones(9))).astype(np.float32)
    enemies=np.array([[1,192,100,2000,6000,0,0x11,1],[2,100,200,100,100,0,0x11,0]],np.float32)
    bodies=np.array([[1,192,100,12,16,1],[2,100,200,8,8,1]],np.float32) if seed%3 else np.empty((0,6),np.float32)
    lasers=np.array([[1,150,30,0,300,.4,3,1],[2,300,100,0,150,-.2,5,0]],np.float32) if seed%4==0 else np.empty((0,8),np.float32)
    old=Frame(100,float(x),float(y),0,False,False,b,(8,16,368,416),True,0,
        bullet_sizes=sizes,player_radius=2.,items=items,enemies=enemies,
        enemy_bodies=bodies,lasers=lasers,resources=GameResources(0,64,3,3,0))
    current=replace(old,tick=102,bullets=np.column_stack((np.arange(count),b[:,1:]+2*v)).astype(np.float32),
        enemy_bodies=bodies.copy(),lasers=lasers.copy())
    if len(bodies):current.enemy_bodies[:,2]+=2
    if len(lasers):current.lasers[:,5]+=.002
    control=np.zeros(28,np.float32);control[seed%18]=control[19]=1;control[26]=.4
    return old,current,control


def compare(old,current,control,beam=2,fast=4.,slow=2.,history=False):
    settings=PlanningConfig(beam_width=beam,collision_backend='rust')
    reference=RollingPlanner(settings,fast_speed=fast,slow_speed=slow)
    reference.search_backend='python'
    actual=RollingPlanner(settings,fast_speed=fast,slow_speed=slow)
    assert actual.search_backend=='rust'
    if history:
        for planner in (reference,actual):
            for duration in (12,28,45,18,63,11):
                planner.record_latency({'post_plan_to_send_ms':duration,'movement_changed':True,
                                        'game_input_match_ms':17,'post_plan_to_input_ms':duration+17})
            planner.last_elapsed_ms=31;planner.last_visible=len(current.bullets)
    expected=reference.plan(current,old,control);result=actual.plan(current,old,control)
    for name in ('features','paths','clearances','collision_times','navigation','decision'):
        np.testing.assert_allclose(getattr(result,name),getattr(expected,name),rtol=2e-5,atol=.001,err_msg=name)
    for name in ('delay_ticks','path_ticks','candidates_checked','feasible_roots','allowed_roots','all_routes_risky',
        'tested_switch_delays','timing_interval_complete','short_escape_priority','body_contact_probe_performed',
        'suggested_movement','rescue_opportunity','nominal_escape_roots','late_control_risk','mobility_probes','exit_measured'):
        assert result.diagnostics[name]==expected.diagnostics[name],name
    return result


@pytest.mark.parametrize('seed',range(30))
def test_owned_search_matches_reference_with_walls_items_enemies_and_lasers(seed):
    compare(*scene(seed),beam=1+seed%4,fast=3.5 if seed%2 else 4.,slow=1.8 if seed%2 else 2.,history=seed%3==1)


def test_real_death_switch_interval_still_excludes_early_collision_direction():
    with np.load(Path(__file__).parent/'fixtures/stage5-switch-window.npz',allow_pickle=False) as d:
        def frame(i,suffix):
            row=d['frames'][i];b=d['bullets_'+suffix]
            return Frame(int(row[0]),float(row[2]),float(row[3]),int(row[4]),bool(row[6]),False,b[:,:3],
                (8,16,368,416),True,0,bullet_sizes=b[:,3:5],player_radius=float(row[7]),
                enemy_bodies=d['enemy_bodies_'+suffix],lasers=d['lasers_'+suffix])
        old,current=frame(0,'previous'),frame(1,'current')
    control=np.zeros(28);control[4]=control[19]=1
    result=compare(old,current,control)
    assert result.features[5,1]==0


def test_independent_search_calls_are_concurrent_and_owned():
    with ThreadPoolExecutor(4) as pool:
        list(pool.map(lambda seed:compare(*scene(seed)),range(8)))


def test_search_rejects_invalid_budgets_before_native_work():
    mod=search_extension()
    args=(0,0,0,2.,3.,32.,[4.,13.,13.],[1.,2.,3.],True,2,18,4.,0.)
    with pytest.raises(ValueError):mod.plan_bytes(b'bad',*args)
    data=np.zeros(88,dtype='<f8');data[6:8]=[376,432]
    data[0]=np.nan
    with pytest.raises(ValueError):mod.plan_bytes(data.tobytes(),*args)
    data[0]=0
    bad=list(args);bad[7]=[3.,2.]
    with pytest.raises(ValueError):mod.plan_bytes(data.tobytes(),*bad)


def test_python_collision_selection_does_not_require_the_optional_search():
    old,current,control=scene(0)
    planner=RollingPlanner(PlanningConfig(collision_backend='python'))
    assert planner.search_backend=='python'
    assert planner.plan(current,old,control).diagnostics['search_backend']=='python'


@pytest.mark.parametrize('seed',range(5))
def test_unknown_obstacles_and_missing_previous_sample_keep_reference_constraints(seed):
    old,current,control=scene(seed+10)
    current=replace(current,player_radius=None,bullet_sizes=None)
    if len(current.enemy_bodies):current.enemy_bodies[:,5]=0
    compare(None,current,control,history=bool(seed%2))
