from dataclasses import replace
import numpy as np
from th06ai.objectives import select_items
from th06ai.native import GameResources
from th06ai.rewards import RewardTracker,DISCOUNT
from test_rewards import frame

def test_power_progress_does_not_vanish_when_point_pickup_budget_is_saturated():
    start=replace(frame(),resources=GameResources(0,32,3,2,0))
    r=RewardTracker();r.reset(start);r.resource_raw_credit=1000.
    after=replace(start,tick=102,resources=replace(start.resources,power=33))
    result=r.step(after,0,False)
    assert result.components['power_progress']>.17

def test_collect_then_die_reclaims_extra_power_potential_without_farming():
    start=replace(frame(),resources=GameResources(0,0,3,2,0))
    r=RewardTracker();r.reset(start)
    collected=replace(start,tick=102,resources=replace(start.resources,power=128))
    first=r.step(collected,0,False)
    death=r.step(replace(collected,tick=104,state=2),0,True)
    assert abs(first.components['power_progress']+DISCOUNT*death.components['power_progress'])<1e-9

def test_nearby_star_cloud_does_not_hide_useful_power_from_model_input():
    stars=np.array([[i,193,385,6,0,1] for i in range(40)],np.float32)
    p=np.array([[900,210,350,0,0,1]],np.float32)
    f=replace(frame(),resources=GameResources(0,16,3,2,0),items=np.concatenate((stars,p)))
    selected=select_items(f,32)
    assert 40 in selected and selected[0]==40
