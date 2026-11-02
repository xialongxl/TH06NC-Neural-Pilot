from dataclasses import replace
import struct
import numpy as np
import pytest
from th06ai.lasers import LASER_CAPACITY,LASER_STRIDE,decode_lasers
from th06ai.body_state import visible_body_state
from th06ai.planning import swept_segments,RollingPlanner
from th06ai.planner_spec import PlanningConfig
from th06ai.acceleration import extension,rust_sweep
from th06ai.risk import collision_risk
from test_rewards import frame

def test_own_build_laser_geometry_uses_quarter_nominal_width_and_excludes_fade():
    data=bytearray(LASER_CAPACITY*LASER_STRIDE)
    struct.pack_into('<f',data,0,32.)
    struct.pack_into('<ff',data,0xC,0.,400.)
    struct.pack_into('<f',data,0x25C,np.pi/2)
    data[0x270]=1;data[0x28C]=1
    struct.pack_into('<ff',data,0x274,192.,16.)
    beams=decode_lasers(data,384,448)
    assert len(beams)==1 and beams[0,6]==8
    data[0x28C]=2
    assert not len(decode_lasers(data,384,448))

@pytest.mark.parametrize('backend',['python','rust'])
def test_rotated_laser_blocks_crossing_without_blocking_parallel_safe_lane(backend):
    if backend=='rust' and extension() is None:pytest.skip('extension not built')
    f=replace(frame(x=172,y=320),player_radius=2.,lasers=np.array([[0,192,16,0,400,np.pi/2,4,1]],np.float32))
    previous=replace(f,tick=f.tick-2)
    bodies=visible_body_state(f,previous)
    args=(np.array([[172,320],[172,320]],np.float32),np.array([[4,0],[0,-4]],np.float32),10.,0.,
          np.array([8,16]),np.array([376,432]),np.empty((0,2),np.float32),np.empty((0,2),np.float32),np.empty(0),np.empty(0),bodies)
    _,gap,hit,_=(rust_sweep if backend=='rust' else swept_segments)(*args)
    assert np.isfinite(hit[0]) and np.isinf(hit[1]) and gap[1]>0

def test_warning_is_available_to_planning_but_not_awarded_as_current_damage_risk():
    warning=replace(frame(),lasers=np.array([[0,192,16,0,400,np.pi/2,8,0]],np.float32))
    assert len(visible_body_state(warning).positions)==1
    assert collision_risk(warning).threat==0
    live=replace(warning,lasers=warning.lasers.copy())
    live.lasers[0,7]=1
    assert collision_risk(live).threat==1

def test_old_checkpoint_plan_features_see_laser_hazard_without_dimension_change():
    f=replace(frame(x=172,y=320),player_radius=2.,lasers=np.array([[0,192,16,0,400,np.pi/2,8,1]],np.float32))
    planner=RollingPlanner(PlanningConfig(collision_backend='auto'))
    result=planner.plan(f,replace(f,tick=f.tick-2))
    assert result.features.shape==(18,8)
    assert not result.features[4,1]  # fast right crosses the active beam
