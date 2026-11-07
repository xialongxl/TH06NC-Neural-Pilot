"""Action timing and emergency-path regressions using synthetic CPU scenes."""
from dataclasses import replace

import numpy as np
import pytest

from th06ai.acceleration import search_extension
from th06ai.navigation import NAV_FIELDS
from th06ai.observations import action_keys
from th06ai.planning import COLLISION_PADDING, RollingPlanner, swept_segments
from th06ai.planning import rectangle_sweep
from th06ai.body_state import visible_body_state
from th06ai.bullet_state import visible_bullet_state
from th06ai.planner_spec import MAX_SEARCH_NODES, PlanningConfig
from test_planning import frame


def test_slow_prediction_matches_verified_two_pixels_per_native_tick():
    planner=RollingPlanner()
    control=np.zeros(28)
    control[13]=control[19]=1  # Existing focused right input.
    result=planner.plan(frame(),control=control)
    delay=result.diagnostics['delay_ticks']
    root=result.diagnostics['path_ticks'][2]-delay
    assert np.allclose(result.paths[13,1],[192+2*delay,320])
    assert np.allclose(result.paths[13,2],[192+2*(delay+root),320])


def test_early_input_can_cross_a_bullet_which_the_nominal_late_path_misses():
    planner=RollingPlanner()
    for t in [16,16,70,70,70,70,70]:
        planner.switch_times.add(t)
    control=np.zeros(28)
    control[1]=control[19]=1  # Old up, new right.
    current=frame(bullets=[(0,204,320)])
    result=planner.plan(current,replace(current,tick=100),control)
    assert np.isinf(result.collision_times[4])  # Nominal route alone was feasible.
    assert not result.features[4,1]  # Early right crosses the current bullet.
    assert result.features[1,1]  # Continuing up is still available.
    assert not result.diagnostics['all_routes_risky']


def test_shared_prefix_risk_does_not_make_resource_seeking_the_emergency_tiebreaker():
    planner=RollingPlanner()
    planner.switch_times.add(80)
    control=np.zeros(28)
    control[4]=control[19]=1
    current=frame(bullets=[(0,204,324)],items=[(0,204,380,0,0,0)])
    result=planner.plan(current,replace(current,tick=100),control)
    assert result.diagnostics['all_routes_risky']
    assert result.diagnostics['prefix_collision_tick'] is not None
    assert result.diagnostics['recovery_ranking']
    assert not result.features[2,1]  # Down moves into the bullet toward the item.
    assert any(result.features[a,1] and 'up' in action_keys(a) for a in range(18))
    assert result.diagnostics['feasible_roots']==0  # Fallback is not called safe.


def test_bullet_prefix_fallback_does_not_reopen_a_path_through_enemy_body():
    planner=RollingPlanner()
    planner.switch_times.add(80)
    control=np.zeros(28)
    control[4]=control[19]=1
    current=frame(bullets=[(0,204,324)])
    current.enemy_bodies=np.array([(0,212,294,8,8,1)],np.float32)
    previous=replace(current,tick=100)
    result=planner.plan(current,previous,control)
    assert result.diagnostics['all_routes_risky']
    assert not result.features[1,1]  # The known body blocks fast up.
    assert result.features[:,1].any()
    assert np.isfinite(result.features).all()


@pytest.mark.parametrize('backend', ['python', 'rust'])
def test_warning_laser_fallback_checks_damaging_hazards_across_the_switch_window(backend):
    if backend == 'rust' and search_extension() is None:
        pytest.skip('optional Rust search not built')
    settings = PlanningConfig(beam_width=1, collision_backend=backend)
    planner = RollingPlanner(settings)
    current = frame(bullets=[(0, 192, 303), (1, 192, 330)])
    current.enemy_bodies = np.array([(0, 172, 320, 6, 6, 1)], np.float32)
    current.lasers = np.array([(0, 100, 320, 0, 200, 0, 3, 0)], np.float32)
    previous = replace(current, tick=100)

    result = planner.plan(current, previous)
    allowed = result.features[:, 1] > 0
    assert result.diagnostics['active_threat_priority']
    assert result.diagnostics['immediate_contact_free_roots'] == 0  # Telegraph covers the player.
    assert result.diagnostics['route_mode'] == 'fallback'
    assert not result.navigation[:, NAV_FIELDS.index('robust_safe')].any()
    assert allowed[result.diagnostics['suggested_movement']]
    assert not allowed[2]  # Fast down crosses the bullet.
    assert not allowed[14]  # Slow up-left crosses the enemy-body gate after the short probe.
    assert allowed[6]  # Fast up-right avoids both over the tested window.

    damaging = replace(current, lasers=current.lasers.copy())
    damaging.lasers[0, 7] = 1
    result = RollingPlanner(settings).plan(damaging, replace(damaging, tick=100))
    assert not result.diagnostics.get('active_threat_priority')
    assert result.diagnostics['nominal_escape_roots'] == 0

    wide_window = RollingPlanner(settings)
    for delay in [16] * 3 + [260] * 10:
        wide_window.switch_times.add(delay)
    result = wide_window.plan(current, previous)
    assert not result.diagnostics['timing_interval_complete']
    assert not result.diagnostics.get('active_threat_priority')


@pytest.mark.parametrize('backend', ['python', 'rust'])
def test_contact_fallback_keeps_the_lane_between_live_lasers(backend):
    if backend == 'rust' and search_extension() is None:
        pytest.skip('optional Rust search not built')
    current = frame(x=200.)
    # The actual lane is open. The forecast margins touch both sides and the
    # warning overlaps the player. Reopening horizontal moves hits live beams.
    current.lasers = np.array([(0, 192.1, 0, 0, 448, np.pi/2, 4, 1),
                               (1, 207.9, 0, 0, 448, np.pi/2, 4, 1),
                               (2, 200, 0, 0, 448, np.pi/2, 4, 0)], np.float32)
    planner = RollingPlanner(PlanningConfig(collision_backend=backend))
    result = planner.plan(current, replace(current, tick=100))
    allowed = np.flatnonzero(result.features[:, 1])
    assert result.diagnostics['contact_priority']
    assert result.diagnostics['route_mode'] == 'fallback'
    assert result.diagnostics['timing_contact_free_roots'] == 0
    assert len(allowed) and set(allowed % 9) <= {0, 1, 2}
    assert not result.navigation[:, NAV_FIELDS.index('robust_safe')].any()


def test_warning_exit_narrows_existing_moves_before_activation():
    current = frame()
    current.lasers = np.array([(0, 192, 0, 0, 448, np.pi/2, 4, 0)], np.float32)
    previous = replace(current, tick=100)
    planner = RollingPlanner()
    result = planner.plan(current, previous)
    # A short active-threat check admits hold and slow left, as in the
    # recorded Stage 1 warning case. Only slow left leaves the warning.
    features = result.features.copy()
    features[:, 1] = 0.
    features[[9, 12], 1] = 1.
    result = replace(result, features=features,
                     diagnostics={**result.diagnostics, 'suggested_movement': 9, 'active_threat_priority': True})
    revised = planner._prefer_warning_exit(result, np.array([192., 320.]), np.zeros(2),
                    np.array([8., 16.]), np.array([376., 432.]), visible_body_state(current, previous),
                    np.array([False]))
    np.testing.assert_array_equal(np.flatnonzero(revised.features[:, 1]), [12])
    assert revised.diagnostics['warning_exit_priority']
    assert revised.diagnostics['suggested_movement'] == 12


def test_bottom_pressure_adds_escape_bias_within_the_existing_mask(monkeypatch):
    current = frame(x=200., y=429., bullets=[(0, 210, 430), (1, 189, 430)])
    planner = RollingPlanner()
    previous = replace(current, tick=100)
    preferred = planner.plan(current, previous)
    assert preferred.diagnostics['bottom_escape_priority']
    assert not RollingPlanner().plan(frame(x=200., y=429.)).diagnostics.get('bottom_escape_priority')
    monkeypatch.setattr(planner, '_prefer_bottom_exit', lambda result, *_: result)
    original = planner.plan(current, previous)
    np.testing.assert_array_equal(preferred.features[:, 1], original.features[:, 1])
    assert preferred.features[1, 1] and preferred.features[0, 1]
    assert preferred.features[1, 7] - preferred.features[0, 7] > original.features[1, 7] - original.features[0, 7]


def test_switch_estimate_uses_pairs_and_forgets_old_outliers():
    planner=RollingPlanner()
    planner.record_latency({'post_plan_to_send_ms':80,'movement_changed':True,
            'game_input_match_ms':20,'post_plan_to_input_ms':100})
    first=planner.plan(frame())
    assert first.diagnostics['estimated_switch_ms']==100
    for _ in range(64):
        planner.record_latency({'post_plan_to_send_ms':2,'movement_changed':True,
                'game_input_match_ms':10,'post_plan_to_input_ms':12})
    planner.last_elapsed_ms=0
    recovered=planner.plan(frame())
    assert recovered.diagnostics['estimated_switch_ms']==12
    assert recovered.diagnostics['delay_ticks'] < first.diagnostics['delay_ticks']
    planner.reset_timing()
    assert not planner.switch_times.values


def test_vectorized_body_clearance_matches_independent_convex_search():
    rng=np.random.default_rng(817)
    relative=rng.uniform(-30,30,(31,7,2)).astype(np.float32)
    velocity=rng.uniform(-5,5,(31,7,2)).astype(np.float32)
    extents=rng.uniform(2,12,(31,7,2)).astype(np.float32)
    length=rng.uniform(1,16,31).astype(np.float32)
    actual,_=rectangle_sweep(relative,velocity,extents,length)
    low=np.zeros((31,7))
    high=np.repeat(length[:,None],7,axis=1).astype(float)
    def gap(t):
        return (np.abs(relative+velocity*t[...,None])-extents).max(axis=2)
    # Convex ternary minimization is independent of the affine-intersection
    # implementation and exercises corners, stationary axes and interior hits.
    for _ in range(70):
        a=low+(high-low)/3
        b=high-(high-low)/3
        left=gap(a)<gap(b)
        high=np.where(left,b,high)
        low=np.where(left,low,a)
    expected=gap((low+high)/2).min(axis=1)
    np.testing.assert_allclose(actual,expected,atol=2e-5)


def test_batched_switch_sweeps_use_each_scenarios_own_duration_and_absolute_time():
    end,_,collision,_=swept_segments(np.zeros((3,2)),np.array([[2.,0]]*3),
        np.array([1.,2.,3.]),np.array([0.,3.,5.]),np.array([0,0]),np.array([100,100]),
        np.array([[12.,0]]),np.array([[-1.,0]]),np.zeros(1),np.zeros(1))
    np.testing.assert_allclose(end[:,0],[2,4,6])
    assert np.isinf(collision[:2]).all()
    np.testing.assert_allclose(collision[2],20/3,atol=1e-5)


def test_low_latency_does_not_create_an_unbounded_extra_search_layer():
    result=RollingPlanner(PlanningConfig(beam_width=4)).plan(frame())
    assert 18 <= result.diagnostics['candidates_checked'] <= MAX_SEARCH_NODES
    assert result.paths.shape[0]==18
    assert result.diagnostics['path_ticks'][-1]==result.diagnostics['horizon_ticks']


def test_vectorized_branch_priorities_preserve_order_and_remove_duplicates():
    result=RollingPlanner._branches(np.array([[100,100],[300,400],[192,224]]),np.array([4,14,0]),8,0,
            np.empty((0,2)),np.empty((0,2)),np.empty(0),np.empty((0,6)),np.empty(0),None,
            np.array([8,16]),np.array([376,432]))
    np.testing.assert_array_equal(result,[[4,8,0,13,6,3,1,2],
            [14,9,5,12,10,17,0,1],[0,9,6,8,3,1,2,4]])


def test_interior_switch_time_can_collide_while_both_endpoints_are_free():
    planner=RollingPlanner()
    args=(np.array([192,320],np.float32),np.array([4,0],np.float32))
    geometry=(6,np.array([8,16]),np.array([376,432]),np.array([[200,312]],np.float32),
              np.zeros((1,2),np.float32),np.zeros(1),np.zeros(1),visible_body_state(frame()))
    endpoints=planner._timing_probe(*args,(1,3),*geometry)
    interval,complete=planner._switch_delays(1,3)
    covered=planner._timing_probe(*args,interval,*geometry)
    assert endpoints[0][1] and complete
    assert not covered[0][1] and 2<covered[1][1]<6
    contact = {'clearance_floor': -COLLISION_PADDING + 1e-4}
    assert planner._timing_probe(*args, (1, 3), *geometry, **contact)[0][1]
    assert not planner._timing_probe(*args, interval, *geometry, **contact)[0][1]


def test_recorded_boss_death_does_not_reopen_early_collision_via_nominal_union():
    from pathlib import Path
    from th06ai.native import Frame
    with np.load(Path(__file__).parent/'fixtures/stage5-switch-window.npz',allow_pickle=False) as data:
        def read(index,suffix):
            row=data['frames'][index];bullets=data['bullets_'+suffix]
            return Frame(int(row[0]),float(row[2]),float(row[3]),int(row[4]),bool(row[6]),False,
                bullets[:,:3],(8,16,368,416),True,0,bullet_sizes=bullets[:,3:5],
                player_radius=float(row[7]),enemy_bodies=data['enemy_bodies_'+suffix],lasers=data['lasers_'+suffix])
        old,current=read(0,'previous'),read(1,'current')
    planner=RollingPlanner(PlanningConfig(beam_width=1,collision_backend='python'))
    for delay in [23]+[45]*8:planner.switch_times.add(delay)
    control=np.zeros(28);control[4]=control[19]=1
    state=visible_bullet_state(current,old);bodies=visible_body_state(current,old)
    def probe(delays):
        return planner._timing_probe(np.array([current.x,current.y]),planner.velocities[4],delays,8,
            np.array([8,16]),np.array([376,432]),current.bullets[:,1:3],state.velocity,
            state.radius,np.zeros(len(state.radius)),bodies)
    assert probe((4,))[0][5]  # The single nominal sample falsely reopened up-left in v6.
    assert not probe((2,))[0][5]  # Actual early dispatch crosses a recorded ordinary bullet.
    result=planner.plan(current,old,control)
    assert result.diagnostics['delay_bounds_ticks']==[1,5]
    assert result.diagnostics['tested_switch_delays']==(1,2,3,4,5)
    assert result.diagnostics['all_routes_risky'] and result.diagnostics['short_escape_priority']
    assert not result.features[5,1] and result.features[:,1].any()
    free=probe((1,2,3,4,5))[0]
    assert np.all(free[result.features[:,1]>.5])


def test_extreme_switch_intervals_have_a_bounded_explicitly_incomplete_grid():
    delays,complete=RollingPlanner._switch_delays(1,300)
    assert len(delays)<=12 and delays[0]==1 and delays[-1]==300 and not complete
