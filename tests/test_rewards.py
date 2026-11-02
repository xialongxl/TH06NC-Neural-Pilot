"""Reward regressions using visible geometry only; no game or GPU access."""
from dataclasses import replace

import numpy as np
import pytest

from th06ai.native import Frame
from th06ai.rewards import DISCOUNT, THREAT_WEIGHT, DEATH_COST, RewardTracker, visible_threat


def frame(tick=100, x=192, y=384, bullets=()):
    bullets = np.asarray(bullets, dtype=np.float32).reshape((-1, 3))
    return Frame(tick, x, y, 3, False, False,
                 bullets, (8, 16, 368, 416), True, 0.0,
                 bullet_sizes=np.full((len(bullets), 2), 4, np.float32), player_radius=2.0)


def test_approaching_bullet_has_higher_threat_than_same_position_receding_bullet():
    current = frame(bullets=[(1, 192, 344)])
    approaching = frame(tick=98, bullets=[(1, 192, 340)])
    receding = frame(tick=98, bullets=[(1, 192, 348)])
    assert visible_threat(current, approaching) > visible_threat(current, receding)


def test_unknown_velocity_and_recycled_slots_do_not_create_phantom_approaches():
    current = frame(bullets=[(1, 192, 344)])
    recycled = frame(tick=98, bullets=[(1, 10, 20)])
    unseen = frame(tick=98, bullets=[(2, 192, 340)])
    stale = frame(tick=50, bullets=[(1, 192, 340)])
    for previous in (recycled, unseen, stale):
        assert visible_threat(current, previous) == pytest.approx(visible_threat(current))


def test_empty_field_has_zero_threat_and_never_rewards_motion_itself():
    start = frame()
    still, moving = RewardTracker(), RewardTracker()
    still.reset(start)
    moving.reset(start)
    a = still.step(frame(tick=102), 0, False)
    b = moving.step(frame(tick=102, x=196), 4, False)
    assert a.total == pytest.approx(b.total)
    assert a.threat == 0
    assert a.components["threat_shaping"] == 0


def test_moving_out_of_an_observed_bullet_path_gets_better_feedback_than_waiting():
    def prepared():
        tracker = RewardTracker()
        tracker.reset(frame(tick=98, bullets=[(1, 192, 336)]))
        tracker.step(frame(bullets=[(1, 192, 340)]), 0, False)
        return tracker
    still = prepared().step(frame(tick=102, bullets=[(1, 192, 344)]), 0, False)
    escape = prepared().step(frame(tick=102, x=208, bullets=[(1, 192, 344)]), 4, False)
    assert escape.threat < still.threat
    assert escape.total > still.total


def test_brief_corner_hold_and_normal_wall_sliding_are_not_treated_as_blocked_input():
    for action in (0, 1):
        tracker = RewardTracker()
        tracker.reset(frame(x=8))
        for i in range(1, 16):
            result = tracker.step(frame(tick=100+2*i, x=8, y=384-(4*i if action else 0)), action, False)
            assert result.components["wall_push"] == 0
        assert tracker.diagnostics()["wall_push_seconds"] == 0


@pytest.mark.parametrize("action, y_step", [(3, 0), (5, -4)])
def test_persistent_ineffective_axis_is_penalized_even_when_diagonal_can_slide(action, y_step):
    tracker = RewardTracker()
    tracker.reset(frame(x=8))
    for i in range(1, 16):
        result = tracker.step(frame(tick=100+2*i, x=8, y=384+i*y_step), action, False)
    assert result.components["wall_push"] < 0
    assert tracker.diagnostics()["wall_push_seconds"] > 0


def test_brief_input_latency_has_a_grace_period_and_movement_clears_blockage():
    tracker = RewardTracker()
    tracker.reset(frame(x=8))
    for i in range(1, 7):
        assert tracker.step(frame(tick=100+2*i, x=8), 3, False).components["wall_push"] == 0
    assert tracker.step(frame(tick=114, x=8), 3, False).components["wall_push"] < 0
    assert tracker.step(frame(tick=116, x=12), 4, False).components["wall_push"] == 0
    assert tracker.step(frame(tick=118, x=12), 4, False).components["wall_push"] == 0


def test_blocked_cost_uses_game_time_instead_of_number_of_samples():
    totals = []
    for stride in (2, 6):
        tracker = RewardTracker()
        tracker.reset(frame(x=8))
        for tick in range(100+stride, 161, stride):
            tracker.step(frame(tick=tick, x=8), 3, False)
        totals.append(tracker.diagnostics()["reward_wall_push"])
    assert totals[0] == pytest.approx(totals[1])


def test_oscillation_cannot_farm_more_discounted_shaping_than_same_end_state():
    start = frame(bullets=[(1, 192, 360)])
    totals = []
    for excursion in (0, 16):
        tracker = RewardTracker()
        tracker.reset(start)
        a = tracker.step(replace(start, tick=102, x=192+excursion), 4 if excursion else 0, False)
        b = tracker.step(replace(start, tick=104), 3 if excursion else 0, False)
        totals.append(a.components["threat_shaping"] + DISCOUNT*b.components["threat_shaping"])
    assert totals[0] == pytest.approx(totals[1])


def test_terminal_potential_is_zero_and_collision_remains_costly():
    start = frame(bullets=[(1, 192, 384)])
    tracker = RewardTracker()
    tracker.reset(start)
    result = tracker.step(replace(start, tick=102, state=2), 0, True)
    assert result.components["death"] == -DEATH_COST
    assert result.components["threat_shaping"] == pytest.approx(THREAT_WEIGHT)
    assert result.total < -7


def test_pause_rebase_clears_motion_history_without_erasing_episode_diagnostics():
    tracker = RewardTracker()
    tracker.reset(frame(x=8))
    for i in range(1, 9):
        tracker.step(frame(tick=100+2*i, x=8), 3, False)
    before = tracker.diagnostics()
    tracker.rebase(frame(tick=150, x=8))
    assert tracker.diagnostics() == before
    assert tracker.step(frame(tick=152, x=8), 3, False).components["wall_push"] == 0


def test_frozen_frame_does_not_generate_survival_or_shaping_rewards():
    tracker = RewardTracker()
    tracker.reset(frame())
    with pytest.raises(RuntimeError, match="游戏帧"):
        tracker.step(frame(), 0, False)
