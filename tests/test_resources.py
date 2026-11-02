from dataclasses import replace

import numpy as np
import pytest

from th06ai.native import ITEM_CAPACITY, ITEM_DTYPE, GameResources, decode_items
from th06ai.rewards import RESOURCE_BUDGET, RESOURCE_COMPONENTS, RewardTracker
from test_rewards import frame


def test_item_pool_decodes_all_seven_types_and_filters_inactive_and_offscreen():
    data = np.zeros(ITEM_CAPACITY, dtype=ITEM_DTYPE)
    for i in range(7):
        data[i] = (1, 100+i, 200, 0, 2, i)
    data[7] = (1, 100, 500, 0, 2, 0)
    items = decode_items(data.tobytes(), 384, 448)
    assert items.shape == (7, 6)
    assert items[:, 3].tolist() == list(range(7))
    assert np.all(items[:, 5] == 2)


def test_invalid_item_types_and_resources_are_rejected():
    data = np.zeros(ITEM_CAPACITY, dtype=ITEM_DTYPE)
    data[0] = (1, 100, 200, 0, 2, 99)
    with pytest.raises(RuntimeError, match="道具类型"):
        decode_items(data.tobytes(), 384, 448)
    with pytest.raises(RuntimeError, match="范围校验"):
        GameResources(0, 129, 3, 2, 0)


def resource_transition(old, new, died=False):
    tracker = RewardTracker()
    tracker.reset(replace(frame(), resources=old))
    result = tracker.step(replace(frame(tick=102), resources=new), 0, died)
    return tracker, result


@pytest.mark.parametrize("changes, component", [
    ({"power": 1}, "power"), ({"bombs": 4}, "bomb_pickup"),
    ({"lives": 3}, "life_pickup"),
    ({"point_items": 1}, "point_items"),
])
def test_actual_resource_increases_receive_separate_credit(changes, component):
    old = GameResources(0, 0, 3, 2, 0)
    _, result = resource_transition(old, replace(old, **changes))
    assert result.components[component] > 0


def test_standing_near_a_drop_or_pressing_bomb_without_spending_does_not_create_resource_rewards():
    state = GameResources(0, 0, 3, 2, 0)
    _, result = resource_transition(state, state)
    assert all(result.components[name] == 0 for name in (*RESOURCE_COMPONENTS, "bomb_use"))


def test_collecting_and_using_bombs_are_distinguished_and_use_cost_is_smaller_than_death():
    old = GameResources(0, 0, 3, 2, 0)
    _, result = resource_transition(old, replace(old, bombs=2))
    assert result.components["bomb_pickup"] == 0
    assert -1 < result.components["bomb_use"] < 0


def test_positive_resource_credit_is_bounded_and_death_cannot_collect_a_last_step_bonus():
    tracker = RewardTracker()
    old = replace(frame(), resources=GameResources(0, 0, 0, 0, 0))
    tracker.reset(old)
    for i in range(1, 11):
        new = replace(old, tick=100+2*i, resources=GameResources(i*100000000, min(128,i*12),
                                                              min(8,i), min(8,i), i*20))
        tracker.step(new, 0, False)
    assert .99 * RESOURCE_BUDGET < tracker.resource_bonus_used <= RESOURCE_BUDGET
    assert sum(tracker.totals[name] for name in RESOURCE_COMPONENTS) <= RESOURCE_BUDGET + 1e-8
    _, death = resource_transition(GameResources(0,0,0,0,0), GameResources(999999,128,8,8,20), died=True)
    assert all(death.components[name] == 0 for name in RESOURCE_COMPONENTS)
    assert death.total < -9


def test_resource_cap_persists_across_update_pauses_and_resets_for_a_new_episode():
    tracker, _ = resource_transition(GameResources(0,0,0,0,0), GameResources(100000000,128,8,8,20))
    before = tracker.resource_bonus_used
    tracker.rebase(replace(frame(tick=104), resources=GameResources(100000000,128,8,8,20)))
    assert tracker.resource_bonus_used == before
    tracker.reset(frame(tick=106))
    assert tracker.resource_bonus_used == 0


def test_raw_score_at_full_power_is_context_only_and_reset_is_not_collection():
    old = GameResources(10000,128,3,2,30)
    _, full = resource_transition(old, replace(old, score=11000))
    assert full.components["power"] == 0
    assert full.components["score"] == 0
    _, point = resource_transition(old, replace(old, point_items=31, score=11000))
    assert point.components["point_items"] > 0
    _, reset = resource_transition(old, GameResources(0,0,3,2,0))
    assert all(reset.components[name] == 0 for name in RESOURCE_COMPONENTS)


def test_point_counter_wrap_is_distinguished_from_reset():
    old = GameResources(10000,0,3,2,65535)
    _, result = resource_transition(old, replace(old, point_items=0, score=11000))
    _, normal = resource_transition(replace(old, point_items=20), replace(old, point_items=21, score=11000))
    assert result.components["point_items"] == pytest.approx(normal.components["point_items"])
    assert result.components["point_items"] > 0
