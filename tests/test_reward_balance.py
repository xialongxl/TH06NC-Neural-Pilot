"""Reward-v7 counterexamples; no game or GPU access."""
from dataclasses import replace
import math

import pytest

from th06ai.native import GameResources
from th06ai.rewards import (RewardTracker, RESOURCE_BUDGET, DEATH_COST,
                            SURVIVAL_PER_SECOND, RESOURCE_COMPONENTS)
from test_rewards import frame
from test_resources import resource_transition
from test_behavior_objective import trajectory


@pytest.mark.parametrize("change,component,legacy_raw,minimum", [
    ({"power": 1}, "power", .2, .9),
    ({"point_items": 1}, "point_items", .05, .24),
    ({"bombs": 4}, "bomb_pickup", .5, 1.8),
    ({"lives": 3}, "life_pickup", 1, 3.5),
])
def test_real_pickups_have_materially_stronger_credit(change, component, legacy_raw, minimum):
    old = GameResources(0, 0, 3, 2, 0)
    _, result = resource_transition(old, replace(old, **change))
    legacy = 6 * -math.expm1(-legacy_raw/6)
    assert result.components[component] >= minimum
    assert result.components[component] > 3.5 * legacy


def test_two_power_and_seven_point_items_are_significant_without_rewarding_death():
    old = GameResources(0, 0, 3, 2, 0)
    new = replace(old, power=2, point_items=7)
    _, result = resource_transition(old, new)
    credit = sum(result.components[name] for name in RESOURCE_COMPONENTS)
    legacy = 6 * -math.expm1(-(.2*2+.05*7)/6)
    assert credit > 4.5*legacy and credit > 3
    assert RESOURCE_BUDGET < DEATH_COST
    _, collision = resource_transition(old, new, died=True)
    assert all(collision.components[name] == 0 for name in RESOURCE_COMPONENTS)
    assert collision.total < 0


def test_pickups_remain_valuable_after_an_initial_resource_batch():
    start = replace(frame(), resources=GameResources(0, 0, 3, 2, 0))
    tracker = RewardTracker()
    tracker.reset(start)
    collected = replace(start, tick=102, resources=replace(start.resources, power=10, point_items=30))
    tracker.step(collected, 0, False)
    next_power = replace(collected, tick=104, resources=replace(collected.resources, power=11))
    result = tracker.step(next_power, 0, False)
    legacy_first_power = 6 * -math.expm1(-.2/6)
    assert result.components["power"] > legacy_first_power
    assert result.components["power"] > 5*SURVIVAL_PER_SECOND


def test_power_then_immediate_death_is_worse_than_staying_alive():
    start = replace(frame(), resources=GameResources(0, 0, 3, 2, 0))
    tracker = RewardTracker()
    tracker.reset(start)
    collected = replace(start, tick=102, resources=replace(start.resources, power=1))
    pickup = tracker.step(collected, 0, False)
    death = tracker.step(replace(collected, tick=104, state=2), 0, True)
    safe = RewardTracker()
    safe.reset(start)
    alive = safe.step(replace(start, tick=104), 0, False)
    assert pickup.total + death.total < alive.total


def test_large_corner_wiggles_do_not_hide_continuous_residence():
    def triangle(i):
        k = i % 24
        return min(k, 24-k)*4
    total, _, diagnostics = trajectory(lambda i: (328+triangle(i), 384+triangle(i+6)), seconds=6)
    assert total < -.5
    assert diagnostics["loiter_seconds"] > 2
    assert -diagnostics["reward_edge_loiter"] > 4*diagnostics["reward_survival"]


def test_corner_grace_restarts_only_after_leaving_the_corner_region():
    tracker = RewardTracker()
    tracker.reset(frame(tick=100, x=376, y=360))
    for tick in range(102, 282, 2):
        tracker.step(frame(tick=tick, x=376, y=360), 0, False)
    # Enter the corner after the episode's initial grace has long elapsed.
    for tick in range(282, 344, 2):
        result = tracker.step(frame(tick=tick, x=376, y=376), 0, False)
        assert result.components["edge_loiter"] == 0
    for tick in range(344, 466, 2):
        result = tracker.step(frame(tick=tick, x=376, y=376), 0, False)
    assert result.components["edge_loiter"] < 0


def test_small_inward_twitch_does_not_restore_wall_push_grace():
    tracker = RewardTracker()
    tracker.reset(frame(x=8))
    for tick in range(102, 122, 2):
        tracker.step(frame(tick=tick, x=8), 3, False)
    assert tracker.totals["wall_push"] < 0
    tracker.step(frame(tick=122, x=12), 4, False)
    tracker.step(frame(tick=124, x=8), 3, False)
    assert tracker.step(frame(tick=126, x=8), 3, False).components["wall_push"] < 0
    # A real departure restores the ordinary short input-latency allowance.
    tracker.step(frame(tick=132, x=32), 4, False)
    tracker.step(frame(tick=138, x=8), 3, False)
    assert tracker.step(frame(tick=140, x=8), 3, False).components["wall_push"] == 0


def test_input_latency_in_open_space_is_not_misclassified_as_wall_push():
    tracker = RewardTracker()
    tracker.reset(frame())
    for tick in range(102, 142, 2):
        result = tracker.step(frame(tick=tick), 3, False)
        assert result.components["wall_push"] == 0
