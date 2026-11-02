"""Counterfactual reward tests: safety/task success matter; graze is diagnostic.

All scenes are synthetic. No game input, game process or CUDA is used.
"""
from dataclasses import replace

import numpy as np
import pytest

from th06ai.native import GameResources
from th06ai.observations import ObservationBuilder
from th06ai.rewards import RewardTracker
from th06ai.risk import collision_risk, visible_threat
from test_rewards import frame


def resources(graze=0, score=0, **changes):
    return GameResources(score, 10, 3, 2, 0, graze=graze, **changes)


def transition(old, new, *, died=False):
    tracker = RewardTracker()
    tracker.reset(old)
    result = tracker.step(new, 0, died)
    return tracker, result


@pytest.mark.parametrize("score_gain", [500, 100000000])
def test_direct_or_multiplier_graze_score_is_not_an_objective_or_resource_budget(score_gain):
    start = replace(frame(), resources=resources())
    plain_tracker, plain = transition(start, replace(start, tick=102))
    graze_tracker, grazed = transition(start, replace(start, tick=102, resources=resources(1, score_gain)))
    assert grazed.graze_gained == 1
    assert grazed.components == plain.components
    assert grazed.total == plain.total
    assert grazed.components["score"] == 0
    assert graze_tracker.resource_raw_credit == plain_tracker.resource_raw_credit == 0
    # Even a huge graze-related score cannot consume or alter later P feedback.
    followups = []
    for tracker in (plain_tracker, graze_tracker):
        new = replace(tracker.frame, tick=104, resources=replace(tracker.frame.resources, power=11))
        followups.append(tracker.step(new, 0, False).components["power"])
    assert followups[0] == followups[1] > 0


def test_a_safe_close_pass_is_neither_rewarded_nor_penalized_for_graze():
    totals = []
    for bullet_x, gain in ((204, 1), (240, 0)):
        # Radius = 4 combined, closest distance = 12 or 48. The first trajectory
        # is inside the native graze ring but outside the collision safety margin.
        tracker = RewardTracker()
        tracker.reset(replace(frame(tick=98, bullets=[(1, bullet_x, 348)]), resources=resources()))
        total = 0
        for i in range(1, 15):
            count = gain if i >= 5 else 0
            f = replace(frame(tick=98+2*i, bullets=[(1, bullet_x, 348+4*i)]),
                        resources=resources(count, count*500))
            result = tracker.step(f, 0, False)
            assert result.threat == 0
            total += result.total
        totals.append(total)
        assert tracker.graze_gained == gain
        assert tracker.graze_with_risk == 0
    assert totals[0] == totals[1]


def test_collision_course_is_risky_while_tangential_near_pass_is_not():
    approaching = frame(tick=98, bullets=[(1, 192, 360)])
    current = frame(bullets=[(1, 192, 364)])
    tangent = frame(tick=98, bullets=[(1, 204, 360)])
    tangent_now = frame(bullets=[(1, 204, 364)])
    assert visible_threat(current, approaching) > .5
    estimate = collision_risk(tangent_now, tangent)
    assert estimate.threat == 0
    assert estimate.min_clearance == pytest.approx(8)
    assert estimate.geometry_coverage == 1


def test_real_bullet_size_changes_collision_feedback_without_rewarding_proximity():
    old = frame(tick=98, bullets=[(1, 204, 360)])
    now = frame(bullets=[(1, 204, 364)])
    large = replace(now, bullet_sizes=np.array([[40, 40]], np.float32))
    assert visible_threat(now, old) == 0
    assert visible_threat(large, old) > .5


@pytest.mark.parametrize("sizes, radius", [(None, 2), ([[float('nan'), 4]], 2),
                                           ([[0, 0]], 2), ([[4, 4]], None), ([[4, 4]], 0)])
def test_missing_or_invalid_geometry_cannot_claim_a_tight_pass_is_safe(sizes, radius):
    now = frame(bullets=[(1, 204, 384)])
    unknown = replace(now, bullet_sizes=None if sizes is None else np.array(sizes), player_radius=radius)
    result = collision_risk(unknown, replace(unknown, tick=98))
    assert visible_threat(now, replace(now, tick=98)) == 0
    assert result.threat == 1
    assert result.geometry_coverage == 0 and result.min_clearance is None


def test_unknown_motion_does_not_claim_predicted_clearance():
    f = frame(bullets=[(1, 204, 384)])
    assert collision_risk(f).min_clearance is None


def test_graze_event_cannot_offset_death_or_lost_boss_damage():
    enemy = np.array([[0, 192, 64, 200, 200, 200, 0x11, 1]], np.float32)
    start = replace(frame(), resources=resources(), enemies=enemy)
    damaged = enemy.copy()
    damaged[0, 3] -= 40
    _, success = transition(start, replace(start, tick=102, enemies=damaged))
    _, grazing_instead = transition(start, replace(start, tick=102, resources=resources(100, 50000)))
    tracker, died = transition(start, replace(start, tick=102, state=2, resources=resources(100, 50000)), died=True)
    assert success.total > grazing_instead.total > died.total
    assert died.total < -19
    assert tracker.graze_with_risk == 100


def test_graze_stats_use_native_increments_and_never_proximity_resets_or_pause_gaps():
    start = replace(frame(bullets=[(1, 204, 384)]), resources=resources(12))
    tracker, result = transition(start, replace(start, tick=102, resources=resources(13, 500)))
    assert result.graze_gained == 1
    for tick in (104, 106):
        same = replace(tracker.frame, tick=tick)
        assert tracker.step(same, 0, False).graze_gained == 0
    reset = replace(tracker.frame, tick=108, resources=resources(0))
    assert tracker.step(reset, 0, False).graze_gained == 0
    unknown = replace(reset, tick=110, resources=resources(None))
    assert tracker.step(unknown, 0, False).graze_gained == 0
    tracker.rebase(replace(reset, tick=150, resources=resources(100)))
    assert tracker.step(replace(tracker.frame, tick=152), 0, False).graze_gained == 0
    assert tracker.diagnostics()["graze_gained"] == 1


def test_graze_counter_is_not_a_policy_input_or_a_goal():
    a = replace(frame(), resources=resources(0))
    b = replace(a, resources=resources(999999))
    first = ObservationBuilder().build(a)
    second = ObservationBuilder().build(b)
    assert first.keys() == second.keys()
    for key in first:
        np.testing.assert_array_equal(first[key], second[key])


def test_point_item_rewards_remain_available_with_graze_score_mixed_in():
    start = replace(frame(), resources=resources())
    a = replace(start, tick=102, resources=replace(resources(), point_items=1, score=10000))
    b = replace(a, resources=replace(a.resources, graze=12, score=16000))
    _, plain = transition(start, a)
    _, mixed = transition(start, b)
    assert plain.components == mixed.components
    assert plain.components["point_items"] > 0


def test_invalid_graze_counter_is_rejected():
    with pytest.raises(RuntimeError, match="范围校验"):
        resources(1000000)


@pytest.mark.parametrize("counter_disabled", [False, True])
def test_native_adapter_reads_graze_and_aligned_collision_geometry(counter_disabled):
    """Exercise the real adapter against bytes, without opening a process."""
    import struct
    from types import SimpleNamespace
    from th06ai import native
    from th06ai.combat import ENEMY_POOL, ENEMY_CAPACITY, ENEMY_STRIDE, BOSS_POINTERS

    base = 0x140000000
    player = bytearray(0x1A0)
    struct.pack_into("<ff", player, 0x20, 192, 384)
    struct.pack_into("<f", player, 0x3C, 2)
    bullets = np.zeros(native.CAPACITY, native.BULLET_DTYPE)
    bullets[7] = (204, 384, 1, 4, 8)
    bullets[9] = (240, 384, 2, 0, 0)  # Unset spawning geometry must stay unknown.
    node = bytearray(0x40)
    struct.pack_into("<Q", node, 8, base + native.BULLET_UPDATE)
    struct.pack_into("<Q", node, 0x38, base + native.BULLET_MANAGER)
    image = {
        native.PLAYER + 0x7710: bytes(player), native.PAUSED: b"\x01",
        native.BOUNDS: struct.pack("<ffff", 8, 16, 368, 416), native.BULLET_NODE: bytes(node),
        native.BULLET_MANAGER: struct.pack("<ii", 100, 100), native.BULLET_MANAGER+8: bullets.tobytes(),
        native.ITEM_POOL: bytes(native.ITEM_CAPACITY*native.ITEM_STRIDE),
        ENEMY_POOL: bytes(ENEMY_CAPACITY*ENEMY_STRIDE), BOSS_POINTERS: bytes(32),
        native.POWER: struct.pack("<H", 16), native.LIVES_BOMBS: struct.pack("<bb", 2, 3),
        native.SCORE: struct.pack("<Q", 10000), native.POINT_ITEMS: struct.pack("<H", 1),
        native.GRAZE_TOTAL: struct.pack("<i", 9), native.GRAZE_DISABLED: bytes([counter_disabled]),
    }

    def read(rva, size):
        value = bytearray(size)
        for address, data in image.items():
            begin, end = max(rva, address), min(rva+size, address+len(data))
            if begin < end:
                value[begin-rva:end-rva] = data[begin-address:end-address]
        return bytes(value)

    reader = native.GameReader.__new__(native.GameReader)
    reader.process = SimpleNamespace(base=base, read=read)
    reader._dtype = native.BULLET_DTYPE
    f = reader.read()
    assert f.paused and f.stage_active
    assert f.resources.graze == (None if counter_disabled else 9)
    assert f.player_radius == 2
    assert f.bullets[:, 0].tolist() == [7, 9]
    np.testing.assert_array_equal(f.bullet_sizes[0], [4, 8])
    assert np.isnan(f.bullet_sizes[1]).all()
    assert collision_risk(f).geometry_coverage == .5


def test_evaluation_records_graze_separately_without_changing_success_metrics(tmp_path):
    from types import SimpleNamespace
    from th06ai.learning import evaluate
    from th06ai.files import read_json

    class EvaluationFixture:
        episode_seconds = 180

        def __init__(self, graze):
            self.graze = graze
            self.paused = False

        def reset(self):
            return {}, {}

        def step(self, action):
            return {}, 0, False, True, {
                "survival_seconds": 180, "episode_diagnostics": {
                    "boss_damage": 6000, "enemy_damage": 200, "power_gained": 16,
                    "point_items_gained": 10, "graze_gained": self.graze,
                    "graze_with_risk": 0, "graze_observed_seconds": 180,
                },
            }

        def pause(self):
            self.paused = True

    model = SimpleNamespace(predict=lambda *args, **kwargs: (0, None))
    reports = []
    for count in (0, 1000):
        env = EvaluationFixture(count)
        run = tmp_path / str(count)
        run.mkdir()
        report = evaluate(model, env, run, lambda **kwargs: None, lambda: None, episodes=1)
        assert env.paused
        assert read_json(run / "evaluation.json")["graze_diagnostics"]["total"] == count
        assert report["graze_diagnostics"]["used_as_objective"] is False
        reports.append(report)
    for metric in ("mean_survival_seconds", "no_hit_rate", "mean_boss_damage", "mean_enemy_damage",
                   "mean_power_gained", "mean_point_items_gained", "loiter_ratio"):
        assert reports[0][metric] == reports[1][metric]
