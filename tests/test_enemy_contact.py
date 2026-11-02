"""Native contact gates and rectangle avoidance; no real game inputs."""
from dataclasses import replace

import numpy as np
import pytest
import torch

from th06ai.combat import BODY_DTYPE, ENEMY_CAPACITY, decode_enemy_bodies, decode_enemies
from th06ai.body_state import BodyState, visible_body_state
from th06ai.planning import RollingPlanner, swept_segments
from th06ai.observations import ObservationBuilder
from th06ai import learning
from test_learning import LearningFixture
from test_planning import frame


def body_frame(tick=102, x=192, y=320, body_x=220, body_y=320, half_width=8, half_height=8, **kwargs):
    return replace(frame(tick=tick, x=x, y=y, **kwargs),
                   enemy_bodies=np.array([[0, body_x, body_y, half_width, half_height, 1]], np.float32))


def test_native_contact_decoder_uses_verified_size_and_ignores_attack_hp_filters():
    data = np.zeros(ENEMY_CAPACITY, BODY_DTYPE)
    data[0] = (220, 320, -128, 7, 0, 60, 12)
    # HP fields are still zero: the existing attack-target decoder returns no
    # target, while the actual contact gate remains enabled in this snapshot.
    assert not len(decode_enemies(data.tobytes(), 384, 448))
    body = decode_enemy_bodies(data.tobytes(), 384, 448)
    assert body.shape == (1, 6)
    assert np.allclose(body[0, 3:5], [20, 4])
    assert body[0, 5] == 1


@pytest.mark.parametrize("flags,extra,active,expected", [
    (7, 0, -128, 1), (0xF, 0, -128, 1), (0x17, 0, -128, 1),
    (0x3F, 8, -128, 0), (6, 0, -128, 0), (5, 0, -128, 0),
    (3, 0, -128, 0), (7, 0, 0, 0)])
def test_native_contact_flags_are_separate_from_shootability(flags, extra, active, expected):
    data = np.zeros(ENEMY_CAPACITY, BODY_DTYPE)
    data[0] = (192, 80, active, flags, extra, 48, 48)
    assert len(decode_enemy_bodies(data.tobytes(), 384, 448)) == expected


def test_partially_visible_body_and_unknown_geometry_are_not_silently_dropped():
    data = np.zeros(ENEMY_CAPACITY, BODY_DTYPE)
    data[0] = (-3, 80, -128, 7, 0, 30, 30)
    data[1] = (192, 80, -128, 7, 0, np.nan, 48)
    body = decode_enemy_bodies(data.tobytes(), 384, 448)
    assert len(body) == 2
    assert body[0, 3] == pytest.approx(10)
    assert body[1, 5] == 0 and np.isnan(body[1, 3:5]).all()
    current = replace(frame(), enemy_bodies=body)
    result = RollingPlanner().plan(current)
    assert np.isfinite(result.features).all()
    assert result.diagnostics["unknown_body_geometry"] == 1


def geometry(x=30, y=0, half_width=20, half_height=4, velocity=(0, 0)):
    return BodyState(np.array([[x, y]], np.float32), np.array([velocity], np.float32),
            np.array([[half_width, half_height]], np.float32), np.ones(1, bool), np.ones(1, bool), 2., np.zeros(1, np.float32))


def test_rectangle_contact_detects_crossing_without_any_bullets_or_endpoint_overlap():
    empty = np.empty((0, 2), np.float32)
    end, clearance, hit, _ = swept_segments(np.array([[0, 6]]), np.array([[10, 0]]), 6, 0,
            np.array([0, 0]), np.array([384, 448]), empty, empty, np.empty(0), np.empty(0), bodies=geometry())
    assert np.allclose(end, [[60, 6]])
    assert clearance[0] < 0 and hit[0] == pytest.approx(.6)


def test_long_thin_body_is_not_replaced_by_an_oversized_circular_radius():
    empty = np.empty((0, 2), np.float32)
    _, clearance, hit, _ = swept_segments(np.array([[0, 16]]), np.array([[10, 0]]), 6, 0,
            np.array([0, 0]), np.array([384, 448]), empty, empty, np.empty(0), np.empty(0), bodies=geometry())
    assert clearance[0] == pytest.approx(8)
    assert np.isinf(hit[0])


def test_moving_enemy_crosses_stationary_player_between_two_clear_endpoints():
    empty = np.empty((0, 2), np.float32)
    body = geometry(x=10, y=100, velocity=(20, 0))
    _, clearance, hit, _ = swept_segments(np.array([[60, 100]]), np.array([[0, 0]]), 5, 0,
            np.array([0, 0]), np.array([384, 448]), empty, empty, np.empty(0), np.empty(0), bodies=body)
    assert clearance[0] < 0 and hit[0] == pytest.approx((50-24)/20)


def test_native_disabled_contact_does_not_create_a_phantom_obstacle():
    data = np.zeros(ENEMY_CAPACITY, BODY_DTYPE)
    data[0] = (220, 320, -128, 0x3F, 8, 48, 24)
    current = replace(frame(items=[(0, 248, 320, 0, 0, 0)]),
                      enemy_bodies=decode_enemy_bodies(data.tobytes(), 384, 448))
    result = RollingPlanner().plan(current, replace(current, tick=100))
    assert result.diagnostics["visible_enemy_bodies"] == 0
    assert result.features[4, 1] == 1


def test_observed_body_motion_and_shape_changes_are_handled_without_guessing_velocity():
    before = body_frame(tick=100, body_x=102)
    current = body_frame(body_x=112)
    state = visible_body_state(current, before)
    assert np.allclose(state.velocity, [[5, 0]]) and state.known_motion.all()
    changed = replace(current, enemy_bodies=current.enemy_bodies.copy())
    changed.enemy_bodies[:, 3] = 20
    assert not visible_body_state(changed, before).known_motion.any()


def test_resource_behind_an_enemy_cannot_lure_the_policy_through_its_body():
    current = body_frame(items=[(0, 248, 320, 0, 0, 0)])
    result = RollingPlanner().plan(current, replace(current, tick=100))
    assert result.diagnostics["visible_bullets"] == 0
    assert result.diagnostics["considered_enemy_bodies"] == 1
    assert result.features[4, 1] == 0
    assert result.features[3, 1] == 1
    torch.set_num_threads(1)
    model = learning.make_model(LearningFixture(), "cpu")
    builder = ObservationBuilder(planner=RollingPlanner())
    builder.build(replace(current, tick=100))
    observation = builder.build(current)
    with torch.no_grad():
        model.policy.action_net.bias[4] = model.policy.action_net.bias[22] = 100
        tensor, _ = model.policy.obs_to_tensor(observation)
        distribution = model.policy.get_distribution(tensor)
        assert distribution.distribution.probs[0, 4] == distribution.distribution.probs[0, 22] == 0
        chosen, values, logs = model.policy(tensor)
        compared, compared_logs, _ = model.policy.evaluate_actions(tensor, chosen)
        assert torch.allclose(values, compared) and torch.allclose(logs, compared_logs)


def test_new_enemy_with_unknown_motion_does_not_make_current_body_a_fallback_shortcut():
    current = body_frame(items=[(0, 248, 320, 0, 0, 0)])
    result = RollingPlanner().plan(current)
    assert result.diagnostics["unknown_body_motion"] == 1
    assert result.diagnostics["immediate_contact_free_roots"] > 0
    assert result.features[4, 1] == 0
    assert result.features[:, 1].sum() > 0


def test_native_bulk_capture_supplies_contact_geometry_to_planner():
    from test_state_capture import UpdatingGameMemory, reader_for
    from th06ai.combat import ENEMY_POOL
    from th06ai import native
    memory = UpdatingGameMemory(advancing=False)
    data = np.zeros(ENEMY_CAPACITY, BODY_DTYPE)
    data[0] = (220, 384, -128, 7, 0, 48, 24)
    offset = ENEMY_POOL-native.ACTORS_START
    memory.actors[offset:offset+data.nbytes] = data.tobytes()
    decoded = reader_for(memory).read()
    assert len(decoded.enemies) == 0 and len(decoded.enemy_bodies) == 1
    assert np.allclose(decoded.enemy_bodies[0, 3:5], [16, 8])
    result = RollingPlanner().plan(decoded, replace(decoded, tick=decoded.tick-2))
    assert result.features[4, 1] == 0
