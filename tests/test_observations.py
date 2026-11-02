import time
from dataclasses import replace

import numpy as np

from th06ai.native import Frame, GameResources
from th06ai.observations import ACTION_COUNT, ObservationBuilder, action_keys, make_observation_space


def frame(bullets=(), tick=100):
    return Frame(tick, 192, 384, 1, False, False, np.array(bullets, np.float32).reshape((-1, 3)),
                 (8, 16, 368, 416), True, time.monotonic())


def test_empty_observation_has_no_nan_and_correct_mask():
    obs = ObservationBuilder().build(frame())
    assert obs["bullets"].shape == (64, 5)
    assert np.isfinite(obs["bullets"]).all()
    assert obs["bullets"].sum() == 0
    assert obs["player"][0] == .5


def test_velocity_is_inferred_from_visible_history_only():
    builder = ObservationBuilder()
    builder.build(frame([(7, 190, 370)]))
    observed = builder.build(frame([(7, 192, 374), (8, 180, 380)], tick=102))["bullets"]
    matched = next(row for row in observed if row[4] and np.isclose(row[0], 0))
    assert np.allclose(matched[2:4], [1/16, 2/16])
    unseen = next(row for row in observed if row[4] and row[0] < 0)
    assert np.allclose(unseen[2:4], [0, 0])


def test_history_is_reset_between_episodes_and_recycled_slots_do_not_teleport():
    builder = ObservationBuilder()
    builder.build(frame([(7, 190, 370)]))
    jumped = builder.build(frame([(7, 10, 20)], tick=102))
    assert np.allclose(jumped["bullets"][0, 2:4], 0)
    builder.reset()
    assert np.allclose(builder.build(frame([(7, 11, 21)], tick=104))["bullets"][0, 2:4], 0)


def test_all_actions_have_valid_directions_focus_and_independent_bomb_choice():
    for action in range(ACTION_COUNT):
        keys = action_keys(action)
        assert "z" in keys
        assert not {"up", "down"} <= keys
        assert not {"left", "right"} <= keys
        assert keys <= {"z", "x", "shift", "up", "down", "left", "right"}
        assert ("shift" in keys) == bool((action // 9) % 2)
        assert ("x" in keys) == (action >= 18)


def test_items_and_current_inventory_are_visible_to_the_model():
    current = frame()
    current.items = np.array([(2, 200, 350, 3, 0, 3), (3, 180, 360, 0, 1, 2)], np.float32)
    current.resources = GameResources(10000, 64, 3, 2, 5)
    observation = ObservationBuilder().build(current)
    assert make_observation_space().contains(observation)
    assert observation["items"][:, -1].sum() == 2
    assert observation["items"][:, 4+3].sum() == 1
    assert observation["items"][:, 4].sum() == 1
    assert np.allclose(observation["resources"][:3], [.5, 3/8, 2/8])
