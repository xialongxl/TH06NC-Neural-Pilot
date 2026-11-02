"""Counterexamples to reward farming; trajectories, not real gameplay claims."""
from dataclasses import replace
import hashlib

import numpy as np
import pytest
import torch

from th06ai import learning
from th06ai.files import atomic_json, read_json
from th06ai.native import GameResources, SHA256
from th06ai.policy_spec import POLICY_VERSION
from th06ai.rewards import DISCOUNT, DEATH_COST, RESOURCE_BUDGET, RESOURCE_COMPONENTS, RewardTracker
from test_rewards import frame
from test_learning import LearningFixture


def trajectory(position, seconds=6, *, die_at_end=False):
    tracker = RewardTracker()
    x, y = position(0)
    tracker.reset(frame(x=x, y=y))
    returns = 0.0
    discounted = 0.0
    for step in range(1, int(seconds * 30) + 1):
        x, y = position(step)
        died = die_at_end and step == int(seconds * 30)
        result = tracker.step(frame(tick=100+step*2, x=x, y=y), 0, died)
        returns += result.total
        discounted += DISCOUNT**(step-1) * result.total
    return returns, discounted, tracker.diagnostics()


def test_idle_or_small_jitter_a_few_pixels_off_corner_cannot_farm_positive_time_reward():
    for position in (lambda _: (8, 432), lambda i: (16+(i % 2)*8, 416+(i % 3)*4)):
        total, discounted, diagnostics = trajectory(position)
        assert total < 0 and discounted < 0
        assert diagnostics["loiter_seconds"] > 2
        assert diagnostics["reward_edge_loiter"] < -diagnostics["reward_survival"]


def test_brief_corner_dodge_safe_open_hold_and_leaving_corner_remain_possible():
    brief, _, diagnostics = trajectory(lambda _: (8, 432), seconds=1)
    assert brief > 0 and diagnostics["reward_edge_loiter"] == 0
    safe, _, safe_diag = trajectory(lambda _: (192, 224))
    moving, _, moving_diag = trajectory(lambda i: (8, 432-i))
    _, _, idle_diag = trajectory(lambda _: (8, 432))
    assert safe > 0 and moving > 0
    assert safe_diag["loiter_seconds"] < .001
    assert moving_diag["loiter_seconds"] < idle_diag["loiter_seconds"] * .1


def test_dying_after_thirteen_seconds_no_longer_earns_net_time_profit():
    total, discounted, diagnostics = trajectory(lambda _: (192, 224), seconds=13, die_at_end=True)
    assert total < 0 and discounted < 0
    assert diagnostics["reward_survival"] < DEATH_COST / 10


def item_frame(tick=100, x=120, *, collected=False, disappeared=False):
    resources = GameResources(0, 1 if collected else 0, 3, 2, 0)
    items = [] if collected or disappeared else [(1, 192, 384, 0, 0, 0)]
    return replace(frame(tick=tick, x=x), resources=resources,
                   items=np.array(items, np.float32).reshape(-1, 6))


def test_approaching_visible_item_then_actual_collection_is_better_than_ignoring_it():
    toward, away = RewardTracker(), RewardTracker()
    toward.reset(item_frame())
    away.reset(item_frame())
    approach = toward.step(item_frame(tick=102, x=128), 4, False)
    retreat = away.step(item_frame(tick=102, x=112), 3, False)
    assert approach.components["item_shaping"] > retreat.components["item_shaping"]
    pickup = toward.step(item_frame(tick=104, x=192, collected=True), 4, False)
    missed = away.step(item_frame(tick=104, x=112, disappeared=True), 0, False)
    assert pickup.components["power"] > 0 and pickup.total > 0
    assert all(missed.components[name] == 0 for name in RESOURCE_COMPONENTS)
    assert missed.components["item_shaping"] <= 0
    assert approach.total + DISCOUNT*pickup.total > retreat.total + DISCOUNT*missed.total


def test_item_approach_retreat_loop_does_not_farm_discounted_potential():
    scores = []
    for excursion in (0, 8):
        tracker = RewardTracker()
        tracker.reset(item_frame())
        a = tracker.step(item_frame(tick=102, x=120+excursion), 4 if excursion else 0, False)
        b = tracker.step(item_frame(tick=104), 3 if excursion else 0, False)
        scores.append(a.components["item_shaping"] + DISCOUNT*b.components["item_shaping"])
    assert scores[0] == pytest.approx(scores[1], abs=1e-12)


def test_alternating_blocked_request_and_neutral_does_not_erase_blockage_grace():
    tracker = RewardTracker()
    tracker.reset(frame(x=8))
    for i in range(1, 61):
        tracker.step(frame(tick=100+2*i, x=8), 3 if i % 2 else 0, False)
    assert tracker.diagnostics()["reward_wall_push"] < 0


def test_pause_rebase_does_not_erase_corner_dwell_history():
    tracker = RewardTracker()
    tracker.reset(frame(x=8, y=432))
    for i in range(1, 101):
        tracker.step(frame(tick=100+2*i, x=8, y=432), 0, False)
    assert tracker.diagnostics()["loiter_seconds"] > 0
    tracker.rebase(frame(tick=302, x=8, y=432))
    result = tracker.step(frame(tick=304, x=8, y=432), 0, False)
    assert result.components["edge_loiter"] < 0


def test_resource_credit_diminishes_but_does_not_switch_off_at_old_hard_budget():
    tracker = RewardTracker()
    original = replace(frame(), resources=GameResources(0, 0, 3, 2, 0))
    tracker.reset(original)
    rich = replace(original, tick=102, resources=GameResources(10000000, 0, 3, 2, 200))
    tracker.step(rich, 0, False)
    # Two hundred actual point-item gains exceed the old hard cap of six.
    pickup = replace(rich, tick=104, resources=replace(rich.resources, power=1))
    result = tracker.step(pickup, 0, False)
    assert result.components["power"] > 0
    assert tracker.resource_bonus_used <= RESOURCE_BUDGET


def test_reward_upgrade_keeps_all_trained_weights_and_counts_and_changes_future_discount(tmp_path, monkeypatch):
    torch.set_num_threads(1)
    game = LearningFixture()
    model = learning.make_model(game, "cpu")
    model.gamma = model.rollout_buffer.gamma = .995
    model.learn(512)
    path = tmp_path / "models" / "old-objective.zip"
    path.parent.mkdir(parents=True)
    model.save(path)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    atomic_json(path.with_suffix(".json"), {"format": 2, "policy_version": POLICY_VERSION,
                "game_sha256": SHA256, "model_sha256": digest, "reward_version": 3,
                "steps": model.num_timesteps, "gradient_updates": model._n_updates})
    monkeypatch.setattr(learning, "ROOT", tmp_path)
    loaded = learning.load_model(path, game, "cpu")
    assert loaded.gamma == DISCOUNT and loaded.rollout_buffer.gamma == DISCOUNT
    assert loaded.num_timesteps == 512 and loaded._n_updates == model._n_updates
    for name, value in model.policy.state_dict().items():
        assert torch.equal(value, loaded.policy.state_dict()[name])
    assert read_json(path.with_suffix(".json"))["reward_version"] == 3
    assert hashlib.sha256(path.read_bytes()).hexdigest() == digest
