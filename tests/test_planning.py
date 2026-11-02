"""Continuous collision, route search and true hybrid-policy regressions."""
from dataclasses import replace

import numpy as np
import pytest
import torch

from th06ai.native import Frame, GameResources
from th06ai.observations import ObservationBuilder, action_keys, make_observation_space
from th06ai.planning import RollingPlanner, swept_segments
from th06ai.planner_spec import HORIZON_TICKS, PlanningConfig
from th06ai import learning
from test_learning import LearningFixture


def frame(x=192., y=320., bullets=(), tick=102, items=(), enemies=()):
    bullets = np.asarray(bullets, np.float32).reshape(-1, 3)
    return Frame(tick, x, y, 0, False, False, bullets, (8, 16, 368, 416), True, 0.,
                 items=np.asarray(items, np.float32).reshape(-1, 6),
                 resources=GameResources(0, 2, 3, 4, 0),
                 enemies=np.asarray(enemies, np.float32).reshape(-1, 8),
                 bullet_sizes=np.full((len(bullets), 2), 4, np.float32), player_radius=2.)


def test_swept_collision_detects_crossing_even_when_both_endpoints_clear():
    end, clearance, hit, _ = swept_segments(np.array([[100, 100]]), np.array([[4, 0]]), 2, 0,
            np.array([0, 0]), np.array([384, 448]), np.array([[104, 100]]), np.zeros((1, 2)),
            np.array([.1]), np.zeros(1))
    assert np.linalg.norm(np.array([100, 100])-[104, 100]) > 2.1
    assert np.linalg.norm(end[0]-[104, 100]) > 2.1
    assert clearance[0] < 0
    assert hit[0] == pytest.approx((4-2.1)/4)


def test_wall_clamp_uses_piecewise_motion_instead_of_a_straight_chord():
    arguments = (np.array([[90, 0]]), np.array([[10, 10]]), 10, 0, np.array([0, 0]), np.array([100, 200]))
    end, clearance, hit, _ = swept_segments(*arguments, np.array([[95, 50]]), np.zeros((1, 2)), np.array([.1]), np.zeros(1))
    assert np.array_equal(end, [[100, 100]])
    assert clearance[0] == pytest.approx(5-2.1)
    assert np.isinf(hit[0])
    _, _, hit, _ = swept_segments(*arguments, np.array([[100, 50]]), np.zeros((1, 2)), np.array([.1]), np.zeros(1))
    assert hit[0] == pytest.approx((50-2.1)/10)


def test_near_collision_excludes_moving_into_bullet_and_searches_turning_routes():
    current = frame(bullets=[(0, 204, 320)])
    result = RollingPlanner().plan(current, replace(current, tick=100))
    assert result.features[4, 1] == 0  # Fast right crosses a known stationary bullet.
    assert result.features[3, 1] == 1
    assert result.paths.shape == (18, 6, 2)
    assert result.diagnostics["candidates_checked"] > 18
    assert not result.diagnostics["all_routes_risky"]
    assert result.diagnostics["horizon_ticks"] >= HORIZON_TICKS


def test_known_pending_movement_is_simulated_before_new_command():
    control = np.zeros(28)
    control[4] = control[19] = 1
    current = frame()
    result = RollingPlanner().plan(current, control=control)
    assert np.allclose(result.paths[:, 1], [200, 320])
    root_duration = result.diagnostics["path_ticks"][2]-result.diagnostics["path_ticks"][1]
    assert np.allclose(result.paths[3, 2], [200-4*root_duration, 320])


def test_plan_cannot_escape_collision_by_assuming_an_instant_next_turn():
    current = frame(bullets=[(0, 216, 320)])
    result = RollingPlanner().plan(current, replace(current, tick=100))
    root_duration = result.diagnostics["path_ticks"][2]-result.diagnostics["path_ticks"][1]
    assert root_duration >= 4
    assert result.features[4, 1] == 0


def test_future_turn_can_make_a_route_feasible_when_constant_motion_would_hit():
    current = frame(bullets=[(0, 228, 320)])
    result = RollingPlanner().plan(current, replace(current, tick=100))
    assert result.features[4, 1] == 1
    assert result.paths[4, 2, 0] > current.x
    # Full-horizon rightward motion crosses x=228. The searched route must
    # turn or stop, and its endpoint remains inside the real player bounds.
    assert not np.allclose(result.paths[4, -1], [current.x+4*(result.diagnostics["horizon_ticks"]-result.diagnostics["delay_ticks"]), current.y])
    assert np.all(result.paths >= [8, 16]) and np.all(result.paths <= [376, 432])


def test_dense_search_returns_complete_horizon_and_still_checks_all_roots():
    angle = np.arange(640)*2*np.pi/640
    bullets = np.column_stack((np.arange(640), 192+70*np.cos(angle), 320+70*np.sin(angle)))
    current = frame(bullets=bullets)
    result = RollingPlanner().plan(current, replace(current, tick=100))
    assert result.features.shape == (18, 8)
    assert result.paths.shape[0] == 18
    assert result.diagnostics["path_ticks"][-1] == result.diagnostics["horizon_ticks"]
    assert result.diagnostics["candidates_checked"] >= 18
    assert result.diagnostics["considered_bullets"] == 640


def test_corner_plan_excludes_futile_outward_push_and_prefers_leaving():
    result = RollingPlanner().plan(frame(x=8, y=432))
    assert result.features[3, 1] == result.features[2, 1] == result.features[7, 1] == 0
    chosen = result.diagnostics["suggested_movement"]
    assert "right" in action_keys(chosen) and "up" in action_keys(chosen)
    assert result.features[chosen, 7] == 0


def test_resource_and_boss_goals_change_the_selected_route_when_open():
    left = RollingPlanner().plan(frame(items=[(0, 152, 320, 0, 0, 0)]))
    right = RollingPlanner().plan(frame(items=[(0, 232, 320, 0, 0, 0)]))
    assert "left" in action_keys(left.diagnostics["suggested_movement"])
    assert "right" in action_keys(right.diagnostics["suggested_movement"])
    boss = RollingPlanner().plan(frame(x=280, enemies=[(0, 192, 80, 500, 1000, 1000, 0x11, 1)]))
    assert "left" in action_keys(boss.diagnostics["suggested_movement"])


def test_all_routes_risky_still_returns_finite_fallback_without_claiming_safety():
    current = frame(bullets=[(0, 192, 320)])
    current.bullet_sizes[:] = 256
    result = RollingPlanner().plan(current, replace(current, tick=100))
    assert result.diagnostics["all_routes_risky"]
    assert result.diagnostics["feasible_roots"] == 0
    assert result.features[:, 1].sum() >= 1
    assert np.isfinite(result.features).all()
    assert result.diagnostics["predicted_collision_tick"] is not None


def test_blank_opening_unknown_velocity_and_disabled_mode_do_not_stop_sampling():
    planner = RollingPlanner()
    for count in (0, 1, 640):
        bullets = [(i, 10+i % 32*11, 20+i//32*18) for i in range(count)]
        current = frame(bullets=bullets)
        observation = ObservationBuilder(planner=planner).build(current)
        assert make_observation_space().contains(observation)
        assert np.isfinite(observation["planning"]).all()
    inactive = ObservationBuilder(planner=RollingPlanner(PlanningConfig(enabled=False))).build(frame())
    assert not inactive["planning"].any()


def test_hybrid_distribution_is_used_for_sampling_logprob_update_and_play():
    torch.set_num_threads(1)
    model = learning.make_model(LearningFixture(), "cpu")
    current = frame(bullets=[(0, 204, 320)])
    builder = ObservationBuilder(planner=RollingPlanner())
    builder.build(replace(current, tick=100))
    observation = builder.build(current)
    tensor, _ = model.policy.obs_to_tensor(observation)
    with torch.no_grad():
        model.policy.action_net.bias[4] = model.policy.action_net.bias[22] = 100
        distribution = model.policy.get_distribution(tensor)
        assert distribution.distribution.probs[0, 4] == distribution.distribution.probs[0, 22] == 0
        action, value, log_prob = model.policy(tensor)
        evaluated_value, evaluated_log_prob, entropy = model.policy.evaluate_actions(tensor, action)
        assert torch.allclose(value, evaluated_value)
        assert torch.allclose(log_prob, evaluated_log_prob)
        assert torch.allclose(log_prob, distribution.log_prob(action))
        assert torch.allclose(entropy, distribution.entropy())
        greedy, _ = model.predict(observation, deterministic=True)
        assert greedy == int(distribution.distribution.probs.argmax(-1))
        for _ in range(100):
            sampled = distribution.get_actions()
            assert observation["planning"][int(sampled) % 18, 1] == 1
    model.policy.optimizer.zero_grad()
    _, probability, _, movement_entropy, bomb_kl, _ = model.policy.evaluate_with_exploration(tensor, action)
    (-probability.mean()+.02*(bomb_kl-movement_entropy).mean()).backward()
    assert all(parameter.grad is None or torch.isfinite(parameter.grad).all() for parameter in model.policy.parameters())


def test_planning_keeps_conditional_bomb_probability_and_empty_inventory_mask():
    torch.set_num_threads(1)
    model = learning.make_model(LearningFixture(), "cpu")
    observation = ObservationBuilder().build(frame())
    with torch.no_grad():
        neutral = model.policy.get_distribution(model.policy.obs_to_tensor(observation)[0]).distribution.probs
        observation["planning"][:, :2] = 1
        observation["planning"][:, 7] = np.linspace(-1, 0, 18)
        planned = model.policy.get_distribution(model.policy.obs_to_tensor(observation)[0]).distribution.probs
        q0 = neutral[:, 18:]/(neutral[:, :18]+neutral[:, 18:])
        q1 = planned[:, 18:]/(planned[:, :18]+planned[:, 18:])
        # A newly created planning encoder may change neural features; compare
        # the direct planning bias with the same latent neural logits.
        tensor, _ = model.policy.obs_to_tensor(observation)
        latent, _ = model.policy._latents(tensor)
        raw = model.policy.action_net(latent).softmax(-1)
        assert torch.allclose(q1, raw[:, 18:]/(raw[:, :18]+raw[:, 18:]), atol=1e-7)
        observation["resources"][1] = 0
        empty = model.policy.get_distribution(model.policy.obs_to_tensor(observation)[0]).distribution.probs
        assert empty[:, 18:].sum() == 0


def test_execution_log_uses_the_plan_that_chose_action_not_the_next_observation():
    from test_sampling_timing import game
    before = frame(bullets=[(0, 204, 320)])
    after = replace(before, tick=104, x=184)
    env, commands, _, _ = game(before, after)
    env.builder = ObservationBuilder(planner=RollingPlanner())
    env.builder.build(replace(before, tick=100))
    env.builder.build(before)
    decision = env.builder.last_plan
    _, _, _, _, info = env.step(3)
    assert commands == [{"z", "left"}]
    assert info["planning"]["chosen_predicted_path"] == decision.paths[3].tolist()
    assert info["planning"]["chosen_movement"] == 3
    assert info["planning_next"] is not info["planning"]


def test_actual_cpu_ppo_rollout_keeps_executed_actions_and_forecast_logprob(tmp_path, monkeypatch):
    class PlannedGame(LearningFixture):
        def __init__(self):
            super().__init__()
            self.builder = ObservationBuilder(planner=RollingPlanner(PlanningConfig(beam_width=1)))
            self.executed = []
        def reset(self, *, seed=None, options=None):
            self.current = frame(tick=0, bullets=[(0, 216, 320)])
            self.builder.reset()
            self.local_steps = 0
            return self.builder.build(self.current), {}
        def step(self, action):
            assert not self.paused
            self.executed.append(int(action))
            velocity = self.builder.planner.velocities[int(action) % 18]
            self.current = replace(self.current, tick=self.current.tick+2,
                    x=float(np.clip(self.current.x+velocity[0]*2, 8, 376)),
                    y=float(np.clip(self.current.y+velocity[1]*2, 16, 432)))
            self.builder.set_control(int(action))
            return self.builder.build(self.current), .01, False, False, {}
    torch.set_num_threads(1)
    game = PlannedGame()
    model = learning.make_model(game, "cpu")
    model.n_steps, model.batch_size, model.n_epochs = 16, 8, 1
    model.rollout_buffer = model.rollout_buffer_class(16, model.observation_space, model.action_space,
            device=model.device, gamma=model.gamma, gae_lambda=model.gae_lambda, n_envs=1)
    checked = []
    class Verify(learning.TrainingCallback):
        def _on_rollout_end(self):
            buffer = self.model.rollout_buffer
            observations = {name: value[:, 0] for name, value in buffer.observations.items()}
            tensor, _ = self.model.policy.obs_to_tensor(observations)
            actions = torch.from_numpy(buffer.actions[:, 0, 0].astype(np.int64))
            with torch.no_grad():
                _, logs, _ = self.model.policy.evaluate_actions(tensor, actions)
            assert np.array_equal(actions.numpy(), game.executed)
            assert np.allclose(logs.numpy(), buffer.log_probs[:, 0], atol=1e-5)
            assert all(observations["planning"][i, int(action) % 18, 1] == 1 for i, action in enumerate(actions))
            checked.append(True)
            super()._on_rollout_end()
    monkeypatch.setattr(learning, "save_model", lambda *args: None)
    model.learn(16, callback=Verify(game, tmp_path, 1, lambda **_: None, lambda: None))
    assert checked == [True] and model._n_updates == 1
