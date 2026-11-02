"""Real CPU PPO + fake game; no game input, GPU work or public run pointer."""
import threading
import time

import pytest
import torch

from th06ai import learning
from th06ai.control import FinishTraining, TrainingControl, set_paused
from test_learning import LearningFixture


def test_user_pause_keeps_partial_rollout_parameters_optimizer_and_episode(tmp_path, monkeypatch):
    torch.set_num_threads(1)
    game = LearningFixture()
    model = learning.make_model(game, "cpu")
    # This checks pause/partial-buffer semantics, not CPU throughput for the
    # production 512-step, four-epoch GPU workload. Use a small real rollout
    # so reaching the pause does not depend on machine speed.
    model.n_steps, model.batch_size, model.n_epochs = 64, 16, 1
    model.rollout_buffer = model.rollout_buffer_class(model.n_steps, model.observation_space,
            model.action_space, device=model.device, gamma=model.gamma, gae_lambda=model.gae_lambda,
            n_envs=model.n_envs, **model.rollout_buffer_kwargs)
    paused = threading.Event()
    records, errors = [], []
    def status(**values):
        records.append(values)
        if values.get("steps") == 72:
            set_paused(tmp_path, True)
        if values.get("user_paused"):
            paused.set()
    control = TrainingControl(tmp_path, lambda: None, status)
    model.training_control, model.training_game = control, game
    monkeypatch.setattr(learning, "save_model", lambda *args: None)
    callback = learning.TrainingCallback(game, tmp_path, 1, status, lambda: None, control)
    def learn():
        try:
            model.learn(total_timesteps=128, callback=callback)
        except BaseException as exc:
            errors.append(exc)
    thread = threading.Thread(target=learn, daemon=True)
    thread.start()
    try:
        assert paused.wait(10), errors
        assert game.paused and model.num_timesteps == 72
        assert model.rollout_buffer.pos == 8 and model._n_updates == 1
        assert model.policy.optimizer.state
        parameters = [p.detach().clone() for p in model.policy.parameters()]
        optimizer_step = next(iter(model.policy.optimizer.state.values()))["step"].clone()
        local_steps, tick, active = game.local_steps, game.current.tick, control.active_seconds
        time.sleep(.2)
        assert model.num_timesteps == 72 and model.rollout_buffer.pos == 8
        assert game.local_steps == local_steps and game.current.tick == tick
        assert control.active_seconds == pytest.approx(active, abs=.01)
        assert all(torch.equal(a, b) for a, b in zip(parameters, model.policy.parameters()))
        assert torch.equal(next(iter(model.policy.optimizer.state.values()))["step"], optimizer_step)
    finally:
        set_paused(tmp_path, False)
        thread.join(10)
    assert not thread.is_alive() and not errors
    assert model.num_timesteps == 128 and model._n_updates == 2
    assert control.paused_seconds >= .2
    assert callback.base_steps == 0
    assert records[-1].get("checkpoint") or (tmp_path / "training_summary.json").is_file()


def test_save_finish_from_pause_is_distinct_from_emergency_stop(tmp_path):
    game = LearningFixture()
    control = TrainingControl(tmp_path, lambda: None, lambda **_: None)
    control.start()
    set_paused(tmp_path, True)
    (tmp_path / "FINISH").touch()
    with pytest.raises(FinishTraining):
        control.gate(game)
    def emergency():
        raise RuntimeError("safety lease revoked")
    control.lease = emergency
    with pytest.raises(RuntimeError, match="safety lease revoked"):
        control.gate(game)


def test_paused_wait_keeps_checking_safety_lease(tmp_path):
    game = LearningFixture()
    calls = []
    def lease():
        calls.append(True)
        if len(calls) == 4:
            raise RuntimeError("GPU fault")
    control = TrainingControl(tmp_path, lease, lambda **_: None, sleep=lambda _: None)
    control.start()
    set_paused(tmp_path, True)
    with pytest.raises(RuntimeError, match="GPU fault"):
        control.gate(game)
    assert game.paused and len(calls) == 4


def test_save_and_finish_preserves_real_updated_model_from_paused_training(tmp_path, monkeypatch):
    from th06ai.config import Config
    torch.set_num_threads(1)
    game = LearningFixture()
    monkeypatch.setattr(learning, "ROOT", tmp_path)
    def status(**values):
        if values.get("steps") == 520:
            set_paused(tmp_path, True)
        if values.get("user_paused"):
            (tmp_path / "FINISH").touch()
    model = learning.train(Config(game_exe="unused", device="cpu"), game, tmp_path, 1,
                           status, lambda: None, resume=None)
    assert model.num_timesteps == 520 and model._n_updates == 4
    assert game.paused
    restored = learning.load_model(tmp_path / "models" / "latest.zip", game, "cpu")
    assert restored.num_timesteps == 520 and restored._n_updates == 4
    assert all(torch.equal(a, b) for a, b in zip(model.policy.parameters(), restored.policy.parameters()))
