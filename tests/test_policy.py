import gymnasium as gym
import numpy as np
import torch

from th06ai.policy import BulletEncoder
from th06ai.observations import make_observation_space

torch.set_num_threads(1)


def make_encoder():
    return BulletEncoder(make_observation_space())


def observation(points, player):
    count = points.shape[0]
    return {"bullets": points, "player": player, "items": torch.zeros(count, 32, 12),
            "resources": torch.zeros(count, 5), "enemies": torch.zeros(count, 32, 8),
            "control": torch.zeros(count, 28), "full_bullets": torch.zeros(count, 640, 10),
            "planning": torch.zeros(count, 18, 8), 'stage': torch.zeros(count, 8),
            'navigation':torch.zeros(count,18,9),'decision':torch.zeros(count,8)}


def test_padding_does_not_generate_nan_or_affect_output():
    encoder = make_encoder()
    points = torch.zeros(2, 64, 5)
    points[1, :, :4] = 1.8  # invalid padding must remain masked
    obs = observation(points, torch.zeros(2, 4))
    obs["items"][1, :, :-1] = 1.8
    result = encoder(obs)
    assert torch.isfinite(result).all()
    assert torch.allclose(result[0], result[1])


def test_bullet_order_does_not_change_decision_features_and_gradients_work():
    encoder = make_encoder()
    points = torch.rand(1, 64, 5)
    points[:, :, 4] = 1
    player = torch.rand(1, 4)
    obs = observation(points, player)
    obs["items"] = torch.rand(1, 32, 12)
    obs["items"][:, :, -1] = 1
    a = encoder(obs)
    b = encoder({**obs, "bullets": points[:, torch.randperm(64)],
                 "items": obs["items"][:, torch.randperm(32)]})
    assert torch.allclose(a, b, atol=1e-6)
    a.square().mean().backward()
    assert any(p.grad is not None and torch.isfinite(p.grad).all() and bool(p.grad.abs().sum() > 0)
               for p in encoder.parameters())
