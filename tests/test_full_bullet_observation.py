"""Near/full-field coverage and learnable threat inputs, without game access."""
from dataclasses import replace

import numpy as np
import pytest
import torch

from th06ai.native import CAPACITY, Frame, GameResources
from th06ai.observations import ObservationBuilder, make_observation_space
from th06ai.policy import BulletEncoder
from th06ai.risk import collision_risk


def scene(bullets, tick=102, sizes=None):
    bullets = np.asarray(bullets, np.float32).reshape(-1, 3)
    return Frame(tick, 192., 320., 0, False, False, bullets, (8, 16, 368, 416), True, 0.,
                 resources=GameResources(0, 2, 3, 4, 0),
                 bullet_sizes=np.full((len(bullets), 2), 4, np.float32) if sizes is None else sizes,
                 player_radius=2.)


def incoming_scene(side):
    angles = np.arange(64)*2*np.pi/64
    unit = np.column_stack((np.cos(angles), np.sin(angles)))
    near = [192., 320.] + unit*(40+np.arange(64) % 5)[:, None]
    now = np.vstack((np.column_stack((np.arange(64), near)), [64, 192+side*130, 320])).astype(np.float32)
    velocity = np.vstack((unit, [-side*12, 0])).astype(np.float32)
    before = now.copy()
    before[:, 1:3] -= 2*velocity
    return scene(before, tick=100), scene(now)


def observed(before, after):
    builder = ObservationBuilder()
    builder.build(before)
    return builder.build(after)


@pytest.mark.parametrize("count", [0, 1, 63, 64, 65, 128, 512, 640])
def test_every_visible_pool_bullet_reaches_network_with_near_branch_unchanged(count):
    indices = np.arange(count)
    current = np.column_stack((indices, 12+indices % 32*11, 24+indices//32*20)).astype(np.float32)
    previous = current.copy()
    previous[:, 1:3] -= [2, 4]
    observation = observed(scene(previous, tick=100), scene(current))
    assert observation["full_bullets"].shape == (CAPACITY, 10)
    assert observation["full_bullets"][:, -1].sum() == count
    assert observation["bullets"][:, -1].sum() == min(count, 64)
    assert make_observation_space().contains(observation)
    full = observation["full_bullets"][:count]
    assert np.allclose(full[:, :2], (current[:, 1:3]-[192, 320])/[384, 448])
    assert np.allclose(full[:, 2:4], [1/16, 2/16])
    assert np.all(full[:, [4, 6, 9]] == 1)
    assert np.allclose(full[:, 5], 4/128)
    assert not observation["full_bullets"][count:].any()
    nearest = np.argsort(np.sum((current[:, 1:3]-[192, 320])**2, axis=1), kind="stable")[:64]
    assert np.allclose(observation["bullets"][:len(nearest), :2], full[nearest, :2])
    assert np.allclose(observation["bullets"][:len(nearest), 2:4], full[nearest, 2:4])


def test_far_incoming_65th_bullet_breaks_previously_identical_inputs():
    before_left, after_left = incoming_scene(-1)
    before_right, after_right = incoming_scene(1)
    left, right = observed(before_left, after_left), observed(before_right, after_right)
    for name in left:
        if name != "full_bullets":
            assert np.array_equal(left[name], right[name])
    assert not np.array_equal(left["full_bullets"], right["full_bullets"])
    for side, observation, before, after in ((-1, left, before_left, after_left), (1, right, before_right, after_right)):
        incoming = observation["full_bullets"][64]
        assert incoming[0] == pytest.approx(side*130/384)
        assert incoming[2] == pytest.approx(-side*12/16)
        assert incoming[4] == incoming[6] == incoming[9] == 1
        assert incoming[7] == pytest.approx((130/12)/24)
        assert incoming[8] == pytest.approx(-4/512)
        assert collision_risk(after, before).threat > .6


def test_unknown_motion_and_geometry_are_distinguished_from_known_stationary():
    frame = scene([[5, 192, 300]])
    unknown = ObservationBuilder().build(frame)["full_bullets"][0]
    stationary = observed(replace(frame, tick=100), frame)["full_bullets"][0]
    assert not unknown[2:5].any()
    assert not stationary[2:4].any() and stationary[4] == 1
    invalid_geometry = replace(frame, bullet_sizes=np.array([[np.nan, 0]], np.float32))
    missing = ObservationBuilder().build(invalid_geometry)["full_bullets"][0]
    assert missing[6] == 0 and missing[5] == pytest.approx(16/128)
    assert np.isfinite(missing).all()
    delayed = observed(replace(frame, tick=50), frame)["full_bullets"][0]
    assert delayed[4] == 0 and delayed[7] == 0


def test_global_branch_is_permutation_and_padding_invariant_and_receives_gradients():
    torch.set_num_threads(1)
    torch.manual_seed(17)
    observation = observed(*incoming_scene(1))
    tensors = {name: torch.from_numpy(values[None].copy()) for name, values in observation.items()}
    tensors["full_bullets"].requires_grad_(True)
    encoder = BulletEncoder(make_observation_space())
    output = encoder(tensors)
    reordered = {**tensors, "full_bullets": tensors["full_bullets"][:, torch.randperm(CAPACITY)]}
    assert torch.allclose(output, encoder(reordered), atol=1e-6)
    padded = tensors["full_bullets"].detach().clone()
    padded[:, 65:, :-1] = 1.7
    assert torch.allclose(output, encoder({**tensors, "full_bullets": padded}), atol=1e-6)
    output.square().mean().backward()
    gradient = tensors["full_bullets"].grad
    assert torch.isfinite(gradient).all()
    assert gradient[0, 64, :-1].abs().sum() > 0
    assert not gradient[0, 65:, :-1].any()
    assert encoder.full_bullet[0].weight.grad.abs().sum() > 0


def test_dense_full_batch_forward_and_backward_are_finite():
    torch.set_num_threads(1)
    space = make_observation_space()
    encoder = BulletEncoder(space)
    observation = {name: torch.zeros(8, *spec.shape) for name, spec in space.spaces.items()}
    generator = torch.Generator().manual_seed(17)
    observation["full_bullets"] = torch.rand(8, CAPACITY, 10, generator=generator)*2-1
    observation["full_bullets"][:, :, -1] = 1
    observation["bullets"][:, :, -1] = 1
    result = encoder(observation)
    assert result.shape == (8, 128) and torch.isfinite(result).all()
    result.square().mean().backward()
    assert all(parameter.grad is None or torch.isfinite(parameter.grad).all() for parameter in encoder.parameters())


def test_native_pool_decoder_environment_and_status_preserve_all_640_bullets():
    from test_state_capture import UpdatingGameMemory, reader_for
    from test_sampling_timing import game
    from th06ai import native
    memory = UpdatingGameMemory(advancing=False)
    for index in range(CAPACITY):
        address = native.BULLET_MANAGER+8+index*native.STRIDE
        memory.put(address+0x30, "ff", 12+index % 32*11, 24+index//32*20)
        memory.put(address+0x44, "H", 1)
        memory.put(address+0x5F4, "ff", 4, 8)
    decoded = reader_for(memory).read()
    assert len(decoded.bullets) == CAPACITY
    env, _, _, statuses = game(decoded, replace(decoded, tick=decoded.tick+2))
    observation, _, _, _, info = env.step(0)
    expected = {"visible": 640, "near": 64, "full": 640, "motion_known": 640, "geometry_known": 640}
    assert info["bullet_observation"] == expected
    assert statuses[-1]["bullet_observation"] == expected
    assert observation["full_bullets"][:, -1].sum() == CAPACITY
