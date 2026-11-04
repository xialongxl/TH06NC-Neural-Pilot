"""Bounded CUDA forward/backward checks; no game input or weight updates."""
from __future__ import annotations
from .i18n import msg

from .config import ROOT
from .files import atomic_json, timestamp


def check_cuda(lease, status):
    status(phase=msg('backend.gpu_environment_check'), message=msg('backend.checking_cuda_forward_and_backward_passes_without'))
    import time
    import torch
    from .policy import BulletEncoder
    from .observations import make_observation_space, OBSERVATION_VERSION
    lease()
    torch.set_num_threads(1)
    if not torch.cuda.is_available():
        raise RuntimeError(msg('backend.cuda_is_unavailable_cpu_execution_cannot_be'))
    torch.cuda.set_per_process_memory_fraction(.15, 0)
    space = make_observation_space()
    encoder = BulletEncoder(space).to("cuda:0")
    observation = {name: torch.zeros((1, *spec.shape), device="cuda:0") for name, spec in space.spaces.items()}
    with torch.no_grad():
        result = encoder(observation)
    torch.cuda.synchronize()
    lease()
    if result.shape != (1, 128) or not bool(torch.isfinite(result).all()):
        raise RuntimeError(msg('backend.the_cuda_forward_pass_failed'))
    # Match the real PPO minibatch size at maximum input capacity. A backward
    # pass tests activation memory and gradients, but there is no optimizer
    # step and neither a model checkpoint nor gameplay samples are produced.
    batch_size = 64
    observation = {name: torch.zeros((batch_size, *spec.shape), device="cuda:0") for name, spec in space.spaces.items()}
    for name in ("bullets", "full_bullets"):
        observation[name][..., -1] = 1
    encoder.zero_grad(set_to_none=True)
    lease()
    started = time.perf_counter()
    dense = encoder(observation)
    torch.cuda.synchronize()
    forward_ms = (time.perf_counter()-started)*1000
    if dense.shape != (batch_size, 128) or not bool(torch.isfinite(dense).all()):
        raise RuntimeError(msg('backend.the_full_capacity_cuda_batch_failed_its'))
    lease()
    started = time.perf_counter()
    dense.square().mean().backward()
    torch.cuda.synchronize()
    backward_ms = (time.perf_counter()-started)*1000
    lease()
    if not all(parameter.grad is not None and bool(torch.isfinite(parameter.grad).all()) for parameter in encoder.parameters()):
        raise RuntimeError(msg('backend.the_full_capacity_cuda_batch_failed_its_2'))
    from gymnasium import spaces
    from .policy import ResourcePolicy
    from .planner_spec import PLANNER_VERSION
    policy = ResourcePolicy(space, spaces.Discrete(36), lambda _: 3e-4,
                features_extractor_class=BulletEncoder, net_arch={"pi": [128, 128], "vf": [128, 128]}).to("cuda:0")
    observation["resources"][:, 1] = 3/8
    observation["planning"][:, :, :2] = 1
    observation["planning"][:, 4, 1] = 0
    with torch.no_grad():
        policy.action_net.bias[4] = policy.action_net.bias[22] = 100
        actions, values, log_probs = policy(observation)
        compared_values, compared_logs, _ = policy.evaluate_actions(observation, actions)
        distribution = policy.get_distribution(observation)
        if (not bool(torch.allclose(log_probs, compared_logs)) or not bool(torch.allclose(values, compared_values))
                or bool((distribution.distribution.probs[:, [4, 22]] > 0).any())):
            raise RuntimeError(msg('backend.cuda_action_probabilities_disagree_with_execution_or'))
    lease()
    info = {"time": timestamp(), "torch": torch.__version__, "cuda_runtime": torch.version.cuda,
            "device": torch.cuda.get_device_name(0), "forward_check": True,
            "backward_check": True, "batch_size": batch_size,
            "planner_version": PLANNER_VERSION, "planner_mask_check": True, "policy_logprob_check": True,
            "dense_forward_ms": round(forward_ms, 3), "dense_backward_ms": round(backward_ms, 3),
            "full_bullet_slots": space["full_bullets"].shape[0],
            "gameplay_trained": False, "allocated_bytes": torch.cuda.max_memory_allocated(0), "format": 1,
            "observation_version": OBSERVATION_VERSION}
    atomic_json(ROOT / "training_backend.json", info)
    status(phase=msg('backend.gpu_check_passed'), device=info["device"], message=msg('backend.full_capacity_cuda_forward_and_backward_passes'))
    return info
