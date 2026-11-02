"""SB3 2.9.2 PPO update with a separate, explicit resource exploration term.

Adapted from stable-baselines3/ppo/ppo.py (MIT). Clipping, GAE targets,
value loss, optimizer and KL stopping retain SB3's behavior. Only the
exploration objective changes; true joint entropy is still logged.
"""
import numpy as np
import torch
from torch.nn import functional as F
from stable_baselines3.common.utils import explained_variance


def update_resource_policy(model):
    model.policy.set_training_mode(True)
    model._update_learning_rate(model.policy.optimizer)
    clip_range = model.clip_range(model._current_progress_remaining)
    clip_range_vf = model.clip_range_vf(model._current_progress_remaining) if model.clip_range_vf else None
    logs = {key: [] for key in ("policy_gradient_loss", "value_loss", "clip_fraction", "joint_entropy",
                                "movement_entropy", "bomb_prior_kl", "bomb_probability", "exploration_loss", "approx_kl")}
    continue_training = True
    for epoch in range(model.n_epochs):
        for data in model.rollout_buffer.get(model.batch_size):
            actions = data.actions.long().flatten()
            values, log_prob, joint_entropy, movement_entropy, bomb_kl, bomb_probability = model.policy.evaluate_with_exploration(data.observations, actions)
            values = values.flatten()
            advantages = data.advantages
            if model.normalize_advantage and len(advantages) > 1:
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
            ratio = torch.exp(log_prob - data.old_log_prob)
            policy_loss = -torch.min(advantages * ratio, advantages * torch.clamp(ratio, 1 - clip_range, 1 + clip_range)).mean()
            values_pred = values if clip_range_vf is None else data.old_values + torch.clamp(values - data.old_values, -clip_range_vf, clip_range_vf)
            value_loss = F.mse_loss(data.returns, values_pred)
            # The resource prior is a soft loss, not an action override.
            exploration_loss = (bomb_kl - movement_entropy).mean()
            loss = policy_loss + model.ent_coef * exploration_loss + model.vf_coef * value_loss
            with torch.no_grad():
                log_ratio = log_prob - data.old_log_prob
                approx_kl = ((torch.exp(log_ratio) - 1) - log_ratio).mean().item()
                row = {"policy_gradient_loss": policy_loss.item(), "value_loss": value_loss.item(),
                       "clip_fraction": (torch.abs(ratio - 1) > clip_range).float().mean().item(),
                       "joint_entropy": joint_entropy.mean().item(), "movement_entropy": movement_entropy.mean().item(),
                       "bomb_prior_kl": bomb_kl.mean().item(), "bomb_probability": bomb_probability.mean().item(),
                       "exploration_loss": exploration_loss.item(), "approx_kl": approx_kl}
                for name, value in row.items():
                    logs[name].append(value)
            if model.target_kl is not None and approx_kl > 1.5 * model.target_kl:
                continue_training = False
                break
            if model.training_control is not None:
                model.training_control.lease()
            model.policy.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.policy.parameters(), model.max_grad_norm)
            model.policy.optimizer.step()
        model._n_updates += 1
        if not continue_training:
            break
    means = {name: float(np.mean(values)) for name, values in logs.items()}
    model.last_policy_diagnostics = means
    for name, value in means.items():
        model.logger.record("train/" + name, value)
    model.logger.record("train/entropy_loss", -means["joint_entropy"])
    model.logger.record("train/loss", loss.item())
    model.logger.record("train/explained_variance", explained_variance(model.rollout_buffer.values.flatten(), model.rollout_buffer.returns.flatten()))
    model.logger.record("train/n_updates", model._n_updates, exclude="tensorboard")
    model.logger.record("train/clip_range", clip_range)
    if clip_range_vf is not None:
        model.logger.record("train/clip_range_vf", clip_range_vf)
