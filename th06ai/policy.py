"""Learned action logits plus a cached online planner, one true distribution."""
import torch
import math
from torch import nn
from stable_baselines3.common.policies import BaseModel, MultiInputActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from .observations import ITEM_FEATURES, RESOURCE_FEATURES, ENEMY_FEATURES, CONTROL_FEATURES, FULL_BULLET_FEATURES
from .policy_spec import POLICY_VERSION, BOMB_PRIOR
from .planner_spec import PLAN_MOVEMENTS, PLAN_FEATURES, PLAN_LOGIT_SCALE
from .stage import STAGE_FEATURES
from .navigation import NAV_FEATURES, DECISION_FEATURES, RESCUE_BOMB_PRIOR
from .game_settings import CONTEXT_FEATURES

class BulletEncoder(BaseFeaturesExtractor):
    """Shared point encoder: insensitive to the ordering of visible bullets."""
    def __init__(self, observation_space):
        super().__init__(observation_space, features_dim=128)
        self.bullet = nn.Sequential(nn.Linear(5, 64), nn.ReLU(), nn.Linear(64, 64), nn.ReLU())
        self.has_items = "items" in observation_space.spaces
        self.has_enemies = "enemies" in observation_space.spaces
        self.has_control = "control" in observation_space.spaces
        self.has_full_bullets = "full_bullets" in observation_space.spaces
        self.has_planning = "planning" in observation_space.spaces
        self.has_stage = 'stage' in observation_space.spaces
        self.has_navigation = 'navigation' in observation_space.spaces
        self.has_game_context = 'game_context' in observation_space.spaces
        if self.has_items:
            self.item = nn.Sequential(nn.Linear(ITEM_FEATURES, 32), nn.ReLU(), nn.Linear(32, 32), nn.ReLU())
        if self.has_enemies:
            self.enemy = nn.Sequential(nn.Linear(ENEMY_FEATURES, 32), nn.ReLU(), nn.Linear(32, 32), nn.ReLU())
        if self.has_full_bullets:
            self.full_bullet = nn.Sequential(nn.Linear(FULL_BULLET_FEATURES, 64), nn.ReLU(), nn.Linear(64, 64), nn.ReLU())
        self.fusion = nn.Sequential(nn.Linear(128 + 4 + (64 + RESOURCE_FEATURES if self.has_items else 0)
                                              + (64 if self.has_enemies else 0)
                                              + (CONTROL_FEATURES if self.has_control else 0)
                                              + (128 if self.has_full_bullets else 0)
                                              + (PLAN_MOVEMENTS*PLAN_FEATURES if self.has_planning else 0)
                                              + (STAGE_FEATURES if self.has_stage else 0)
                                              + (18*NAV_FEATURES+DECISION_FEATURES if self.has_navigation else 0)
                                              + (CONTEXT_FEATURES if self.has_game_context else 0), 128), nn.ReLU())

    @staticmethod
    def pool(points, encoder):
        mask = points[..., -1:]
        encoded = encoder(points)
        mean = (encoded * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        maximum = encoded.masked_fill(mask == 0, -1e9).max(dim=1).values
        maximum = torch.where(mask.sum(dim=1) > 0, maximum, torch.zeros_like(maximum))
        return torch.cat([mean, maximum], dim=1)

    def forward(self, observations):
        if not self.has_items:
            return self.fusion(torch.cat([self.pool(observations["bullets"], self.bullet),
                                          observations["player"]], dim=1))
        features = [self.pool(observations["bullets"], self.bullet),
                                      self.pool(observations["items"], self.item),
                                      observations["player"], observations["resources"]]
        if self.has_enemies:
            features.append(self.pool(observations["enemies"], self.enemy))
        if self.has_control:
            features.append(observations["control"])
        if self.has_full_bullets:
            features.append(self.pool(observations["full_bullets"], self.full_bullet))
        if self.has_planning:
            features.append(observations["planning"].flatten(start_dim=1))
        if self.has_stage:
            features.append(observations['stage'])
        if self.has_navigation:
            features.extend((observations['navigation'].flatten(start_dim=1),observations['decision']))
        if self.has_game_context:
            features.append(observations['game_context'])
        return self.fusion(torch.cat(features, dim=1))


def exploration_terms(probabilities, bombs_available, prior=None):
    """Separate movement entropy from a conditional Bernoulli bomb prior.

    Joint Shannon entropy would reward a 50/50 bomb decision. Instead use
    H(movement) - E_m KL(P(bomb|m) || Bernoulli(prior)). The PPO advantage
    can still learn to outweigh this soft regularizer in useful states.
    """
    if probabilities.shape[-1] != 36:
        raise ValueError("炸弹策略必须保留 36 类动作")
    movement = probabilities[..., :18] + probabilities[..., 18:]
    safe_movement = movement.clamp_min(torch.finfo(movement.dtype).tiny)
    entropy = -(movement * safe_movement.log()).sum(-1)
    q = probabilities[..., 18:] / safe_movement
    safe_q = q.clamp_min(torch.finfo(q.dtype).tiny)
    safe_not_q = (1 - q).clamp_min(torch.finfo(q.dtype).tiny)
    prior = torch.full_like(bombs_available,BOMB_PRIOR,dtype=probabilities.dtype) if prior is None else prior
    kl = q * (safe_q.log() - prior[...,None].log())
    kl += (1 - q) * (safe_not_q.log() - torch.log1p(-prior[...,None]))
    bomb_kl = (movement * kl).sum(-1)
    bomb_kl = torch.where(bombs_available, bomb_kl, torch.zeros_like(bomb_kl))
    return entropy, bomb_kl


class ResourcePolicy(MultiInputActorCriticPolicy):
    """Same 36 actions; planning changes the distribution before sampling."""
    def _build(self, lr_schedule):
        super()._build(lr_schedule)
        if self.action_space.n != 36 or "resources" not in self.observation_space.spaces:
            raise ValueError("资源策略需要资源状态和 36 类动作")
        self.reset_bomb_prior()

    def reset_bomb_prior(self):
        """Retain the non-bomb movement head, reinitialize only bomb rows."""
        with torch.no_grad():
            self.action_net.weight[18:].copy_(self.action_net.weight[:18])
            self.action_net.bias[18:].copy_(self.action_net.bias[:18] + math.log(BOMB_PRIOR / (1 - BOMB_PRIOR)))
        # Old momentum must not push the repaired bomb rows back immediately.
        for parameter in (self.action_net.weight, self.action_net.bias):
            for value in self.optimizer.state.get(parameter, {}).values():
                if isinstance(value, torch.Tensor) and value.shape == parameter.shape:
                    value[18:].zero_()

    def _latents(self, obs):
        features = self.extract_features(obs)
        if self.share_features_extractor:
            return self.mlp_extractor(features)
        pi_features, vf_features = features
        return self.mlp_extractor.forward_actor(pi_features), self.mlp_extractor.forward_critic(vf_features)

    def _bomb_context(self, obs):
        available = obs['resources'][...,1]>0
        rescue = torch.zeros_like(available)
        if 'decision' in obs:
            context = obs['decision']
            known = context[...,0]>.5
            available &= ~known | (context[...,4]>.5)
            rescue = known & (context[...,1]>.5) & (context[...,2]<.5) & (context[...,3]>.5) & available
            if 'control' in obs:
                rescue &= obs['control'][...,18]<.5
        prior = torch.where(rescue,torch.full_like(available,RESCUE_BOMB_PRIOR,dtype=torch.float32),
                            torch.full_like(available,BOMB_PRIOR,dtype=torch.float32))
        return available,prior

    def _resource_distribution(self, latent_pi, obs):
        logits = self.action_net(latent_pi)
        allowed = torch.ones_like(logits[..., :18], dtype=torch.bool)
        if "planning" in obs:
            forecast = obs["planning"]
            enabled = forecast[..., 0, 0] > .5
            allowed = torch.where(enabled[..., None], forecast[..., 1] > .5, allowed)
            # Normal planner output always has a fallback. Keep synthetic or
            # unavailable all-zero forecasts finite rather than making NaNs.
            allowed = torch.where(allowed.any(-1, keepdim=True), allowed, torch.ones_like(allowed))
            bias = torch.where(enabled[..., None], forecast[..., 7]*PLAN_LOGIT_SCALE,
                               torch.zeros_like(logits[..., :18]))
            logits = logits+torch.cat([bias, bias], dim=-1)
        logits = logits.masked_fill(~torch.cat([allowed, allowed], dim=-1), torch.finfo(logits.dtype).min)
        available,prior = self._bomb_context(obs)
        odds_shift = torch.log(prior)-torch.log1p(-prior)-(math.log(BOMB_PRIOR)-math.log1p(-BOMB_PRIOR))
        masked_bomb_logits = (logits[...,18:]+odds_shift[...,None]).masked_fill(~available[..., None],torch.finfo(logits.dtype).min)
        final_logits = torch.cat([logits[..., :18], masked_bomb_logits], -1)
        if getattr(self, '_fast_inference', False):
            # PolicyRunner validates finite CPU observations and results. Avoid
            # distribution validation's CPU synchronization during graph capture.
            self.action_dist.distribution = torch.distributions.Categorical(logits=final_logits, validate_args=False)
            return self.action_dist
        return self.action_dist.proba_distribution(action_logits=final_logits)

    def forward(self, obs, deterministic=False):
        latent_pi, latent_vf = self._latents(obs)
        distribution = self._resource_distribution(latent_pi, obs)
        self.last_bomb_probability = distribution.distribution.probs[..., 18:].sum(-1).detach()
        actions = distribution.get_actions(deterministic=deterministic)
        return actions.reshape((-1, *self.action_space.shape)), self.value_net(latent_vf), distribution.log_prob(actions)

    def get_distribution(self, obs):
        features = BaseModel.extract_features(self, obs, self.pi_features_extractor)
        return self._resource_distribution(self.mlp_extractor.forward_actor(features), obs)

    def evaluate_with_exploration(self, obs, actions):
        latent_pi, latent_vf = self._latents(obs)
        distribution = self._resource_distribution(latent_pi, obs)
        available,prior = self._bomb_context(obs)
        entropy, bomb_kl = exploration_terms(distribution.distribution.probs, available, prior)
        return (self.value_net(latent_vf), distribution.log_prob(actions), distribution.entropy(),
                entropy, bomb_kl, distribution.distribution.probs[..., 18:].sum(-1))

    def evaluate_actions(self, obs, actions):
        # Preserve SB3's public contract: this is actual joint Shannon entropy.
        return self.evaluate_with_exploration(obs, actions)[:3]
