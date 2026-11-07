"""SB3's rollout protocol with a cooperative gate before each policy action.

Based on stable-baselines3 2.9's OnPolicyAlgorithm.collect_rollouts (MIT).
No rollout reset, counter reset or optimizer recreation occurs on a UI pause.
"""
from .i18n import msg
import numpy as np
import torch
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.utils import obs_as_tensor

from .control import SamplingInterrupted
from .config import READBACK_AB_STEPS
from .input_timing import clock_ns


class PausablePPO(PPO):
    training_control = None
    training_game = None
    decision_runner = None

    def train(self):
        from .policy import ResourcePolicy
        if not isinstance(self.policy, ResourcePolicy):
            return super().train()
        from .ppo_update import update_resource_policy
        return update_resource_policy(self)

    def _excluded_save_params(self):
        return super()._excluded_save_params() + ["training_control", "training_game", "decision_runner"]

    def _refresh_observation(self):
        observation = self.training_game.builder.build(self.training_game.current)
        self._last_obs = {name: value[None, ...] for name, value in observation.items()}

    def collect_rollouts(self, env, callback, rollout_buffer, n_rollout_steps):
        if self.training_control is None:
            return super().collect_rollouts(env, callback, rollout_buffer, n_rollout_steps)
        if env.num_envs != 1 or not isinstance(self.action_space, spaces.Discrete):
            raise RuntimeError(msg('backend.live_pause_control_requires_one_game_environment'))
        assert self._last_obs is not None
        self.policy.set_training_mode(False)
        rollout_buffer.reset()
        rollout_buffer.gae_ready = False
        callback.on_rollout_start()
        runner = self.decision_runner
        readback_trial = getattr(runner, 'readback_trial', None)
        if readback_trial is not None:
            runner.readback_trial = None
        n_steps = 0
        refresh_needed = False
        while n_steps < n_rollout_steps:
            resumed = self.training_control.gate(self.training_game)
            pending_boundary = getattr(self.training_game, 'pending_stage_boundary', False)
            if resumed or pending_boundary or refresh_needed:
                if pending_boundary:
                    if n_steps and not self._last_episode_starts[0]:
                        with torch.no_grad():
                            boundary_value = self.policy.predict_values(obs_as_tensor(self._last_obs, self.device))
                        rollout_buffer.rewards[n_steps-1] += self.gamma * boundary_value.cpu().numpy().reshape(-1)
                    self._last_episode_starts = np.ones(env.num_envs, dtype=bool)
                    self.training_game.pending_stage_boundary = False
                self._refresh_observation()
                refresh_needed = False
            policy_started = getattr(self.training_game, 'begin_policy_timing', clock_ns)()
            if runner is not None:
                if readback_trial is not None and n_steps < READBACK_AB_STEPS:
                    decision = readback_trial.predict(runner, self._last_obs, n_steps, self.training_control.lease)
                else:
                    decision = runner.predict(self._last_obs, lease=self.training_control.lease)
                actions,values,log_probs = decision.actions, decision.values, decision.log_probs
                bomb_probability = decision.bomb_probability
            else:
                with torch.no_grad():
                    obs_tensor = obs_as_tensor(self._last_obs, self.device)
                    actions, values, log_probs = self.policy(obs_tensor)
                actions = actions.cpu().numpy()
                bomb = getattr(self.policy, 'last_bomb_probability', None)
                bomb_probability = float(bomb.cpu().numpy().reshape(-1)[0]) if bomb is not None else None
            if hasattr(self.training_game, 'record_policy_timing'):
                self.training_game.record_policy_timing(policy_started, clock_ns())
            try:
                if hasattr(self.training_game,'refresh_decision'):
                    self.training_game.refresh_decision()
                if runner is not None and hasattr(self.training_game,'record_decision'):
                    self.training_game.record_decision(decision,self.num_timesteps+1)
                new_obs, rewards, dones, infos = env.step(actions)
            except SamplingInterrupted as interruption:
                if interruption.episode_end is not None:
                    self._last_episode_starts = np.ones(env.num_envs, dtype=bool)
                    self._update_info_buffer([interruption.episode_end], self._last_episode_starts)
                    if hasattr(callback, 'record_episode_end'):
                        callback.record_episode_end(interruption.episode_end)
                    self._last_obs = env.reset()
                    continue
                # Focus loss, a scene change or a native clock reset can
                # interrupt an action. Re-observe without a fabricated sample.
                if interruption.bootstrap_boundary:
                    # Cut GAE at the last actual old-stage observation. A
                    # value bootstrap preserves its continuation estimate;
                    # rewards from the next stage must not label old actions.
                    if n_steps and not self._last_episode_starts[0]:
                        rollout_buffer.rewards[n_steps-1] += self.gamma * values.detach().cpu().numpy().reshape(-1)
                    self._last_episode_starts = np.ones(env.num_envs, dtype=bool)
                # Resume can discover a stage boundary. Keep _last_obs until
                # the loop has bootstrapped the last accepted transition.
                refresh_needed = True
                continue
            self.num_timesteps += env.num_envs
            if bomb_probability is not None:
                infos[0]['bomb_probability'] = bomb_probability
            if runner is not None:
                infos[0]['inference'] = decision.timing
            self._update_info_buffer(infos, dones)
            n_steps += 1
            actions = actions.reshape(-1, 1)
            for idx, done in enumerate(dones):
                if done and infos[idx].get("terminal_observation") is not None and infos[idx].get("TimeLimit.truncated", False):
                    terminal_obs = self.policy.obs_to_tensor(infos[idx]["terminal_observation"])[0]
                    with torch.no_grad():
                        terminal_value = self.policy.predict_values(terminal_obs)[0]
                    rewards[idx] += self.gamma * terminal_value
            buffer_started = clock_ns()
            rollout_buffer.add(self._last_obs, actions, rewards, self._last_episode_starts, values, log_probs)
            if hasattr(self.training_game, 'record_pre_policy_work'):
                self.training_game.record_pre_policy_work(buffer_add_ms=(clock_ns()-buffer_started)/1e6)
            self._last_obs = new_obs
            self._last_episode_starts = dones
            # The environment action has already happened. Keep its actual
            # transition even when this callback reaches the wall-time budget
            # or the user finishes; otherwise counters/logs get one step ahead
            # of the saved partial rollout.
            callback.update_locals(locals())
            callback_started = clock_ns()
            continuing = callback.on_step()
            if hasattr(self.training_game, 'record_pre_policy_work'):
                self.training_game.record_pre_policy_work(callback_ms=(clock_ns()-callback_started)/1e6)
            if not continuing:
                return False
        with torch.no_grad():
            values = self.policy.predict_values(obs_as_tensor(self._last_obs, self.device))
        rollout_buffer.compute_returns_and_advantage(last_values=values, dones=self._last_episode_starts)
        rollout_buffer.gae_ready = True
        callback.update_locals(locals())
        callback.on_rollout_end()
        return True
