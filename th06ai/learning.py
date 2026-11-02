from __future__ import annotations

import csv
import hashlib
import json
import time
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback

from .config import Config, ROOT
from .files import atomic_json, timestamp
from .native import SHA256
from .policy import BulletEncoder, ResourcePolicy, POLICY_VERSION, BOMB_PRIOR
from .observations import (OBSERVATION_VERSION, ACTION_COUNT, ACTION_TIMING_VERSION, make_observation_space,
                           NEAREST, FULL_BULLETS, FULL_BULLET_FIELDS)
from .rewards import DIAGNOSTIC_FIELDS, DISCOUNT, REWARD_VERSION
from .control import FinishTraining, TrainingControl, SamplingInterrupted
from .model_store import inspect_checkpoint, preserve_checkpoint
from .pausable_ppo import PausablePPO
from .planner_spec import PLANNER_VERSION, PLAN_FIELDS
from .input_timing import clock_ns
from .log_rows import action_row, episode_row, serialized_row
from .navigation import NAV_FIELDS, DECISION_FIELDS, RESCUE_BOMB_PRIOR


def make_model(env, device="cpu", seed=17):
    inference_only = env is None
    if inference_only:
        # Construct the new network/optimizer for a CPU-only legacy load.
        # This supplies shapes only; it cannot produce fake game experience.
        import gymnasium as gym
        env = gym.Env()
        env.observation_space = make_observation_space()
        env.action_space = gym.spaces.Discrete(ACTION_COUNT)
    model = PausablePPO(ResourcePolicy, env, device=device, seed=seed,
               policy_kwargs={"features_extractor_class": BulletEncoder,
                              "net_arch": {"pi": [128, 128], "vf": [128, 128]}},
               learning_rate=3e-4, n_steps=512, batch_size=64, n_epochs=4,
               gamma=DISCOUNT, gae_lambda=.95, ent_coef=.02, clip_range=.2,
               verbose=1)
    model.policy_version = POLICY_VERSION
    model.reward_version = REWARD_VERSION
    model.action_timing_version = ACTION_TIMING_VERSION
    model.planner_version = PLANNER_VERSION
    if inference_only:
        model.env = None
    return model


def adapt_objective(model):
    """New rewards/discount apply to future rollouts without resetting weights."""
    model.gamma = DISCOUNT
    model.rollout_buffer.gamma = DISCOUNT
    model.reward_version = REWARD_VERSION
    model.action_timing_version = ACTION_TIMING_VERSION
    model.planner_version = PLANNER_VERSION


def save_model(model, path: Path, run_dir: Path, steps: int):
    if model._n_updates <= 0:
        raise RuntimeError("模型还没有完成参数更新，不保存为已训练模型")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.stem + ".pending.zip")
    model.save(temporary)
    preserve_checkpoint(path, ROOT / "models")
    # Checkpoints are only written while the game is confirmed paused.
    temporary.replace(path)
    atomic_json(path.with_suffix(".json"), {"time": timestamp(), "game_sha256": SHA256,
                "model_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "steps": int(steps), "gradient_updates": model._n_updates,
                "run_dir": str(run_dir), "algorithm": "PPO", "format": OBSERVATION_VERSION,
                "reward_version": REWARD_VERSION,
                "gamma": float(model.gamma),
                "action_timing_version": ACTION_TIMING_VERSION,
                "planner_version": PLANNER_VERSION, "planner_fields": PLAN_FIELDS,
                "bullet_inputs": {"near": NEAREST, "full": FULL_BULLETS, "full_fields": FULL_BULLET_FIELDS},
                "source_checkpoint": getattr(model, "source_checkpoint", None),
                "migrated_from": getattr(model, "migrated_from", None),
                "policy_version": getattr(model, "policy_version", None),
                "bomb_prior": BOMB_PRIOR if isinstance(model.policy, ResourcePolicy) else None,
                'rescue_bomb_prior':RESCUE_BOMB_PRIOR,'navigation_fields':NAV_FIELDS,'decision_fields':DECISION_FIELDS,
                'training_goal':getattr(model,'training_goal','survival'),
                "policy_upgrade": getattr(model, "policy_upgrade", None)})
    return preserve_checkpoint(path, ROOT / "models")


def load_model(path: Path, env, device="cpu"):
    path = path.resolve()
    if not path.is_relative_to((ROOT / "models").resolve()):
        raise RuntimeError("仅加载本项目 models 目录中自行生成的模型")
    metadata_path = path.with_suffix(".json")
    if not path.is_file() or not metadata_path.is_file():
        raise RuntimeError("尚无已训练的模型；请先完成训练，不能用随机权重冒充模型试玩")
    checkpoint = inspect_checkpoint(path, verify=True)
    if not checkpoint.usable:
        raise RuntimeError(f"模型无法加载：{checkpoint.compatibility}")
    if checkpoint.metadata.get("format") in (1, 2, 3, 4, 5, 6, 7):
        model = migrate_legacy(path, env, device)
        upgraded = checkpoint.metadata.get("format") == 1 or (checkpoint.metadata.get("policy_version") or 1) < POLICY_VERSION
        if upgraded:
            model.policy.reset_bomb_prior()
    else:
        upgraded = (checkpoint.metadata.get("policy_version") or 1) < POLICY_VERSION
        model = PausablePPO.load(path, env=env, device=device,
                                custom_objects={"policy_class": ResourcePolicy} if upgraded else None)
        if not isinstance(model.policy, ResourcePolicy):
            raise RuntimeError("模型策略类型与版本不符，拒绝带入旧炸弹逻辑")
        if upgraded:
            # Keep learned movement/encoder/value weights and compatible Adam
            # state. Only the faulty bomb branch and its moments are rebased.
            model.policy.reset_bomb_prior()
    model.policy_version = POLICY_VERSION
    model.bomb_policy_upgraded = upgraded
    if upgraded:
        model.policy_upgrade = {"type": "bomb_exploration_repair", "version": POLICY_VERSION,
                                "source": str(path), "source_sha256": checkpoint.metadata["model_sha256"],
                                "bomb_prior": BOMB_PRIOR,
                                "note": "仅重置炸弹分支；保留移动/编码器/价值参数、兼容优化器状态和累计训练量"}
    model.source_checkpoint = str(path)
    model.source_checkpoint_sha256 = checkpoint.metadata.get('model_sha256')
    model.source_checkpoint_steps = checkpoint.metadata.get('steps')
    adapt_objective(model)
    return model


def migrate_legacy(path: Path, env, device="cpu"):
    """Extend v1..v7 encoders with zero-influence new input columns.

    Existing encoder/actor/value parameters and compatible Adam moments are
    copied. An 18-action source also gets a rare bomb branch; a 36-action
    source keeps its learned bomb rows until a separately required policy fix.
    """
    import torch
    import math
    old = PausablePPO.load(path, device=device)
    if old.action_space.n not in (18, 36):
        raise RuntimeError("旧模型动作数无法安全迁移")
    model = make_model(env, device)
    source, target = old.policy.state_dict(), model.policy.state_dict()

    def expand(name, value, original, *, bias_prior=False):
        result = torch.zeros_like(value)
        if value.shape == original.shape:
            return original.clone()
        if name.endswith("fusion.0.weight"):
            if original.shape[1] == 132:
                result[:, :128] = original[:, :128]
                result[:, 192:196] = original[:, 128:132]
            elif original.shape[1] in (201, 265, 293, 421, 565, 573):
                result[:, :original.shape[1]] = original
            else:
                raise RuntimeError("未知的旧观测编码布局")
        elif name == "action_net.weight":
            result[:18] = original
            if bias_prior:
                result[18:] = original
        elif name == "action_net.bias":
            result[:18] = original
            if bias_prior:
                result[:18] += math.log1p(-BOMB_PRIOR)
                result[18:] = original + math.log(BOMB_PRIOR)
        else:
            raise RuntimeError(f"无法迁移参数 {name}：{original.shape} → {value.shape}")
        return result

    for name, value in target.items():
        if name in source:
            target[name] = expand(name, value, source[name], bias_prior=True)
        elif ".full_bullet." in name:
            # Warm-start the added global point encoder from the existing
            # near encoder. New geometry/motion-availability columns start at
            # zero. The appended fusion columns also start at zero, preserving
            # every old action/value output until future policy updates.
            original = source.get(name.replace(".full_bullet.", ".bullet."))
            if original is not None:
                if name.endswith("full_bullet.0.weight"):
                    target[name] = torch.zeros_like(value)
                    target[name][:, :4] = original[:, :4]
                    target[name][:, -1] = original[:, 4]
                else:
                    target[name] = original.clone()
    model.policy.load_state_dict(target)
    old_parameters = dict(old.policy.named_parameters())
    # Keep Adam moments for learned parameters; new dimensions have no history.
    for name, parameter in model.policy.named_parameters():
        original = old_parameters.get(name)
        if original is None or original not in old.policy.optimizer.state:
            continue
        state = old.policy.optimizer.state[original]
        model.policy.optimizer.state[parameter] = {
            key: (expand(name, parameter, value) if isinstance(value, torch.Tensor) and value.ndim else
                  value.clone() if isinstance(value, torch.Tensor) else value)
            for key, value in state.items()}
    model.num_timesteps = old.num_timesteps
    model._n_updates = old._n_updates
    model.migrated_from = str(path)
    if hasattr(old, "policy_upgrade"):
        model.policy_upgrade = old.policy_upgrade
    return model


class TrainingCallback(BaseCallback):
    def __init__(self, env, run_dir: Path, minutes: float, status, lease, control=None):
        super().__init__()
        self.game = env
        self.run_dir = run_dir
        self.minutes = minutes
        self.status = status
        self.lease = lease
        self.started = None
        self.last_checkpoint = 0
        self.last_checkpoint_updates = 0
        self.pending_checkpoint = False
        self.rollouts = 0
        self.episode_rows = []
        self.pending_episode_rows = []
        self.action_rows = []
        self.bomb_requests = 0
        self.bombs_used = 0
        self.control = control
        self.base_steps = 0
        self.finish_reason = "本次时长已完成"
        self.rollout_checkpoint = None
        self.rollout_first_step = None

    def elapsed(self):
        return self.control.active_seconds if self.control else time.monotonic() - self.started

    def pause_game(self):
        self.game.pause_for_user() if hasattr(self.game, "pause_for_user") else self.game.pause()

    def _on_training_start(self):
        self.started = time.monotonic()
        self.base_steps = self.model.num_timesteps
        self.last_checkpoint_updates = self.model._n_updates
        if self.control:
            self.control.start()
        self.pause_game()
        import torch
        observation = self.game.builder.build(self.game.current)
        tensor, _ = self.model.policy.obs_to_tensor(observation)
        with torch.no_grad():
            self.model.policy(tensor)
        runner = getattr(self.model, 'decision_runner', None)
        if runner is not None:
            runner.warmup(observation, self.lease)
        if self.model.device.type == "cuda":
            torch.cuda.synchronize()
        self.status(phase="神经网络训练", message="PPO：采样 → 暂停游戏 → 更新权重",
                    device=str(self.model.device), reward_version=REWARD_VERSION,
                    observation_version=OBSERVATION_VERSION, action_timing_version=ACTION_TIMING_VERSION,
                    steps=self.base_steps, base_steps=self.base_steps, session_steps=0,
                    updates=self.model._n_updates, budget_seconds=self.minutes * 60,
                    model_source=getattr(self.model, "source_checkpoint", None),
                    migrated_from=getattr(self.model, "migrated_from", None),
                    policy_version=POLICY_VERSION, bomb_prior=BOMB_PRIOR,
                    goal=getattr(self.game,'goal','survival'),
                    inference=runner.diagnostics() if runner is not None else None,
                    bomb_policy_upgraded=getattr(self.model, "bomb_policy_upgraded", False))

    def _on_rollout_start(self):
        self.lease()
        self.rollout_first_step = int(self.num_timesteps)+1
        elapsed = self.elapsed()
        # Each completed update is saved while the game remains paused.
        # Don't overwrite latest with an unchanged checkpoint at session start.
        if self.model._n_updates > self.last_checkpoint_updates:
            path = save_model(self.model, ROOT / "models" / "latest.zip", self.run_dir, self.num_timesteps)
            self.rollout_checkpoint = str(path) if path else None
            self.last_checkpoint_updates = self.model._n_updates
            self.last_checkpoint = elapsed
            self.status(checkpoint=str(path or ROOT / "models" / "latest.zip"))
        if self.control:
            window = getattr(getattr(self.game, "reader", None), "window", None)
            if window is not None and not window.focused:
                self.control.request_pause("游戏失去焦点，已自动暂停；点击恢复继续当前回合")
            self.control.gate(self.game, resume_game=False)
        self.game.resume_for_user() if hasattr(self.game, "resume_for_user") else self.game.resume()
        # Menu transitions consume real frames; collect the next action from
        # a fresh observation, not the frame saved before the PPO update.
        observation = self.game.builder.build(self.game.current)
        self.model._last_obs = {name: value[None, ...] for name, value in observation.items()}
        self.status(phase="神经网络采样", updates=self.model._n_updates,
                    policy_diagnostics=getattr(self.model, "last_policy_diagnostics", {}))

    def _on_step(self):
        self.lease()
        elapsed = self.elapsed()
        for info in self.locals.get("infos", []):
            if "action" in info:
                self.bomb_requests += int(info["action"] >= 18)
                self.bombs_used += info.get("bomb_consumed", 0)
                self.action_rows.append(action_row(info, int(self.num_timesteps)))
        for done, info in zip(self.locals.get("dones", []), self.locals.get("infos", [])):
            if done:
                row = episode_row(info,self.num_timesteps,round(elapsed,3))
                self.episode_rows.append(row)
                self.pending_episode_rows.append(row)
        self.status(steps=int(self.num_timesteps), training_seconds=round(elapsed, 1),
                    session_steps=int(self.num_timesteps - self.base_steps),
                    bomb_requests=self.bomb_requests, bombs_used=self.bombs_used,
                    fps=round((self.num_timesteps - self.base_steps) / max(elapsed, .001), 1))
        return elapsed < self.minutes * 60

    def flush_actions(self):
        if self.pending_episode_rows:
            path = self.run_dir/'episodes.csv'
            new = not path.exists()
            with path.open('a',encoding='utf-8',newline='') as stream:
                writer = csv.DictWriter(stream,fieldnames=list(self.pending_episode_rows[0]))
                if new:writer.writeheader()
                writer.writerows(serialized_row(row) for row in self.pending_episode_rows)
            self.pending_episode_rows.clear()
        if not self.action_rows:
            return
        path = self.run_dir / "actions.csv"
        new = not path.exists()
        with path.open("a", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(self.action_rows[0]))
            if new:
                writer.writeheader()
            writer.writerows(serialized_row(row) for row in self.action_rows)
        self.action_rows.clear()

    def flush_observations(self):
        """Save the actual PPO input, not a reconstructed scene or a demo.

        Runs only while paused. The buffer already contains CPU arrays; this
        adds no GPU round trip or disk write to the per-action inference path.
        Traces are diagnostic; an I/O failure must not kill game training.
        """
        buffer = self.model.rollout_buffer
        count = buffer.buffer_size if buffer.full else buffer.pos
        gae_ready = bool(getattr(buffer, 'gae_ready', buffer.full))
        if not count or self.rollout_first_step is None:
            return
        first_step = self.rollout_first_step
        last_step = first_step+count-1
        path = self.run_dir / f"observations-{first_step:09d}-{last_step:09d}.npz"
        if path.exists():
            return
        try:
            payload = {"obs_"+name: values[:count].copy() for name, values in buffer.observations.items()}
            for name in ("actions", "rewards", "values", "log_probs", "episode_starts"):
                payload[name] = getattr(buffer, name)[:count].copy()
            # A finish request can interrupt collection before GAE runs.
            # Do not present zero/uncomputed targets as valid returns.
            if gae_ready:
                for name in ("returns", "advantages"):
                    payload[name] = getattr(buffer, name)[:count].copy()
            payload["steps"] = np.arange(first_step, last_step+1, dtype=np.int64)
            payload["metadata"] = np.array(json.dumps({"observation_version": OBSERVATION_VERSION,
                    'stage_fields': ['Stage 1', 'Stage 2', 'Stage 3', 'Stage 4', 'Stage 5', 'Stage 6', 'EX', 'known'],
                    'stage_return_boundary': 'bootstrap old-stage value; cut GAE before new-stage samples',
                    'freshness_return_boundary':'bootstrap previous value; cut GAE across discarded stale decisions',
                    'navigation_fields':NAV_FIELDS,'decision_fields':DECISION_FIELDS,
                    'goal':getattr(self.game,'goal','survival'),
                    'episode_seconds': getattr(self.game, 'episode_seconds', None),
                    "planner_version": PLANNER_VERSION, "planner_fields": PLAN_FIELDS,
                    "reward_version": REWARD_VERSION, "action_timing_version": ACTION_TIMING_VERSION,
                    "policy_gradient_epochs": int(self.model._n_updates),
                    "bullet_inputs": {"near": NEAREST, "full": FULL_BULLETS, "full_fields": FULL_BULLET_FIELDS},
                    "gae_ready": gae_ready,
                    "policy_checkpoint": self.rollout_checkpoint or getattr(self.model, "source_checkpoint", None),
                    "note": "Actual on-policy buffer before its update; not demonstrations or independent evaluation. Rewards may include time-limit value bootstrapping."}, ensure_ascii=False))
            pending = path.with_suffix(".pending.npz")
            np.savez_compressed(pending, **payload)
            pending.replace(path)
        except (OSError, ValueError) as error:
            self.status(trajectory_error=str(error))

    def _on_rollout_end(self):
        self.pause_game()
        self.flush_actions()
        self.flush_observations()
        if self.control:
            self.control.gate(self.game, resume_game=False)
        self.rollouts += 1
        self.status(phase="PPO 更新神经网络", updates=self.model._n_updates)

    def _on_training_end(self):
        self.pause_game()
        self.lease()
        self.flush_actions()
        self.flush_observations()
        if self.model._n_updates > 0:
            path = save_model(self.model, ROOT / "models" / "latest.zip", self.run_dir, self.num_timesteps)
            self.status(checkpoint=str(path or ROOT / "models" / "latest.zip"))
        else:
            self.status(message="尚未完成一次参数更新，未保存随机模型")
        self.status(updates=self.model._n_updates, steps=int(self.num_timesteps),
                    session_steps=int(self.num_timesteps - self.base_steps),
                    training_seconds=round(self.elapsed(), 2))
        atomic_json(self.run_dir / "training_summary.json", {
            "algorithm": "PPO", "steps": int(self.num_timesteps), "rollouts": self.rollouts,
            "gradient_updates": self.model._n_updates,
            "reward_version": REWARD_VERSION,
            "observation_version": OBSERVATION_VERSION,
            'episode_seconds': getattr(self.game, 'episode_seconds', None),
            'stage_aware': True,
            "elapsed_seconds": self.elapsed(), "episodes": len(self.episode_rows),
            "session_steps": int(self.num_timesteps - self.base_steps), "base_steps": self.base_steps,
            "paused_seconds": self.control.paused_seconds if self.control else 0,
            "finish_reason": self.finish_reason,
            "source_checkpoint": getattr(self.model, "source_checkpoint", None),
            "policy_version": POLICY_VERSION, "bomb_requests": self.bomb_requests, "bombs_used": self.bombs_used,
            'goal':getattr(self.game,'goal','survival'),
            'inference':self.model.decision_runner.diagnostics() if getattr(self.model,'decision_runner',None) else None,
            'diagnostics':self.game.diagnostics.status() if getattr(self.game,'diagnostics',None) else None,
            "policy_diagnostics": getattr(self.model, "last_policy_diagnostics", {}),
            "mean_survival_seconds": float(np.mean([r["survival_seconds"] for r in self.episode_rows])) if self.episode_rows else None,
            "note": "训练日志不等于独立评估；请用 evaluate 测量模型表现。"})


def train(config: Config, env, run_dir: Path, minutes: float, status, lease, resume: str | None):
    import torch
    torch.set_num_threads(1)
    if config.device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("当前为 CPU 版 PyTorch，未启用 CUDA；不会静默切换训练设备")
        torch.cuda.set_per_process_memory_fraction(.15)
    model = load_model(Path(resume), env, config.device) if resume else make_model(env, config.device)
    adapt_objective(model)
    model.training_goal = config.goal
    from .inference import PolicyRunner
    model.decision_runner = PolicyRunner(model.policy, config.inference_backend)
    control = TrainingControl(run_dir, lease, status)
    env.training_control = control
    model.training_control, model.training_game = control, env
    callback = TrainingCallback(env, run_dir, minutes, status, lease, control)
    # Wall time, not an assumed FPS, controls the user's one-hour budget.
    try:
        model.learn(total_timesteps=10_000_000, callback=callback, reset_num_timesteps=not bool(resume))
    except FinishTraining:
        callback.finish_reason = "用户保存并结束"
        control.finishing = True
        status(phase="正在保存并结束", finishing=True, user_paused=False)
        if control.started is not None:
            callback.on_training_end()
        else:
            status(phase='已取消启动', message='启动已取消；未更新或覆盖模型', finishing=False)
    finally:
        model.training_control = model.training_game = None
        model.decision_runner = None
        env.training_control = None
    return model


def evaluate(model, env, run_dir: Path, status, lease, episodes=10):
    rows = []
    # Preserve completed episodes even if focus, F8 or a later reset interrupts
    # the session. Only the final report is marked complete.
    atomic_json(run_dir / 'evaluation_progress.json', evaluation_report(model, env, rows, episodes))
    control = getattr(env, 'training_control', None)
    for i in range(episodes):
        obs, _ = env.reset()
        if control and control.started is None:
            control.start()
        while True:
            lease()
            if control and control.gate(env):
                obs = env.builder.build(env.current)
            policy_started = clock_ns()
            runner = getattr(model,'decision_runner',None)
            if runner is not None:
                decision = runner.predict(obs,deterministic=True,lease=lease)
                action = decision.actions[0]
            else:
                action, _ = model.predict(obs, deterministic=True)
            if hasattr(env, 'record_policy_timing'):
                env.record_policy_timing(policy_started, clock_ns())
            try:
                if hasattr(env,'refresh_decision'):env.refresh_decision()
                if runner is not None and hasattr(env,'record_decision'):env.record_decision(decision)
                obs, reward, terminated, truncated, info = env.step(int(action))
            except SamplingInterrupted:
                if control:
                    control.gate(env)
                obs = env.builder.build(env.current)
                continue
            status(phase="独立模型评估", evaluation_episode=i + 1, evaluation_episodes=episodes)
            if terminated or truncated:
                rows.append({"episode": i + 1, **episode_row(info,getattr(env,'total_steps',0)),
                             **info.get("episode_diagnostics", {})})
                atomic_json(run_dir / 'evaluation_progress.json', evaluation_report(model, env, rows, episodes))
                break
    env.pause()
    report = evaluation_report(model, env, rows, episodes, complete=True)
    atomic_json(run_dir / 'evaluation_progress.json', report)
    atomic_json(run_dir / "evaluation.json", report)
    return report


def evaluation_report(model, env, rows, requested, complete=False):
    average = lambda key: float(np.mean([r.get(key, 0) for r in rows])) if rows else None
    report = {"episodes": list(rows), 'complete':complete,
              'episodes_requested':requested, 'episodes_completed':len(rows),
              'checkpoint':{'path':getattr(model,'source_checkpoint',None),
                            'sha256':getattr(model,'source_checkpoint_sha256',None),
                            'steps':getattr(model,'source_checkpoint_steps',None)},
              'planner_version':PLANNER_VERSION,
              'goal':getattr(env,'goal','survival'),
              'bombs_used':sum(r.get('bombs_used',0) for r in rows),
              'nmnb_clean_observation_rate':float(np.mean([not r['died'] and not r.get('bombs_used',0) for r in rows])) if rows else None,
              'nmnb_note':'Observed windows only; neither time-limit survival nor an unfinished run proves a full NMNB clear.',
              "mean_survival_seconds": average('survival_seconds'),
              'median_survival_seconds':float(np.median([r['survival_seconds'] for r in rows])) if rows else None,
              "no_hit_rate": float(np.mean([not r["died"] for r in rows])) if rows else None,
              'time_limit_completions':sum(bool(env.episode_seconds) and not r['died'] and r['survival_seconds']>=env.episode_seconds for r in rows),
              "max_seconds": env.episode_seconds,
              "game_sha256": SHA256, "reward_version": REWARD_VERSION, "time": timestamp()}
    report.update(mean_power_gained=average('power_gained'),
                  mean_enemy_damage=average('enemy_damage'),
                  mean_boss_damage=average('boss_damage'),
                  mean_point_items_gained=average('point_items_gained'),
                  loiter_ratio=float(sum(r.get("loiter_seconds", 0) for r in rows) /
                                     max(1e-6, sum(r["survival_seconds"] for r in rows))))
    # Graze is evidence to inspect, never a "best model" score or objective.
    report["graze_diagnostics"] = {
        "total": sum(r.get("graze_gained", 0) for r in rows),
        "with_observed_risk": sum(r.get("graze_with_risk", 0) for r in rows),
        "observed_seconds": sum(r.get("graze_observed_seconds", 0) for r in rows),
        "used_as_objective": False,
    }
    return report
