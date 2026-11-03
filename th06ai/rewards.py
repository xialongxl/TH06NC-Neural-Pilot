"""Training feedback only. This module never chooses or replaces an action."""
from __future__ import annotations

from dataclasses import dataclass
from collections import deque
import math

import numpy as np

from .native import Frame
from .observations import ACTION_COUNT, DIRECTIONS
from .combat import damage_feedback, aim_potential
from .risk import collision_risk, visible_threat
from .objectives import power_potential

REWARD_VERSION = 9
# Nominal 30 decisions/sec: an effective horizon near 33 sec, instead of 6.7.
DISCOUNT = .999
SURVIVAL_PER_SECOND = .05
DEATH_COST = 20.0
THREAT_WEIGHT = 2.0  # Bounded below the 20-point death penalty.
BLOCKED_GRACE_TICKS = 12.0
BLOCKED_COST_PER_SECOND = 2.0
RESOURCE_BUDGET = 18.0  # Larger pickup credit, still bounded below one death.
POWER_CREDIT = 1.0
POINT_CREDIT = .25
BOMB_CREDIT = 2.0
LIFE_CREDIT = 4.0
ITEM_POTENTIAL_LIMIT = .1
LOITER_WINDOW_TICKS = 180
LOITER_GRACE_TICKS = 120
LOITER_COST_PER_SECOND = 1.0
CORNER_INNER_PIXELS = 32.0
CORNER_EXIT_PIXELS = 64.0
WALL_RELEASE_PIXELS = 16.0
RESOURCE_COMPONENTS = ("power", "bomb_pickup", "life_pickup", "score", "point_items")
COMPONENTS = ("survival", "death", "threat_shaping", "item_shaping", "power_progress", "aim_shaping", "damage", "wall_push", "edge_loiter", *RESOURCE_COMPONENTS, "bomb_use")
DIAGNOSTIC_FIELDS = ("wall_push_seconds", "mean_threat", "corner_exposure_seconds", "edge_exposure_seconds", "loiter_seconds",
                     "moving_seconds", "power_gained", "point_items_gained", "enemy_damage", "boss_damage",
                     "graze_gained", "graze_observed_seconds", "graze_with_risk", "geometry_coverage",
                     *("reward_" + x for x in COMPONENTS))


def edge_exposure(frame: Frame) -> float:
    left, top, width, height = frame.bounds
    distances = np.maximum(0, [frame.x-left, left+width-frame.x, frame.y-top, top+height-frame.y])
    return float(1 - np.prod(1 - np.exp(-.5 * (distances / 32)**2)))


def corner_exposure(frame: Frame) -> float:
    """Continuous exposure to two walls; no special exemption a pixel off a wall."""
    left, top, width, height = frame.bounds
    dx = max(0, min(frame.x - left, left + width - frame.x))
    dy = max(0, min(frame.y - top, top + height - frame.y))
    return math.exp(-.5 * ((dx / 48)**2 + (dy / 64)**2))


def residence_age(frame: Frame, positions: np.ndarray) -> float:
    """Observed residence clock for policy input, including the free grace.

    This does not change residence costs. Supplying only the charged level
    would hide the difference between just entering and being about to reach
    the two-second grace boundary.
    """
    if not len(positions):
        return 0.
    entered = 0
    left, top, width, height = frame.bounds
    corner = np.array([left if frame.x < left+width/2 else left+width,
                       top if frame.y < top+height/2 else top+height])
    if np.all(np.abs(np.array([frame.x, frame.y])-corner) < CORNER_EXIT_PIXELS):
        inside = np.all(np.abs(positions[:, 1:3]-corner) < CORNER_EXIT_PIXELS, axis=1)
        outside = np.flatnonzero(~inside)
        entered = int(outside[-1])+1 if len(outside) else 0
    return float(np.clip((frame.tick-positions[entered, 0]) / LOITER_WINDOW_TICKS, 0, 1))


def residence_level(frame: Frame, positions: np.ndarray) -> float:
    """Local corner residence, or low-progress travel along a single wall.

    Corner grace is measured since entering that region. Moving around inside
    it never clears residence; leaving the outer region does. A genuine wall
    traverse outside corners can still reduce the low-progress cost.
    """
    covered = frame.tick - positions[0, 0]
    extent = float(np.linalg.norm(np.ptp(positions[:, 1:3], axis=0)))
    dwell = float(np.clip((covered - LOITER_GRACE_TICKS) / 60, 0, 1))
    progress_scale = math.exp(-.5 * (extent / 32)**2)
    left, top, width, height = frame.bounds
    corner = np.array([left if frame.x < left+width/2 else left+width,
                       top if frame.y < top+height/2 else top+height])
    distance = np.abs(np.array([frame.x, frame.y]) - corner)
    if np.all(distance < CORNER_EXIT_PIXELS):
        inside = np.all(np.abs(positions[:, 1:3] - corner) < CORNER_EXIT_PIXELS, axis=1)
        outside = np.flatnonzero(~inside)
        entered = int(outside[-1])+1 if len(outside) else 0
        local_ticks = frame.tick - positions[entered, 0]
        dwell = float(np.clip((local_ticks - LOITER_GRACE_TICKS) / 60, 0, 1))
        closeness = float(np.prod(np.clip((CORNER_EXIT_PIXELS-distance) /
                                         (CORNER_EXIT_PIXELS-CORNER_INNER_PIXELS), 0, 1)))
        return dwell * max(edge_exposure(frame) * progress_scale, closeness)
    return edge_exposure(frame) * dwell * progress_scale


def item_potential(frame: Frame, threat: float, raw_credit: float) -> float:
    """Bounded state potential; never chooses an action or awards a pickup.

    Approaching a visible, useful item supplies short-range learning feedback.
    Its disappearance removes the potential regardless of cause. Only real
    resource counter changes earn pickup credit, preventing disappearance farms.
    """
    if not len(frame.items) or frame.resources is None:
        return 0.0
    r = frame.resources
    desire = np.array([float(r.power < 128), 1.0, float(r.power < 128),
                       float(r.bombs < 8), float(r.power < 128), float(r.lives < 8), .01])
    types = frame.items[:, 3].astype(np.int64)
    distance = np.linalg.norm(frame.items[:, 1:3] - [frame.x, frame.y], axis=1)
    # Observed item velocity only. Items already falling out of reach contribute
    # less; this is a learning signal, not a scripted collection route.
    falling = frame.items[:, 5] > 0
    bottom = frame.bounds[1] + frame.bounds[3]
    remaining_ticks = (bottom - frame.items[:, 2]) / np.maximum(frame.items[:, 5], .01)
    reachable = np.ones(len(distance))
    speeds = getattr(frame, 'movement_speeds', None)
    # Use verified maximum movement, since reaching an item can release Shift.
    # Unknown speed retains the original 4px/tick baseline, not a made-up readout.
    fast = float(speeds[0]) if speeds is not None and 0 < speeds[1] <= speeds[0] < 20 else 4.
    reachable[falling] = np.clip(remaining_ticks[falling] * fast / np.maximum(distance[falling], 1), 0, 1)
    attraction = desire[types] * np.exp(-distance / 96) * reachable
    return float(ITEM_POTENTIAL_LIMIT * math.exp(-raw_credit / RESOURCE_BUDGET)
                 * max(.05, 1 - threat) * attraction.max())


@dataclass(frozen=True)
class RewardStep:
    total: float
    components: dict[str, float]
    threat: float
    dx: float
    dy: float
    blocked_fraction: float
    corner_exposure: float
    loiter: float
    item_potential: float
    graze_gained: int
    min_clearance: float | None
    geometry_coverage: float


class RewardTracker:
    def __init__(self, *, feature_cache=None):
        self.feature_cache = feature_cache
        self.reset()

    def reset(self, frame: Frame | None = None):
        self.frame = frame
        self.threat = visible_threat(frame) if frame is not None else 0.0
        self.blocked_ticks = np.zeros(2, dtype=np.float64)
        self.previous_direction = np.zeros(2, dtype=int)
        self.total_ticks = 0
        self.threat_ticks = 0.0
        self.wall_push_ticks = 0.0
        self.totals = dict.fromkeys(COMPONENTS, 0.0)
        self.resource_bonus_used = 0.0
        self.resource_raw_credit = 0.0
        self.item_potential = item_potential(frame, self.threat, 0) if frame is not None else 0.0
        self.aim_potential = aim_potential(frame, self.threat) if frame is not None else 0.0
        self.power_potential = power_potential(frame)
        self.positions = deque([(frame.tick, frame.x, frame.y)] if frame is not None else [])
        self.corner_ticks = self.loiter_ticks = self.moving_ticks = 0.0
        self.edge_ticks = self.enemy_damage = self.boss_damage = 0.0
        self.power_gained = self.point_items_gained = 0
        self.graze_gained = self.graze_with_risk = 0
        self.graze_observed_ticks = self.geometry_ticks = 0.0

    def rebase(self, frame: Frame):
        # Menu transitions are outside action sampling. Discard their motion
        # history and pending blockage; retain this episode's totals.
        self.frame = frame
        self.threat = visible_threat(frame)
        self.blocked_ticks[:] = 0
        self.previous_direction[:] = 0
        self.item_potential = item_potential(frame, self.threat, self.resource_raw_credit)
        self.aim_potential = aim_potential(frame, self.threat)
        self.power_potential = power_potential(frame)
        # Preserve dwell history across a managed update pause. Game restart
        # uses reset(); update pauses must not provide a free loiter reset.
        if self.positions and frame.tick < self.positions[-1][0]:
            self.positions.clear()
        self.positions.append((frame.tick, frame.x, frame.y))

    def step(self, frame: Frame, action: int, died: bool) -> RewardStep:
        if self.frame is None:
            raise RuntimeError("奖励状态尚未初始化")
        if not 0 <= int(action) < ACTION_COUNT:
            raise ValueError("模型动作必须是 0 到 35")
        before = self.frame
        ticks = frame.tick - before.tick
        if ticks < 0 or (ticks == 0 and not died):
            raise RuntimeError("奖励计算需要推进的游戏帧；计时重置时应先对齐观测")
        risk = collision_risk(frame, before, cache=self.feature_cache)
        threat = risk.threat
        graze_gained = 0
        if before.resources is not None and frame.resources is not None:
            old_graze, new_graze = before.resources.graze, frame.resources.graze
            if old_graze is not None and new_graze is not None and new_graze >= old_graze:
                # Count actual native events only. Repeated proximity, resets,
                # disabled counters and menu rebases never fabricate grazes.
                graze_gained = new_graze - old_graze
                self.graze_observed_ticks += ticks
                self.graze_gained += graze_gained
                self.graze_with_risk += graze_gained if max(self.threat, threat) > .1 or died else 0
        old_potential = -THREAT_WEIGHT * self.threat
        # A true terminal state has zero potential. A time-limit truncation
        # remains nonterminal and is bootstrapped by PPO.
        new_potential = 0.0 if died else -THREAT_WEIGHT * threat
        shaping = DISCOUNT * new_potential - old_potential
        displacement = np.array([frame.x - before.x, frame.y - before.y])
        keys = DIRECTIONS[int(action) % 9]
        direction = np.array([int("right" in keys) - int("left" in keys),
                              int("down" in keys) - int("up" in keys)])
        left, top, width, height = frame.bounds
        low = np.array([left, top])
        high = low + [width, height]
        position = np.array([frame.x, frame.y])
        against_wall = ((direction < 0) & (position-low <= .5) |
                        (direction > 0) & (high-position <= .5))
        blocked = against_wall & (np.abs(displacement) <= .05) & (not died)
        # A tiny inward twitch does not renew the blockage grace. Releasing
        # the wall region clears it; other intervals decay it gradually.
        released_wall = np.minimum(position-low, high-position) >= WALL_RELEASE_PIXELS
        self.blocked_ticks = np.where(released_wall, 0,
                                      np.where(blocked, self.blocked_ticks + ticks,
                                               np.maximum(0, self.blocked_ticks - ticks * .25)))
        active_axes = max(1, int(np.count_nonzero(direction)))
        blocked_fraction = float(np.count_nonzero(blocked) / active_axes)
        charged_ticks = np.where(blocked, np.minimum(ticks, np.maximum(self.blocked_ticks - BLOCKED_GRACE_TICKS, 0)), 0)
        wall_push = -BLOCKED_COST_PER_SECOND * float(charged_ticks.sum()) / (60 * active_axes)
        self.positions.append((frame.tick, frame.x, frame.y))
        while len(self.positions) > 1 and self.positions[1][0] <= frame.tick - LOITER_WINDOW_TICKS:
            self.positions.popleft()
        positions = np.array(self.positions)
        exposure = corner_exposure(frame)
        edge = edge_exposure(frame)
        loiter = residence_level(frame, positions)
        components = {"survival": SURVIVAL_PER_SECOND * ticks / 60, "death": -DEATH_COST if died else 0.0,
                      "edge_loiter": -LOITER_COST_PER_SECOND * loiter * ticks / 60,
                      "threat_shaping": shaping, "wall_push": wall_push}
        resource_rewards = dict.fromkeys(RESOURCE_COMPONENTS, 0.0)
        bomb_use = 0.0
        if not died and before.resources is not None and frame.resources is not None:
            old, new = before.resources, frame.resources
            point_gain = new.point_items - old.point_items
            if point_gain < 0 and old.point_items > 64511 and new.point_items < 1024:
                point_gain += 65536
            power_gain = max(0, new.power - old.power)
            point_gain = max(0, min(1024, point_gain))
            self.power_gained += power_gain
            self.point_items_gained += point_gain
            resource_rewards = {"power": POWER_CREDIT * power_gain,
                                "bomb_pickup": BOMB_CREDIT * max(0, new.bombs - old.bombs),
                                "life_pickup": LIFE_CREDIT * max(0, new.lives - old.lives),
                                # Raw score includes direct graze points and
                                # graze-dependent bonuses. It is display/context
                                # only; actual point-item gains remain rewarded.
                                "score": 0.0,
                                "point_items": POINT_CREDIT * point_gain}
            total = sum(resource_rewards.values())
            credit = (RESOURCE_BUDGET * math.exp(-self.resource_raw_credit / RESOURCE_BUDGET)
                      * -math.expm1(-total / RESOURCE_BUDGET))
            scale = credit / max(total, 1e-12)
            resource_rewards = {name: amount * scale for name, amount in resource_rewards.items()}
            self.resource_bonus_used += sum(resource_rewards.values())
            self.resource_raw_credit += total
            bomb_use = -.1 * max(0, old.bombs - new.bombs)
        potential = 0.0 if died else item_potential(frame, threat, self.resource_raw_credit)
        aimed = 0.0 if died else aim_potential(frame, threat)
        power_progress=0. if died else power_potential(frame)
        enemy_damage, boss_damage, damage_reward = damage_feedback(before, frame) if not died else (0.0, 0.0, 0.0)
        components.update(resource_rewards, bomb_use=bomb_use,
                          power_progress=DISCOUNT*power_progress-self.power_potential,
                          damage=damage_reward, aim_shaping=DISCOUNT * aimed - self.aim_potential,
                          item_shaping=DISCOUNT * potential - self.item_potential)
        components = {name: float(value) for name, value in components.items()}
        for name, value in components.items():
            self.totals[name] += value
        self.total_ticks += ticks
        self.threat_ticks += threat * ticks
        self.geometry_ticks += risk.geometry_coverage * ticks
        self.wall_push_ticks += blocked_fraction * ticks
        self.corner_ticks += exposure * ticks
        self.edge_ticks += edge * ticks
        self.enemy_damage += enemy_damage
        self.boss_damage += boss_damage
        self.loiter_ticks += loiter * ticks
        self.moving_ticks += ticks * (float(np.linalg.norm(displacement)) > .05)
        self.item_potential = potential
        self.aim_potential = aimed
        self.power_potential = power_progress
        self.frame, self.threat, self.previous_direction = frame, threat, direction
        return RewardStep(float(sum(components.values())), components, threat,
                          float(displacement[0]), float(displacement[1]), blocked_fraction, exposure, loiter, potential,
                          graze_gained, risk.min_clearance, risk.geometry_coverage)

    def diagnostics(self):
        return {"wall_push_seconds": self.wall_push_ticks / 60,
                "mean_threat": self.threat_ticks / max(1, self.total_ticks),
                "corner_exposure_seconds": self.corner_ticks / 60, "loiter_seconds": self.loiter_ticks / 60,
                "edge_exposure_seconds": self.edge_ticks / 60,
                "enemy_damage": self.enemy_damage, "boss_damage": self.boss_damage,
                "moving_seconds": self.moving_ticks / 60, "power_gained": self.power_gained,
                "point_items_gained": self.point_items_gained,
                "graze_gained": self.graze_gained, "graze_observed_seconds": self.graze_observed_ticks / 60,
                "graze_with_risk": self.graze_with_risk,
                "geometry_coverage": self.geometry_ticks / max(1, self.total_ticks),
                **{"reward_" + name: value for name, value in self.totals.items()}}
