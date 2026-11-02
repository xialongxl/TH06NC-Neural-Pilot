from __future__ import annotations

import numpy as np
from dataclasses import replace

from .native import Frame, CAPACITY
from .bullet_state import visible_velocities, visible_bullet_state, PREDICTION_TICKS
from .planner_spec import PLAN_MOVEMENTS, PLAN_FEATURES
from .input_timing import clock_ns
from .stage import STAGE_FEATURES, stage_vector, context_changed
from .navigation import NAV_FEATURES, DECISION_FEATURES

NEAREST = 64
FULL_BULLETS = CAPACITY
FULL_BULLET_FIELDS = ("dx", "dy", "vx", "vy", "motion_known", "collision_radius",
                      "geometry_known", "closest_time", "predicted_clearance", "valid")
FULL_BULLET_FEATURES = len(FULL_BULLET_FIELDS)
NEAREST_ITEMS = 32
ITEM_FEATURES = 12
RESOURCE_FEATURES = 5
NEAREST_ENEMIES = 32
ENEMY_FEATURES = 8
CONTROL_FEATURES = 28
OBSERVATION_VERSION = 8
ACTION_TIMING_VERSION = 3
ACTION_COUNT = 36


def make_observation_space(version=OBSERVATION_VERSION):
    from gymnasium import spaces
    fields = {"bullets": spaces.Box(-2, 2, (NEAREST, 5), np.float32),
                        "player": spaces.Box(0, 1, (4,), np.float32),
                        "items": spaces.Box(-2, 2, (NEAREST_ITEMS, ITEM_FEATURES), np.float32),
                        "resources": spaces.Box(0, 1, (RESOURCE_FEATURES,), np.float32)}
    if version >= 3:
        fields["enemies"] = spaces.Box(-2, 2, (NEAREST_ENEMIES, ENEMY_FEATURES), np.float32)
    if version >= 4:
        fields["control"] = spaces.Box(-2, 2, (CONTROL_FEATURES,), np.float32)
    if version >= 5:
        fields["full_bullets"] = spaces.Box(-2, 2, (FULL_BULLETS, FULL_BULLET_FEATURES), np.float32)
    if version >= 6:
        fields["planning"] = spaces.Box(-1, 1, (PLAN_MOVEMENTS, PLAN_FEATURES), np.float32)
    if version >= 7:
        fields['stage'] = spaces.Box(0, 1, (STAGE_FEATURES,), np.float32)
    if version >= 8:
        fields['navigation'] = spaces.Box(-1,1,(PLAN_MOVEMENTS,NAV_FEATURES),np.float32)
        fields['decision'] = spaces.Box(0,1,(DECISION_FEATURES,),np.float32)
    return spaces.Dict(fields)


class ObservationBuilder:
    """Geometry/temporal preprocessing only. Never selects or modifies actions."""
    def __init__(self, planner=None, goal='survival'):
        self.planner = planner
        if goal not in ('survival','nmnb'):raise ValueError('未知会话目标')
        self.goal = goal
        self.reset()

    def reset(self):
        self.previous = None
        self._motion_previous = None
        if self.planner is not None:
            self.planner.reset_timing()
        self.control = np.zeros(CONTROL_FEATURES, np.float32)
        self.last_plan = None
        self.last_build_ms = None

    def set_control(self, action, *, blocked_ticks=(0, 0), residence=0., resource_saturation=0.):
        """Observed control/reward history, never a recommendation or override.

        0:18 last requested movement; 18 bomb request; 19 request known;
        20:22 measured self velocity; 22 sample duration; 23 motion known;
        24:26 wall-blockage memory; 26 residence age; 27 resource saturation.
        """
        self.control[:20] = 0
        if action is not None:
            if not 0 <= int(action) < ACTION_COUNT:
                raise ValueError("模型动作必须是 0 到 35")
            self.control[int(action) % 18] = 1
            self.control[18:20] = (int(action) >= 18, 1)
        self.control[24:26] = np.clip(np.asarray(blocked_ticks) / 180, 0, 1)
        self.control[26] = np.clip(residence, 0, 1)
        self.control[27] = np.clip(resource_saturation, 0, 1)

    def build(self, frame: Frame) -> dict[str, np.ndarray]:
        started_ns = clock_ns()
        if self.previous is not None and (frame.tick < self.previous.tick or context_changed(self.previous, frame)):
            self.reset()
        # Refreshing a paused/resumed observation at the same native tick must
        # not erase measured velocity. Keep the preceding distinct sample.
        previous = (self._motion_previous if self.previous is not None and frame.tick == self.previous.tick
                    else self.previous)
        width = frame.bounds[2] + 2 * frame.bounds[0]
        height = frame.bounds[3] + 2 * frame.bounds[1]
        if width <= 0 or height <= 0:
            raise RuntimeError("场地尺寸不可用")
        bullets = np.zeros((NEAREST, 5), dtype=np.float32)
        full_bullets = np.zeros((FULL_BULLETS, FULL_BULLET_FEATURES), dtype=np.float32)
        current = frame.bullets
        state = None
        if len(current):
            relative = current[:, 1:3] - np.array([frame.x, frame.y], np.float32)
            indices = np.argsort(np.sum(relative * relative, axis=1), kind="stable")[:NEAREST]
            n = len(indices)
            bullets[:n, :2] = relative[indices] / (width, height)
            state = visible_bullet_state(frame, previous)
            bullets[:n, 2:4] = state.velocity[indices] / 16
            bullets[:n, 4] = 1
            # Keep ALL visible rows in the verified pool. The separate near
            # branch retains its original geometry and pooling distribution.
            n_full = len(current)
            full_bullets[:n_full, :2] = relative / (width, height)
            full_bullets[:n_full, 2:4] = state.velocity / 16
            full_bullets[:n_full, 4] = state.known_motion
            full_bullets[:n_full, 5] = state.radius / 128
            full_bullets[:n_full, 6] = state.known_geometry
            full_bullets[:n_full, 7] = state.closest_tick / PREDICTION_TICKS
            full_bullets[:n_full, 8] = state.predicted_clearance / 512
            full_bullets[:n_full, 9] = 1
        player = np.array([frame.x / width, frame.y / height,
                           float(frame.focused), min(len(current) / 640, 1)], dtype=np.float32)
        items = np.zeros((NEAREST_ITEMS, ITEM_FEATURES), dtype=np.float32)
        if len(frame.items):
            relative_items = frame.items[:, 1:3] - [frame.x, frame.y]
            from .objectives import select_items
            selected = select_items(frame,NEAREST_ITEMS)
            n = len(selected)
            types = frame.items[selected, 3].astype(np.int64)
            if np.any((types < 0) | (types > 6)):
                raise RuntimeError("模型观测收到未支持的道具类型")
            items[:n, :2] = relative_items[selected] / (width, height)
            items[:n, 2:4] = frame.items[selected, 4:6] / 16
            items[np.arange(n), 4 + types] = 1
            items[:n, -1] = 1
        resources = np.zeros(RESOURCE_FEATURES, dtype=np.float32)
        enemies = np.zeros((NEAREST_ENEMIES, ENEMY_FEATURES), dtype=np.float32)
        if len(frame.enemies):
            current_enemies = frame.enemies
            relative_enemies = current_enemies[:, 1:3] - [frame.x, frame.y]
            selected = np.lexsort((np.sum(relative_enemies**2, axis=1), -current_enemies[:, 7]))[:NEAREST_ENEMIES]
            n = len(selected)
            enemies[:n, :2] = relative_enemies[selected] / (width, height)
            previous_enemies = {int(row[0]): row for row in previous.enemies} if previous is not None else {}
            ticks = frame.tick - previous.tick if previous is not None else 0
            for index, slot in enumerate(selected):
                row = current_enemies[slot]
                old = previous_enemies.get(int(row[0]))
                if old is not None and 0 < ticks <= 15 and np.array_equal(row[4:6], old[4:6]):
                    velocity = (row[1:3] - old[1:3]) / ticks
                    if np.max(np.abs(velocity)) < 20:
                        enemies[index, 2:4] = velocity / 16
            enemies[:n, 4] = current_enemies[selected, 3] / np.maximum(1, current_enemies[selected, 4])
            enemies[:n, 5] = current_enemies[selected, 7]
            enemies[:n, 6] = ((current_enemies[selected, 6].astype(np.int64) & 0x11) == 0x11)
            enemies[:n, 7] = 1
        if frame.resources is not None:
            r = frame.resources
            resources[:] = [r.power / 128, r.bombs / 8, r.lives / 8,
                            np.log1p(r.score) / np.log1p(1e12), r.point_items / 1000]
        control = self.control.copy()
        ticks = frame.tick - previous.tick if previous is not None else 0
        control[20:24] = 0
        if ticks > 0:
            control[22] = min(ticks / 60, 1)
            velocity = (np.array([frame.x, frame.y]) - [previous.x, previous.y]) / ticks
            if ticks <= 15 and np.all(np.abs(velocity) < 20):
                control[20:22] = velocity / 16
                control[23] = 1
        self._motion_previous = previous
        self.previous = replace(frame)
        planning = np.zeros((PLAN_MOVEMENTS, PLAN_FEATURES), np.float32)
        navigation = np.zeros((PLAN_MOVEMENTS,NAV_FEATURES),np.float32)
        decision = np.zeros(DECISION_FEATURES,np.float32)
        if self.planner is not None:
            self.last_plan = self.planner.plan(frame, previous, control, state=state)
            planning = self.last_plan.features
            if self.last_plan.navigation is not None:navigation = self.last_plan.navigation
            if self.last_plan.decision is not None:decision = self.last_plan.decision.copy()
        decision[0] = 1
        decision[4] = float(self.goal!='nmnb')
        result = {"bullets": bullets, "player": np.clip(player, 0, 1),
                "items": np.clip(items, -2, 2), "resources": np.clip(resources, 0, 1),
                "enemies": np.clip(enemies, -2, 2), "control": np.clip(control, -2, 2),
                "full_bullets": np.clip(full_bullets, -2, 2), "planning": planning,
                'stage': stage_vector(frame.stage_id), 'navigation':navigation,'decision':decision}
        self.last_build_ms = (clock_ns()-started_ns)/1e6
        return result


DIRECTIONS = ((), ("up",), ("down",), ("left",), ("right",),
              ("up", "left"), ("up", "right"), ("down", "left"), ("down", "right"))


def action_keys(action: int) -> set[str]:
    if not 0 <= int(action) < ACTION_COUNT:
        raise ValueError("模型动作必须是 0 到 35")
    keys = {"z", *DIRECTIONS[int(action) % 9]}
    if (int(action) // 9) % 2:
        keys.add("shift")
    if int(action) >= 18:
        keys.add("x")
    return keys
