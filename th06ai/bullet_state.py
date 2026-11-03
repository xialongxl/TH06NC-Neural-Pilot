"""Visible bullet geometry shared by observations and reward diagnostics.

Only observed current/past states are used. Projection assumes straight-line
motion and a stationary player; it is not future game state or an action rule.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .native import Frame

PREDICTION_TICKS = 24.0
UNKNOWN_COLLISION_RADIUS = 16.0


def visible_velocities(frame: Frame, previous: Frame | None):
    """Match visible pool slots; missing/recycled motion stays unknown."""
    velocities = np.zeros((len(frame.bullets), 2), dtype=np.float32)
    known = np.zeros(len(frame.bullets), dtype=bool)
    if (previous is None or not len(frame.bullets) or not len(previous.bullets)
            or not 0 < frame.tick - previous.tick <= 15):
        return velocities, known
    ticks = frame.tick - previous.tick
    old_slots = previous.bullets[:, 0].astype(np.int64)
    slots = frame.bullets[:, 0].astype(np.int64)
    order = np.argsort(old_slots, kind="stable")
    matches = np.searchsorted(old_slots[order], slots)
    current_indices = np.flatnonzero(matches < len(order))
    previous_indices = order[matches[current_indices]]
    found = old_slots[previous_indices] == slots[current_indices]
    current_indices, previous_indices = current_indices[found], previous_indices[found]
    delta = (frame.bullets[current_indices, 1:3] - previous.bullets[previous_indices, 1:3]) / ticks
    plausible = np.all(np.abs(delta) < 20, axis=1)
    matched = current_indices[plausible]
    velocities[matched] = delta[plausible]
    known[matched] = True
    return velocities, known


@dataclass(frozen=True)
class BulletState:
    relative: np.ndarray
    velocity: np.ndarray
    known_motion: np.ndarray
    radius: np.ndarray
    known_geometry: np.ndarray
    closest_tick: np.ndarray
    current_clearance: np.ndarray
    predicted_clearance: np.ndarray


class BulletStateCache:
    """Two immutable snapshot pairs, not a tick-based cache across refreshes."""
    def __init__(self):
        self.entries = []

    @staticmethod
    def _key(frame):
        if frame is None:
            return None
        def array_key(value):
            if value is None:return None
            # Native captures are read-only; writable external snapshots also
            # need exact contents, including same-tick in-place edits.
            return id(value), value.tobytes() if value.flags.writeable else None
        return (frame.tick, frame.x, frame.y, frame.player_radius,
                array_key(frame.bullets), array_key(frame.bullet_sizes))

    def get(self, frame, previous=None):
        key = (self._key(frame), self._key(previous))
        for cached_key, _, state in self.entries:
            if cached_key == key:
                return state
        state = visible_bullet_state(frame, previous)
        # Keep references alive so CPython cannot recycle an array id in a key.
        self.entries = [(key, (frame, previous), state), *self.entries[:1]]
        return state

    def clear(self):
        self.entries.clear()


def visible_bullet_state(frame: Frame, previous: Frame | None = None) -> BulletState:
    relative = frame.bullets[:, 1:3].astype(np.float64) - [frame.x, frame.y]
    velocity, known_motion = visible_velocities(frame, previous)
    velocity64 = velocity.astype(np.float64)
    radius = np.full(len(relative), UNKNOWN_COLLISION_RADIUS, dtype=np.float64)
    known_geometry = np.zeros(len(relative), dtype=bool)
    if frame.bullet_sizes is not None and np.shape(frame.bullet_sizes) == (len(relative), 2):
        sizes = np.asarray(frame.bullet_sizes, dtype=np.float64)
        if frame.player_radius is not None and np.isfinite(frame.player_radius) and 0 < frame.player_radius <= 32:
            known_geometry = np.all(np.isfinite(sizes) & (sizes > 0) & (sizes <= 256), axis=1)
            radius[known_geometry] = sizes[known_geometry].min(axis=1) * .5 + frame.player_radius
    speed_squared = np.sum(velocity64 * velocity64, axis=1)
    approach = -np.sum(relative * velocity64, axis=1)
    closest_tick = np.clip(approach / np.maximum(speed_squared, 1e-12), 0, PREDICTION_TICKS)
    closest_tick[~known_motion] = 0
    closest = relative + velocity64 * closest_tick[:, None]
    return BulletState(relative, velocity, known_motion, radius, known_geometry, closest_tick,
                       np.linalg.norm(relative, axis=1)-radius,
                       np.linalg.norm(closest, axis=1)-radius)
