"""Visible enemy decoding and conservative damage accounting; no action rules."""
from __future__ import annotations
from .i18n import msg

import math
import numpy as np

ENEMY_POOL = 0xAEE0B8
ENEMY_CAPACITY = 256
ENEMY_STRIDE = 0x10B0
BOSS_POINTERS = 0xBFA198
ENEMY_DTYPE = np.dtype({"names": ["x", "y", "active", "flags", "hp", "max_hp", "phase_hp"],
                        "formats": ["<f4", "<f4", "i1", "u1", "<i4", "<i4", "<i4"],
                        "offsets": [0xB0, 0xB4, 0xBC, 0xBD, 0x234, 0x238, 0x23C],
                        "itemsize": ENEMY_STRIDE})
BODY_DTYPE = np.dtype({"names": ["x", "y", "active", "flags", "extra_flags", "width", "height"],
                       "formats": ["<f4", "<f4", "i1", "u1", "u1", "<f4", "<f4"],
                       "offsets": [0xB0, 0xB4, 0xBC, 0xBD, 0xBE, 0x2C0, 0x2C4],
                       "itemsize": ENEMY_STRIDE})
# Own-build 0x3933c scales the full hit size by 2/3; 0x6c09e
# halves that size for the axis-aligned contact rectangle.
BODY_HALF_SCALE = float(np.float32(2/3) * np.float32(.5))


def decode_enemy_bodies(raw, width, height, player_radius=0.):
    """slot,x,y,half-width,half-height,geometry-known for contact obstacles.

    Body contact gates are independent of shootability, HP and the Boss HUD.
    Include rectangles intersecting the visible field even with offscreen
    centers. Missing sizes stay unknown, never silently become zero hazards.
    """
    if len(raw) != ENEMY_CAPACITY*ENEMY_STRIDE:
        raise RuntimeError(msg('backend.unexpected_enemy_collision_pool_size'))
    data = np.frombuffer(raw, BODY_DTYPE)
    contact = ((data["active"] < 0) & ((data["flags"] & 7) == 7)
               & ((data["extra_flags"] & 8) == 0))
    slots = np.flatnonzero(contact)
    if not slots.size:
        return np.empty((0, 6), np.float32)
    x, y = data["x"][slots], data["y"][slots]
    sizes = np.column_stack((data["width"][slots], data["height"][slots]))
    known = np.all(np.isfinite(sizes) & (sizes >= 0) & (sizes <= 4096), axis=1)
    half = sizes*BODY_HALF_SCALE
    # Unknown geometry is represented explicitly. A whole-field envelope is
    # only an uncertainty fallback, not a guessed physical enemy radius.
    visible_half = np.where(known[:, None], half, [width, height])
    margin = float(player_radius)
    visible = (np.isfinite(x) & np.isfinite(y)
               & (x+visible_half[:, 0]+margin >= 0) & (x-visible_half[:, 0]-margin <= width)
               & (y+visible_half[:, 1]+margin >= 0) & (y-visible_half[:, 1]-margin <= height))
    selected_half = half[visible]
    selected_half[~known[visible]] = np.nan
    return np.column_stack((slots[visible], x[visible], y[visible], selected_half, known[visible])).astype(np.float32)


def decode_enemies(raw, width, height, boss_rvas=()):
    """Rows: slot, x, y, HP, max HP, phase HP reference, combat flags, Boss."""
    if len(raw) != ENEMY_CAPACITY * ENEMY_STRIDE:
        raise RuntimeError(msg('backend.unexpected_enemy_pool_size'))
    data = np.frombuffer(raw, dtype=ENEMY_DTYPE)
    slots = np.flatnonzero(data["active"] < 0)
    if not slots.size:
        return np.empty((0, 8), np.float32)
    x, y = data["x"][slots], data["y"][slots]
    if not np.all(np.isfinite(x) & np.isfinite(y)):
        raise RuntimeError(msg('backend.invalid_coordinates_for_an_active_enemy'))
    hp, max_hp = data["hp"][slots], data["max_hp"][slots]
    visible = ((x >= 0) & (x <= width) & (y >= 0) & (y <= height)
               & (hp > 0) & (max_hp > 0) & (hp <= max_hp))
    slots = slots[visible]
    flags = data["flags"][slots]
    boss = (flags & 8) != 0
    if boss.any():
        boss &= np.isin(ENEMY_POOL + slots * ENEMY_STRIDE, boss_rvas)
    return np.column_stack((slots, x[visible], y[visible], hp[visible], max_hp[visible],
                            data["phase_hp"][slots], flags, boss)).astype(np.float32)


def damage_feedback(before, after):
    """Count only plausible nonterminal HP loss in a stable damageable phase.

    A disappearance, HP reset, phase change, invulnerability, teleport or large
    scripted drop supplies no damage credit. Final-hit/clear bonuses are not
    inferred from disappearance. Up to 10+70 damage/native tick is pinned in
    this build's player-hit handlers; other changes are not credited.
    """
    ticks = after.tick - before.tick
    if not 0 < ticks <= 15:
        return 0.0, 0.0, 0.0
    old = {int(row[0]): row for row in before.enemies}
    regular_damage = boss_damage = reward = 0.0
    for row in after.enemies:
        previous = old.get(int(row[0]))
        if previous is None or row[3] <= 0 or previous[3] <= 0 or not ((int(row[6]) & 0x11) == 0x11 and (int(previous[6]) & 0x11) == 0x11):
            continue
        if not np.array_equal(row[4:6], previous[4:6]) or row[7] != previous[7]:
            continue
        delta = float(previous[3] - row[3])
        if not 0 < delta <= 80 * ticks or np.linalg.norm(row[1:3] - previous[1:3]) > 20 * ticks:
            continue
        if row[7]:
            boss_damage += delta
            reward += 4.0 * delta / float(row[4])
        else:
            regular_damage += delta
            reward += .25 * delta / float(row[4])
    return regular_damage, boss_damage, reward


def aim_potential(frame, threat):
    candidates = [row for row in frame.enemies if row[2] < frame.y and (int(row[6]) & 0x11) == 0x11]
    if not candidates:
        return 0.0
    bosses = [row for row in candidates if row[7]]
    targets = bosses or candidates
    alignment = max(math.exp(-abs(float(row[1]) - frame.x) / 64) for row in targets)
    return .01 * alignment * max(.05, 1 - threat)
