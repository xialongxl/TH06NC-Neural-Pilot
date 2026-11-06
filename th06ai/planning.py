"""Receding-horizon geometry search. Never sends keys or replaces an action.

The cached forecast joins the policy's actual categorical distribution, so PPO
samples, executes and evaluates the same action and log probability.
"""
from __future__ import annotations
from .i18n import msg

from dataclasses import dataclass, replace
import time

import numpy as np

from .bullet_state import visible_bullet_state
from .body_state import visible_body_state
from .input_timing import TimingWindow, FRAME_MS, clock_ns
from .navigation import NAV_FIELDS, DECISION_FIELDS, MOBILITY_WEIGHT, EXIT_WEIGHT, EXIT_PROBE_LIMIT, EXIT_TICKS
from .planner_spec import (PlanningConfig, PLANNER_VERSION, PLAN_MOVEMENTS, PLAN_FEATURES,
                           PLAN_LOGIT_SCALE, HORIZON_TICKS, DELAY_TICKS, SEGMENT_TICKS, MAX_SEARCH_NODES)

DIRECTIONS = np.array([(0, 0), (0, -1), (0, 1), (-1, 0), (1, 0),
                       (-1, -1), (1, -1), (-1, 1), (1, 1)], dtype=np.float32)
DIRECTIONS /= np.maximum(1, np.linalg.norm(DIRECTIONS, axis=1))[:, None]
_SEARCH_DIRECTIONS = np.vstack((DIRECTIONS,DIRECTIONS))
CLEARANCE = 6.
COLLISION_PADDING = 2.
SELF_SPEED_ERROR = .25
UNKNOWN_MOTION_ERROR = 4.  # Practical forecast uncertainty, not a verified speed bound.
_RECTANGLE_I = np.array([0, 0, 0, 1, 1, 2])
_RECTANGLE_J = np.array([1, 2, 3, 2, 3, 3])


@dataclass(frozen=True)
class PlanResult:
    features: np.ndarray
    paths: np.ndarray
    diagnostics: dict
    clearances: np.ndarray | None = None
    collision_times: np.ndarray | None = None
    completed_ns: int = 0
    started_ns: int = 0
    navigation: np.ndarray | None = None
    decision: np.ndarray | None = None


def rectangle_sweep(relative, velocity, extents, length):
    """Continuous slab entry and minimum signed L-infinity separation.

    Expanding a contact rectangle by the player's circular radius is a
    conservative envelope of the game's rounded rectangle. Its corners can
    exclude a close pass, but cannot create a corner gap through an enemy.
    """
    moving = np.abs(velocity) > 1e-7
    lower = np.divide(-extents-relative, velocity, out=np.full_like(relative, -np.inf), where=moving)
    upper = np.divide(extents-relative, velocity, out=np.full_like(relative, np.inf), where=moving)
    entry_axis, exit_axis = np.minimum(lower, upper), np.maximum(lower, upper)
    outside = (~moving) & (np.abs(relative) > extents)
    entry_axis[outside], exit_axis[outside] = np.inf, -np.inf
    entry = np.maximum(entry_axis.max(axis=2), 0)
    exit_time = np.minimum(exit_axis.min(axis=2), length[:, None])
    first = np.where(entry <= exit_time, entry, np.inf)
    constants = np.stack((relative[..., 0]-extents[..., 0], -relative[..., 0]-extents[..., 0],
                          relative[..., 1]-extents[..., 1], -relative[..., 1]-extents[..., 1]), axis=-1)
    slopes = np.stack((velocity[..., 0], -velocity[..., 0], velocity[..., 1], -velocity[..., 1]), axis=-1)
    minimum = np.minimum(constants.max(axis=2), (constants+slopes*length[:, None, None]).max(axis=2))
    # A convex piecewise-linear maximum reaches its minimum at an endpoint
    # or an intersection of two of its four affine pieces.
    denominator = slopes[..., _RECTANGLE_I]-slopes[..., _RECTANGLE_J]
    t = np.divide(constants[..., _RECTANGLE_J]-constants[..., _RECTANGLE_I], denominator,
                  out=np.zeros_like(denominator), where=np.abs(denominator) > 1e-7)
    t = np.clip(t, 0, length[:, None, None])
    gap_x = np.abs(relative[..., 0, None]+velocity[..., 0, None]*t)-extents[..., 0, None]
    gap_y = np.abs(relative[..., 1, None]+velocity[..., 1, None]*t)-extents[..., 1, None]
    minimum = np.minimum(minimum, np.maximum(gap_x, gap_y).min(axis=2))
    return minimum.min(axis=1), first.min(axis=1)


def swept_segments(starts, velocities, duration, start_tick, low, high, bullets, bullet_velocity, radii, uncertainty, bodies=None):
    """Continuous relative-motion tests, split at each axis's wall-hit time.

    Testing only endpoints misses bullets crossed between observations. A
    clamped diagonal also changes its velocity when one axis reaches a wall.
    """
    starts = np.asarray(starts, np.float32)
    velocities = np.asarray(velocities, np.float32)
    durations = np.broadcast_to(np.asarray(duration, np.float32), (len(starts),))
    start_ticks = np.broadcast_to(np.asarray(start_tick, np.float32), (len(starts),))
    ends = np.clip(starts+velocities*durations[:, None], low, high)
    requested = np.linalg.norm(velocities*durations[:, None], axis=1)
    blocked = 1-np.linalg.norm(ends-starts, axis=1)/np.maximum(requested, 1e-6)
    blocked[requested < 1e-6] = 0
    minimum = np.full(len(starts), np.inf, np.float32)
    collision = np.full(len(starts), np.inf, np.float32)
    if not len(bullets) and (bodies is None or not len(bodies.positions)):
        return ends, minimum, collision, np.clip(blocked, 0, 1)
    wall = np.where(velocities > 0, high, low)
    hit = np.divide(wall-starts, velocities,
                    out=np.broadcast_to(durations[:, None], starts.shape).copy(), where=np.abs(velocities) > 1e-6)
    times = np.sort(np.column_stack((np.zeros(len(starts), np.float32),
                    np.minimum(np.maximum(hit, 0), durations[:, None]), durations)), axis=1)
    # Bound temporary candidate/obstacle pairs, rather than forcing tiny
    # 128-row batches even when each row contains only a few obstacles.
    work = len(bullets)+4*(len(bodies.positions) if bodies is not None else 0)
    batch_rows = max(32, min(2048, 131072//max(1, work)))
    for piece in range(3):
        live = np.flatnonzero(times[:, piece+1]-times[:, piece] > 1e-6)
        for offset in range(0, len(live), batch_rows):
            rows = live[offset:offset+batch_rows]
            begin, end = times[rows, piece], times[rows, piece+1]
            absolute_begin, absolute_end = start_ticks[rows]+begin, start_ticks[rows]+end
            length = end-begin
            player = np.clip(starts[rows]+velocities[rows]*begin[:, None], low, high)
            finish = np.clip(starts[rows]+velocities[rows]*end[:, None], low, high)
            actual_velocity = (finish-player)/length[:, None]
            if len(bullets):
                relative = bullets[None]+bullet_velocity[None]*absolute_begin[:, None, None]-player[:, None]
                relative_velocity = bullet_velocity[None]-actual_velocity[:, None]
                radius = radii[None]+COLLISION_PADDING+uncertainty[None]*absolute_end[:, None]
                a = np.einsum('ijk,ijk->ij', relative_velocity, relative_velocity)
                b = np.einsum('ijk,ijk->ij', relative, relative_velocity)
                c = np.einsum('ijk,ijk->ij', relative, relative)-radius*radius
                closest = np.clip(-b/np.maximum(a, 1e-12), 0, length[:, None])
                near = relative+relative_velocity*closest[..., None]
                clearance = np.sqrt(np.einsum('ijk,ijk->ij', near, near))-radius
                minimum[rows] = np.minimum(minimum[rows], clearance.min(axis=1))
                discriminant = b*b-a*c
                entry = (-b-np.sqrt(np.maximum(discriminant, 0)))/np.maximum(a, 1e-12)
                intersects = (a > 1e-12) & (discriminant >= 0) & (entry >= 0) & (entry <= length[:, None])
                entry = np.where(c <= 0, 0., np.where(intersects, entry, np.inf))
                collision[rows] = np.minimum(collision[rows], absolute_begin+entry.min(axis=1))
            if bodies is not None and len(bodies.positions):
                relative = bodies.positions[None]+bodies.velocity[None]*absolute_begin[:, None, None]-player[:, None]
                relative_velocity = bodies.velocity[None]-actual_velocity[:, None]
                angles=getattr(bodies,'angles',None)
                if angles is not None and np.any(angles):
                    c,s=np.cos(angles),np.sin(angles)
                    def rotate(v):
                        return np.stack((v[...,0]*c+v[...,1]*s,-v[...,0]*s+v[...,1]*c),axis=-1)
                    relative,relative_velocity=rotate(relative),rotate(relative_velocity)
                padding = (bodies.player_radius+COLLISION_PADDING
                           +bodies.uncertainty[None]*absolute_end[:, None])
                extents = bodies.half_extents[None]+padding[..., None]
                clearance, entry = rectangle_sweep(relative, relative_velocity, extents, length)
                minimum[rows] = np.minimum(minimum[rows], clearance)
                collision[rows] = np.minimum(collision[rows], absolute_begin+entry)
    return ends, minimum, collision, np.clip(blocked, 0, 1)


class RollingPlanner:
    def __init__(self, settings=None, *, fast_speed=4., slow_speed=2.):
        self.settings = settings or PlanningConfig()
        from .acceleration import extension,search_extension
        available=extension() is not None
        if self.settings.collision_backend=='rust' and not available:
            raise RuntimeError(msg('backend.build_the_rust_collision_extension_or_select'))
        self.collision_backend='rust' if available and self.settings.collision_backend in ('rust','auto') else 'python'
        self.search_backend='rust' if self.collision_backend=='rust' and search_extension() is not None else 'python'
        try:
            self.fast_speed, self.slow_speed = float(fast_speed), float(slow_speed)
        except (TypeError, ValueError):
            self.fast_speed, self.slow_speed = 4., 2.
        if not 0 < self.slow_speed <= self.fast_speed < 20:
            self.fast_speed, self.slow_speed = 4., 2.
        self.velocities = np.vstack((DIRECTIONS*self.fast_speed, DIRECTIONS*self.slow_speed))
        self.last_elapsed_ms = 0.
        self.last_visible = 0
        self.tail_times = TimingWindow()
        self.confirmation_times = TimingWindow()
        self.switch_times = TimingWindow()

    def _sweep(self, *args, **kwargs):
        from .acceleration import rust_sweep
        if self.collision_backend=='rust':
            return rust_sweep(*args, **kwargs)
        return swept_segments(*args, **kwargs)

    def _rank_warning_fallback(self, result, start, pending, low, high, bodies, damaging):
        diagnostics = result.diagnostics
        if (not diagnostics['all_routes_risky'] or not diagnostics['timing_interval_complete']
                or diagnostics['immediate_contact_free_roots'] != 0 or damaging.all()):
            return result

        # The nominal probe covers known damaging hazards at every tested
        # switch time. Its entire short window lies inside the horizon.
        nav = result.navigation
        nominal = (nav[:, NAV_FIELDS.index('nominal_collision')] >= 1.) & (result.features[:, 4] == 0)
        if not nominal.any():
            return result

        live = bodies.subset(damaging)
        if len(live.positions):
            live = replace(live, uncertainty=np.full_like(live.uncertainty, SELF_SPEED_ERROR))
            empty = np.empty((0, 2), np.float32)
            live_free, _, _, _ = self._timing_probe(start, pending, diagnostics['tested_switch_delays'],
                diagnostics['timing_probe_hold_ticks'], low, high, empty, empty, np.empty(0), np.empty(0), live)
            nominal &= live_free

        old_allowed = result.features[:, 1] > 0
        if nominal.any() and (old_allowed & ~nominal).any():
            allowed = old_allowed & nominal
            if not allowed.any():
                # Keep the full forecast, including warning beams, for ranking
                # when the old fallback has no direction left in this subset.
                horizon = diagnostics['horizon_ticks']
                hit = nav[:, NAV_FIELDS.index('action_collision')] * horizon
                allowed = nominal & (hit >= hit[nominal].max() - .5)
                clear = nav[:, NAV_FIELDS.index('action_clearance')] * 64
                allowed &= clear >= clear[allowed].max() - 1.
            features = result.features.copy()
            features[:, 1] = allowed
            best = diagnostics['suggested_movement']
            if not allowed[best]:
                utility = features[:, 6]
                best = int(np.argmax(np.where(allowed, utility, -np.inf)))
                features[:, 7] = np.clip((utility - utility[best]) * (64 / PLAN_LOGIT_SCALE), -1, 0)
            decision = result.decision.copy()
            decision[DECISION_FIELDS.index('safe_exit_fraction')] = nav[best, NAV_FIELDS.index('exit_fraction')]
            diagnostics = {**diagnostics, 'active_threat_priority': True, 'allowed_roots': int(allowed.sum()),
                           'suggested_movement': best,
                           'wall_space': float(nav[best, NAV_FIELDS.index('wall_space')]),
                           'exit_fraction': float(nav[best, NAV_FIELDS.index('exit_fraction')]),
                           'exit_measured': bool(nav[best, NAV_FIELDS.index('exit_measured')]),
                           'estimated_clearance': float(result.clearances[best]) if np.isfinite(result.clearances[best]) else None,
                           'predicted_collision_tick': float(result.collision_times[best]) if np.isfinite(result.collision_times[best]) else None}
            result = replace(result, features=features, decision=decision, diagnostics=diagnostics)
        completed = clock_ns()
        self.last_elapsed_ms = (completed - result.started_ns) / 1e6
        return replace(result, completed_ns=completed,
                       diagnostics={**result.diagnostics, 'elapsed_ms': round(self.last_elapsed_ms, 3)})

    def set_speeds(self, fast, slow):
        if not (np.isfinite(fast) and np.isfinite(slow) and 0 < slow <= fast < 20):
            raise ValueError(msg('backend.invalid_player_movement_speed'))
        if (fast, slow) != (self.fast_speed, self.slow_speed):
            self.fast_speed, self.slow_speed = float(fast), float(slow)
            self.velocities = np.vstack((DIRECTIONS*fast, DIRECTIONS*slow))
            self.reset_timing()

    def reset_timing(self):
        self.tail_times.clear()
        self.confirmation_times.clear()
        self.switch_times.clear()
        self.last_elapsed_ms = self.last_visible = 0

    def record_latency(self, timing):
        self.tail_times.add(timing.get('post_plan_to_send_ms'))
        if timing.get('movement_changed'):
            self.confirmation_times.add(timing.get('game_input_match_ms'))
        tail = timing.get('post_plan_to_send_ms')
        if tail is not None and np.isfinite(tail) and tail >= 0:
            # Keep paired end-to-end measurements. Summing two independent
            # P95s made a late-switch envelope the sole predicted trajectory.
            confirmation = timing.get('game_input_match_ms')
            if confirmation is None:
                confirmation = self.confirmation_times.quantile(.5, FRAME_MS)
            self.switch_times.add(timing.get('post_plan_to_input_ms') or tail+confirmation)

    def _timing_probe(self, start, pending, delays, hold_ticks, low, high,
                      bullets, velocity, radii, uncertainty, bodies):
        # Timing probes cover only the first held command. The full search
        # still sees every horizon-relevant obstacle; don't repeat its whole
        # 32-frame obstacle set for these much shorter checks.
        end_tick = max(delays)+hold_ticks
        reach_speed = max(self.fast_speed, float(np.linalg.norm(pending)))
        relative = bullets-start
        closest = np.clip(-np.sum(relative*velocity, axis=1)/
                          np.maximum(np.sum(velocity*velocity, axis=1), 1e-12), 0, end_tick)
        reach = reach_speed*end_tick+radii+COLLISION_PADDING+uncertainty*end_tick
        keep = np.linalg.norm(relative+velocity*closest[:, None], axis=1) <= reach
        bullets, velocity, radii, uncertainty = bullets[keep], velocity[keep], radii[keep], uncertainty[keep]
        relative = bodies.positions-start
        closest = np.clip(-np.sum(relative*bodies.velocity, axis=1)/
                          np.maximum(np.sum(bodies.velocity*bodies.velocity, axis=1), 1e-12), 0, end_tick)
        reach = (reach_speed*end_tick+np.linalg.norm(bodies.half_extents, axis=1)+bodies.player_radius
                 +COLLISION_PADDING+bodies.uncertainty*end_tick)
        bodies = bodies.subset(np.linalg.norm(relative+bodies.velocity*closest[:, None], axis=1) <= reach)
        free = np.ones(18, bool)
        hits = np.full(18, np.inf, np.float32)
        action_hits = hits.copy()
        action_clearance = hits.copy()
        if not len(bullets) and not len(bodies.positions):
            return free, hits, action_hits, action_clearance
        delays = np.asarray(sorted(set(delays)), np.float32)
        count = len(delays)
        prefix, _, prefix_hit, _ = self._sweep(np.repeat(start[None], count, axis=0),
                np.repeat(pending[None], count, axis=0), delays, 0,
                low, high, bullets, velocity, radii, uncertainty, bodies=bodies)
        _, clearance, hit, _ = self._sweep(np.repeat(prefix, 18, axis=0), np.tile(self.velocities, (count, 1)),
                hold_ticks, np.repeat(delays, 18), low, high, bullets, velocity, radii, uncertainty, bodies=bodies)
        hit, clearance = hit.reshape(count, 18), clearance.reshape(count, 18)
        full_hit = np.minimum(prefix_hit[:, None], hit)
        return np.isinf(full_hit).all(axis=0), full_hit.min(axis=0), hit.min(axis=0), clearance.min(axis=0)

    @staticmethod
    def _switch_delays(early, late):
        # Collision need not be monotone in switch time: both endpoints can
        # miss a bullet while an intermediate switch crosses it. Cover each
        # native tick in normal envelopes, with a bounded grid for outliers.
        if late-early<12:return tuple(range(early,late+1)),True
        return tuple(np.unique(np.linspace(early,late,12,dtype=int)).tolist()),False

    def _goals(self, frame, previous):
        items = frame.items
        if len(items):
            from .objectives import item_weights
            weights = item_weights(frame)
            distance = np.linalg.norm(items[:, 1:3]-[frame.x, frame.y], axis=1)
            useful = np.flatnonzero(weights > .05)
            useful = useful[np.argsort(-weights[useful]*np.exp(-distance[useful]/64), kind="stable")[:32]]
            items, weights = items[useful], weights[useful]
        else:
            weights = np.empty(0)
        enemies = frame.enemies
        targets = enemies[((enemies[:, 6].astype(int) & 0x11) == 0x11) & (enemies[:, 2] < frame.y)] if len(enemies) else enemies
        target = None
        if len(targets):
            order = np.lexsort((np.abs(targets[:, 1]-frame.x), -targets[:, 7]))
            row = targets[order[0]]
            velocity = 0.
            if previous is not None and 0 < frame.tick-previous.tick <= 15:
                found = previous.enemies[previous.enemies[:, 0] == row[0]]
                if len(found) and np.array_equal(found[0, 4:6], row[4:6]):
                    estimate = (row[1]-found[0, 1])/(frame.tick-previous.tick)
                    if abs(estimate) < 20:
                        velocity = float(estimate)
            target = (float(row[1]), velocity, 3. if row[7] else .3)
        return items, weights, target

    def _mobility(self, positions, scores, allowed, low, high, tick,
                  bullets, velocity, radii, bodies, confidence):
        """Wall reachability for all roots; bounded hazard probes for finalists."""
        reach = max(1., self.fast_speed*8)
        extent = np.minimum(high,positions+reach)-np.maximum(low,positions-reach)
        area = np.clip(np.prod(np.maximum(extent,0),axis=1)/(4*reach*reach),0,1)
        speeds = self.velocities[10:18]
        ends = np.clip(positions[:,None]+speeds[None]*EXIT_TICKS,low,high)
        moves = np.linalg.norm(ends-positions[:,None],axis=-1)>.1
        wall_exits = moves.mean(axis=1)
        exits, measured = wall_exits.copy(), np.zeros(18,bool)
        adjusted = scores+MOBILITY_WEIGHT*(area-.5)
        probes = 0
        if len(bullets) or len(bodies.positions):
            for _ in range(EXIT_PROBE_LIMIT):
                best = int(np.argmax(np.where(allowed,adjusted,-np.inf)))
                if measured[best]:break
                _,clearance,hit,_ = self._sweep(np.repeat(positions[best:best+1],8,axis=0),
                    speeds,EXIT_TICKS,tick,low,high,bullets,velocity,radii,np.zeros(len(bullets)),bodies=bodies)
                exits[best] = float((np.isinf(hit)&moves[best]).mean())
                measured[best] = True;probes += 1
                # Hazard probing only removes optimistic wall-only mobility;
                # unmeasured candidates retain an explicit unknown bit.
                adjusted[best] += EXIT_WEIGHT*confidence*(exits[best]-wall_exits[best])
        else:
            measured[:] = True
        return adjusted,area,exits,measured,probes

    @staticmethod
    def _goal_quality(starts, ends, duration, tick, items, weights, target, bounds):
        quality = np.zeros(len(starts), np.float32)
        if len(items):
            item_start = items[:, 1:3]+items[:, 4:6]*tick
            relative = item_start[None]-starts[:, None]
            velocity = items[None, :, 4:6]-(ends-starts)[:, None]/duration
            t = np.clip(-np.sum(relative*velocity, axis=2)/np.maximum(np.sum(velocity*velocity, axis=2), 1e-12), 0, duration)
            distance = np.linalg.norm(relative+velocity*t[..., None], axis=2)
            item_y = item_start[None, :, 1]+items[None, :, 5]*t
            item_x = item_start[None, :, 0]+items[None, :, 4]*t
            visible = ((item_y >= 0) & (item_y <= bounds[1]*2+bounds[3]) &
                       (item_x >= 0) & (item_x <= bounds[0]*2+bounds[2]))
            quality = np.max(weights[None]*np.exp(-distance/64-.04*(tick+t))*visible, axis=1).astype(np.float32)
        aim = np.zeros(len(starts), np.float32)
        if target is not None:
            x, vx, weight = target
            aim = (weight*np.exp(-.5*((ends[:, 0]-(x+vx*(tick+duration)))/48)**2)).astype(np.float32)
        return quality, aim

    @staticmethod
    def _edge(starts, ends, low, high):
        midpoint = (starts+ends)/2
        distance = np.minimum(midpoint-low, high-midpoint)
        exposure = np.exp(-.5*(distance/32)**2)
        return exposure.max(axis=1), exposure.prod(axis=1)

    @staticmethod
    def _branches(positions, last_actions, count, tick, bullets, velocity, radii, items, weights, target, low, high):
        if count == 18:
            return np.tile(np.arange(18), (len(positions), 1))
        goal = (low+high)/2-positions
        if len(items):
            locations = items[:, 1:3]+items[:, 4:6]*tick
            relative = locations[None]-positions[:, None]
            index = np.argmax(weights[None]*np.exp(-np.linalg.norm(relative, axis=2)/64), axis=1)
            goal = relative[np.arange(len(positions)), index]
        elif target is not None:
            goal = np.column_stack((target[0]+target[1]*tick-positions[:, 0], np.zeros(len(positions))))
        escape = goal.copy()
        if len(bullets):
            relative = bullets[None]+velocity[None]*tick-positions[:, None]
            clearance = np.linalg.norm(relative, axis=2)-radii[None]
            nearest = np.argmin(clearance, axis=1)
            close = clearance[np.arange(len(positions)), nearest] < 32
            escape[close] = -relative[np.arange(len(positions)), nearest][close]
        def heading(vectors):
            result = np.argmax(np.sum(vectors[:, None]*DIRECTIONS[None, 1:], axis=2), axis=1)+1
            result[np.linalg.norm(vectors, axis=1) < 1e-4] = 0
            return result
        goal_direction, escape_direction = heading(goal), heading(escape)
        ring = np.array([4, 8, 2, 7, 3, 5, 1, 6])
        ring_index = np.zeros(9, int)
        ring_index[ring] = np.arange(8)
        last = np.asarray(last_actions, int)
        focus, around = last//9*9, ring_index[last % 9]
        choices = np.column_stack((last, focus+escape_direction, focus, focus+goal_direction,
                (last+9) % 18, focus+ring[(around-1) % 8], focus+ring[(around+1) % 8],
                focus+ring[(around+4) % 8], np.tile(np.arange(18), (len(last), 1))))
        # Stable first occurrence per action reproduces dict.fromkeys for
        # every row, without a Python loop for every beam node.
        first = np.full((len(last), 18), choices.shape[1], int)
        np.minimum.at(first, (np.arange(len(last))[:, None], choices), np.arange(choices.shape[1])[None])
        return np.argsort(first, axis=1, kind='stable')[:, :count]

    def plan(self, frame, previous=None, control=None, state=None):
        started_ns = clock_ns()
        started = time.perf_counter()
        neutral = np.zeros((PLAN_MOVEMENTS, PLAN_FEATURES), np.float32)
        if not self.settings.enabled or frame.paused or not frame.stage_active or frame.state == 2:
            return PlanResult(neutral, np.empty((18, 0, 2), np.float32), {"enabled": False, "version": PLANNER_VERSION})
        control = np.zeros(28) if control is None else control
        state = state if state is not None else visible_bullet_state(frame, previous)
        low = np.array(frame.bounds[:2], np.float32)
        high = low+frame.bounds[2:]
        start = np.array([frame.x, frame.y], np.float32)
        captured_ns = getattr(frame, 'capture_started_ns', 0)
        age_ms = (max(0., (clock_ns()-captured_ns)/1e6) if captured_ns > 0 else
                  max(0., (time.monotonic()-frame.observed_at)*1000) if frame.observed_at > 0 else 0.)
        scene_load = len(frame.bullets)+4*(len(frame.enemy_bodies)+len(frame.lasers))
        cold_per_obstacle=.004 if self.collision_backend=='rust' else .04
        estimate_ms = max(cold_per_obstacle*scene_load, self.last_elapsed_ms*(scene_load+32)/(self.last_visible+32))
        # Age already includes capture and pre-planning work. Add predicted
        # planning, measured POST-planning work, and game-input observation
        # latency once each. The cold start retains the previous two frames.
        tail_ms = self.tail_times.upper(DELAY_TICKS*FRAME_MS/2)
        confirmation_ms = max(FRAME_MS, self.confirmation_times.upper(DELAY_TICKS*FRAME_MS/2))
        cold_switch = tail_ms+confirmation_ms
        switch_ms = self.switch_times.quantile(.5, cold_switch)
        delay = max(1, int(np.ceil((age_ms+estimate_ms+switch_ms)/FRAME_MS)))
        early = max(0, int(np.floor((age_ms+estimate_ms+
                    self.switch_times.quantile(.1, cold_switch))/FRAME_MS))-1)
        late = max(delay, int(np.ceil((age_ms+estimate_ms+
                   self.switch_times.upper(cold_switch))/FRAME_MS))+1)
        # Keep a bounded forecast window. A slow previous call reduces search
        # depth rather than increasing both horizon and computation forever.
        horizon = max(HORIZON_TICKS, late*2+SEGMENT_TICKS[0])
        root_duration = SEGMENT_TICKS[0]+delay
        remaining = horizon-delay-root_duration
        future_steps = min(3, remaining//root_duration)
        if future_steps:
            base, extra = divmod(remaining, future_steps)
            durations = (root_duration, *(base+int(i >= future_steps-extra) for i in range(future_steps)))
        else:
            durations = (horizon-delay,)
        uncertainty = (SELF_SPEED_ERROR+np.where(state.known_motion, 0., UNKNOWN_MOTION_ERROR)).astype(np.float32)
        # Triangle inequality: omit only bullets outside the player's entire
        # reachable disc under this same finite-horizon motion assumption.
        closest_tick = np.clip(-np.sum(state.relative*state.velocity, axis=1)/
                               np.maximum(np.sum(state.velocity**2, axis=1), 1e-12), 0, horizon)
        closest_distance = np.linalg.norm(state.relative+state.velocity*closest_tick[:, None], axis=1)
        relevant = closest_distance <= self.fast_speed*horizon+state.radius+COLLISION_PADDING+uncertainty*horizon
        bullets = frame.bullets[relevant, 1:3].astype(np.float32)
        velocity = state.velocity[relevant]
        radii, uncertainty = state.radius[relevant].astype(np.float32), uncertainty[relevant]
        all_bodies = visible_body_state(frame, previous)
        body_relative = all_bodies.positions-start
        closest_body_tick = np.clip(-np.sum(body_relative*all_bodies.velocity, axis=1)/
                np.maximum(np.sum(all_bodies.velocity**2, axis=1), 1e-12), 0, horizon)
        body_distance = np.linalg.norm(body_relative+all_bodies.velocity*closest_body_tick[:, None], axis=1)
        body_reach = (self.fast_speed*horizon+np.linalg.norm(all_bodies.half_extents, axis=1)
                      +all_bodies.player_radius+COLLISION_PADDING+all_bodies.uncertainty*horizon)
        body_relevant = body_distance <= body_reach
        bodies = all_bodies.subset(body_relevant)
        damaging = np.r_[np.ones(len(frame.enemy_bodies), bool), frame.lasers[:, 7] == 1][body_relevant]
        work_size = len(bullets)+4*len(bodies.positions)
        if work_size > 128 and future_steps:
            # Dense scenes use a complete one-turn forecast instead of a
            # deeper partial tree. Every candidate still reaches the full
            # horizon; a cheap prefix is never mislabeled a safe full route.
            durations = (root_duration, horizon-delay-root_duration)
            future_steps = 1
        # Bound candidate/bullet pairs per decision. All 18 immediate moves
        # remain; only future branching and beam width are narrowed in dense
        # scenes. This is a compute budget, never a sampling STOP condition.
        beam = self.settings.beam_width if work_size <= 64 else 1
        # Shorter measured latency can add a future turn. Keep it from
        # automatically spending that gain on another full 18-way expansion.
        # All immediate actions and all reachable obstacles are still tested.
        node_budget = min(MAX_SEARCH_NODES, 120000/max(1, work_size))
        branches = int(np.clip((node_budget-18)/(18*beam*max(1, future_steps)), 3, 18))
        if self.search_backend=='rust':
            from .acceleration import rust_plan
            known=state.known_motion[relevant] & state.known_geometry[relevant]
            body_known=bodies.known_motion & bodies.known_geometry & damaging
            pending=self.velocities[int(np.argmax(control[:18]))] if control[19]>.5 else np.asarray(control[20:22])*16
            items,weights,target=self._goals(frame,previous)
            switch_delays,timing_complete=self._switch_delays(early,late)
            features,paths,minimum,collision,navigation,decision,meta=rust_plan(
                start,pending,low,high,self.velocities,_SEARCH_DIRECTIONS,control[26],frame.bounds,items,weights,target,
                bullets,velocity,radii,uncertainty,bodies,known,body_known,
                delay,late,horizon,durations,switch_delays,timing_complete,beam,branches,self.fast_speed,age_ms)
            (prefix_hit,timing_free,checked,feasible,allowed,fallback,nominal_free,rescue,late_risk,
             known_prefix_hit,body_probe,immediate_free,probes,confidence,wall_space,exit_fraction,
             exit_measured,short_priority,best)=meta
            best=int(best);fallback=bool(fallback);body_probe=bool(body_probe);late_risk=bool(late_risk)
            elapsed_ms=(time.perf_counter()-started)*1000
            self.last_elapsed_ms,self.last_visible=elapsed_ms,scene_load
            diagnostics={"enabled":True,"version":PLANNER_VERSION,"horizon_ticks":horizon,
                'collision_backend':self.collision_backend,'search_backend':'rust','visible_lasers':len(frame.lasers),
                'delay_ticks':delay,'beam_width':beam,'branch_limit':branches,'search_node_budget':MAX_SEARCH_NODES,
                'delay_bounds_ticks':[early,late],'estimated_switch_ms':round(switch_ms,3),
                'tested_switch_delays':switch_delays,'timing_interval_complete':timing_complete,
                'short_escape_priority':bool(short_priority),'timing_probe_hold_ticks':late+SEGMENT_TICKS[0],
                'timing_contact_free_roots':int(timing_free),'prefix_collision_tick':prefix_hit if np.isfinite(prefix_hit) else None,
                'recovery_ranking':fallback,'observation_age_ms':round(age_ms,3),'estimated_plan_ms':round(estimate_ms,3),
                'estimated_post_plan_ms':round(tail_ms,3),'input_confirmation_reserve_ms':round(confirmation_ms,3),
                'latency_samples':len(self.tail_times.values),'path_ticks':np.cumsum((0,delay,*durations)).tolist(),
                'fast_speed':self.fast_speed,'slow_speed':self.slow_speed,'visible_bullets':len(frame.bullets),
                'considered_bullets':len(bullets),'visible_enemy_bodies':len(all_bodies.positions),
                'considered_enemy_bodies':len(bodies.positions),'unknown_body_motion':int((~all_bodies.known_motion).sum()),
                'unknown_body_geometry':int((~all_bodies.known_geometry).sum()),'body_contact_probe_performed':body_probe,
                'immediate_contact_free_roots':int(immediate_free) if body_probe else None,
                'unknown_motion':int((~state.known_motion).sum()),'unknown_geometry':int((~state.known_geometry).sum()),
                'candidates_checked':int(checked),'feasible_roots':int(feasible),'allowed_roots':int(allowed),
                'all_routes_risky':fallback,'route_mode':'fallback' if fallback else 'safe','nominal_escape_roots':int(nominal_free),
                'rescue_opportunity':bool(rescue),'late_control_risk':late_risk,
                'nominal_prefix_collision_tick':known_prefix_hit if late_risk else None,
                'mobility_probes':int(probes),'mobility_probe_limit':EXIT_PROBE_LIMIT,'mobility_hazard_confidence':round(confidence,3),
                'wall_space':wall_space,'exit_fraction':exit_fraction,'exit_measured':bool(exit_measured),
                'suggested_movement':best,'estimated_clearance':float(minimum[best]) if np.isfinite(minimum[best]) else None,
                'predicted_collision_tick':float(collision[best]) if np.isfinite(collision[best]) else None,'elapsed_ms':round(elapsed_ms,3)}
            result = PlanResult(features,paths,diagnostics,minimum,collision,completed_ns=clock_ns(),started_ns=started_ns,
                                navigation=navigation,decision=decision)
            return self._rank_warning_fallback(result, start, pending, low, high, bodies, damaging)
        hint_positions = np.concatenate((bullets, bodies.positions))
        hint_velocity = np.concatenate((velocity, bodies.velocity))
        hint_radii = np.concatenate((radii, np.linalg.norm(bodies.half_extents, axis=1)+bodies.player_radius))
        pending = self.velocities[int(np.argmax(control[:18]))] if control[19] > .5 else np.asarray(control[20:22])*16
        prefix, prefix_clear, prefix_hit, _ = self._sweep(start[None], pending[None], delay, 0,
                low, high, bullets, velocity, radii, uncertainty, bodies=bodies)
        # Earlier dispatch also permits an earlier next decision. Compare the
        # same held-command duration, not a common absolute endpoint that
        # would falsely force an early command to keep moving extra frames.
        probe_hold = late+SEGMENT_TICKS[0]
        switch_delays, timing_complete = self._switch_delays(early,late)
        timing_free, timing_hit, timing_action_hit, timing_action_clear = self._timing_probe(
                start, pending, switch_delays, probe_hold, low, high, bullets, velocity, radii, uncertainty, bodies)
        immediate_free = np.ones(18, bool)
        body_probe = False
        positions = np.repeat(prefix, 18, axis=0)
        roots = np.arange(18)
        last_actions = roots.copy()
        paths = np.tile(np.stack((start, prefix[0]))[None], (18, 1, 1))
        minimum = np.full(18, prefix_clear[0])
        collision = np.full(18, prefix_hit[0])
        action_collision = np.full(18, np.inf, np.float32)
        action_minimum = action_collision.copy()
        wall_cost = np.zeros(18)
        edge_cost = np.zeros(18)
        corner_cost = np.zeros(18)
        aim_score = np.zeros(18)
        goal_best = np.zeros(18)
        movement_cost = np.zeros(18)
        items, weights, target = self._goals(frame, previous)
        tick = delay
        candidates_checked = 0
        root_blocked = np.zeros(18, bool)
        root_effort = np.zeros(18)
        for depth, duration in enumerate(durations):
            if depth == 0:
                parents = np.arange(18)
                actions = roots.copy()
            else:
                options = self._branches(positions, last_actions, branches, tick, hint_positions, hint_velocity, hint_radii,
                                         items, weights, target, low, high)
                parents = np.repeat(np.arange(len(positions)), branches)
                actions = options.ravel()
            starts = positions[parents]
            ends, clearance, hit, effort = self._sweep(starts, self.velocities[actions], duration, tick,
                    low, high, bullets, velocity, radii, uncertainty, bodies=bodies)
            candidates_checked += len(ends)
            if depth == 0:
                root_effort = effort.copy()
                root_blocked = (np.linalg.norm(ends-starts, axis=1) < .1) & (np.linalg.norm(self.velocities, axis=1) > .1)
            next_roots = roots[parents]
            next_minimum = np.minimum(minimum[parents], clearance)
            next_collision = np.minimum(collision[parents], hit)
            next_action_collision = np.minimum(action_collision[parents], hit)
            next_action_minimum = np.minimum(action_minimum[parents], clearance)
            edge, corner = self._edge(starts, ends, low, high)
            goal, aim = self._goal_quality(starts, ends, duration, tick, items, weights, target, frame.bounds)
            weight = duration/horizon
            next_edge = edge_cost[parents]+edge*weight
            next_corner = corner_cost[parents]+corner*weight
            next_wall = wall_cost[parents]+effort*weight
            next_aim = aim_score[parents]+aim*weight
            next_goal = np.maximum(goal_best[parents], goal)
            next_movement = movement_cost[parents]+np.linalg.norm(ends-starts, axis=1)/(self.fast_speed*horizon)
            utility = (1.4*next_goal+.5*next_aim-1.2*next_edge
                       -.6*(1+control[26])*next_corner-1.5*next_wall-.03*next_movement
                       -2*root_effort[next_roots]-.5*np.clip((CLEARANCE-next_minimum)/CLEARANCE, 0, 2))
            width = 1 if depth == len(durations)-1 else beam
            # Every root retains the same beam width and branch count. Keep
            # the existing root-major layout and sort all 18 groups at once.
            # This removes dozens of tiny Python/NumPy dispatches per plan.
            per_root = len(next_roots)//18
            order = np.lexsort((-utility.reshape(18, per_root),
                    -np.minimum(next_action_collision, horizon+1).reshape(18, per_root),
                    -np.minimum(next_collision, horizon+1).reshape(18, per_root)), axis=1)
            keep = (np.arange(18)[:, None]*per_root+order[:, :width]).ravel()
            paths = np.concatenate((paths[parents[keep]], ends[keep, None]), axis=1)
            positions, roots = ends[keep], next_roots[keep]
            last_actions = actions[keep]
            minimum, collision = next_minimum[keep], next_collision[keep]
            action_minimum, action_collision = next_action_minimum[keep], next_action_collision[keep]
            edge_cost, corner_cost, wall_cost = next_edge[keep], next_corner[keep], next_wall[keep]
            aim_score, goal_best, movement_cost = next_aim[keep], next_goal[keep], next_movement[keep]
            scores = utility[keep]
            tick += duration
        safe = np.isinf(collision) & timing_free & timing_complete
        allowed = safe & ~root_blocked
        fallback = not bool(allowed.any())
        known = state.known_motion[relevant] & state.known_geometry[relevant]
        confirmed_bullets,confirmed_velocity,confirmed_radii = bullets[known],velocity[known],radii[known]
        body_known = bodies.known_motion & bodies.known_geometry & damaging
        # Warning lasers remain robust planning obstacles, but are not evidence
        # of an already damaging beam for rescue exploration.
        confirmed_bodies = replace(bodies.subset(body_known),uncertainty=np.zeros(int(body_known.sum()),np.float32))
        nominal_free,nominal_hit,nominal_action_hit,nominal_action_clear = self._timing_probe(
                start,pending,switch_delays,max(8,SEGMENT_TICKS[0]),low,high,
                confirmed_bullets,confirmed_velocity,confirmed_radii,np.zeros(int(known.sum())),confirmed_bodies)
        _,_,known_prefix_hit,_ = self._sweep(start[None],pending[None],late,0,low,high,
                confirmed_bullets,confirmed_velocity,confirmed_radii,np.zeros(int(known.sum())),bodies=confirmed_bodies)
        if fallback:
            if len(bodies.positions):
                # Only fallback needs this nominal body-only gate. Feasible
                # full routes already passed the stronger uncertain envelope.
                # Keep it independent of a bullet's shared-prefix collision.
                body_probe = True
                nominal_bodies = replace(bodies, uncertainty=np.full_like(bodies.uncertainty, SELF_SPEED_ERROR))
                empty = np.empty((0, 2), np.float32)
                immediate_free, _, _, _ = self._timing_probe(start, pending, switch_delays,
                        probe_hold, low, high, empty, empty, np.empty(0), np.empty(0), nominal_bodies)
            eligible = immediate_free & ~root_blocked if len(bodies.positions) else np.ones(18, bool)
            if not eligible.any():
                eligible = np.ones(18, bool)
            combined_hit = np.minimum(collision, timing_hit)
            latest = float(np.max(np.where(eligible, np.minimum(combined_hit, horizon+1), -np.inf)))
            allowed = eligible & (np.minimum(combined_hit, horizon+1) >= latest-.5)
            # A shared old-key prefix may collide for every candidate. Rank
            # the portion new keys can actually change rather than falling
            # back to item/aim scores among 18 equally doomed prefix times.
            recovery_hit = np.minimum(action_collision, timing_action_hit)
            latest_recovery = float(np.max(np.where(allowed, np.minimum(recovery_hit, horizon+1), -np.inf)))
            allowed &= np.minimum(recovery_hit, horizon+1) >= latest_recovery-.5
            recovery_clear = np.minimum(action_minimum, timing_action_clear)
            best_clear = np.max(np.where(allowed, recovery_clear, -np.inf))
            if np.isfinite(best_clear):
                allowed &= recovery_clear >= best_clear-1.
            if np.any(allowed & ~root_blocked):
                allowed &= ~root_blocked
            # Short escapes must pass EVERY tested switch time. A union of a
            # single nominal-time result with the recovery mask used to reopen
            # directions already known to collide at earlier dispatch times.
            # Prefer the surviving short window over long-horizon utility;
            # this remains fallback, not a certified complete safe route.
            short_escapes = nominal_free & immediate_free & ~root_blocked
            if short_escapes.any():allowed = short_escapes
        if not allowed.any():
            allowed[int(np.argmax(scores))] = True
        confidence = (len(confirmed_bullets)+4*len(confirmed_bodies.positions))/max(1,len(bullets)+4*len(bodies.positions))
        scores,wall_space,exit_fraction,exit_measured,mobility_probes = self._mobility(
            positions,scores,allowed,low,high,horizon,confirmed_bullets,confirmed_velocity,
            confirmed_radii,confirmed_bodies,confidence)
        best = int(np.argmax(np.where(allowed, scores, -np.inf)))
        bias = np.clip((scores-scores[best])*8, -PLAN_LOGIT_SCALE, 0)/PLAN_LOGIT_SCALE
        features = np.column_stack((np.ones(18), allowed, np.clip(minimum/64, -1, 1),
                np.minimum(collision, horizon)/horizon, root_blocked,
                np.clip(goal_best/1.5, 0, 1), np.clip(scores/8, -1, 1), bias)).astype(np.float32)
        late_risk = bool(np.isfinite(known_prefix_hit[0]))
        best_contact = float(np.max(np.minimum(nominal_hit,horizon)))
        rescue = bool(fallback and not nominal_free.any() and not late_risk
                      and best_contact>late+2 and np.max(nominal_action_clear)<-COLLISION_PADDING)
        navigation = np.column_stack((safe & ~root_blocked,np.full(18,fallback),
                np.full(18,np.isfinite(prefix_hit[0])),np.clip(action_minimum/64,-1,1),
                np.minimum(action_collision,horizon)/horizon,wall_space,exit_fraction,exit_measured,
                np.minimum(nominal_hit,horizon)/horizon)).astype(np.float32)
        decision = np.array([1.,fallback,late_risk,rescue,1.,min(best_contact/horizon,1.),
                             min(age_ms/(FRAME_MS*horizon),1.),float(exit_fraction[best])],np.float32)
        elapsed_ms = (time.perf_counter()-started)*1000
        self.last_elapsed_ms, self.last_visible = elapsed_ms, scene_load
        result = PlanResult(features, paths.astype(np.float32), {
                "enabled": True, "version": PLANNER_VERSION, "horizon_ticks": horizon,
                'collision_backend':self.collision_backend,'search_backend':'python','visible_lasers':len(frame.lasers),
                "delay_ticks": delay, "beam_width": beam, "branch_limit": branches,
                "search_node_budget": MAX_SEARCH_NODES,
                "delay_bounds_ticks": [early, late], "estimated_switch_ms": round(switch_ms, 3),
                'tested_switch_delays':switch_delays,'timing_interval_complete':timing_complete,
                'short_escape_priority':bool(fallback and (nominal_free & immediate_free & ~root_blocked).any()),
                "timing_probe_hold_ticks": probe_hold,
                "timing_contact_free_roots": int(timing_free.sum()),
                "prefix_collision_tick": float(prefix_hit[0]) if np.isfinite(prefix_hit[0]) else None,
                "recovery_ranking": fallback,
                "observation_age_ms": round(age_ms, 3), "estimated_plan_ms": round(estimate_ms, 3),
                "estimated_post_plan_ms": round(tail_ms, 3), "input_confirmation_reserve_ms": round(confirmation_ms, 3),
                "latency_samples": len(self.tail_times.values),
                "path_ticks": np.cumsum((0, delay, *durations)).tolist(),
                "fast_speed": self.fast_speed, "slow_speed": self.slow_speed,
                "visible_bullets": len(frame.bullets), "considered_bullets": len(bullets),
                "visible_enemy_bodies": len(all_bodies.positions), "considered_enemy_bodies": len(bodies.positions),
                "unknown_body_motion": int((~all_bodies.known_motion).sum()),
                "unknown_body_geometry": int((~all_bodies.known_geometry).sum()),
                "body_contact_probe_performed": body_probe,
                "immediate_contact_free_roots": int(immediate_free.sum()) if body_probe else None,
                "unknown_motion": int((~state.known_motion).sum()),
                "unknown_geometry": int((~state.known_geometry).sum()),
                "candidates_checked": candidates_checked, "feasible_roots": int(safe.sum()),
                "allowed_roots": int(allowed.sum()), "all_routes_risky": fallback,
                'route_mode':'fallback' if fallback else 'safe',
                'nominal_escape_roots':int(nominal_free.sum()),'rescue_opportunity':rescue,
                'late_control_risk':late_risk,'nominal_prefix_collision_tick':float(known_prefix_hit[0]) if late_risk else None,
                'mobility_probes':mobility_probes,'mobility_probe_limit':EXIT_PROBE_LIMIT,
                'mobility_hazard_confidence':round(confidence,3),'wall_space':float(wall_space[best]),
                'exit_fraction':float(exit_fraction[best]),'exit_measured':bool(exit_measured[best]),
                "suggested_movement": best, "estimated_clearance": float(minimum[best]) if np.isfinite(minimum[best]) else None,
                "predicted_collision_tick": float(collision[best]) if np.isfinite(collision[best]) else None,
                "elapsed_ms": round(elapsed_ms, 3)}, minimum, collision,
                completed_ns=clock_ns(), started_ns=started_ns,navigation=navigation,decision=decision)
        return self._rank_warning_fallback(result, start, pending, low, high, bodies, damaging)
