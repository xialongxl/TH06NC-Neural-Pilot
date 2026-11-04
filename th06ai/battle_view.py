"""Optional, latest-only visualization of an already executed decision."""
from __future__ import annotations

import asyncio
import json
import math
import os
import secrets
import time

from .console_stream import encode
from .files import read_json
from .navigation import NAV_FIELDS
from .planner_spec import PLAN_FIELDS, PLAN_MOVEMENTS

DEMAND = 'battle-view-demand.json'
LATEST = 'battle-view-latest.json'
PERIOD = .1
MAX_AGE_MS = 350
DEMAND_TTL = 1.5


def _write_latest(path, payload):
    # A competing reader or transient lock can cost a frame, never a retry
    # on the diagnostic pipe or an exception in the control loop.
    pending = path.with_name(f'{path.name}.{os.getpid()}.tmp')
    try:
        pending.write_text(json.dumps(payload, allow_nan=False, separators=(',', ':')), encoding='utf-8')
        pending.replace(path)
    finally:
        pending.unlink(missing_ok=True)


class BattleViewPublisher:
    """One reference slot from the control loop to the diagnostic sender."""

    def __init__(self, root):
        self.root = root
        self.demand = None
        self.latest = self.sent = None
        self.next_check = self.next_capture = 0.
        self.in_flight = False

    def capture(self, frame, plan, action, input_state, sequence):
        demand = self.demand
        if demand is None:
            return
        now = time.monotonic()
        if not self.next_capture <= now < demand[1] or frame.capture_started_ns <= 0:
            return
        self.next_capture = now + PERIOD
        # Reader arrays and completed plan results belong to this decision.
        self.latest = (demand[0], frame, plan, action, input_state, sequence)

    def next_job(self):
        now = time.monotonic()
        if now >= self.next_check:
            self.next_check = now + .25
            marker = self.root / DEMAND
            data = read_json(marker) if marker.is_file() else {}
            if not isinstance(data, dict):
                data = {}
            generation, deadline = data.get('generation'), data.get('expires')
            self.demand = (generation, deadline) if (isinstance(generation, str)
                and isinstance(deadline, (int, float)) and now < deadline <= now + DEMAND_TTL) else None
            if self.demand is None:
                self.latest = self.sent = None
        sample = self.latest
        if (self.in_flight or self.demand is None or sample is None or sample is self.sent
                or sample[0] != self.demand[0] or now >= self.demand[1]):
            return None
        generation, frame, plan, action, input_state, sequence = sample
        self.sent = sample
        if (time.perf_counter_ns() - frame.capture_started_ns) / 1e6 >= MAX_AGE_MS:
            return None
        geometry = (frame.bounds, frame.x, frame.y, frame.tick, frame.stage_id,
                    frame.bullets, frame.enemy_bodies, frame.lasers)
        planning = (plan.features, plan.navigation, plan.paths, plan.collision_times,
                    plan.diagnostics.get('search_backend'),
                    plan.diagnostics.get('timing_interval_complete')) if plan is not None else None
        self.in_flight = True
        return ('battle_view', generation, frame.capture_started_ns, sequence, geometry,
                planning, action, input_state)


def _coordinates(array, limit, columns):
    import numpy as np

    rows = array[:limit, columns]
    return np.round(rows[np.isfinite(rows).all(axis=1)].astype(np.float64), 3).tolist()


def publish_battle_view(root, generation, captured_ns, sequence, geometry, planning, action, input_state):
    """Build a bounded wire frame in the CPU-only diagnostic child."""
    import numpy as np

    bounds, x, y, tick, stage, bullets, bodies, lasers = geometry
    low_x, low_y, span_x, span_y = bounds
    width, height = span_x + 2 * low_x, span_y + 2 * low_y
    if not all(map(math.isfinite, (*bounds, width, height, x, y))) or min(width, height) <= 0:
        return
    candidates = [['unknown', None] for _ in range(PLAN_MOVEMENTS)]
    route = []
    backend = None
    if planning is not None:
        features, navigation, paths, collision_times, backend, complete = planning
        enabled, allowed, blocked, utility = (PLAN_FIELDS.index(key)
                                               for key in ('enabled', 'allowed', 'blocked', 'utility'))
        robust, fallback, collision = (NAV_FIELDS.index(key)
                                       for key in ('robust_safe', 'fallback', 'action_collision'))
        for index, row in enumerate(features[:PLAN_MOVEMENTS]):
            if not row[enabled] or navigation is None:
                continue
            nav = navigation[index]
            if complete and nav[robust]:
                risk = 'clear'
            elif row[allowed] and nav[fallback]:
                risk = 'fallback'
            elif row[blocked] or nav[collision] < 1 or (
                    collision_times is not None and math.isfinite(collision_times[index])):
                risk = 'risk'
            else:
                risk = 'unknown'
            score = float(row[utility])
            candidates[index] = [risk, round(score, 3) if math.isfinite(score) else None]
        if features[action % PLAN_MOVEMENTS, enabled]:
            route = _coordinates(paths[action % PLAN_MOVEMENTS], 32, slice(None))
    xy = bullets[:640, 1:3]
    visible = (xy >= 0).all(axis=1) & (xy <= (width, height)).all(axis=1)
    payload = {
        'generation': generation, 'captured_ns': captured_ns, 'sequence': sequence,
        'frame': {
            'viewport': [width, height], 'bounds': list(bounds), 'player': [x, y],
            'tick': tick, 'stage': stage, 'action': action, 'input': input_state,
            'backend': backend, 'candidates': candidates, 'route': route,
            'bullets': np.round(xy[visible].astype(np.float64), 3).tolist(),
            'bodies': _coordinates(bodies, 256, slice(1, 6)),
            'lasers': _coordinates(lasers, 64, slice(1, 8)),
        },
    }
    _write_latest(root / LATEST, payload)


class BattleViewStream:
    """Independent loopback SSE. All subscribers share one latest-file read."""

    def __init__(self, run):
        self.run = run
        self.enabled = self.closed = False
        self.clients = self.generation = 0
        self.task = None
        self.changed = asyncio.Event()
        self.key = ('off',)
        self.packet = encode({'version': 1, 'state': 'off'})
        # The following state is used only by serial background polls.
        self._generation = None
        self._run_dir = self._demand_id = None
        self._next_session = self._next_demand = 0.

    def _publish(self, key, **data):
        if key != self.key:
            self.packet = encode({'version': 1, **data})
            self.key = key
            self.changed.set()
            self.changed = asyncio.Event()

    def set_enabled(self, enabled):
        if self.enabled != enabled:
            self.enabled = enabled
            self.generation += 1
            state = 'waiting' if enabled else 'off'
            self._publish((state,), state=state)
            self._start()

    def _start(self):
        if self.enabled and self.clients and not self.closed and self.task is None:
            self.task = asyncio.create_task(self._sample())

    def _poll(self, generation):
        now = time.monotonic()
        if now >= self._next_session or self._generation != generation:
            path = self.run()
            if path != self._run_dir or self._generation != generation:
                self._revoke()
                self._run_dir = path
                self._demand_id = secrets.token_hex(8)
                self._next_demand = 0.
            self._generation = generation
            self._next_session = now + .25
        path = self._run_dir
        if path is None:
            return None
        if now >= self._next_demand:
            _write_latest(path / DEMAND, {'generation': self._demand_id, 'expires': now + DEMAND_TTL})
            self._next_demand = now + .5
        data = read_json(path / LATEST)
        if not isinstance(data, dict) or data.get('generation') != self._demand_id:
            return None
        captured, sequence, frame = data.get('captured_ns'), data.get('sequence'), data.get('frame')
        if not isinstance(captured, int) or not isinstance(sequence, int) or not isinstance(frame, dict):
            return None
        age_ms = (time.perf_counter_ns() - captured) / 1e6
        if not 0 <= age_ms < MAX_AGE_MS:
            return None
        return (self._demand_id, sequence), frame, round(age_ms)

    def _revoke(self):
        if self._run_dir is None:
            return
        marker = self._run_dir / DEMAND
        try:
            data = read_json(marker)
            if isinstance(data, dict) and data.get('generation') == self._demand_id:
                _write_latest(marker, {'generation': self._demand_id, 'expires': 0})
        except OSError:
            pass  # The existing demand expires even if the marker is busy.
        self._run_dir = None
        self._generation = None

    async def _sample(self):
        try:
            while self.enabled and self.clients and not self.closed:
                generation = self.generation
                started = time.monotonic()
                try:
                    data = await asyncio.to_thread(self._poll, generation)
                except (OSError, ValueError, KeyError, TypeError):
                    data = None
                if generation == self.generation and not self.closed:
                    if data is None:
                        self._publish(('waiting',), state='waiting')
                    else:
                        identity, frame, age = data
                        self._publish(('live', *identity), state='live', frame=frame, age_ms=age,
                                      expires_at_ms=round(time.time() * 1000) + MAX_AGE_MS - age)
                while (remaining := started + PERIOD - time.monotonic()) > 0:
                    await asyncio.sleep(max(.001, remaining))
        finally:
            await asyncio.to_thread(self._revoke)
            self.task = None
            self._start()

    async def events(self, request):
        self.clients += 1
        self._start()
        previous = None
        try:
            while not self.closed and not await request.is_disconnected():
                if self.key != previous:
                    previous = self.key
                    yield self.packet
                    continue
                try:
                    await asyncio.wait_for(self.changed.wait(), timeout=3)
                except asyncio.TimeoutError:
                    yield ': keepalive\n\n'
        finally:
            self.clients -= 1
            if self.clients == 0:
                self.generation += 1
                if self.enabled:
                    self._publish(('waiting',), state='waiting')

    async def close(self):
        self.closed = True
        self.changed.set()
        if self.task is not None:
            await self.task
