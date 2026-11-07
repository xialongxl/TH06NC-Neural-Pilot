"""Opt-in visualization and commentary for already observed gameplay."""
from __future__ import annotations

import asyncio
import json
import math
import os
import secrets
import time
from dataclasses import dataclass

from .console_stream import encode
from .files import read_json
from .navigation import NAV_FIELDS
from .planner_spec import PLAN_FIELDS, PLAN_MOVEMENTS

DEMAND = 'battle-view-demand.json'
LATEST = 'battle-view-latest.json'
COMMENTARY_LATEST = 'battle-view-commentary.json'
PERIOD = .1
MAX_AGE_MS = 350
DEMAND_TTL = 1.5
COMMENTARY_LIMIT = 32
COMMENTARY_TTL_NS = 5_000_000_000


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
    """Latest decision and a bounded event buffer for the diagnostic sender."""

    def __init__(self, root):
        self.root = root
        self.demand = None
        self.latest = self.sent = None
        self.next_check = self.next_capture = 0.
        self.in_flight = False
        self.miss_count = 0
        self._last_missed_attempt = None
        self._events = self._sent_events = ()
        self._next_event_send = 0.

    def capture(self, frame, plan, action, input_state, sequence, *, after=None):
        demand = self.demand
        if demand is None:
            return
        now = time.monotonic()
        if not self.next_capture <= now < demand[1] or frame.capture_started_ns <= 0:
            return
        self.next_capture = now + PERIOD
        # Reader arrays and completed plan results belong to this decision.
        self.latest = (demand[0], frame, plan, action, input_state, sequence, after, self.miss_count)

    def record_miss(self, attempt, stage, tick):
        if attempt == self._last_missed_attempt:
            return
        self._last_missed_attempt = attempt
        self.miss_count += 1
        self._record_event('miss', stage, tick, miss_count=self.miss_count)

    def record_resource_gain(self, before, after):
        old, new = before.resources, after.resources
        if old is None or new is None:
            return
        power = max(0, new.power - old.power)
        bombs = max(0, new.bombs - old.bombs)
        points = new.point_items - old.point_items
        if points < 0 and old.point_items > 64511 and new.point_items < 1024:
            points += 65536  # The native point-item counter is 16-bit.
        if power or bombs or points > 0:
            self._record_event('resource_gain', after.stage_id, after.tick,
                               power_delta=int(power), point_items_delta=int(max(0, points)),
                               bombs_delta=int(bombs))

    def _record_event(self, kind, stage, tick, **values):
        demand = self.demand
        if demand is None or time.monotonic() >= demand[1]:
            return
        events = self._events
        seq = events[-1]['seq'] + 1 if events else 1
        event = {'seq': seq, 'captured_ns': time.perf_counter_ns(), 'kind': kind,
                 'stage': stage, 'tick': tick, **values}
        # One producer replaces the tuple. The sender only reads snapshots of it.
        self._events = (*events[-(COMMENTARY_LIMIT - 1):], event)

    def next_commentary_job(self, *, force=False):
        events = self._events
        if events is self._sent_events:
            return None
        now = time.monotonic()
        if now < self._next_event_send and not force:
            return None
        self._sent_events = events
        self._next_event_send = now + PERIOD
        return ('battle_commentary', events)

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
        generation, frame, plan, action, input_state, sequence, after, miss_count = sample
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
                planning, action, input_state, _frame_context(after, miss_count) if after is not None else None)


def _frame_context(frame, miss_count):
    bosses = [row for row in frame.enemies if row[7]]
    hp_percent = None
    if len(bosses) == 1:
        hp, max_hp = map(float, bosses[0][3:5])
        if math.isfinite(max_hp) and max_hp > 0 and 0 <= hp <= max_hp:
            hp_percent = round(100 * hp / max_hp, 1)
    return {'tick': frame.tick, 'boss_hp_percent': hp_percent, 'current_spellcard': None,
            'bombs_remaining': frame.resources.bombs if frame.resources is not None else None,
            'miss_count': miss_count}


def _coordinates(array, limit, columns):
    import numpy as np

    rows = array[:limit, columns]
    return np.round(rows[np.isfinite(rows).all(axis=1)].astype(np.float64), 3).tolist()


def publish_battle_view(root, generation, captured_ns, sequence, geometry, planning, action, input_state, context=None):
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
    if context is not None:
        payload['frame']['context'] = context
    _write_latest(root / LATEST, payload)


def publish_battle_commentary(root, events):
    _write_latest(root / COMMENTARY_LATEST, {'events': events})


@dataclass(frozen=True, slots=True)
class CommentaryPacket:
    run_id: str
    seq: int
    deadline_ns: int
    wire: str


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
        self.commentary = ()
        # The following state is used only by serial background polls.
        self._generation = None
        self._run_dir = self._demand_id = None
        self._next_session = self._next_demand = 0.
        self._commentary_run = self._commentary_stamp = None
        self._commentary_cache = ()

    def _notify(self):
        self.changed.set()
        self.changed = asyncio.Event()

    def _publish(self, key, **data):
        if key != self.key:
            self.packet = encode({'version': 1, **data})
            self.key = key
            self._notify()

    def _publish_commentary(self, packets):
        if packets != self.commentary:
            self.commentary = packets
            self._notify()

    def set_enabled(self, enabled):
        if self.enabled != enabled:
            self.enabled = enabled
            self.generation += 1
            self._publish_commentary(())
            state = 'waiting' if enabled else 'off'
            self._publish((state,), state=state)
            self._start()

    def _start(self):
        if self.enabled and self.clients and not self.closed and self.task is None:
            self.task = asyncio.create_task(self._sample())

    def _poll(self, generation):
        now = time.monotonic()
        if now >= self._next_session or self._generation != generation:
            self._commentary_run, live = self.run()
            path = self._commentary_run if live else None
            if path != self._run_dir or self._generation != generation:
                self._revoke()
                self._run_dir = path
                self._demand_id = secrets.token_hex(8)
                self._next_demand = 0.
            self._generation = generation
            self._next_session = now + .25
        return self._poll_frame(now), self._poll_commentary()

    def _poll_frame(self, now):
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

    def _poll_commentary(self):
        run = self._commentary_run
        if run is None:
            return ()
        path = run / COMMENTARY_LATEST
        try:
            stat = path.stat()
        except OSError:
            return ()
        stamp = path, stat.st_mtime_ns, stat.st_size, stat.st_ino
        if stamp == self._commentary_stamp:
            return self._commentary_cache
        data = read_json(path)
        events = data.get('events') if isinstance(data, dict) else None
        if not isinstance(events, list):
            return ()
        now_ns, now_ms = time.perf_counter_ns(), time.time_ns() // 1_000_000
        packets = []
        for event in events[-COMMENTARY_LIMIT:]:
            if not isinstance(event, dict):
                continue
            kind = event.get('kind')
            if kind == 'miss':
                keys = ('miss_count',)
            elif kind == 'resource_gain':
                keys = ('power_delta', 'point_items_delta', 'bombs_delta')
            else:
                continue
            counts = {key: event.get(key) for key in keys}
            seq, captured_ns = event.get('seq'), event.get('captured_ns')
            tick, stage = event.get('tick'), event.get('stage')
            if (type(seq) is not int or seq <= 0 or type(captured_ns) is not int or type(tick) is not int
                    or (stage is not None and (type(stage) is not int or not 1 <= stage <= 7))
                    or any(type(count) is not int or count < 0 for count in counts.values())):
                continue
            age = now_ns - captured_ns
            if not 0 <= age < COMMENTARY_TTL_NS:
                continue
            event_id = f'{run.name}:{seq}'
            payload = {'version': 1, 'run_id': run.name, 'seq': seq, 'event_id': event_id,
                       'kind': kind, 'stage': stage, 'tick': tick, **counts,
                       'expires_at_ms': now_ms + (COMMENTARY_TTL_NS - age) // 1_000_000}
            wire = f'event: commentary\nid: {event_id}\n' + encode(payload)
            packets.append(CommentaryPacket(run.name, seq, captured_ns + COMMENTARY_TTL_NS, wire))
        self._commentary_stamp = stamp
        self._commentary_cache = tuple(packets)
        return self._commentary_cache

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
                    data, commentary = await asyncio.to_thread(self._poll, generation)
                except (OSError, ValueError, KeyError, TypeError):
                    data, commentary = None, ()
                if generation == self.generation and not self.closed:
                    if data is None:
                        self._publish(('waiting',), state='waiting')
                    else:
                        identity, frame, age = data
                        self._publish(('live', *identity), state='live', frame=frame, age_ms=age,
                                      expires_at_ms=round(time.time() * 1000) + MAX_AGE_MS - age)
                    self._publish_commentary(commentary)
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
        last_id = getattr(request, 'headers', {}).get('last-event-id', '')
        last_run, _, raw_seq = last_id.rpartition(':')
        last_seq = int(raw_seq) if len(raw_seq) <= 20 and raw_seq.isascii() and raw_seq.isdecimal() else 0
        try:
            while not self.closed and not await request.is_disconnected():
                if previous is None:
                    previous = self.key
                    yield self.packet
                    continue
                now_ns = time.perf_counter_ns()
                event = next((packet for packet in self.commentary if packet.deadline_ns > now_ns
                              and (packet.run_id != last_run or packet.seq > last_seq)), None)
                if event is not None:
                    last_run, last_seq = event.run_id, event.seq
                    yield event.wire
                    continue
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
                self._publish_commentary(())
                if self.enabled:
                    self._publish(('waiting',), state='waiting')

    async def close(self):
        self.closed = True
        self.changed.set()
        if self.task is not None:
            await self.task
