"""Shared, bounded UI snapshots and incremental SSE. No training controls."""
from __future__ import annotations
from .i18n import msg

import asyncio
from contextlib import suppress
from dataclasses import dataclass
import json
import time

VISIBLE_SECONDS = .5
HIDDEN_SECONDS = 5.
HEARTBEAT_SECONDS = 3.
HISTORY_LIMIT = 180

# A display DTO deliberately excludes worker/driver heartbeat clocks and
# internal training diagnostics. Those remain available from GET /api/state.
GPU_FIELDS = ("name", "temperature_c", "power_w", "memory_used", "memory_total", "utilization",
              "fan_percent", "memory_temperature_c")
WORKER_FIELDS = ("training_seconds", "budget_seconds", "paused_seconds", "steps", "session_steps", "updates",
                 'game_settings','requested_game_setup',
                 'stage_id', 'stage_label', 'stage_seconds', 'stage_changes', 'stage_transition',
                 'game_phase','player_shots','visible_lasers','goal','episode_bombs_used',
                 "episodes", "survival_seconds", "user_paused", "phase", "message", "model_source", "checkpoint",
                 "resources", "bomb_requests", "bombs_used", "boss_hp", "boss_max_hp", "boss_damage", "power_gained",
                 "point_items_gained", "loiter_seconds", "planning", "bullet_observation", "input_latency", "focus_pending", "error")
LATENCY_FIELDS = ('observation_to_send_ms', 'observation_to_send_p95_ms', 'policy_ms', 'policy_p95_ms',
                  'send_ms', 'send_p95_ms', 'input_events', 'game_input_state', 'game_input_match_ms')


def display_state(raw):
    state = {key: value for key, value in raw.items() if key != "gpu_age_seconds"}
    session = raw.get("session") or {}
    shown = {key: session[key] for key in ("state", "mode", "reason") if key in session}
    if not raw.get("active") and "time" in session:
        shown["time"] = session["time"]
    gpu, worker = session.get("gpu") or {}, session.get("worker") or {}
    shown["gpu"] = {key: gpu[key] for key in GPU_FIELDS if key in gpu}
    shown["worker"] = {key: worker[key] for key in WORKER_FIELDS if key in worker}
    if isinstance(shown['worker'].get('input_latency'), dict):
        timing = shown['worker']['input_latency']
        shown['worker']['input_latency'] = {key: timing[key] for key in LATENCY_FIELDS if key in timing}
    state["session"] = shown
    return state


def _diff(before, after, path, sets, removals):
    if before is after:
        return
    if isinstance(before, dict) and isinstance(after, dict):
        for key in before.keys()-after.keys():
            removals.append([*path, key])
        for key, value in after.items():
            target = [*path, key]
            if key not in before:
                sets.append([target, value])
            else:
                _diff(before[key], value, target, sets, removals)
    elif before != after:
        sets.append([path, after])


def history_delta(before, after, same_run):
    if same_run and (before is after or before == after):
        return None
    if same_run and before:
        for index in range(len(after)-1, -1, -1):
            if after[index] == before[-1]:
                return {"mode": "append", "after": before[-1].get("stamp"), "points": after[index+1:]}
    return {"mode": "reset", "points": after[-HISTORY_LIMIT:]}


def make_delta(before, after, base, revision):
    sets, removals = [], []
    _diff({k: v for k, v in before.items() if k != "gpu_history"},
          {k: v for k, v in after.items() if k != "gpu_history"}, [], sets, removals)
    history = history_delta(before.get("gpu_history", []), after.get("gpu_history", []),
                            before.get("run_dir") == after.get("run_dir"))
    if not sets and not removals and history is None:
        return None
    frame = {"protocol": 2, "kind": "delta", "base": base, "revision": revision, "set": sets, "remove": removals}
    if history is not None:
        frame["history"] = history
    return frame


def encode(frame):
    return "data: " + json.dumps(frame, ensure_ascii=False, allow_nan=False, separators=(",", ":")) + "\n\n"


@dataclass
class Published:
    revision: int
    state: dict
    delta: dict | None
    delta_text: str | None
    full_text: str | None = None

    def full(self):
        if self.full_text is None:
            self.full_text = encode({"protocol": 2, "kind": "snapshot", "revision": self.revision, "state": self.state})
        return self.full_text


class SnapshotBroker:
    """One sampler per API process; clients retain only their last snapshot.

    Slow clients diff against the latest snapshot rather than queuing every
    intermediate frame. A reconnect always receives a fresh complete state.
    """
    def __init__(self, service, battle_view_enabled=None):
        self.service = service
        self.battle_view_enabled = battle_view_enabled
        self.latest = None
        self.raw = None
        self.samples = 0
        self.clients = {}
        self._serial = 0
        self._task = None
        self._wake = asyncio.Event()
        self._condition = asyncio.Condition()
        self._error = None
        self._closed = False
        self._shutdown_complete = False
        self._last_sample_at = -float("inf")

    async def _run(self):
        try:
            while self.clients:
                self._wake.clear()
                # New connections can wake a hidden sampler, but cannot make
                # several tabs multiply the normal 2 Hz sampling rate.
                # Windows asyncio timers may wake early. Use the high
                # resolution clock and recheck before sampling again.
                remaining = VISIBLE_SECONDS-(time.perf_counter()-self._last_sample_at)
                while remaining > 0:
                    await asyncio.sleep(max(.001, remaining))
                    remaining = VISIBLE_SECONDS-(time.perf_counter()-self._last_sample_at)
                if not self.clients:
                    break
                raw = await asyncio.to_thread(self.service.snapshot)
                if self.battle_view_enabled is not None:
                    raw['battle_view_enabled'] = self.battle_view_enabled()
                self._last_sample_at = time.perf_counter()
                state = display_state(raw)
                self.raw = raw
                self.samples += 1
                previous = self.latest
                revision = previous.revision+1 if previous else 1
                delta = make_delta(previous.state, state, previous.revision, revision) if previous else None
                if previous is None or delta is not None:
                    self.latest = Published(revision, state, delta, encode(delta) if delta else None)
                async with self._condition:
                    self._condition.notify_all()
                interval = HIDDEN_SECONDS if all(self.clients.values()) else VISIBLE_SECONDS
                try:
                    await asyncio.wait_for(self._wake.wait(), timeout=interval)
                except asyncio.TimeoutError:
                    pass
        except asyncio.CancelledError:
            raise
        except Exception as error:
            self._error = error
            async with self._condition:
                self._condition.notify_all()

    async def events(self, request, *, hidden=False, protocol=2):
        if self._closed:
            return
        self._serial += 1
        identity = self._serial
        self.clients[identity] = hidden
        if self._task is None or self._task.done():
            self._error = None
            self.latest, self.raw = None, None
            self._task = asyncio.create_task(self._run())
        self._wake.set()
        previous = None
        sent = 0.
        try:
            while not self._closed and not await request.is_disconnected():
                if self._error is not None:
                    raise RuntimeError(msg('backend.could_not_read_the_console_state')) from self._error
                published = self.latest
                now = time.perf_counter()
                if published is None:
                    async with self._condition:
                        await self._condition.wait_for(lambda: self._closed or self.latest is not None or self._error is not None)
                    continue
                if protocol == 1:
                    yield encode(self.raw)
                    previous, sent = published, now
                elif previous is None:
                    yield published.full()
                    previous, sent = published, now
                elif previous.revision != published.revision:
                    if published.delta and published.delta["base"] == previous.revision:
                        text = published.delta_text
                    else:
                        delta = make_delta(previous.state, published.state, previous.revision, published.revision)
                        text = encode(delta or {"protocol": 2, "kind": "delta", "base": previous.revision,
                                               "revision": published.revision, "set": [], "remove": []})
                    yield text
                    previous, sent = published, now
                elif now-sent >= (HIDDEN_SECONDS if hidden else HEARTBEAT_SECONDS):
                    yield encode({"protocol": 2, "kind": "heartbeat", "revision": published.revision})
                    sent = now
                if hidden or protocol == 1:
                    await asyncio.sleep(HIDDEN_SECONDS if hidden else VISIBLE_SECONDS)
                else:
                    async with self._condition:
                        try:
                            await asyncio.wait_for(self._condition.wait_for(lambda:
                                self._closed or self._error is not None or (self.latest is not None
                                    and self.latest.revision != previous.revision)),
                                timeout=max(.01, HEARTBEAT_SECONDS-(time.perf_counter()-sent)))
                        except asyncio.TimeoutError:
                            pass
            if self._shutdown_complete:
                yield "event: shutdown\n" + encode({"message": msg("console.backend_shut_down")})
        finally:
            self.clients.pop(identity, None)
            self._wake.set()

    async def close(self, *, shutdown=False):
        self._shutdown_complete = self._shutdown_complete or shutdown
        self._closed = True
        async with self._condition:
            self._condition.notify_all()
        if self._task is not None:
            self._task.cancel()
            with suppress(asyncio.CancelledError):
                await self._task
