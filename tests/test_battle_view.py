"""Sidecar snapshots use completed decisions and never require a game or GPU."""
import asyncio
import json
import time
from types import SimpleNamespace

import numpy as np

from th06ai.battle_view import DEMAND, LATEST, BattleViewStream, publish_battle_view
from th06ai.diagnostic_writer import DiagnosticWriter
from th06ai.files import atomic_json
from th06ai.navigation import NAV_FIELDS
from th06ai.planner_spec import PLAN_FIELDS


def geometry():
    return ((8., 16., 368., 416.), 192., 390., 42, 2,
            np.array([[0, 1, 2], [1, 389, 2], [2, 3, np.nan]], dtype=np.float32),
            np.zeros((0, 6), dtype=np.float32), np.zeros((0, 8), dtype=np.float32))


def test_wire_frame_separates_viewport_bounds_risk_and_action(tmp_path):
    features = np.zeros((18, len(PLAN_FIELDS)))
    navigation = np.ones((18, len(NAV_FIELDS)))
    features[:, PLAN_FIELDS.index('enabled')] = 1
    features[1, PLAN_FIELDS.index('allowed')] = 1
    navigation[1:, NAV_FIELDS.index('robust_safe')] = 0
    navigation[2, NAV_FIELDS.index('action_collision')] = .5
    paths = np.zeros((18, 2, 2))
    paths[15] = [[192, 390], [194, 388]]
    planning = (features, navigation, paths, np.full(18, np.inf), 'rust', True)
    publish_battle_view(tmp_path, 'id', 1000, 5, geometry(), planning, 33, 'matched')
    frame = json.loads((tmp_path / LATEST).read_text())['frame']
    assert frame['viewport'] == [384, 448] and frame['bounds'] == [8, 16, 368, 416]
    assert frame['bullets'] == [[1, 2]]
    assert frame['action'] == 33 and frame['input'] == 'matched'
    assert frame['route'] == [[192, 390], [194, 388]]
    assert [risk for risk, _ in frame['candidates'][:4]] == ['clear', 'fallback', 'risk', 'unknown']
    publish_battle_view(tmp_path, 'id', 1000, 6, geometry(), (*planning[:-1], False), 33, 'held')
    assert json.loads((tmp_path / LATEST).read_text())['frame']['candidates'][0][0] == 'unknown'


def test_diagnostic_child_publishes_optional_frame_without_counting_it_as_a_log(tmp_path):
    writer = DiagnosticWriter(tmp_path)
    try:
        bounds, x, y, tick, stage, bullets, bodies, lasers = geometry()
        frame = SimpleNamespace(bounds=bounds, x=x, y=y, tick=tick, stage_id=stage,
                                bullets=bullets, enemy_bodies=bodies, lasers=lasers, capture_started_ns=1)
        writer.battle_view.capture(frame, None, 4, 'held', 1)
        assert writer.battle_view.latest is None
        atomic_json(tmp_path / DEMAND, {'generation': 'test', 'expires': time.monotonic() + 1.4})
        deadline = time.monotonic() + 4
        while writer.battle_view.demand is None and time.monotonic() < deadline:
            time.sleep(.01)
        assert writer.battle_view.demand is not None
        frame.capture_started_ns = time.perf_counter_ns()
        writer.battle_view.capture(frame, None, 4, 'held', 2)
        writer.battle_view.capture(frame, None, 5, 'held', 3)  # Rate-limited.
        assert writer.submit(('event', {'kind': 'regular'}))
        while not (tmp_path / LATEST).exists() and time.monotonic() < deadline:
            time.sleep(.01)
        packet = json.loads((tmp_path / LATEST).read_text())
        assert packet['sequence'] == 2 and packet['frame']['action'] == 4
        assert writer.status()['dropped_records'] == 0
    finally:
        assert writer.close()
    assert (tmp_path / 'sampling_events.jsonl').is_file()


def test_stream_shares_one_demand_and_discards_expired_frames(tmp_path):
    class Request:
        async def is_disconnected(self):
            return False

    def decoded(packet):
        return json.loads(packet.removeprefix('data: '))

    async def exercise():
        stream = BattleViewStream(lambda: tmp_path)
        first, second = stream.events(Request()), stream.events(Request())
        try:
            assert decoded(await anext(first))['state'] == 'off'
            assert decoded(await anext(second))['state'] == 'off'
            assert stream.task is None and not (tmp_path / DEMAND).exists()
            stream.set_enabled(True)
            assert decoded(await anext(first))['state'] == 'waiting'
            assert decoded(await anext(second))['state'] == 'waiting'
            deadline = time.monotonic() + 3
            while not (tmp_path / DEMAND).exists() and time.monotonic() < deadline:
                await asyncio.sleep(.01)
            demand = json.loads((tmp_path / DEMAND).read_text())
            publish_battle_view(tmp_path, demand['generation'], time.perf_counter_ns(),
                                1, geometry(), None, 4, 'held')
            event = await asyncio.wait_for(anext(first), 1)
            assert decoded(event)['state'] == 'live'
            assert await asyncio.wait_for(anext(second), 1) == event
            assert decoded(await asyncio.wait_for(anext(first), 1))['state'] == 'waiting'
            stream.set_enabled(False)
            assert decoded(await anext(second))['state'] == 'off'
        finally:
            await first.aclose()
            await second.aclose()
            await stream.close()
        assert json.loads((tmp_path / DEMAND).read_text())['expires'] == 0

    asyncio.run(exercise())
