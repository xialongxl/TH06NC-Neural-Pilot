"""Sidecar snapshots use completed decisions and never require a game or GPU."""
import asyncio
import json
import time
from types import SimpleNamespace

import numpy as np

from th06ai.battle_view import (COMMENTARY_LATEST, DEMAND, LATEST, BattleViewPublisher, BattleViewStream,
                                _frame_context, publish_battle_commentary, publish_battle_view)
from th06ai.diagnostic_writer import DiagnosticWriter
from th06ai.files import atomic_json
from th06ai.navigation import NAV_FIELDS
from th06ai.native import GameResources
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
        writer.battle_view.record_miss(1, stage, tick - 1)
        assert not (tmp_path / COMMENTARY_LATEST).exists()
        atomic_json(tmp_path / DEMAND, {'generation': 'test', 'expires': time.monotonic() + 1.4})
        deadline = time.monotonic() + 4
        while writer.battle_view.demand is None and time.monotonic() < deadline:
            time.sleep(.01)
        assert writer.battle_view.demand is not None
        frame.capture_started_ns = time.perf_counter_ns()
        after = SimpleNamespace(tick=tick + 2, enemies=np.array([[0, 192, 50, 1500, 6000, 0, 0, 1]]),
                                resources=GameResources(0, 0, 0, 2, 0))
        writer.battle_view.capture(frame, None, 4, 'held', 2, after=after)
        writer.battle_view.capture(frame, None, 5, 'held', 3)  # Rate-limited.
        assert writer.submit(('event', {'kind': 'regular'}))
        while not (tmp_path / LATEST).exists() and time.monotonic() < deadline:
            time.sleep(.01)
        packet = json.loads((tmp_path / LATEST).read_text())
        assert packet['sequence'] == 2 and packet['frame']['action'] == 4
        assert packet['frame']['tick'] == 42
        assert packet['frame']['context'] == {'tick': 44, 'boss_hp_percent': 25.0, 'current_spellcard': None,
                                              'bombs_remaining': 0, 'miss_count': 1}
        assert writer.status()['dropped_records'] == 0
        writer.battle_view.record_miss(2, stage, tick + 3)
    finally:
        assert writer.close()
    assert (tmp_path / 'sampling_events.jsonl').is_file()
    event, = json.loads((tmp_path / COMMENTARY_LATEST).read_text())['events']
    assert (event['kind'], event['seq'], event['tick'], event['miss_count']) == ('miss', 1, 45, 2)
    assert writer.status()['written_actions'] == writer.status()['written_deaths'] == 0


def test_commentary_reports_net_resources_and_deduplicates_misses(tmp_path):
    publisher = BattleViewPublisher(tmp_path)
    publisher.demand = ('test', time.monotonic() + 1)
    before = SimpleNamespace(resources=GameResources(0, 10, 2, 1, 65535))
    after = SimpleNamespace(resources=GameResources(100, 18, 3, 1, 1), stage_id=2, tick=44)
    publisher.record_resource_gain(before, after)
    publisher.record_resource_gain(after, after)
    publisher.record_miss(4, 2, 45)
    publisher.record_miss(4, 2, 46)
    _, (gain, miss) = publisher.next_commentary_job()
    assert {key: value for key, value in gain.items() if key != 'captured_ns'} == {
        'seq': 1, 'kind': 'resource_gain', 'stage': 2, 'tick': 44,
        'power_delta': 8, 'point_items_delta': 2, 'bombs_delta': 1}
    assert miss['tick'] == 45 and miss['miss_count'] == 1
    assert publisher.next_commentary_job(force=True) is None


def test_unknown_or_ambiguous_boss_health_is_null():
    boss = [0, 192, 50, 1500, 6000, 0, 0, 1]
    frame = SimpleNamespace(tick=44, resources=None, enemies=[boss, boss])
    assert _frame_context(frame, 0) == {'tick': 44, 'boss_hp_percent': None, 'current_spellcard': None,
                                       'bombs_remaining': None, 'miss_count': 0}
    frame.enemies = [[*boss[:4], 0, *boss[5:]]]
    assert _frame_context(frame, 0)['boss_hp_percent'] is None


def test_stream_shares_one_demand_and_discards_expired_frames(tmp_path):
    class Request:
        async def is_disconnected(self):
            return False

    def decoded(packet):
        return json.loads(packet.removeprefix('data: '))

    async def exercise():
        stream = BattleViewStream(lambda: (tmp_path, True))
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


def test_commentary_delivers_final_event_and_resumes_after_last_event_id(tmp_path):
    class Request:
        def __init__(self, headers=None):
            self.headers = headers or {}

        async def is_disconnected(self):
            return False

    def miss(seq, age_seconds=0):
        return {'seq': seq, 'captured_ns': time.perf_counter_ns() - age_seconds * 1_000_000_000,
                'kind': 'miss', 'stage': 3, 'tick': 200, 'miss_count': seq}

    async def exercise():
        run = tmp_path / 'fixture-train'
        run.mkdir()
        publish_battle_commentary(run, [miss(1, age_seconds=6), miss(2), miss(3)])
        stream = BattleViewStream(lambda: (run, False))  # Finalized run, no new frame demand.
        stream.set_enabled(True)
        first = stream.events(Request())
        resumed = stream.events(Request({'last-event-id': 'fixture-train:2'}))
        try:
            for subscriber in (first, resumed):
                assert json.loads((await anext(subscriber)).removeprefix('data: '))['state'] == 'waiting'
            second = await asyncio.wait_for(anext(first), 1)
            third = await asyncio.wait_for(anext(resumed), 1)
            assert second.startswith('event: commentary\nid: fixture-train:2\n')
            assert third.startswith('event: commentary\nid: fixture-train:3\n')
            assert await asyncio.wait_for(anext(first), 1) == third
            payload = json.loads(third.split('data: ', 1)[1])
            assert payload.pop('expires_at_ms') > time.time_ns() // 1_000_000
            assert payload == {'version': 1, 'run_id': 'fixture-train', 'seq': 3, 'event_id': 'fixture-train:3',
                               'kind': 'miss', 'stage': 3, 'tick': 200, 'miss_count': 3}
            assert not (run / DEMAND).exists()
            stream.set_enabled(False)
            assert json.loads((await anext(resumed)).removeprefix('data: '))['state'] == 'off'
        finally:
            await first.aclose()
            await resumed.aclose()
            await stream.close()

    asyncio.run(exercise())
