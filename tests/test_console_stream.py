"""Delta correctness, shared sampling and caches; never starts game/GPU work."""
import asyncio
import copy
import json
import time

from th06ai import console_stream
from th06ai.console_stream import SnapshotBroker, display_state, history_delta, make_delta
from th06ai.files import atomic_json
from test_web_console import setup, session


def state():
    return {"active": True, "preview": True, "busy": True, "run_dir": "run-a", "gpu_age_seconds": 5,
            "models": [{"name": "model", "steps": 42}], "readiness": {"ready": True},
            "gpu_history": [{"stamp": 1, "temperature_c": 49}],
            "session": {"time": "clock", "monotonic": 1, "state": "running", "mode": "train",
                        "gpu": {"temperature_c": 49, "monotonic": 1},
                        "worker": {"steps": 42, "user_paused": False, "time": "clock", "monotonic": 1}}}


def test_display_dto_ignores_heartbeat_clocks_but_keeps_control_state():
    old = state()
    new = copy.deepcopy(old)
    new["gpu_age_seconds"] = 9
    new["session"]["time"] = "next"
    new["session"]["monotonic"] = 100
    new["session"]["gpu"]["monotonic"] = 100
    new["session"]["worker"].update(time="next", monotonic=100)
    assert make_delta(display_state(old), display_state(new), 1, 2) is None
    new["session"]["worker"]["user_paused"] = True
    patch = make_delta(display_state(old), display_state(new), 1, 2)
    assert patch["set"] == [[["session", "worker", "user_paused"], True]]


def test_display_stream_retains_the_actual_session_goal():
    raw=state()
    raw['session']['worker'].update(goal='nmnb',episode_bombs_used=0)
    shown=display_state(raw)
    assert shown['session']['worker']['goal']=='nmnb'
    assert shown['session']['worker']['episode_bombs_used']==0


def test_live_delta_does_not_repeat_catalog_configuration_or_history():
    old = display_state(state())
    new = copy.deepcopy(old)
    new["session"]["worker"]["steps"] = 43
    new["gpu_history"].append({"stamp": 2, "temperature_c": 50})
    patch = make_delta(old, new, 1, 2)
    assert patch["set"] == [[["session", "worker", "steps"], 43]]
    assert patch["history"] == {"mode": "append", "after": 1, "points": [{"stamp": 2, "temperature_c": 50}]}
    assert "models" not in json.dumps(patch) and "readiness" not in json.dumps(patch)
    assert old["gpu_history"] == [{"stamp": 1, "temperature_c": 49}]


def test_latency_display_keeps_real_durations_but_drops_internal_clock_values():
    raw = state()
    raw['session']['worker']['input_latency'] = {'observation_to_send_ms': 32.,
        'send_started_ns': 12345678901234567, 'policy_started_ns': 123, 'game_input_state': 'unavailable'}
    shown = display_state(raw)
    assert shown['session']['worker']['input_latency'] == {
        'observation_to_send_ms': 32., 'game_input_state': 'unavailable'}
    raw['session']['worker']['input_latency']['send_started_ns'] += 1
    assert make_delta(shown, display_state(raw), 1, 2) is None


def test_history_resets_on_run_change_or_evicted_cursor_and_handles_missing_samples():
    old = [{"stamp": 1, "temperature_c": 49}]
    assert history_delta(old, [], False) == {"mode": "reset", "points": []}
    latest = [{"stamp": i, "temperature_c": 50} for i in range(2, 202)]
    reset = history_delta(old, latest, True)
    assert reset["mode"] == "reset" and len(reset["points"]) == 180
    assert history_delta(old, copy.deepcopy(old), True) is None
    gap = history_delta(old, [*old, {"stamp": 8, "temperature_c": 50}], True)
    assert gap["points"][0]["stamp"] == 8


def test_null_and_removal_have_distinct_operations():
    patch = make_delta({"a": {"old": 1, "reason": "old"}}, {"a": {"reason": None}}, 1, 2)
    assert patch["remove"] == [["a", "old"]]
    assert patch["set"] == [[["a", "reason"], None]]


class Request:
    async def is_disconnected(self):
        return False


class Source:
    def __init__(self):
        self.value = state()
        self.calls = []

    def snapshot(self):
        self.calls.append(time.perf_counter())
        return self.value


def event(text):
    return json.loads(text.removeprefix("data: ").strip())


def test_two_clients_share_one_sampler_and_reconnect_receives_a_full_snapshot(monkeypatch):
    monkeypatch.setattr(console_stream, "VISIBLE_SECONDS", .02)
    monkeypatch.setattr(console_stream, "HEARTBEAT_SECONDS", .05)
    async def run():
        source = Source()
        broker = SnapshotBroker(source)
        a, b = broker.events(Request()), broker.events(Request())
        try:
            first = event(await asyncio.wait_for(anext(a), 1))
            second = event(await asyncio.wait_for(anext(b), 1))
            assert first == second and first["kind"] == "snapshot"
            source.value = copy.deepcopy(source.value)
            source.value["session"]["worker"]["steps"] = 43
            delta_a, delta_b = await asyncio.wait_for(asyncio.gather(anext(a), anext(b)), 1)
            assert delta_a == delta_b and event(delta_a)["kind"] == "delta"
            assert len(source.calls) <= 3
            assert all(y-x >= .017 for x, y in zip(source.calls, source.calls[1:]))
            fresh = broker.events(Request())
            try:
                reset = event(await asyncio.wait_for(anext(fresh), 1))
                assert reset["kind"] == "snapshot" and reset["state"]["session"]["worker"]["steps"] == 43
            finally:
                await fresh.aclose()
        finally:
            await a.aclose(); await b.aclose(); await broker.close()
        assert not broker.clients
    asyncio.run(run())


def test_unchanged_state_sends_only_heartbeat_and_hidden_clients_sample_slowly(monkeypatch):
    monkeypatch.setattr(console_stream, "VISIBLE_SECONDS", .01)
    monkeypatch.setattr(console_stream, "HIDDEN_SECONDS", .06)
    async def run():
        source = Source()
        broker = SnapshotBroker(source)
        stream = broker.events(Request(), hidden=True)
        try:
            assert event(await anext(stream))["kind"] == "snapshot"
            assert event(await asyncio.wait_for(anext(stream), .5))["kind"] == "heartbeat"
            assert len(source.calls) <= 3
        finally:
            await stream.aclose(); await broker.close()
    asyncio.run(run())


def test_slow_client_gets_combined_latest_delta_without_queued_history(monkeypatch):
    monkeypatch.setattr(console_stream, "VISIBLE_SECONDS", .01)
    monkeypatch.setattr(console_stream, "HIDDEN_SECONDS", .03)
    async def run():
        source = Source()
        broker = SnapshotBroker(source)
        a, slow = broker.events(Request()), broker.events(Request(), hidden=True)
        try:
            await anext(a); await anext(slow)
            for stamp in (2, 3, 4):
                source.value = copy.deepcopy(source.value)
                source.value["gpu_history"].append({"stamp": stamp, "temperature_c": 50})
                await asyncio.wait_for(anext(a), 1)
            frame = event(await asyncio.wait_for(anext(slow), 1))
            assert frame["kind"] == "delta" and [p["stamp"] for p in frame["history"]["points"]] == [2, 3, 4]
            assert "models" not in json.dumps(frame)
        finally:
            await a.aclose(); await slow.aclose(); await broker.close()
    asyncio.run(run())


def test_cached_files_invalidate_after_atomic_replacement_and_old_snapshots_stay_immutable(setup):
    root, checkpoint, service, client, calls, initial = setup
    run = session(root, worker={"steps": 42}, gpu={"monotonic": 1, "temperature_c": 49})
    before = service.snapshot()
    same = service.snapshot()
    assert same["models"] is before["models"]
    assert same["gpu_history"] is before["gpu_history"]
    atomic_json(run / "status.json", {"state": "running", "mode": "train", "monotonic": time.monotonic(),
        "worker": {"steps": 43}, "gpu": {"monotonic": 2, "temperature_c": 50}})
    after = service.snapshot()
    assert after["session"]["worker"]["steps"] == 43
    assert before["session"]["worker"]["steps"] == 42
    assert len(after["gpu_history"]) == 2 and len(before["gpu_history"]) == 1
    assert not calls


def test_unchanged_logs_return_304_and_new_data_invalidates_etag(setup):
    root, checkpoint, service, client, calls, initial = setup
    run = session(root)
    (run / "worker.log").write_text("first", encoding="utf-8")
    first = client.get("/api/logs")
    assert first.status_code == 200
    assert client.get("/api/logs", headers={"If-None-Match": first.headers["etag"]}).status_code == 304
    (run / "worker.log").write_text("first\nsecond", encoding="utf-8")
    changed = client.get("/api/logs", headers={"If-None-Match": first.headers["etag"]})
    assert changed.status_code == 200 and "second" in changed.json()["worker"]
    assert client.get("/api/bootstrap?state=false").json().keys() == {"token", "protocol"}
    assert not calls


def test_http_query_accepts_browser_protocol_strings_and_passes_visibility(setup):
    root, checkpoint, service, client, calls, initial = setup
    received = []

    async def finite_events(request, *, protocol, hidden):
        received.append((protocol, hidden))
        yield 'data: {}\n\n'

    client.app.state.stream.events = finite_events
    assert client.get('/api/events?protocol=2&hidden=true').status_code == 200
    assert client.get('/api/events?protocol=1').status_code == 200
    assert client.get('/api/events').status_code == 200
    assert received == [(2, True), (1, False), (1, False)]
    assert client.get('/api/events?protocol=3').status_code == 422
    assert not calls
