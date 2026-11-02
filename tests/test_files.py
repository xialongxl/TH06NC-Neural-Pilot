import threading
import json
import time

import numpy as np
import pytest

from th06ai.files import atomic_json, read_json


@pytest.mark.parametrize("value", [np.float16(.125), np.float32(.125), np.float64(.125),
                                    np.int32(7), np.int64(2**40), np.uint64(2**60), np.bool_(True)])
def test_numpy_metrics_remain_json_numbers_booleans_and_arrays(tmp_path, value):
    path = tmp_path / "worker.json"
    atomic_json(path, {"metrics": {"value": value}, "history": np.array([.125, .5], np.float32)})
    data = json.loads(path.read_text(encoding="utf-8"))
    assert type(data["metrics"]["value"]) is type(value.item())
    assert data["metrics"]["value"] == value.item()
    assert data["history"] == [.125, .5]


def test_unknown_objects_are_not_silently_stringified_or_overwrite_previous_status(tmp_path):
    path = tmp_path / "worker.json"
    atomic_json(path, {"valid": True})
    with pytest.raises(TypeError, match="not JSON serializable"):
        atomic_json(path, {"invalid": object()})
    assert read_json(path) == {"valid": True}


@pytest.mark.parametrize("boss", [False, True])
def test_first_real_damage_feedback_flows_through_background_heartbeat(tmp_path, boss):
    from th06ai.worker import Heartbeat
    from th06ai.lease import SharedLease
    from th06ai.rewards import RewardTracker
    from test_combat import enemy_frame
    tracker = RewardTracker()
    tracker.reset(enemy_frame(hp=100, maximum=100, phase=100, boss=boss))
    hit = tracker.step(enemy_frame(tick=102, hp=90, maximum=100, phase=100, boss=boss), 0, False)
    json.dumps({"reward_components": hit.components, "diagnostics": tracker.diagnostics()})
    owner = SharedLease(create=True)
    heartbeat = Heartbeat(tmp_path, owner)
    try:
        owner.publish(time.monotonic(), "GPU-test")
        heartbeat.update(reward_components=hit.components, diagnostics=tracker.diagnostics(),
                         model_metrics={"value": np.float32(.125), "steps": np.int64(1)})
        deadline = time.monotonic() + 2
        while time.monotonic() < deadline:
            data = read_json(tmp_path / "worker.json")
            if data.get("reward_components", {}).get("damage", 0) > 0 or heartbeat.error:
                break
            time.sleep(.01)
        assert heartbeat.error is None
        assert data["reward_components"]["damage"] > 0
        assert data["diagnostics"]["boss_damage" if boss else "enemy_damage"] == 10
        assert data["model_metrics"] == {"value": .125, "steps": 1}
        heartbeat.lease()  # Valid numeric metrics must not invalidate protection.
    finally:
        heartbeat.close()
        owner.close()


def test_concurrent_dashboard_reads_do_not_break_atomic_status_updates(tmp_path):
    path = tmp_path / "status.json"
    atomic_json(path, {"sequence": -1})
    done = threading.Event()
    errors = []
    def reader():
        while not done.is_set():
            item = read_json(path)
            if item and not isinstance(item.get("sequence"), int):
                errors.append(item)
    thread = threading.Thread(target=reader)
    thread.start()
    try:
        for i in range(300):
            atomic_json(path, {"sequence": i})
    finally:
        done.set()
        thread.join(timeout=1)
    assert not errors
    assert read_json(path) == {"sequence": 299}
