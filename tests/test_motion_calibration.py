"""CPU/fake snapshots only. These tests never connect to a game or send keys."""
from dataclasses import replace
from types import SimpleNamespace

import pytest

from th06ai.motion import held_speed, measured_speed
from th06ai.worker import measure_movement
from test_planning import frame


def test_unacknowledged_switch_is_not_counted_as_a_slow_movement_frame():
    before = replace(frame(x=180, tick=100), input_bits=0x40)
    confirmed = replace(before, x=182, tick=103, input_bits=0x84)
    after = replace(confirmed, x=186, tick=105)
    assert held_speed(before, confirmed, {'right','shift'}) is None
    assert held_speed(confirmed, after, {'right','shift'}) == 2
    assert held_speed(confirmed, replace(after, input_bits=0x80), {'right','shift'}) is None


def test_motion_measurement_rejects_walls_menus_death_and_inconsistent_samples():
    before = replace(frame(x=374, tick=100), input_bits=0x80)
    assert held_speed(before, replace(before, x=376, tick=102), {'right'}) is None
    before = replace(before, x=180)
    for end in (replace(before, tick=102, paused=True), replace(before, tick=102, state=2),
                replace(before, tick=99), replace(before, tick=102, input_bits=None)):
        assert held_speed(before, end, {'right'}) is None
    assert measured_speed([None,2,2,2,2,2,2,2.5]) == 2
    with pytest.raises(ValueError):
        measured_speed([1,2,3,4,5,6,7,8,9])


def test_real_calibration_probe_measures_only_after_game_input_confirmation():
    start = replace(frame(x=180, tick=100), input_bits=0x40)
    snapshots = [start, replace(start,tick=101,x=176), replace(start,tick=102,x=172)]
    snapshots += [replace(start,tick=103+i,x=174+2*i,input_bits=0x84) for i in range(10)]
    captures = iter(snapshots)
    sent, released = [], []
    env = SimpleNamespace(reader=SimpleNamespace(read=lambda:next(captures)),
            keys=SimpleNamespace(set=lambda keys:sent.append(keys), release=lambda:released.append(True)),
            lease=lambda:None, _wait=lambda _:None)
    speed, probe = measure_movement(env, focused=True)
    assert speed == 2 and probe['measurement_version'] == 2
    assert probe['samples_px_per_tick'] == [2]*9
    assert sent == [{'right','shift'}] and released == [True]


def test_calibration_probe_releases_keys_on_interruption():
    captures = iter([frame(), replace(frame(),state=2)])
    released=[]
    env=SimpleNamespace(reader=SimpleNamespace(read=lambda:next(captures)),
        keys=SimpleNamespace(set=lambda _:None,release=lambda:released.append(True)),
        lease=lambda:None,_wait=lambda _:None)
    with pytest.raises(RuntimeError):
        measure_movement(env)
    assert released == [True]
