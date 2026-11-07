"""Real input encoding with a mocked OS boundary; never sends keys."""
from dataclasses import replace
from types import SimpleNamespace

import pytest

from th06ai import win32
from th06ai.input_timing import InputTiming, TimingWindow
from th06ai.planning import RollingPlanner
from test_planning import frame
from test_state_capture import UpdatingGameMemory, reader_for


def packets(events):
    return [(item.data.ki.wVk, item.data.ki.wScan, item.data.ki.dwFlags) for item in events]


def test_direction_and_shift_switch_is_one_ordered_batch_and_held_z_is_not_repeated(monkeypatch):
    calls = []
    def send(count, events, size):
        calls.append(packets(events))
        return count
    monkeypatch.setattr(win32, 'send_input', send)
    controller = win32.KeyController(SimpleNamespace(focused=True))
    controller.held = {'z', 'up', 'left'}
    timing = controller.set({'z', 'down', 'right', 'shift'})
    assert calls == [[(0, 0x48, 0xB), (0, 0x4B, 0xB), (0x10, 0x2A, 0),
                      (0, 0x50, 9), (0, 0x4D, 9)]]
    assert timing['events'] == 5 and timing['completed_ns'] >= timing['started_ns']
    assert controller.set(controller.held)['events'] == 0
    assert len(calls) == 1
    controller.release()
    assert len(calls) == 2 and all(flags & 2 for _, _, flags in calls[-1])
    assert not controller.held


def test_partial_batch_releases_new_and_old_keys_and_reports_failure(monkeypatch):
    calls = []
    def send(count, events, size):
        calls.append(packets(events))
        return count-1 if len(calls) == 1 else count
    monkeypatch.setattr(win32, 'send_input', send)
    controller = win32.KeyController(SimpleNamespace(focused=True))
    controller.held = {'left', 'z'}
    with pytest.raises(RuntimeError, match='键盘输入失败'):
        controller.set({'right', 'z'})
    assert not controller.held
    assert any(scan == 0x4D and flags & 2 for row in calls[1:] for _, scan, flags in row)
    assert any(vk == 0x5A and flags & 2 for row in calls[1:] for vk, _, flags in row)


def test_losing_focus_releases_only_owned_keys_before_refusing_the_change(monkeypatch):
    calls = []
    monkeypatch.setattr(win32, 'send_input', lambda count, events, size: calls.append(packets(events)) or count)
    controller = win32.KeyController(SimpleNamespace(focused=False))
    controller.held = {'right'}
    with pytest.raises(win32.GameFocusLost):
        controller.set({'left'})
    assert calls == [[(0, 0x4D, 0xB)]] and not controller.held


def test_stage_measurements_and_game_input_match_are_distinct_from_send_completion():
    timing = InputTiming()
    before = replace(frame(), capture_started_ns=1_000_000_000)
    plan = SimpleNamespace(completed_ns=1_025_000_000)
    timing.policy(1_030_000_000, 1_049_000_000)
    timing.begin(before, plan, {'right', 'shift', 'z'}, {'left', 'z'},
        {'started_ns': 1_050_000_000, 'completed_ns': 1_052_000_000, 'send_ms': 2., 'events': 3})
    assert timing.latest['observation_to_send_ms'] == 52
    assert timing.latest['post_plan_to_send_ms'] == 27
    assert timing.latest['policy_ms'] == 19
    timing.observe(replace(before, input_bits=0x40, input_observed_ns=1_053_000_000))
    timing.observe(replace(before, x=196, input_bits=0x85, input_observed_ns=1_060_000_000,
                           capture_finished_ns=1_061_000_000))
    result = timing.finish()
    assert result['game_input_state'] == 'matched'
    assert result['game_input_match_ms'] == 8  # Extra shot bit is independent.
    assert result['first_motion_observed_ms'] == 9
    assert timing.policy_ms is None


def test_missing_confirmation_is_unknown_and_cannot_invent_a_zero_delay():
    timing = InputTiming()
    timing.begin(frame(), None, {'up'}, {'down'},
        {'started_ns': 1, 'completed_ns': 2, 'send_ms': .001, 'events': 2})
    timing.observe(frame())
    result = timing.finish()
    assert result['game_input_state'] == 'unavailable'
    assert result['game_input_match_ms'] is None and result['observation_to_send_ms'] is None


def test_latency_window_recovers_after_an_outlier_and_ignores_invalid_samples():
    window = TimingWindow()
    for value in (None, -1, float('nan'), float('inf')):
        window.add(value)
    assert window.upper(16) == 16
    window.add(200)
    assert window.upper(16) == 200
    for _ in range(64):
        window.add(2)
    assert window.upper(16) == 2


def test_planning_prefix_uses_measured_tail_without_counting_policy_twice_and_can_recover():
    planner = RollingPlanner()
    current = frame()
    cold = planner.plan(current).diagnostics['delay_ticks']
    planner.record_latency({'post_plan_to_send_ms': 80, 'policy_ms': 70,
                            'movement_changed': True, 'game_input_match_ms': 20})
    planner.last_elapsed_ms = 0
    slow = planner.plan(current)
    assert slow.diagnostics['estimated_post_plan_ms'] == 80
    assert slow.diagnostics['input_confirmation_reserve_ms'] == 20
    assert slow.diagnostics['delay_ticks'] > cold
    assert slow.diagnostics['delay_ticks'] <= 7  # Policy is already in the tail.
    for _ in range(64):
        planner.record_latency({'post_plan_to_send_ms': 2})
    planner.last_elapsed_ms = 0
    assert planner.plan(current).diagnostics['delay_ticks'] < slow.diagnostics['delay_ticks']
    planner.reset_timing()
    assert not planner.tail_times.values and not planner.confirmation_times.values


def test_live_snapshot_exposes_read_bounds_and_game_input_without_requiring_an_ack():
    memory = UpdatingGameMemory(advancing=False)
    reader = reader_for(memory)
    captured = reader.read()
    assert 0 < captured.capture_started_ns <= captured.input_observed_ns <= captured.capture_finished_ns
    assert captured.input_bits == 0x84
    original = memory.read
    def missing(address, size):
        from th06ai.native import INPUT_WORD
        if address == INPUT_WORD:
            raise OSError('optional input diagnostic unavailable')
        return original(address, size)
    memory.read = missing
    latest = reader.read()
    assert latest.stage_active and latest.input_bits is None
    assert reader.snapshot_diagnostics['total_elapsed_ms'] >= 0


def test_new_latency_feedback_only_changes_the_next_observation_and_preserves_sampled_plan(monkeypatch):
    from th06ai import planning as plan_module, environment
    from th06ai.observations import ObservationBuilder
    from test_sampling_timing import game
    monkeypatch.setattr(plan_module, 'clock_ns', lambda: 1_000_000_000)
    monkeypatch.setattr(environment, 'clock_ns', lambda: 1_040_000_000)
    before = replace(frame(tick=100), capture_started_ns=1_000_000_000,
                     capture_finished_ns=1_000_000_000)
    dispatched = replace(before, input_bits=0x80, input_observed_ns=1_041_000_000,
                          capture_finished_ns=1_042_000_000)
    completed = replace(dispatched, tick=102, x=200, capture_finished_ns=1_080_000_000)
    env, commands, _, _ = game(before, completed)
    env.builder = ObservationBuilder(planner=RollingPlanner())
    observation = env.builder.build(before)
    plan_copy = observation['planning'].copy()
    original_delay = env.builder.last_plan.diagnostics['delay_ticks']
    reads = iter((dispatched, completed))
    env.reader.read = lambda: next(reads)
    def send(keys):
        commands.append(keys)
        return {'started_ns': 1_039_000_000, 'completed_ns': 1_040_000_000, 'send_ms': 1., 'events': 2}
    env.keys.set = send
    env.record_policy_timing(1_010_000_000, 1_038_000_000)
    _, _, _, _, info = env.step(4)
    assert commands == [{'right', 'z'}]
    assert info['planning']['delay_ticks'] == original_delay
    assert info['planning_next']['delay_ticks'] > original_delay
    assert info['input_latency']['post_plan_to_send_ms'] == 40
    assert info['input_latency']['game_input_match_ms'] == 1
    assert (observation['planning'] == plan_copy).all()


def test_thread_profile_joins_capture_plan_policy_and_dispatch_for_one_action(monkeypatch):
    from th06ai.observations import ObservationBuilder
    from th06ai.thread_profile import ThreadMark, ThreadProfile
    from th06ai.log_rows import action_row
    from test_sampling_timing import game

    profile = ThreadProfile(1)
    stamps = iter(ThreadMark(i * 1_000_000, 0, i * 10) for i in range(1, 9))
    monkeypatch.setattr(profile, 'mark', lambda: next(stamps) if profile.remaining else None)
    reader = reader_for(UpdatingGameMemory(advancing=False))
    reader.thread_profile = profile
    before = reader.read()
    env, commands, _, _ = game(before, replace(before, tick=before.tick + 2))
    env.thread_profile = profile
    env.builder = ObservationBuilder(planner=RollingPlanner())
    env.builder.thread_profile = profile
    env.builder.build(before)
    started = env.begin_policy_timing()
    env.record_policy_timing(started, started + 1)
    _, _, _, _, info = env.step(4)

    assert commands == [{'right', 'z'}]
    trace = action_row(info, 1)['input_thread_timing']
    assert trace['wall_ns'] == [i * 1_000_000 for i in range(1, 9)]
    assert trace['cycles'] == [i * 10 for i in range(1, 9)]
    assert profile.remaining == 0 and profile.mark() is None


def test_thread_profile_rejects_a_policy_run_on_a_replaced_observation():
    from th06ai.thread_profile import ThreadProfile

    profile = ThreadProfile(2)
    before = replace(frame(), capture_thread_marks=(profile.mark(), profile.mark()))
    plan = object()
    profile.planned(before, plan, profile.mark())
    refreshed = replace(before)
    profile.begin_policy(refreshed)
    profile.end_policy(refreshed)
    assert profile.action(before, plan, profile.mark(), profile.mark()) is None
    assert profile.remaining == 2


def test_thread_report_ranks_the_same_slow_actions_and_preserves_missing_counters(tmp_path):
    import csv
    import json
    from th06ai.thread_profile import summarize_actions

    path = tmp_path / 'actions.csv'
    first = {'version': 1, 'thread_id': 5, 'wall_ns': [i * 1_000_000 for i in range(8)],
             'cpu_ns': [0] * 8, 'cycles': [None] * 8}
    second = dict(first, wall_ns=[2_000_000_000 + i * 2_000_000 for i in range(8)], cpu_ns=[50_000_000] * 8,
                  cycles=[100 + 10 * i for i in range(8)])
    with path.open('w', encoding='utf-8', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['input_thread_timing'])
        writer.writerows([[json.dumps(first)], ['broken'], [json.dumps(second)], ['']])

    report = summarize_actions(path)
    assert (report['actions'], report['profiled_actions'], report['discarded'], report['slowest_actions']) == (4, 2, 1, 1)
    assert report['window'] == {'wall_ms': 2014., 'thread_cpu_ms': 50.}
    assert report['capture_to_dispatch']['slowest_mean_wall_ms'] == 14.
    assert report['phases']['prepare']['thread_cycles']['samples'] == 1
    assert report['phases']['prepare']['slowest_mean_thread_cycles'] == 10


def test_readback_report_pairs_only_adjacent_same_scene_actions(tmp_path):
    import csv
    import json
    from th06ai.thread_profile import summarize_readback_actions

    path = tmp_path / 'actions.csv'
    with path.open('w', encoding='utf-8', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['steps', 'decision_stage_id', 'game_phase', 'input_observation_to_send_ms', 'inference'])
        for pair, phases in enumerate((('combat', 'combat'), ('combat', 'boss'), ('boss',))):
            for position, phase in enumerate(phases):
                method = ('synchronous', 'blocking_event')[position]
                inference = {'readback_ab': {'pair': pair, 'position': position, 'method': method},
                             'result_wait_copy_ms': 5 - position}
                writer.writerow([10 + pair * 2 + position, 1, phase, 12 + position, json.dumps(inference)])

    report = summarize_readback_actions(path)
    assert (report['tagged_actions'], report['matched_pairs'], report['excluded_pairs']) == (5, 1, 2)
    assert report['scenes'] == {'1/combat': 1}
    assert report['paired_delta_event_minus_sync']['readback_ms']['mean'] == -1
    assert report['paired_delta_event_minus_sync']['observation_to_send_ms']['p95'] == 1
    assert report['methods']['blocking_event']['policy_thread_cycles'] is None
