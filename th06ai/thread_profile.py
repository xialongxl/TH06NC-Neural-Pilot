"""Opt-in control-thread counters. No scheduling, input or GPU control."""
from dataclasses import dataclass
from threading import get_native_id
import time


PROFILE_VERSION = 1
PHASES = ('capture', 'prepare', 'plan', 'post_plan', 'policy', 'pre_dispatch', 'dispatch')


@dataclass(frozen=True, slots=True)
class ThreadMark:
    wall_ns: int
    cpu_ns: int | None
    cycles: int | None


def thread_cycle_reader():
    import ctypes as ct
    from ctypes import wintypes as wt

    try:
        # Keep the GIL for these short counter queries so instrumentation
        # does not introduce a new handoff at every timing boundary.
        kernel = ct.PyDLL('kernel32')
        current = kernel.GetCurrentThread
        current.argtypes, current.restype = [], wt.HANDLE
        query = kernel.QueryThreadCycleTime
        query.argtypes = [wt.HANDLE, ct.POINTER(ct.c_ulonglong)]
        query.restype = wt.BOOL
    except (AttributeError, OSError):
        return None
    handle = current()
    value = ct.c_ulonglong()

    def read():
        return value.value if query(handle, ct.byref(value)) else None

    return read


class ThreadProfile:
    def __init__(self, steps):
        self.remaining = steps
        self.thread_id = get_native_id()
        self._cycles = thread_cycle_reader()
        self._plan = None
        self._policy = None

    def mark(self):
        if not self.remaining or get_native_id() != self.thread_id:
            return None
        wall_ns = time.perf_counter_ns()
        try:
            cpu_ns = time.thread_time_ns()
        except OSError:
            cpu_ns = None
        return ThreadMark(wall_ns, cpu_ns, self._cycles() if self._cycles is not None else None)

    def planned(self, frame, plan, started):
        self._plan = frame, plan, started, self.mark()

    def begin_policy(self, frame):
        started = self.mark()
        self._policy = (frame, started, None) if started is not None else None

    def end_policy(self, frame):
        if self._policy is not None and self._policy[0] is frame:
            self._policy = frame, self._policy[1], self.mark()

    def action(self, frame, plan, dispatch_started, dispatch_finished):
        policy, self._policy = self._policy, None
        capture = frame.capture_thread_marks
        planning = self._plan
        if (capture is None or planning is None or policy is None
                or planning[0] is not frame or planning[1] is not plan or policy[0] is not frame):
            return None
        marks = (*capture, *planning[2:], *policy[1:], dispatch_started, dispatch_finished)
        if any(mark is None for mark in marks):
            return None
        wall = [mark.wall_ns for mark in marks]
        if any(end < start for start, end in zip(wall, wall[1:])):
            return None
        return {'version': PROFILE_VERSION, 'thread_id': self.thread_id, 'wall_ns': wall,
                'cpu_ns': [mark.cpu_ns for mark in marks], 'cycles': [mark.cycles for mark in marks]}

    def commit(self):
        self.remaining -= 1
        if not self.remaining:
            self._plan = self._policy = None


def _parse_sample(text):
    import json

    try:
        sample = json.loads(text)
    except ValueError:
        return None
    if not isinstance(sample, dict) or sample.get('version') != PROFILE_VERSION:
        return None
    thread = sample.get('thread_id')
    if type(thread) is not int or thread <= 0:
        return None
    for key in ('wall_ns', 'cpu_ns', 'cycles'):
        values = sample.get(key)
        if not isinstance(values, list) or len(values) != len(PHASES) + 1:
            return None
        if any((type(value) is not int or value < 0) and not (value is None and key != 'wall_ns')
               for value in values):
            return None
    wall = sample['wall_ns']
    if wall[-1] == wall[0] or any(end < start for start, end in zip(wall, wall[1:])):
        return None
    return sample


def _delta(first, last):
    return last - first if first is not None and last is not None and last >= first else None


def _distribution(values):
    from statistics import median

    ordered = sorted(value for value in values if value is not None)
    if not ordered:
        return None
    rank95 = (95 * len(ordered) + 99) // 100
    return {'samples': len(ordered), 'p50': median(ordered), 'p95': ordered[rank95 - 1]}


def _phase_summary(samples, slowest, start, end):
    from statistics import mean

    def milliseconds(sample):
        return (sample['wall_ns'][end] - sample['wall_ns'][start]) / 1e6

    def cycles_in_phase(sample):
        return _delta(sample['cycles'][start], sample['cycles'][end])

    cycles = map(cycles_in_phase, slowest)
    known = [value for value in cycles if value is not None]
    return {
        'wall_ms': _distribution(milliseconds(sample) for sample in samples),
        'thread_cycles': _distribution(map(cycles_in_phase, samples)),
        'slowest_mean_wall_ms': mean(map(milliseconds, slowest)),
        'slowest_mean_thread_cycles': mean(known) if known else None,
    }


def summarize_actions(path):
    """Read action CSV rows offline, including logs that predate thread profiling."""
    import csv

    samples = []
    actions = discarded = 0
    with path.open(encoding='utf-8-sig', newline='') as stream:
        for row in csv.DictReader(stream):
            actions += 1
            if not (raw := row.get('input_thread_timing')):
                continue
            sample = _parse_sample(raw)
            if sample is None:
                discarded += 1
            else:
                samples.append(sample)
    report = {'actions': actions, 'profiled_actions': len(samples), 'discarded': discarded}
    if not samples:
        return report
    slow_count = (len(samples) + 19) // 20
    slowest = sorted(samples, key=lambda s: s['wall_ns'][-1] - s['wall_ns'][0], reverse=True)[:slow_count]
    first = min(samples, key=lambda s: s['wall_ns'][0])
    last = max(samples, key=lambda s: s['wall_ns'][-1])
    wall_ns = last['wall_ns'][-1] - first['wall_ns'][0]
    threads = sorted({sample['thread_id'] for sample in samples})
    cpu_ns = (_delta(first['cpu_ns'][0], last['cpu_ns'][-1])
              if len(threads) == 1 and wall_ns >= 1_000_000_000 else None)
    report.update(
        thread_ids=threads,
        slowest_actions=slow_count,
        window={'wall_ms': wall_ns / 1e6, 'thread_cpu_ms': cpu_ns / 1e6 if cpu_ns is not None else None},
        capture_to_dispatch=_phase_summary(samples, slowest, 0, -1),
        phases={name: _phase_summary(samples, slowest, index, index + 1) for index, name in enumerate(PHASES)},
    )
    return report


@dataclass(frozen=True, slots=True)
class _ReadbackSample:
    pair: int
    position: int
    method: str
    step: int
    stage: str | None
    phase: str | None
    died: bool
    metrics: dict


def _readback_sample(row):
    import json
    import math

    try:
        inference = json.loads(row.get('inference') or '{}')
        trial = inference['readback_ab']
        step = int(row['steps'])
        pair, position, method = trial['pair'], trial['position'], trial['method']
    except (ValueError, TypeError, KeyError):
        return None
    if (type(pair) is not int or pair < 0 or type(position) is not int or position not in (0, 1)
            or method not in ('synchronous', 'blocking_event')):
        return None

    def number(value):
        try:
            value = float(value)
        except (TypeError, ValueError):
            return None
        return value if math.isfinite(value) and value >= 0 else None

    trace = _parse_sample(row.get('input_thread_timing') or 'null')
    cycles = trace['cycles'] if trace is not None else [None] * (len(PHASES) + 1)
    metrics = {
        'observation_to_send_ms': number(row.get('input_observation_to_send_ms')),
        'policy_ms': number(row.get('input_policy_ms')),
        'readback_ms': number(inference.get('result_wait_copy_ms')),
        'policy_thread_cycles': _delta(cycles[4], cycles[5]),
        'readback_thread_cycles': number(inference.get('readback_thread_cycles')),
    }
    return _ReadbackSample(pair, position, method, step, row.get('decision_stage_id'),
                           row.get('game_phase'), bool(row.get('death_evidence')), metrics)


def summarize_readback_actions(path):
    """Compare adjacent A/B actions from the same sampled stage and phase."""
    import csv
    from collections import Counter, defaultdict
    from statistics import mean

    groups = defaultdict(list)
    with path.open(encoding='utf-8-sig', newline='') as stream:
        for row in csv.DictReader(stream):
            sample = _readback_sample(row)
            if sample is not None:
                groups[sample.pair].append(sample)
    matched = []
    scenes = Counter()
    for group in groups.values():
        if len(group) != 2:
            continue
        first, second = sorted(group, key=lambda sample: sample.position)
        if (first.position != 0 or second.position != 1 or first.method == second.method
                or second.step != first.step + 1 or first.died or second.died
                or not first.stage or first.stage != second.stage
                or first.phase not in ('combat', 'boss') or first.phase != second.phase):
            continue
        scenes[f'{first.stage}/{first.phase}'] += 1
        matched.append({first.method: first.metrics, second.method: second.metrics})
    report = {'tagged_actions': sum(len(group) for group in groups.values()),
              'matched_pairs': len(matched), 'excluded_pairs': len(groups) - len(matched),
              'scenes': dict(scenes)}
    if not matched:
        return report

    def summary(values):
        known = [value for value in values if value is not None]
        return {**_distribution(known), 'mean': mean(known)} if known else None

    baseline, trial = 'synchronous', 'blocking_event'
    keys = matched[0][baseline]
    report['methods'] = {method: {} for method in (baseline, trial)}
    report['paired_delta_event_minus_sync'] = {}
    for key in keys:
        values = [(pair[baseline][key], pair[trial][key]) for pair in matched]
        paired = [(a, b) for a, b in values if a is not None and b is not None]
        report['methods'][baseline][key] = summary(a for a, _ in paired)
        report['methods'][trial][key] = summary(b for _, b in paired)
        report['paired_delta_event_minus_sync'][key] = summary(b - a for a, b in paired)
    return report


def main():
    import argparse
    import json
    from pathlib import Path

    parser = argparse.ArgumentParser(description='Summarize control-thread timing from actions.csv')
    parser.add_argument('actions_csv', type=Path)
    parser.add_argument('--readback-ab', action='store_true', help='Compare matched live readback pairs')
    args = parser.parse_args()
    report = summarize_readback_actions if args.readback_ab else summarize_actions
    print(json.dumps(report(args.actions_csv), indent=2))


if __name__ == '__main__':
    main()
