"""Measured input timing only; no OS calls, waits, action changes or stops."""
from collections import deque
import math
import time

clock_ns = time.perf_counter_ns
FRAME_MS = 1000/60
MOVEMENT_MASK = 0xF4
# NC's pinned movement path: up/down/left/right=10/20/40/80, focus=04.
# Right=80 and focused right=84 also match the retained real calibration.
INPUT_BITS = {"up": 0x10, "down": 0x20, "left": 0x40, "right": 0x80, "shift": 4}


class TimingWindow:
    def __init__(self, size=64):
        self.values = deque(maxlen=size)
        self._ordered = None

    def add(self, milliseconds):
        if milliseconds is not None and math.isfinite(milliseconds) and milliseconds >= 0:
            self.values.append(float(milliseconds))
            self._ordered = None

    def upper(self, default):
        return self.quantile(.95, default)

    def quantile(self, fraction, default):
        if not self.values:
            return default
        if self._ordered is None:self._ordered = sorted(self.values)
        ordered = self._ordered
        return ordered[max(0, min(len(ordered)-1, math.ceil(fraction*len(ordered))-1))]

    def clear(self):
        self.values.clear()
        self._ordered = None


class InputTiming:
    def __init__(self):
        self.policy_ms = None
        self.latest = {}
        self.observation_times = TimingWindow()
        self.policy_times = TimingWindow()
        self.send_times = TimingWindow()
        self.policy_bounds = (None, None)
        self._work_frame = None
        self._work = {}

    def record_work(self, frame, **milliseconds):
        if frame is not self._work_frame:
            self._work_frame, self._work = frame, {}
        self._work.update(milliseconds)

    def policy(self, started_ns, completed_ns):
        self.policy_ms = (completed_ns-started_ns)/1e6
        self.policy_bounds = (started_ns, completed_ns)

    def begin(self, frame, plan, keys, previous_keys, dispatch):
        self.sent_ns = dispatch['completed_ns']
        self.expected = sum(INPUT_BITS.get(key, 0) for key in keys)
        self.changed = any((key in keys) != (key in previous_keys) for key in INPUT_BITS)
        self.start_position = None
        self.motion_ms = None
        self.match_ms = None
        self.samples = 0
        captured = getattr(frame, 'capture_started_ns', 0)
        planned = getattr(plan, 'completed_ns', 0)
        self.latest = {'version': 3, 'policy_ms': self.policy_ms,
                'pre_policy_work': dict(self._work) if frame is self._work_frame else {},
                'read_breakdown': dict(getattr(frame, 'capture_metrics', {})),
                'policy_started_ns': self.policy_bounds[0], 'policy_completed_ns': self.policy_bounds[1],
                'plan_started_ns': getattr(plan, 'started_ns', 0) or None,
                'plan_completed_ns': planned or None,
                'observation_started_ns': captured or None, 'send_started_ns': dispatch['started_ns'],
                'send_completed_ns': self.sent_ns,
                'observation_to_send_ms': (self.sent_ns-captured)/1e6 if 0 < captured <= self.sent_ns else None,
                'post_plan_to_send_ms': (self.sent_ns-planned)/1e6 if 0 < planned <= self.sent_ns else None,
                'send_ms': dispatch['send_ms'], 'input_events': dispatch['events'],
                'movement_changed': self.changed, 'expected_movement_bits': self.expected,
                'game_input_state': 'pending' if self.changed else 'held',
                'game_input_match_ms': None, 'game_input_match_tick': None,
                'observation_native_tick': frame.tick, 'first_motion_observed_ms': None}
        self.observation_times.add(self.latest['observation_to_send_ms'])
        self.policy_times.add(self.policy_ms)
        if dispatch['events']:
            self.send_times.add(dispatch['send_ms'])
        self.latest.update(observation_to_send_p95_ms=self.observation_times.upper(None),
                policy_p95_ms=self.policy_times.upper(None), send_p95_ms=self.send_times.upper(None))
        # An old policy measurement cannot silently leak across pause/reset.
        self.policy_ms = None
        self.policy_bounds = (None, None)
        self._work_frame, self._work = None, {}

    def observe(self, frame):
        self.samples += 1
        bits = getattr(frame, 'input_bits', None)
        sampled = getattr(frame, 'input_observed_ns', 0)
        if bits is not None:
            self.latest['observed_game_input_bits'] = int(bits)
            if self.changed and self.match_ms is None and sampled >= self.sent_ns and (bits & MOVEMENT_MASK) == self.expected:
                self.match_ms = (sampled-self.sent_ns)/1e6
                self.latest.update(game_input_state='matched', game_input_match_ms=self.match_ms,
                                   game_input_match_tick=frame.tick,
                                   post_plan_to_input_ms=(sampled-self.latest['plan_completed_ns'])/1e6
                                   if self.latest['plan_completed_ns'] and sampled >= self.latest['plan_completed_ns'] else None)
        if self.start_position is None:
            self.start_position = (frame.x, frame.y)
        elif self.motion_ms is None and getattr(frame, 'capture_finished_ns', 0) >= self.sent_ns:
            dx, dy = frame.x-self.start_position[0], frame.y-self.start_position[1]
            if dx*dx+dy*dy > .01:
                self.motion_ms = (frame.capture_finished_ns-self.sent_ns)/1e6
                self.latest['first_motion_observed_ms'] = self.motion_ms
        self.latest['samples_after_send'] = self.samples

    def finish(self):
        if self.changed and self.match_ms is None:
            self.latest['game_input_state'] = 'not_observed' if 'observed_game_input_bits' in self.latest else 'unavailable'
        return dict(self.latest)
