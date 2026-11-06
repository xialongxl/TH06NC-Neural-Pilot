"""Evidence from committed observations during a fixed-checkpoint evaluation."""
from collections import Counter, defaultdict
from statistics import median

from .game_phase import phase_of
from .game_settings import MENU_PHASES, observed_settings
from .stage import decode_stage

EVALUATION_REPORT_VERSION = 1
BOSS_ABSENCE_TICKS = 120


class EvaluationRound:
    def __init__(self, frame, setup):
        self.start_settings = observed_settings(frame) if frame is not None and frame.stage_active else None
        self.start_matches_requested = None
        if setup is not None and self.start_settings is not None and all(
                self.start_settings[key] is not None for key in ('mode', 'difficulty', 'character', 'stage')):
            expected_stage = 7 if setup.difficulty == 'extra' else setup.stage
            self.start_matches_requested = setup.matches(frame) and frame.stage_id == expected_stage
        self.stages = {}
        self.transitions = []
        self._frame = None
        self._previous_stage = None
        self._boss_stage = None
        self._boss_absent_since = self._boss_absent_tick = None
        self.observe(frame)

    def observe(self, frame):
        if frame is None or frame is self._frame:
            return False
        self._frame = frame
        phase = phase_of(frame)
        if (phase in MENU_PHASES and phase != 'menu_transition') or phase == 'continue':
            self._previous_stage = None
            self._boss_absent_since = self._boss_absent_tick = None
            return False
        stage = decode_stage(frame.stage_id) if frame.stage_active else None
        if stage is None:
            return False
        identity = (frame.game_mode, frame.difficulty, frame.character)
        record = self.stages.get(stage)
        changed = record is None
        if changed:
            record = {'stage_id': stage, 'first_observed_tick': frame.tick, 'first_sample': None,
                      'first_matched_input_tick': None, 'boss_power_observations': []}
            self.stages[stage] = record
        if all(identity):
            if self._previous_stage is not None:
                previous_stage, previous_tick, previous_identity = self._previous_stage
                if (identity == previous_identity and identity[0] == 'story' and 1 <= previous_stage < 6
                        and stage == previous_stage + 1
                        and all(event['to_stage'] != stage for event in self.transitions)):
                    self.transitions.append({'from_stage': previous_stage, 'to_stage': stage,
                                             'before_tick': previous_tick, 'after_tick': frame.tick})
                    changed = True
            self._previous_stage = (stage, frame.tick, identity)

        if stage != self._boss_stage:
            self._boss_stage = stage
            self._boss_absent_since = self._boss_absent_tick = None
        if frame.paused or phase not in ('combat', 'boss', 'dialogue', 'script_wait'):
            return changed
        bosses = frame.enemies[frame.enemies[:, 7] != 0, 0]
        contiguous = self._boss_absent_tick is not None and 0 <= frame.tick - self._boss_absent_tick <= 15
        if not len(bosses):
            if phase == 'combat':
                if not contiguous:
                    self._boss_absent_since = frame.tick
                self._boss_absent_tick = frame.tick
            return changed

        # A new visibility window needs observed Boss-free combat. A slot
        # change, a missing frame, or a phase HP reset alone is insufficient.
        separated = contiguous and self._boss_absent_tick - self._boss_absent_since >= BOSS_ABSENCE_TICKS
        self._boss_absent_since = self._boss_absent_tick = None
        observations = record['boss_power_observations']
        if not observations or separated:
            observations.append({'appearance': len(observations) + 1, 'native_tick': frame.tick,
                                 'power': frame.resources.power if frame.resources is not None else None,
                                 'slots': [int(slot) for slot in bosses]})
            return True
        return changed

    def record_action(self, info):
        stage = decode_stage(info.get('stage_id'))
        if stage not in self.stages or stage != info.get('decision_stage_id'):
            return False
        record = self.stages[stage]
        timing = info.get('input_latency') or {}
        changed = record['first_sample'] is None
        if changed:
            record['first_sample'] = {'decision_tick': info.get('native_tick_before'),
                                      'dispatch_tick': info.get('native_tick_dispatch'),
                                      'sample_tick': info.get('native_tick'),
                                      'input_state': timing.get('game_input_state')}
        matched_tick = timing.get('game_input_match_tick')
        if record['first_matched_input_tick'] is None and timing.get('game_input_state') == 'matched' and matched_tick is not None:
            record['first_matched_input_tick'] = matched_tick
            return True
        return changed

    def report(self):
        return {'start_game_settings': self.start_settings, 'start_matches_requested': self.start_matches_requested,
                'max_stage_reached': max(self.stages, default=None),
                'confirmed_stage_transitions': list(self.transitions),
                'stage_progress': [{**record, 'boss_power_observations': list(record['boss_power_observations'])}
                                   for record in self.stages.values()]}


def summarize_rounds(rows):
    reached, cleared = Counter(), Counter()
    boss_power = defaultdict(list)
    for row in rows:
        cleared.update(event['from_stage'] for event in row.get('confirmed_stage_transitions', []))
        for stage in row.get('stage_progress', []):
            stage_id = stage['stage_id']
            reached[stage_id] += 1
            for boss in stage['boss_power_observations']:
                if boss['power'] is not None:
                    boss_power[stage_id, boss['appearance']].append(boss['power'])
    return {'max_stage_reached': max(reached, default=None), 'misses': sum(bool(row['died']) for row in rows),
            'round_end_counts': dict(Counter(row.get('episode_end_reason') or 'unknown' for row in rows)),
            'rounds_reaching_stage': dict(sorted(reached.items())),
            'rounds_clearing_stage': dict(sorted(cleared.items())),
            'boss_power_summary': [{'stage_id': stage, 'appearance': appearance,
                                    'samples': len(values), 'median_power': median(values)}
                                   for (stage, appearance), values in sorted(boss_power.items())]}
