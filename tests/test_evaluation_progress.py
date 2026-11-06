"""Completed evaluation episodes survive a later interruption. No real game."""
import json
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from th06ai import learning
from th06ai.control import SamplingInterrupted
from th06ai.evaluation import EvaluationRound
from th06ai.game_settings import GameSetup
from th06ai.native import GameResources
from test_learning import LearningFixture
from test_rewards import frame


def test_interrupted_evaluation_keeps_completed_episodes_and_checkpoint_identity(tmp_path):
    env=LearningFixture()
    env.episode_seconds=180
    original=env.reset
    calls=[]
    def reset():
        calls.append(True)
        if len(calls)>1:
            raise RuntimeError('simulated reset interrupted')
        return original()
    env.reset=reset
    model=SimpleNamespace(predict=lambda *a,**k:(0,None),source_checkpoint='history.zip',
            source_checkpoint_sha256='example-hash',source_checkpoint_steps=43914)
    with pytest.raises(RuntimeError,match='simulated reset'):
        learning.evaluate(model,env,tmp_path,lambda **_:None,lambda:None,episodes=3)
    progress=json.loads((tmp_path/'evaluation_progress.json').read_text(encoding='utf-8'))
    assert not progress['complete']
    assert progress['episodes_completed']==1 and progress['episodes_requested']==3
    assert progress['checkpoint']['steps']==43914
    assert progress['checkpoint']['sha256']=='example-hash'
    assert not (tmp_path/'evaluation.json').exists()


def test_completed_evaluation_records_median_and_does_not_call_time_cap_a_clear(tmp_path):
    env=LearningFixture()
    env.episode_seconds=180
    model=SimpleNamespace(predict=lambda *a,**k:(0,None))
    result=learning.evaluate(model,env,tmp_path,lambda **_:None,lambda:None,episodes=2)
    assert result['complete'] and result['episodes_completed']==2
    assert result['median_survival_seconds']==result['mean_survival_seconds']
    assert result['time_limit_completions']==0
    assert (tmp_path/'evaluation.json').exists()


def story_frame(stage, tick, *, power=14, boss_slot=None):
    enemies = [] if boss_slot is None else [[boss_slot, 180, 90, 100, 100, 100, 31, 1]]
    return replace(frame(tick=tick), stage_id=stage, phase='combat' if boss_slot is None else 'boss',
                   game_mode='story', difficulty='normal', character='reimu_a',
                   resources=GameResources(0, power, 3, 2, 0),
                   enemies=np.asarray(enemies, dtype=np.float32).reshape(-1, 8))


def test_evaluation_separates_stage_entry_input_confirmation_and_terminal_miss(tmp_path):
    class Story(LearningFixture):
        episode_seconds = 0
        game_setup = GameSetup()

        def reset(self):
            self.current = story_frame(1, 9019, power=28, boss_slot=3)
            self.frames = iter((
                replace(self.current, tick=0, stage_id=None, stage_active=False, phase='unknown'),
                replace(story_frame(2, 2), difficulty=None),
                story_frame(2, 4),
                replace(story_frame(2, 6), state=2, phase='dead'),
            ))
            return self.builder.build(self.current), {}

        def step(self, action):
            self.current = next(self.frames)
            if self.current.state != 2:
                raise SamplingInterrupted()
            info = {'stage_id': 2, 'decision_stage_id': 2, 'native_tick_before': 4,
                    'native_tick_dispatch': 5, 'native_tick': 6, 'died': True,
                    'episode_end_reason': 'death', 'episode_bombs_used': 1,
                    'input_latency': {'game_input_state': 'matched', 'game_input_match_tick': 6}}
            return self.builder.build(self.current), 0., True, False, info

    seen = []
    env = Story()
    model = SimpleNamespace(predict=lambda obs, *, deterministic: (0, None) if deterministic else pytest.fail())
    report = learning.evaluate(model, env, tmp_path, lambda **state: seen.append(state), lambda: None, episodes=1)
    saved = json.loads((tmp_path / 'evaluation.json').read_text(encoding='utf-8'))
    row = saved['episodes'][0]
    assert report['misses'] == 1 and report['bombs_used'] == 1 and saved['max_stage_reached'] == 2
    assert row['start_matches_requested'] is True and saved['rounds_clearing_stage'] == {'1': 1}
    assert row['confirmed_stage_transitions'] == [{'from_stage': 1, 'to_stage': 2, 'before_tick': 9019, 'after_tick': 4}]
    stage2 = row['stage_progress'][1]
    assert stage2['first_observed_tick'] == 2
    assert stage2['first_sample'] == {'decision_tick': 4, 'dispatch_tick': 5, 'sample_tick': 6, 'input_state': 'matched'}
    assert stage2['first_matched_input_tick'] == 6
    assert saved['boss_power_summary'] == [{'stage_id': 1, 'appearance': 1, 'samples': 1, 'median_power': 28}]
    assert any(state.get('evaluation_round', {}).get('max_stage_reached') == 2 for state in seen)


def test_boss_power_groups_use_observed_combat_between_appearances():
    start = story_frame(1, 100, power=19, boss_slot=3)
    progress = EvaluationRound(start, GameSetup())
    progress.observe(story_frame(1, 110, power=24, boss_slot=4))
    progress.observe(story_frame(1, 300, power=34, boss_slot=3))
    for tick in range(306, 427, 6):
        progress.observe(story_frame(1, tick, power=39))
    progress.observe(replace(story_frame(1, 432, boss_slot=3), resources=None))
    observations = progress.report()['stage_progress'][0]['boss_power_observations']
    assert [(boss['native_tick'], boss['power']) for boss in observations] == [(100, 19), (432, None)]


@pytest.mark.parametrize('invalid', ('stage_skip', 'settings_change', 'verified_menu'))
def test_unverified_stage_changes_do_not_count_as_clears(invalid):
    progress = EvaluationRound(story_frame(1, 9019), GameSetup())
    if invalid == 'verified_menu':
        progress.observe(replace(story_frame(1, 0), phase='title', stage_active=False))
    stage = 3 if invalid == 'stage_skip' else 2
    arriving = story_frame(stage, 2)
    if invalid == 'settings_change':
        arriving = replace(arriving, difficulty='hard')
    progress.observe(arriving)
    report = progress.report()
    assert report['max_stage_reached'] == stage and report['confirmed_stage_transitions'] == []


def test_extra_start_uses_its_native_stage_and_does_not_count_as_a_clear():
    start = replace(story_frame(7, 2), difficulty='extra')
    report = EvaluationRound(start, GameSetup(difficulty='extra')).report()
    assert report['start_matches_requested'] is True
    assert report['max_stage_reached'] == 7 and report['confirmed_stage_transitions'] == []


def test_practice_return_is_a_distinct_outcome_and_finishes_without_another_game_read(tmp_path):
    class Practice(LearningFixture):
        episode_seconds = 0
        game_setup = GameSetup(mode='practice')

        def reset(self):
            self.current = replace(story_frame(1, 2), game_mode='practice')
            return self.builder.build(self.current), {}

        def step(self, action):
            raise SamplingInterrupted(episode_end={'episode_end_reason': 'practice_return', 'stage_id': 1})

        def release_input(self):
            self.released = True

        def pause(self):
            pytest.fail('The final report must not need a game read')

    env = Practice()
    report = learning.evaluate(SimpleNamespace(predict=lambda *args, **kwargs: (0, None)), env, tmp_path,
                               lambda **_: None, lambda: None, episodes=1)
    assert env.released and report['complete']
    assert report['round_end_counts'] == {'practice_return': 1}
    assert report['rounds_clearing_stage'] == {} and report['misses'] == 0
