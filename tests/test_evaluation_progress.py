"""Completed evaluation episodes survive a later interruption. No real game."""
import json
from types import SimpleNamespace

import pytest

from th06ai import learning
from test_learning import LearningFixture


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
