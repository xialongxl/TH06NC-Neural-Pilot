"""Cooperative training controls. The independent hardware guard stays active."""
from __future__ import annotations
from .i18n import msg

import time
from contextlib import contextmanager
from pathlib import Path

from .game_phase import phase_of


STAGE_OVERTIME_LIMIT_SECONDS = 600
SCENE_WAIT_LIMIT_SECONDS = 60


class SceneWaitBudget:
    """Cap automatic wait credit until another real sample is accepted."""

    def __init__(self):
        self._step = -1
        self._deducted = 0.0

    def remaining(self, step):
        if type(step) is not int or step < 0 or step < self._step:
            return 0.0
        if step > self._step:
            return SCENE_WAIT_LIMIT_SECONDS
        return SCENE_WAIT_LIMIT_SECONDS - self._deducted

    def deduct(self, step, elapsed):
        remaining = self.remaining(step)
        if remaining and step > self._step:
            self._step, self._deducted = step, 0.0
        seconds = min(remaining, max(0.0, elapsed))
        self._deducted += seconds
        return seconds


class SessionStopwatch:
    """Account for elapsed time using the last observed wait state."""

    def __init__(self, now):
        self._updated = now
        self._paused = False
        self._scene_step = None
        self._scene_wait = SceneWaitBudget()
        self.active_seconds = 0.0
        self.paused_seconds = 0.0
        self.scene_wait_seconds = 0.0

    def advance(self, now):
        elapsed = now - self._updated
        self._updated = now
        credited = 0.0
        if self._paused:
            credited = elapsed
        elif self._scene_step is not None:
            credited = self._scene_wait.deduct(self._scene_step, elapsed)
            self.scene_wait_seconds += credited
        self.paused_seconds += credited
        self.active_seconds += elapsed - credited

    def set_wait(self, now, *, paused=False, scene_step=None):
        self.advance(now)
        self._paused, self._scene_step = paused, scene_step


class SamplingInterrupted(Exception):
    """Re-observe after a pause/clock reset, without inventing a transition."""
    def __init__(self, *args, stage_boundary=False, bootstrap_boundary=False, episode_end=None):
        super().__init__(*args)
        self.stage_boundary = stage_boundary
        self.bootstrap_boundary = bootstrap_boundary or stage_boundary
        self.episode_end = episode_end


class FinishTraining(Exception):
    """A normal training finish, with a reason for the saved run summary."""
    def __init__(self, reason='manual'):
        super().__init__(reason)
        self.reason = reason


class TrainingControl:
    def __init__(self, run_dir: Path, lease, status, *, budget_seconds=None,
                 session_time=None, clock=time.monotonic, sleep=time.sleep):
        self.run_dir, self.lease, self.status = run_dir, lease, status
        self.clock, self.sleep = clock, sleep
        self.started = False
        self.session_time = session_time
        self._local_clock = None
        self._waiting = False
        self.busy = False
        self.finishing = False
        self.budget_seconds = budget_seconds
        self.overtime_stage = None
        self.pause_reason = msg('backend.paused_by_the_user')

    def start(self):
        if self.started:
            return
        self.started = True
        if self.session_time is None:
            self._local_clock = SessionStopwatch(self.clock())
        else:
            self.status(clock_started=True)

    def timing(self):
        if self.session_time is not None:
            return self.session_time()
        clock = self._local_clock
        if clock is not None:
            clock.advance(self.clock())
        return {'training_seconds': clock.active_seconds if clock else 0.0,
                'paused_seconds': clock.paused_seconds if clock else 0.0,
                'budget_seconds': self.budget_seconds}

    @property
    def paused_seconds(self):
        return self.timing().get('paused_seconds')

    @property
    def active_seconds(self):
        return self.timing().get('training_seconds')

    def _active_budget(self):
        timing = self.timing()
        return timing.get('training_seconds'), timing.get('budget_seconds')

    @property
    def target_reached(self):
        elapsed, budget = self._active_budget()
        return self.started and elapsed is not None and budget is not None and elapsed >= budget

    @property
    def budget_reached(self):
        return self.budget_seconds is not None and self.target_reached

    def check_budget(self, frame=None, *, boundary=None):
        if not self.started or self.budget_seconds is None:
            return
        elapsed, budget = self._active_budget()
        if elapsed is None or budget is None or elapsed < budget:
            return
        if boundary is None:
            stage = getattr(frame, 'stage_id', None)
            if elapsed >= budget + STAGE_OVERTIME_LIMIT_SECONDS:
                boundary = 'overtime_limit'
            elif frame is None or not frame.stage_active or type(stage) is not int or not 1 <= stage <= 7:
                boundary = 'time_limit'
            elif phase_of(frame) not in ('combat', 'boss', 'pause', 'dialogue', 'script_wait'):
                boundary = 'time_limit'
            elif self.overtime_stage is not None and self.overtime_stage != stage:
                boundary = 'stage_ended'
            else:
                if self.overtime_stage is None:
                    self.overtime_stage = stage
                    self.status(overtime_stage=stage)
                return
        self.lease()
        raise FinishTraining('manual' if (self.run_dir / 'FINISH').exists() else boundary)

    def request_pause(self, reason=msg('backend.paused_by_the_user')):
        self.pause_reason = reason
        (self.run_dir / "PAUSE").touch()

    @contextmanager
    def suspended_budget(self, *, scene_step=None):
        if self._waiting or (not self.started and self.session_time is None):
            yield
            return
        self._waiting = True
        try:
            if self._local_clock is not None:
                self._local_clock.set_wait(self.clock(), paused=scene_step is None, scene_step=scene_step)
            if self.session_time is not None or scene_step is not None:
                self.status(clock_paused=scene_step is None, scene_wait_step=scene_step)
            yield
        finally:
            self._waiting = False
            if self._local_clock is not None:
                self._local_clock.set_wait(self.clock())
            if self.session_time is not None or scene_step is not None:
                self.status(clock_paused=False, scene_wait_step=None)

    def gate(self, game, *, resume_game=True):
        self.lease()  # STOP or a lost safety lease must win over save/pause.
        if (self.run_dir / "FINISH").exists():
            raise FinishTraining()
        if not (self.run_dir / "PAUSE").exists():
            return False
        self.busy = True
        try:
            with self.suspended_budget():
                return self._wait_paused(game, resume_game)
        finally:
            self.busy = False
            self.status(user_paused=False)

    def _wait_paused(self, game, resume_game):
        game.pause_for_user() if hasattr(game, "pause_for_user") else game.pause()
        while (self.run_dir / "PAUSE").exists():
            self.lease()
            self.status(user_paused=True, phase=msg('backend.training_paused'), message=self.pause_reason)
            if (self.run_dir / "FINISH").exists():
                raise FinishTraining()
            self.sleep(.05)
        self.lease()
        if (self.run_dir / "FINISH").exists():
            raise FinishTraining()
        if resume_game:
            game.resume_for_user() if hasattr(game, "resume_for_user") else game.resume()
        return True


def set_paused(run_dir: Path, paused: bool):
    marker = run_dir / "PAUSE"
    if paused:
        marker.touch()
        (run_dir / 'FOCUS').unlink(missing_ok=True)
    else:
        (run_dir / 'FOCUS').touch()
        marker.unlink(missing_ok=True)
