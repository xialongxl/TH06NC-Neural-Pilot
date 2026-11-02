"""Cooperative training controls. The independent hardware guard stays active."""
from __future__ import annotations

import time
from contextlib import contextmanager
from pathlib import Path


class SamplingInterrupted(Exception):
    """Re-observe after a pause/clock reset, without inventing a transition."""
    def __init__(self, *args, stage_boundary=False, bootstrap_boundary=False):
        super().__init__(*args)
        self.stage_boundary = stage_boundary
        self.bootstrap_boundary = bootstrap_boundary or stage_boundary


class FinishTraining(Exception):
    """An explicit save-and-finish request, never a hardware emergency."""


class TrainingControl:
    def __init__(self, run_dir: Path, lease, status, *, clock=time.monotonic, sleep=time.sleep):
        self.run_dir, self.lease, self.status = run_dir, lease, status
        self.clock, self.sleep = clock, sleep
        self.started = None
        self.paused_total = 0.0
        self.pause_started = None
        self.busy = False
        self.finishing = False
        self.pause_reason = "用户暂停"

    def start(self):
        self.started = self.clock()

    @property
    def paused_seconds(self):
        return self.paused_total + (self.clock() - self.pause_started if self.pause_started is not None else 0)

    @property
    def active_seconds(self):
        return max(0.0, self.clock() - self.started - self.paused_seconds) if self.started is not None else 0.0

    def request_pause(self, reason="用户暂停"):
        self.pause_reason = reason
        (self.run_dir / "PAUSE").touch()

    @contextmanager
    def suspended_budget(self):
        own = self.started is not None and self.pause_started is None
        if own:
            self.pause_started = self.clock()
        try:
            yield
        finally:
            if own:
                self.paused_total += self.clock()-self.pause_started
                self.pause_started = None

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
            self.status(user_paused=False, paused_seconds=round(self.paused_total, 2))

    def _wait_paused(self, game, resume_game):
        game.pause_for_user() if hasattr(game, "pause_for_user") else game.pause()
        while (self.run_dir / "PAUSE").exists():
            self.lease()
            self.status(user_paused=True, phase="训练已暂停", message=self.pause_reason,
                        training_seconds=round(self.active_seconds, 2),
                        paused_seconds=round(self.paused_seconds, 2))
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
