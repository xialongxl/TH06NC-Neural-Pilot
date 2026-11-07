from __future__ import annotations
from .i18n import msg

import time
import json
from contextlib import nullcontext
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Callable

import gymnasium as gym
import numpy as np

from .config import Config, READBACK_AB_STEPS, ROOT
from .files import read_json
from .planning import PlanResult, RollingPlanner
from .native import Frame, GameReader, INPUT_WORD
from .observations import ACTION_COUNT, ACTION_TIMING_VERSION, ObservationBuilder, action_keys, make_observation_space, movement_action
from .rewards import REWARD_VERSION, RESOURCE_BUDGET, RewardTracker, residence_age
from .win32 import KeyController, VIRTUAL_KEYS, get_async_key, GameFocusLost, foreground, restore_foreground
from .control import FinishTraining, SamplingInterrupted
from .input_timing import InputTiming, clock_ns
from .game_phase import phase_of, combat_frame, PHASE_NAMES
from .stage import context_changed, stage_label
from .game_settings import GameSetup, MENU_PHASES, menu_key, observed_settings


@dataclass(slots=True)
class _ActionSample:
    action: int
    before: Frame
    after: Frame
    dispatch_tick: int
    death_probe_tick: int | None
    previous_action: int | None
    previous_keys: set[str]
    actual_previous_action: int | None
    decision_plan: PlanResult | None


class TouhouEnv(gym.Env):
    """A single real-time game, two native frames per learned action.

    Menu automation is separate from the policy. An unconfirmed reset is an
    error, never a fabricated successful environment reset.
    """
    metadata = {"render_modes": []}

    def __init__(self, config: Config, lease: Callable, status: Callable):
        super().__init__()
        self.reader = GameReader(config.game_path)
        if config.capture_backend != 'auto':
            self.reader.process.configure_reads(config.capture_backend)
        self.reader.checkpoint = lease
        self.reader.window.checkpoint = lease
        self.episode_seconds = config.episode_seconds
        from .acceleration import key_controller
        self.keys = key_controller(self.reader.window, config.input_backend)
        self.lease = lease
        self.status = status
        calibration = read_json(ROOT / "native_calibration.json")
        self.builder = ObservationBuilder(planner=RollingPlanner(config.planning,
                fast_speed=calibration.get("fast_px_per_tick", 4.),
                slow_speed=calibration.get("slow_px_per_tick", 2.)),goal=config.goal)
        self.reward_tracker = RewardTracker(feature_cache=self.builder.feature_cache)
        self.action_space = gym.spaces.Discrete(ACTION_COUNT)
        self.observation_space = make_observation_space()
        self.current: Frame | None = None
        self.start_tick = 0
        self.episode_ticks = 0
        self.first_reset = True
        self.episode_count = 0
        self.total_steps = 0
        self.episode_return = 0.0
        self.episode_steps = 0
        self.managed_pause = False
        self.training_control = None
        self._sampling = False
        self.last_action = None
        self.input_timing = InputTiming()
        self.stage_ticks = 0
        self.stage_changes = 0
        self.pending_stage_boundary = False
        self._practice_origin = None
        self._decision_refresh_used = False
        self.decision_refreshes = 0
        self.decision_context = {}
        self.goal = config.goal
        self.game_setup = config.game_setup
        self.episode_bombs_used = 0
        self.diagnostics = None
        self.diagnostics_config = config.diagnostics

    def enable_diagnostics(self, run_dir, mode):
        from .diagnostics import SessionDiagnostics
        self.diagnostics = SessionDiagnostics(run_dir, mode)
        self.reader.observer = self.diagnostics.observe
        self.reader.control_observer = self.diagnostics.observe_control
        profile_steps = self.diagnostics_config.thread_profile_steps
        if mode == 'train' and self.diagnostics_config.readback_ab:
            profile_steps = max(profile_steps, READBACK_AB_STEPS)
        if profile_steps:
            from .thread_profile import ThreadProfile
            self.thread_profile = ThreadProfile(profile_steps)
            self.reader.thread_profile = self.builder.thread_profile = self.thread_profile

    def _timing(self):
        if not hasattr(self, 'input_timing'):
            self.input_timing = InputTiming()
        return self.input_timing

    def _read_control(self):
        reader = getattr(self.reader, 'read_control', None)
        return reader() if reader is not None else self.reader.read()

    def _full_frame(self, sample):
        return sample if isinstance(sample, Frame) else self.reader.read()

    def begin_policy_timing(self):
        profile = getattr(self, 'thread_profile', None)
        if profile is not None:
            profile.begin_policy(self.current)
        return clock_ns()

    def record_policy_timing(self, started_ns, completed_ns):
        self._timing().policy(started_ns, completed_ns)
        profile = getattr(self, 'thread_profile', None)
        if profile is not None:
            profile.end_policy(self.current)

    def record_pre_policy_work(self, **milliseconds):
        self._timing().record_work(self.current, **milliseconds)

    def record_decision(self, decision, training_step=None):
        self.decision_context = {'training_step':training_step,'bomb_probability':decision.bomb_probability,
                                 'inference':dict(decision.timing)}

    def refresh_decision(self):
        """One bounded re-observation if actual native age exceeds the forecast.

        No new key is sent, no fabricated transition is stored. A value-bootstrap
        boundary keeps the unobserved old-control gap out of PPO advantages.
        """
        if not hasattr(self.reader,'read_control') or self.current is None:return
        if not self.reader.window.focused:
            self._pause_sampling(msg('backend.the_game_lost_focus_keys_were_released'))
        before = self.current
        if phase_of(before) in ('result', 'practice_stage', 'title', 'title_prompt'):
            self._round_result(before, before)
        probe = self.reader.read_control()
        self._sampling = True
        try:
            self._check_sampling_scene(probe, before)
            if probe.state == 2:return  # Preserve a real Miss for the normal step path.
            if phase_of(probe) in ('dialogue','script_wait'):
                self._advance_dialogue()
        finally:
            self._sampling = False
        if probe.paused or not probe.stage_active:
            self._pause_sampling(msg('backend.the_game_is_paused_or_changing_scenes'))
        plan = self.builder.last_plan
        diag = plan.diagnostics if plan is not None else {}
        age_ticks = probe.tick-before.tick
        late = (diag.get('delay_bounds_ticks') or [0,2])[-1]
        clearance = diag.get('estimated_clearance')
        near = diag.get('all_routes_risky') or clearance is not None and clearance<32
        changed = context_changed(before,probe) or probe.tick<before.tick
        expired = age_ticks>late+1 and (near or age_ticks>diag.get('horizon_ticks',32)/2)
        if not changed and not expired:return
        if not changed and getattr(self,'_decision_refresh_used',False):
            self.freshness_warning = {'age_ticks':age_ticks,'forecast_late_ticks':late,'retry_exhausted':True}
            diagnostics = getattr(self,'diagnostics',None)
            if diagnostics is not None:diagnostics.event(event='decision_refresh_budget',stage_id=before.stage_id,**self.freshness_warning)
            return
        latest = self.reader.read()
        self._check_sampling_scene(latest, before)
        if latest.state == 2:return
        if context_changed(before,latest) or latest.tick<before.tick:
            self._rebase_sampling(latest,before)
        if latest.paused or not latest.stage_active:
            self._pause_sampling(msg('backend.the_game_is_paused_or_changing_scenes'))
        self._account_menu_gap(latest)
        self.current = latest
        self.reward_tracker.rebase(latest)
        self._observe_control(latest,getattr(self,'last_action',None))
        self.input_timing = InputTiming()
        self._decision_refresh_used = True
        self.decision_refreshes = getattr(self,'decision_refreshes',0)+1
        diagnostics = getattr(self,'diagnostics',None)
        if diagnostics is not None:
            diagnostics.event(event='decision_refresh',stage_id=latest.stage_id,before_tick=before.tick,
                              after_tick=latest.tick,forecast_late_ticks=late,next_training_step=self.total_steps+1)
        self.status(decision_refreshes=self.decision_refreshes,
                    sampling_timing={'event':'decision_refresh','before_tick':before.tick,'after_tick':latest.tick,
                                     'forecast_late_ticks':late,'retries':1})
        raise SamplingInterrupted(bootstrap_boundary=True)

    def _consume_focus_request(self):
        control = getattr(self, 'training_control', None)
        run_dir = control.run_dir if control else getattr(self, 'run_dir', None)
        marker = run_dir / 'FOCUS' if run_dir is not None else None
        if marker is not None and marker.exists():
            marker.unlink(missing_ok=True)
            return True
        return False

    def _acquire_focus(self):
        """Finish an authorized transaction, or wait with hardware guard live."""
        self._consume_focus_request()
        if self.reader.window.focused:
            return
        self.keys.release()
        control = getattr(self, 'training_control', None)
        with control.suspended_budget() if control else nullcontext():
            if self.reader.window.focus_for_start():
                return
            try:
                while not self.reader.window.focused:
                    self.lease()
                    if control and not control.finishing and (control.run_dir / 'FINISH').exists():
                        raise FinishTraining()
                    if not getattr(getattr(self.reader, 'process', None), 'alive', True):
                        raise GameFocusLost(msg('backend.the_game_exited_keys_were_released'))
                    self.status(phase=msg('backend.waiting_for_game_focus'), message=msg('backend.automatic_focus_failed_select_focus_game_or'),
                                user_paused=True, focus_pending=True,
                                focus_diagnostics=getattr(self.reader.window, 'focus_diagnostics', {}))
                    if self._consume_focus_request():
                        self.reader.window.focus_for_start()
                    time.sleep(.05)
            finally:
                self.status(focus_pending=False,
                            user_paused=bool(control and (control.run_dir / 'PAUSE').exists()))

    def _check_sampling_control(self):
        control = getattr(self, 'training_control', None)
        if control and getattr(self, '_sampling', False) and not control.busy:
            if (control.run_dir / 'FINISH').exists():
                self.keys.release()
                raise FinishTraining()
            if (control.run_dir / 'PAUSE').exists():
                self.keys.release()
                raise SamplingInterrupted()

    def _ensure_transaction_focus(self):
        if not self.reader.window.focused:
            self.keys.release()
            control = getattr(self, "training_control", None)
            if control is None:
                if getattr(self, '_sampling', False):
                    raise GameFocusLost(msg('backend.the_game_lost_focus_the_ai_stopped'))
                self._acquire_focus()
                return
            if getattr(self, "_sampling", False) and not control.busy:
                control.request_pause(msg('backend.the_game_lost_focus_and_was_paused'))
                raise SamplingInterrupted()
            if not control.busy and control.started:
                control.request_pause(msg('backend.the_session_paused_while_you_used_the'))
            self._acquire_focus()

    def _wait(self, seconds: float):
        deadline = time.monotonic() + seconds
        focus_wait = 0.
        while time.monotonic() < deadline:
            self.lease()
            self._check_sampling_control()
            if not self.reader.window.focused:
                lost_at = time.monotonic()
                self._ensure_transaction_focus()
                elapsed = time.monotonic()-lost_at
                deadline += elapsed
                focus_wait += elapsed
            time.sleep(min(.01, max(0, deadline - time.monotonic())))
        return focus_wait

    def wait_for_focus(self):
        self.status(phase=msg('backend.focusing_the_game'), message=msg('backend.activating_the_game_window'), game_pid=self.reader.window.pid)
        self._acquire_focus()

    def tap(self, key: str, hold=.16, gap=.10):
        self.lease()
        self._ensure_transaction_focus()
        self.keys.set({key})
        self._wait(hold)
        self.status(last_input={"key": key, "os_down": bool(get_async_key(VIRTUAL_KEYS[key]) & 0x8000),
                                "game_bits": int.from_bytes(self.reader.process.read(INPUT_WORD, 4), "little")})
        self.keys.release()
        self._wait(gap)

    def ensure_game_started(self, restart=True):
        self.wait_for_focus()
        self.status(phase=msg('backend.checking_stage_state'), message=msg('backend.detecting_pause_stage_and_menu_state'))
        self.lease()
        f = self.reader.read()
        self.status(game_phase=phase_of(f), game_phase_evidence=f.phase_evidence,
                    game_settings=observed_settings(f), native_tick=f.tick)
        setup = getattr(self, 'game_setup', None)
        if phase_of(f) in MENU_PHASES or phase_of(f) == 'result':
            self._enter_from_menu()
            self.current=self.reader.read()
            return
        # A real pause can freeze the very first tick (including tick 0).
        # Native phase/activity identifies an entered stage, not its age.
        if not f.stage_active:
            raise RuntimeError(msg('backend.no_active_stage_detected_enter_a_stage'))
        if setup is not None and not setup.matches(f):
            self._return_to_title()
            self._enter_from_menu()
            return
        # NC offers a documented R shortcut while paused. Start from a fresh
        # run instead of depending on the operator's current menu selection.
        if restart:
            self._restart()
        else:
            self.resume()
        self.current = self.reader.read()
        self._verify_setup(self.current)

    def _verify_setup(self, frame, *, stage_boundary=False):
        setup = getattr(self, 'game_setup', None)
        self.status(game_settings=observed_settings(frame),
                    requested_game_setup=setup.to_dict() if setup else None)
        if setup is not None and not setup.matches(frame):
            self._pause_sampling(msg('backend.the_current_difficulty_character_or_practice_stage'),
                                 stage_boundary=stage_boundary)

    def _return_to_title(self):
        self.pause()
        deadline = time.monotonic()+4
        while time.monotonic() < deadline:
            f = self.reader.read()
            if phase_of(f) in MENU_PHASES:
                return
            if f.paused and f.menu_ready:
                self.tap('q', hold=.08, gap=.12)
                break
            self._wait(.025)
        deadline = time.monotonic()+8
        while time.monotonic() < deadline:
            f = self.reader.read()
            if phase_of(f) in MENU_PHASES:
                return
            self._wait(.025)
        self._pause_sampling(msg('backend.could_not_confirm_a_return_to_the'))

    def _enter_from_menu(self):
        """Only confirmed NC screens receive keys; wait through animations."""
        deadline=time.monotonic()+25
        last_command=None
        control = getattr(self, 'training_control', None)
        while time.monotonic()<deadline:
            self.lease()
            if control:
                control.check_budget(boundary='stage_ended')
            f=self.reader.read()
            phase=phase_of(f)
            self.status(phase=msg('backend.entering_the_game'),game_phase=phase,game_phase_evidence=f.phase_evidence,
                        game_settings=observed_settings(f),
                        message=PHASE_NAMES.get(phase,phase))
            setup = getattr(self, 'game_setup', None)
            if combat_frame(f) and f.tick>0:
                if setup is not None and any(getattr(f, key) is None for key in (
                        'stage_id', 'difficulty', 'character', 'game_mode')):
                    self._wait(.025)
                    continue
                self.current=f
                self._verify_setup(f)
                return
            if phase == 'result':
                if last_command != ('result',):
                    self.tap('z', hold=.08, gap=.1)
                    last_command = ('result',)
                else:
                    self._wait(.05)
                continue
            if phase in MENU_PHASES and phase != 'menu_transition' and f.menu_ready:
                selection=f.menu_selection
                try:
                    if setup is not None:
                        key = menu_key(f, setup)
                    else:
                        target=0 if phase in ('title','run_mode') else None
                        if selection is None or not 0<=selection<=10:
                            raise ValueError(msg('backend.menu_option_not_recognized'))
                        key=('up' if selection>target else 'down') if target is not None and selection!=target else 'z'
                except ValueError as error:
                    self._pause_sampling(msg('backend.check_game_before_resuming', error))
                signature=(phase,selection,key)
                if signature!=last_command:
                    self.tap(key,hold=.08,gap=.05)
                    last_command=signature
                else:
                    # A held/repeated Z must never race into the next screen.
                    self._wait(.05)
            else:
                self._wait(.025)
        self._pause_sampling(msg('backend.stage_entry_could_not_be_confirmed_enter'))

    def _frozen(self):
        first = self.reader.read()
        self._wait(.08)
        second = self.reader.read()
        return first.stage_active and second.stage_active and first.tick == second.tick

    def release_input(self):
        self.keys.release()
        self.last_action = None

    def _at_round_menu(self, frame):
        phase = phase_of(frame)
        return phase == 'continue' or (phase == 'result' and (
            getattr(self, '_practice_origin', None) is not None
            or self._story_end_reason(frame, getattr(self, 'current', None)) is not None))

    def pause(self):
        self.release_input()
        f = self.reader.read()
        if f.stage_active and not self._at_round_menu(f):
            if not f.paused:
                self._ensure_transaction_focus()
                self.keys.set({'esc'})
                deadline = time.monotonic()+1.5
                try:
                    while not f.paused:
                        deadline += self._wait(.005) or 0.
                        f = self.reader.read()
                        if f.paused or not f.stage_active or self._at_round_menu(f):
                            break
                        if 'esc' not in getattr(self.keys, 'held', {'esc'}):
                            self.keys.set({'esc'})
                            deadline = time.monotonic()+1.5
                        if time.monotonic() >= deadline:
                            f=self._await_manual_pause()
                            break
                finally:
                    self.keys.release()
            self.managed_pause = True
        self._remember_pause_frame(f)
        self.status(paused=bool(f.paused),game_phase=phase_of(f),game_phase_evidence=f.phase_evidence)

    def _await_manual_pause(self):
        self.keys.release()
        control=getattr(self,'training_control',None)
        reason=msg('backend.could_not_confirm_the_pause_menu_keys')
        if control is None:raise RuntimeError(reason)
        control.request_pause(reason)
        with control.suspended_budget():
            while True:
                self.lease()
                f=self.reader.read()
                self.status(phase=msg('backend.waiting_for_the_game_to_pause'),message=reason,user_paused=True,
                            game_phase=phase_of(f),game_phase_evidence=f.phase_evidence)
                if f.paused or not f.stage_active or self._at_round_menu(f):return f
                if (control.run_dir/'FINISH').exists():raise FinishTraining()
                time.sleep(.05)

    def _account_menu_gap(self,frame):
        """Count advancing combat frames between the last action and a pause.

        No fabricated reward/action is attributed to that unsampled interval.
        Native ticks do not advance while the actual pause menu is open.
        """
        previous=getattr(self,'current',None)
        if previous is None or not getattr(self,'episode_count',0):return
        if combat_frame(previous) and frame.stage_active and frame.tick>=previous.tick and not context_changed(previous, frame):
            gap=frame.tick-previous.tick
            self.episode_ticks=getattr(self,'episode_ticks',0)+gap
            self.stage_ticks=getattr(self,'stage_ticks',0)+gap
            self.unsampled_ticks=getattr(self,'unsampled_ticks',0)+gap

    def _remember_pause_frame(self, frame):
        previous = getattr(self, 'current', None)
        self._account_menu_gap(frame)
        # F7 or a PPO update can land on either side of the loading screen.
        # Keep the old context until resume can record the real boundary.
        if self._stage_transition_pending(frame, previous) or self._story_end_reason(frame, previous) or (
                getattr(previous, 'game_mode', None) == 'story' and frame.game_mode == 'story'
                and context_changed(previous, frame)):
            return
        self.current = frame

    def resume(self):
        self.last_action = None
        previous = getattr(self, 'current', None)
        f = self.reader.read()
        control = getattr(self, 'training_control', None)
        if control and control.budget_reached and self._confirmed_context_change(previous, f):
            self._record_stage_context(f, previous)
        if f.stage_active and (f.paused or self.managed_pause and not f.phase):
            # Observed NC menu: Z confirms "解除暂停"; Escape does not resume.
            previous_tick = f.tick
            if f.paused and 'pause_animation' in f.phase_evidence:
                # NC can freeze gameplay before the menu accepts confirmation.
                # An early held Z has no fresh edge when the menu becomes ready.
                menu_deadline=time.monotonic()+3
                while f.paused and not f.menu_ready:
                    self.status(phase=msg('backend.waiting_for_the_pause_menu'),game_phase=phase_of(f),message=msg('backend.waiting_for_the_menu_animation'))
                    if time.monotonic()>=menu_deadline:
                        self._pause_sampling(msg('backend.the_pause_menu_is_not_ready_check'))
                    self._wait(.005)
                    f=self.reader.read()
                if not f.stage_active:
                    self._pause_sampling(msg('backend.the_game_left_the_current_stage_enter'))
            if f.paused and f.menu_selection is not None:
                # Selection is observable; never confirm "return to title".
                for _ in range(6):
                    if f.menu_selection==0:break
                    self.tap('up',hold=.06,gap=.05)
                    f=self.reader.read()
                if f.menu_selection!=0:
                    self._pause_sampling(msg('backend.select_resume_in_the_pause_menu_before'))
            self._ensure_transaction_focus()
            self.status(phase=msg('backend.waiting_for_the_resume_animation'), message=msg('backend.waiting_for_game_frames_to_advance'))
            paused_frame = f
            deadline = time.monotonic() + 3
            self.keys.set({'z'})
            try:
                while True:
                    f = self.reader.read()
                    if getattr(self, '_practice_origin', None) is not None and self._practice_end_reason(f):
                        break
                    if self._stage_transition_pending(f, previous):
                        break
                    # A fixed tap and coarse three-frame check left the
                    # player idle after PPO updates. Confirm semantic resume
                    # and its first advancing native frame instead.
                    if f.stage_active and not f.paused and (f.tick > previous_tick or (
                            paused_frame.game_mode == 'story' and f.game_mode == 'story'
                            and f.stage_id is not None and f.stage_id != paused_frame.stage_id)):
                        self.status(last_input={'key':'z','game_bits':getattr(f,'input_bits',None),
                                                'native_tick':f.tick,'confirmed':True})
                        break
                    if time.monotonic() >= deadline:
                        raise RuntimeError(msg('backend.could_not_verify_that_the_game_resumed'))
                    deadline += self._wait(.002) or 0.
                    if 'z' not in getattr(self.keys, 'held', {'z'}):
                        self.keys.set({'z'})
            finally:
                self.keys.release()
        self.managed_pause = False
        self.status(paused=False)
        if self._stage_transition_pending(f, previous):
            # The sampler can cut the rollout when the new stage is known.
            # Keep the last playable frame across an F7 pause in the handoff.
            return
        if previous is not None and previous.paused and combat_frame(f) and f.tick>=previous.tick and getattr(self,'episode_count',0) and not context_changed(previous, f):
            gap=f.tick-previous.tick
            self.episode_ticks=getattr(self,'episode_ticks',0)+gap
            self.unsampled_ticks=getattr(self,'unsampled_ticks',0)+gap
        if self._story_end_reason(f, previous):
            # Let the sampler close the episode with the last known stage.
            return
        self.current = f
        if getattr(self, '_practice_origin', None) is not None and (
                self._practice_end_reason(f) or phase_of(f) == 'menu_transition'):
            # Resolve the Practice exit in the sampler, where a terminal
            # boundary can cut GAE without bootstrapping from the menu.
            return
        tracker = getattr(self, "reward_tracker", None)
        if context_changed(previous, f):
            self._record_stage_context(f, previous)
            self.pending_stage_boundary = True
        elif previous is not None and previous.paused and combat_frame(f) and f.tick>=previous.tick and getattr(self,'episode_count',0):
            self.stage_ticks=getattr(self,'stage_ticks',0)+(f.tick-previous.tick)
        if tracker is not None:
            tracker.rebase(f)
            self.builder.reset()
            self.input_timing = InputTiming()
            self._observe_control(f, None)

    def _observe_control(self, frame, action):
        """Expose pending input and existing feedback memory to the network."""
        tracker = self.reward_tracker
        residence = residence_age(frame, np.asarray(tracker.positions))
        self.builder.set_control(action, blocked_ticks=tracker.blocked_ticks,
                                 residence=residence,
                                 resource_saturation=tracker.resource_bonus_used / RESOURCE_BUDGET)

    def pause_for_user(self):
        """Confirm an actual game pause, then return focus to the user's panel."""
        self.release_input()
        observed = self.reader.read()
        if observed.paused or not observed.stage_active or self._at_round_menu(observed):
            self.keys.release()
            self._remember_pause_frame(observed)
            self.managed_pause = bool(observed.paused)
            self.status(paused=bool(observed.paused),game_phase=phase_of(observed),game_phase_evidence=observed.phase_evidence)
            return
        previous = foreground()
        control = getattr(self, "training_control", None)
        was_busy = control.busy if control else False
        if control and not was_busy and not self.reader.window.focused:
            control.request_pause(msg('backend.the_game_lost_focus_and_was_paused_2'))
        if control:
            control.busy = True
        try:
            self._acquire_focus()
            self.pause()
        finally:
            if control:
                control.busy = was_busy
            if previous and previous != self.reader.window.hwnd:
                restore_foreground(previous)

    def resume_for_user(self):
        self._acquire_focus()
        self.resume()

    def _restart(self):
        existing=self.reader.read()
        if phase_of(existing) in MENU_PHASES or phase_of(existing) == 'result':
            self._enter_from_menu()
            return
        if phase_of(existing)=='continue':
            self.tap('x',hold=.08,gap=.1)
            self._enter_from_menu()
            return
        self.pause()
        self._wait(.6)  # Menu animation settling belongs to restart, not F7 pause.
        old = self.reader.read()
        if old.tick <= 0:
            self.managed_pause = True
            self.resume()
            self._wait(.3)
            self.pause()
            old = self.reader.read()
        # Observe the reset while R is down. Waiting until after a short tap
        # can miss frame 0, or miss the shortcut during a menu transition.
        control = getattr(self, 'training_control', None)
        if control:
            control.check_budget(boundary='stage_ended')
        self.keys.set({"r"})
        reset_seen = False
        press_deadline = time.monotonic() + .6
        try:
            while time.monotonic() < press_deadline:
                self._wait(.01)
                probe = self.reader.read()
                if old.tick > 0 and 0 <= probe.tick < old.tick:
                    reset_seen = True
                    break
        finally:
            self.keys.release()
        deadline = time.monotonic() + 8
        last_tick = -1
        last_change = time.monotonic()
        resumed_after_reset = False
        while time.monotonic() < deadline:
            self.lease()
            f = self.reader.read()
            if f.stage_active and 0 <= f.tick < old.tick:
                reset_seen = True
            if f.tick != last_tick:
                last_tick, last_change = f.tick, time.monotonic()
            if (reset_seen and f.stage_active and f.tick >= 5 and
                    time.monotonic() - last_change > .3 and not resumed_after_reset):
                self.managed_pause = True
                self.resume()
                resumed_after_reset = True
                last_change = time.monotonic()
                continue
            if reset_seen and combat_frame(f) and f.tick >= 2:
                self.current = f
                self.managed_pause = False
                if self._frozen():
                    self.managed_pause = True
                    self.resume()
                    self.current = self.reader.read()
                self._verify_setup(self.current)
                return
            self._wait(.05)
        latest = self.reader.read()
        self._pause_sampling(msg('backend.restart_not_confirmed_previous_frame_0_current', old.tick, latest.tick))

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        terminal = getattr(self, '_terminal_observation', None)
        self._terminal_observation = None
        control = getattr(self, 'training_control', None)
        while True:
            try:
                if control:
                    if not self.first_reset:
                        control.check_budget(boundary='stage_ended')
                    control.gate(self, resume_game=False)
                if self.first_reset:
                    self.ensure_game_started(restart=(options or {}).get("restart", True))
                    self.first_reset = False
                else:
                    self._restart()
                break
            except SamplingInterrupted:
                if control is None:
                    raise
                control.gate(self, resume_game=False)
            except FinishTraining as finish:
                if terminal is None or finish.reason == 'manual':
                    raise
                # SB3 resets inside step(). Deliver the real last transition
                # even when the timer expires during the reset transaction.
                self.release_input()
                return terminal, {}
        self.builder.reset()
        self.input_timing = InputTiming()
        self.current = self.reader.read()
        if self.current.resources is None:
            raise RuntimeError(msg('backend.verified_in_game_resources_are_required_for'))
        self._practice_origin = self.current if getattr(self.current, 'game_mode', None) == 'practice' else None
        self.reward_tracker.reset(self.current)
        self.last_action = None
        self.start_tick = self.current.tick
        self.episode_ticks = 0
        self.unsampled_ticks = 0
        self.stage_ticks = 0
        self.stage_changes = 0
        self.pending_stage_boundary = False
        self.episode_return = 0.0
        self.episode_steps = 0
        self.episode_bombs_used = 0
        self._decision_refresh_used = False
        self.decision_context = {}
        self.freshness_warning = None
        self.episode_count += 1
        self.status(phase=msg('backend.collecting_training_samples'), episodes=self.episode_count, paused=False,
                    stage_id=self.current.stage_id, stage_label=stage_label(self.current.stage_id),
                    stage_seconds=0., stage_changes=0, native_tick=self.current.tick,
                    game_phase=phase_of(self.current), game_phase_evidence=self.current.phase_evidence,
                    game_settings=observed_settings(self.current))
        return self.builder.build(self.current), {"native_tick": self.current.tick,
                'stage_id': self.current.stage_id, 'stage_label': stage_label(self.current.stage_id),
                'game_settings': observed_settings(self.current)}

    @staticmethod
    def _practice_end_reason(frame):
        phase = phase_of(frame)
        if phase == 'result':
            return 'practice_result'
        if phase == 'practice_stage' and getattr(frame, 'phase_evidence', {}).get('practice_menu') is True:
            return 'practice_return'
        return None

    def _practice_result(self, frame, *, death_probe_tick=None):
        origin = getattr(self, '_practice_origin', getattr(self, 'current', None))
        if getattr(origin, 'game_mode', None) != 'practice':
            return
        reason = self._practice_end_reason(frame)
        if reason is not None:
            self._finish_round(frame, origin, reason, death_probe_tick)

    @staticmethod
    def _story_end_reason(frame, before):
        if getattr(before, 'game_mode', None) != 'story':
            return None
        last_stage = {'easy': 5, 'extra': 7}.get(before.difficulty, 6)
        if before.stage_id != last_stage:
            return None
        phase = phase_of(frame)
        if phase == 'result':
            return 'story_result'
        if not frame.stage_active and phase in ('title', 'title_prompt'):
            return 'story_return'
        return None

    def _round_result(self, frame, before, *, death_probe_tick=None):
        self._practice_result(frame, death_probe_tick=death_probe_tick)
        reason = self._story_end_reason(frame, before)
        if reason is not None:
            self._finish_round(frame, before, reason, death_probe_tick)
        if phase_of(frame) == 'result' and getattr(before, 'game_mode', None) == 'story':
            control = getattr(self, 'training_control', None)
            if control:
                control.check_budget(boundary='stage_result')

    def _finish_round(self, frame, origin, reason, death_probe_tick):
        self.release_input()
        died = death_probe_tick is not None or frame.state == 2
        info = {'episode_end_reason':'death' if died else reason, 'died':died,
                'survival_seconds':getattr(self,'episode_ticks',0)/60,
                'stage_id':origin.stage_id, 'stage_label':stage_label(origin.stage_id),
                'game_settings':observed_settings(origin), 'goal':self.goal,
                'episode_bombs_used':getattr(self,'episode_bombs_used',0),
                'episode':{'r':self.episode_return,'l':self.episode_steps,'t':self.episode_ticks/60},
                'episode_diagnostics':self.reward_tracker.diagnostics()}
        self.current = self._full_frame(frame)
        self._practice_origin = None
        self.pending_stage_boundary = False
        self.status(phase=msg('backend.stage_round_complete'), game_phase=phase_of(self.current),
                    game_phase_evidence=self.current.phase_evidence, game_settings=observed_settings(self.current),
                    message=msg('backend.stage_round_boundary_confirmed'))
        diagnostics = getattr(self, 'diagnostics', None)
        if diagnostics is not None:
            diagnostics.event(event=reason, **info)
            view = getattr(diagnostics, 'battle_view', None)
            if view is not None and died:
                view.record_miss(self.episode_count, origin.stage_id,
                                 frame.tick if death_probe_tick is None else death_probe_tick)
        # A menu observation is not a fabricated zero-reward combat action.
        raise SamplingInterrupted(episode_end=info)

    def step(self, action):
        self._sampling = True
        try:
            return self._step(action)
        except GameFocusLost:
            control = getattr(self, "training_control", None)
            if control is None:
                raise
            self.keys.release()
            control.request_pause(msg('backend.the_game_lost_focus_and_was_paused_2'))
            raise SamplingInterrupted() from None
        finally:
            self._sampling = False

    def _pause_sampling(self, reason, *, stage_boundary=False):
        """Keep a training session alive while its game is paused/unavailable."""
        self.keys.release()
        self.last_action = None
        control = getattr(self, "training_control", None)
        if control is None:
            raise RuntimeError(reason)
        control.request_pause(reason)
        raise SamplingInterrupted(stage_boundary=stage_boundary)

    def _record_stage_context(self, frame, before):
        event = {'event': 'stage_context_change', 'before_stage': before.stage_id,
                 'stage_id': frame.stage_id, 'stage_label': stage_label(frame.stage_id),
                 'before_tick': before.tick, 'after_tick': frame.tick,
                 'previous_stage_seconds': getattr(self, 'stage_ticks', 0)/60,
                 'episode_seconds': getattr(self, 'episode_ticks', 0)/60}
        self.stage_ticks = 0
        self.stage_changes = getattr(self, 'stage_changes', 0) + int(before.stage_id is not None and frame.stage_id is not None)
        self.last_stage_transition = event
        positions = getattr(self.reward_tracker, 'positions', None)
        if positions is not None:
            positions.clear()
        run_dir = getattr(self, 'run_dir', None)
        if run_dir is not None:
            try:
                with (run_dir/'stage_events.jsonl').open('a', encoding='utf-8') as stream:
                    stream.write(json.dumps(event, ensure_ascii=False)+'\n')
            except OSError as error:
                self.status(stage_log_error=str(error))
        self.status(stage_id=frame.stage_id, stage_label=stage_label(frame.stage_id),
                    stage_seconds=0., stage_changes=self.stage_changes, stage_transition=event,
                    game_phase=phase_of(frame), game_phase_evidence=frame.phase_evidence,
                    native_tick=frame.tick, game_settings=observed_settings(frame))
        control = getattr(self, 'training_control', None)
        if control and control.budget_reached and self._confirmed_context_change(before, frame):
            self.release_input()
            self.current = frame
            story_clear = (before.game_mode == frame.game_mode == 'story' and before.stage_id < 6
                           and frame.stage_id == before.stage_id + 1
                           and before.difficulty == frame.difficulty and before.character == frame.character)
            control.check_budget(boundary='stage_complete' if story_clear else 'stage_ended')

    @staticmethod
    def _confirmed_context_change(before, frame):
        return (before is not None and before.stage_id is not None and frame.stage_id is not None
                and frame.stage_active and context_changed(before, frame)
                and all(getattr(before, key) is None or getattr(frame, key) is not None
                        for key in ('difficulty', 'character', 'game_mode')))

    def _stage_transition_pending(self, frame, before):
        if getattr(before, 'game_mode', None) not in ('story', 'practice') or before.stage_id is None or frame.paused:
            return False
        phase = phase_of(frame)
        if phase == 'result':
            return before.game_mode == 'story' and self._story_end_reason(frame, before) is None
        if not frame.stage_active:
            return phase in ('unknown', 'menu_transition')
        if phase not in ('unknown', 'combat', 'boss', 'script_wait'):
            return False
        return frame.stage_id is None or any(getattr(before, key) is not None and getattr(frame, key) is None
                                             for key in ('difficulty', 'character', 'game_mode'))

    def _check_sampling_scene(self, frame, before, *, death_probe_tick=None):
        self._round_result(frame, before, death_probe_tick=death_probe_tick)
        if self._stage_transition_pending(frame, before):
            self._rebase_sampling(frame, before, death_probe_tick=death_probe_tick)
        phase = phase_of(frame)
        control = getattr(self, 'training_control', None)
        if phase == 'continue' and control:
            control.check_budget(boundary='stage_ended')
        if phase in ('dialogue', 'script_wait', 'pause') and self._confirmed_context_change(before, frame):
            if not frame.paused or (control and control.budget_reached):
                self._rebase_sampling(self._full_frame(frame), before, death_probe_tick=death_probe_tick)

    def _await_stage_transition(self, frame, before, death_probe_tick):
        """Hold the old context until the native stage or result is known."""
        self.status(phase=msg('console.stage_transition'), game_phase='stage_transition',
                    stage_id=None, stage_label=None, stage_seconds=None,
                    game_settings=None, game_phase_evidence=getattr(frame, 'phase_evidence', {}),
                    message=msg('backend.waiting_for_stage_or_result'))
        control = getattr(self, 'training_control', None)
        was_sampling = getattr(self, '_sampling', False)
        self._sampling = True
        try:
            with control.suspended_budget(scene_step=self.total_steps) if control else nullcontext():
                deadline = time.monotonic() + 25
                frame = self._full_frame(frame)
                while True:
                    self._check_sampling_control()
                    self._round_result(frame, before, death_probe_tick=death_probe_tick)
                    if not self._stage_transition_pending(frame, before) or time.monotonic() >= deadline:
                        break
                    self._wait(.025)
                    frame = self.reader.read()
                    if frame.state == 2 and death_probe_tick is None:
                        death_probe_tick = frame.tick
        finally:
            self._sampling = was_sampling
        if not frame.stage_active or frame.paused or frame.stage_id is None or self._stage_transition_pending(frame, before):
            if control:
                control.check_budget(boundary='time_limit')
            self._pause_sampling(msg('backend.the_game_is_paused_or_changing_scenes'))
        return frame

    def _rebase_sampling(self, frame, before, *, death_probe_tick=None):
        # A reset of the native manager clock does not mean the process/model
        # failed. The interrupted action cannot be attributed to either side
        # of that boundary, so re-observe without inventing a reward or reset.
        self._round_result(frame, before, death_probe_tick=death_probe_tick)
        self.keys.release()
        if self._stage_transition_pending(frame, before):
            frame = self._await_stage_transition(frame, before, death_probe_tick)
        stage_boundary = context_changed(before, frame)
        if stage_boundary:
            self._record_stage_context(frame, before)
        self.current = frame
        self.builder.reset()
        self.input_timing = InputTiming()
        self.reward_tracker.rebase(frame)
        self.last_action = None
        self._observe_control(frame, None)
        self.status(phase=msg('backend.realigning_the_game_observation'), message=msg('backend.the_stage_changed_training_segments_were_separated') if stage_boundary else msg('backend.the_game_timer_reset_sampling_resumed_with'),
                    game_settings=observed_settings(frame), game_phase=phase_of(frame),
                    game_phase_evidence=frame.phase_evidence, native_tick=frame.tick,
                    stage_id=frame.stage_id, stage_label=stage_label(frame.stage_id),
                    stage_seconds=round(getattr(self, 'stage_ticks', 0)/60, 2),
                    sampling_timing={"event": "stage_context_change" if stage_boundary else "clock_reset", "before_tick": before.tick,
                                     "after_tick": frame.tick, "delta": frame.tick-before.tick})
        setup = getattr(self, 'game_setup', None)
        if setup is not None and not setup.matches(frame):
            self._verify_setup(frame, stage_boundary=stage_boundary)
        raise SamplingInterrupted(stage_boundary=stage_boundary)

    def _advance_dialogue(self):
        """Dialogue is a game transaction, not a learned movement sample."""
        self.keys.release()
        self.last_action=None
        control=getattr(self,'training_control',None)
        previous = getattr(self, 'current', None)
        with control.suspended_budget(scene_step=self.total_steps) if control else nullcontext():
            while True:
                self.lease()
                self._check_sampling_control()
                f=self.reader.read()
                self._check_sampling_scene(f, previous)
                phase=phase_of(f)
                if phase not in ('dialogue','script_wait'):
                    break
                self.status(phase=PHASE_NAMES[phase],game_phase=phase,game_phase_evidence=f.phase_evidence,
                            message=msg('backend.movement_is_suspended_during_dialogue_the_game'),paused=False)
                if phase=='dialogue':
                    # Never leave arrows/Shift/X held across a dialogue.
                    self.keys.set({'z'})
                    self._wait(.08)
                    self.keys.release()
                    self._wait(.04)
                else:
                    self.keys.release()
                    self._wait(.025)
        self.keys.release()  # Force a new Z down edge on return to combat.
        if context_changed(previous, f):
            self._rebase_sampling(f, previous)
        self.current=f
        self.builder.reset()
        self.input_timing=InputTiming()
        self.reward_tracker.rebase(f)
        self._observe_control(f,None)
        raise SamplingInterrupted()

    def _step(self, action):
        before, decision_plan = self._prepare_action()
        sample = self._capture_action(action, before, decision_plan)
        return self._record_transition(sample)

    def _prepare_action(self):
        """Check authorization, budget, and game context before any policy input."""
        self.lease()
        self._check_sampling_control()
        before = self.current or self.reader.read()
        self._check_sampling_scene(before, before)
        control = getattr(self, 'training_control', None)
        if control:
            control.check_budget(before)
        decision_plan = self.builder.last_plan
        if phase_of(before) in ('dialogue','script_wait'):
            self._advance_dialogue()
        if before.paused or not before.stage_active:
            self._pause_sampling(msg('backend.the_game_is_paused_or_changing_scenes_2'))
        setup = getattr(self, 'game_setup', None)
        if setup is not None and not setup.matches(before):
            self._verify_setup(before)
        return before, decision_plan

    def _check_action_capture(self, frame, before, baseline, timing, death_probe_tick):
        self._check_sampling_scene(frame, before, death_probe_tick=death_probe_tick)
        timing.observe(frame)
        if phase_of(frame) in ('dialogue', 'script_wait'):
            self._advance_dialogue()
        if frame.paused or not frame.stage_active:
            self._pause_sampling(msg('backend.the_game_is_paused_or_changing_scenes_2'))
        if frame.tick < baseline.tick or context_changed(baseline, frame):
            self._rebase_sampling(self._full_frame(frame), before)

    def _capture_action(self, action, before, decision_plan):
        """Send the sampled action and validate its post-dispatch captures."""
        previous_action = getattr(self, "last_action", None)
        previous_keys = (set(self.keys.held) if hasattr(self.keys, 'held') else
                         action_keys(previous_action) if previous_action is not None else set())
        actual_previous_action = movement_action(previous_keys)
        action = int(action)
        requested_keys = action_keys(action)
        if hasattr(self.keys,'reconcile'):
            self.keys.reconcile('z')
        profile = getattr(self, 'thread_profile', None)
        thread_started = profile.mark() if profile is not None else None
        switch_started = clock_ns()
        dispatch = self.keys.set(requested_keys)
        switch_completed = clock_ns()
        thread_finished = profile.mark() if thread_started is not None else None
        # Test doubles and third-party controllers may not return OS timing.
        if not isinstance(dispatch, dict):
            dispatch = {'started_ns': switch_started, 'completed_ns': switch_completed,
                        'send_ms': None, 'events': None}
        timing = self._timing()
        timing.begin(before, decision_plan, requested_keys,
                     previous_keys, dispatch)
        if thread_started is not None:
            trace = profile.action(before, decision_plan, thread_started, thread_finished)
            if trace is not None:
                timing.latest['thread_timing'] = trace
        timing.latest['key_switch_ms'] = (switch_completed-switch_started)/1e6
        timing.latest['observation_build_ms'] = self.builder.last_build_ms
        timing.latest['read_total_ms'] = (before.capture_finished_ns-before.capture_started_ns)/1e6 if before.capture_started_ns else None
        # Inference/status work happens while the previous keys remain held.
        # Its elapsed time must not satisfy this command's minimum hold time.
        # Start the hold clock from a fresh capture AFTER dispatch. The full
        # transition still includes real pre-dispatch events; they are logged
        # and the pending previous action is part of the policy observation.
        dispatched = self._read_control()
        death_probe_tick = dispatched.tick if dispatched.state == 2 else None
        self._check_action_capture(dispatched, before, before, timing, death_probe_tick)
        deadline = time.monotonic() + .4
        after = dispatched
        hold_until_tick = dispatched.tick + 2
        while after.tick < hold_until_tick and after.state != 2:
            self._wait(.002)
            after = self._read_control()
            if after.state == 2:
                death_probe_tick = after.tick
            self._check_action_capture(after, before, dispatched, timing, death_probe_tick)
            if after.tick >= hold_until_tick or after.state == 2:
                break
            if time.monotonic() > deadline:
                self._pause_sampling(msg('backend.the_game_timer_has_not_advanced_resume'))
        if not isinstance(after, Frame):
            after = self._full_frame(after)
            timing.observe(after)
        self._check_sampling_scene(after, before, death_probe_tick=death_probe_tick)
        if phase_of(after) in ('dialogue', 'script_wait'):
            self._advance_dialogue()
        if after.paused or not after.stage_active:
            self._pause_sampling(msg('backend.the_game_is_paused_or_changing_scenes'))
        if after.tick < before.tick or context_changed(before, after):
            self._rebase_sampling(after, before)
        return _ActionSample(action=action, before=before, after=after, dispatch_tick=dispatched.tick,
                             death_probe_tick=death_probe_tick, previous_action=previous_action,
                             previous_keys=previous_keys, actual_previous_action=actual_previous_action,
                             decision_plan=decision_plan)

    def _record_transition(self, sample):
        """Commit a valid transition and prepare the next policy observation."""
        before, f, action = sample.before, sample.after, sample.action
        delta = f.tick - before.tick
        # A lightweight sample can observe the real Miss before the complete
        # pool capture reaches the respawn frame. Do not erase that event.
        died = f.state == 2 or sample.death_probe_tick is not None
        # Scheduling/inference delays naturally make an action last longer.
        # Use elapsed native ticks instead of treating >15 as a fatal error.
        # Accumulation also survives native counter resets and excludes menus.
        elapsed_ticks = getattr(self, "episode_ticks", max(0, before.tick-self.start_tick)) + delta
        seconds = elapsed_ticks / 60
        feedback = self.reward_tracker.step(f, action, died)
        reward = feedback.total
        truncated = bool(self.episode_seconds and seconds >= self.episode_seconds)
        self.current = f
        self.episode_ticks = elapsed_ticks
        self.stage_ticks = getattr(self, 'stage_ticks', 0) + delta
        self.episode_return += reward
        self.total_steps += 1
        self.episode_steps += 1
        self._decision_refresh_used = False
        self.last_action = action
        self._observe_control(f, action)
        latency = self._timing().finish()
        if 'thread_timing' in latency:
            self.thread_profile.commit()
        diagnostics = getattr(self,'diagnostics',None)
        view = getattr(diagnostics, 'battle_view', None)
        if view is not None:
            if died:
                view.record_miss(self.episode_count, f.stage_id,
                                 f.tick if sample.death_probe_tick is None else sample.death_probe_tick)
            elif (view.demand is not None and phase_of(before) in ('combat', 'boss')
                  and phase_of(f) in ('combat', 'boss')):
                view.record_resource_gain(before, f)
                if not truncated:
                    view.capture(before, sample.decision_plan, action, latency['game_input_state'],
                                 self.total_steps, after=f)
        planner = self.builder.planner
        if planner is not None:
            planner.record_latency(latency)
        build_started = clock_ns()
        observation = self.builder.build(f)
        latency['next_observation_build_ms'] = (clock_ns()-build_started)/1e6
        info_started = clock_ns()
        info = self._transition_info(sample, observation, feedback, latency, seconds, died, truncated)
        self.episode_bombs_used = getattr(self, 'episode_bombs_used', 0) + info['bomb_consumed']
        info['episode_bombs_used'] = self.episode_bombs_used
        if died or truncated:
            self.keys.release()
            info['episode'] = {'r': self.episode_return, 'l': self.episode_steps, 't': seconds}
            info['episode_diagnostics'] = self.reward_tracker.diagnostics()
            info['episode_diagnostics'].update(stage_id=info['stage_id'], stage_label=info['stage_label'],
                                               stage_seconds=info['stage_seconds'], stage_changes=info['stage_changes'])
        if diagnostics is not None:
            diagnostics.record(info, getattr(self, 'decision_context', {}).get('training_step') or self.total_steps)
        self._publish_transition(f, reward, info, diagnostics)
        self.freshness_warning = None
        self.record_pre_policy_work(env_diagnostics_ms=(clock_ns()-info_started)/1e6)
        if died or truncated:
            self._terminal_observation = observation
        return observation, reward, died, truncated, info

    def _transition_info(self, sample, observation, feedback, latency, seconds, died, truncated):
        before, f, decision_plan = sample.before, sample.after, sample.decision_plan
        decision_planning = dict(decision_plan.diagnostics) if decision_plan is not None else {"enabled": False}
        if decision_plan is not None and decision_plan.diagnostics.get("enabled"):
            movement = sample.action % 18
            decision_planning.update(chosen_movement=movement,
                    chosen_allowed=bool(decision_plan.features[movement, 1]),
                    chosen_predicted_path=decision_plan.paths[movement].tolist(),
                    chosen_clearance=float(decision_plan.clearances[movement]) if np.isfinite(decision_plan.clearances[movement]) else None,
                    chosen_collision_tick=float(decision_plan.collision_times[movement]) if np.isfinite(decision_plan.collision_times[movement]) else None)
        full = observation["full_bullets"]
        coverage = {"visible": len(f.bullets),
                    "near": int(observation["bullets"][:, -1].sum()),
                    "full": int(full[:, -1].sum()),
                    "motion_known": int(full[:, 4].sum()),
                    "geometry_known": int(full[:, 6].sum())}
        planning = self.builder.last_plan.diagnostics if self.builder.last_plan is not None else {"enabled": False}
        movement = {"dx": feedback.dx, "dy": feedback.dy,
                    "blocked_fraction": feedback.blocked_fraction}
        return {"survival_seconds": seconds, "died": died, "native_tick": f.tick,
                'game_settings':observed_settings(f),
                'stage_id': f.stage_id, 'stage_label': stage_label(f.stage_id),
                'decision_stage_id': before.stage_id, 'stage_seconds': self.stage_ticks/60,
                'stage_changes': getattr(self, 'stage_changes', 0),
                'decision_refreshes':getattr(self,'decision_refreshes',0),
                'freshness_warning':getattr(self,'freshness_warning',None),
                'inference':getattr(self,'decision_context',{}).get('inference'),
                'game_phase':phase_of(f),'game_phase_evidence':f.phase_evidence,
                'player_shots':f.player_shots,'visible_lasers':len(f.lasers),
                'unsampled_combat_seconds':getattr(self,'unsampled_ticks',0)/60,
                'decision_lasers':before.lasers,'next_lasers':f.lasers,
                'episode_end_reason':'death' if died else 'time_limit' if truncated else None,
                "native_tick_before": before.tick, "native_tick_delta": f.tick-before.tick,
                "native_tick_dispatch": sample.dispatch_tick,
                "pre_dispatch_ticks": sample.dispatch_tick-before.tick,
                "action_hold_ticks": f.tick-sample.dispatch_tick,
                "previous_action": sample.previous_action,
                'actual_previous_action':sample.actual_previous_action,
                'actual_previous_keys':sorted(sample.previous_keys),
                'goal':getattr(self,'goal','survival'),
                'death_evidence':'full_snapshot' if f.state==2 else 'control_sample' if died else None,
                'death_probe_tick':sample.death_probe_tick,
                'bomb_probability':getattr(self,'decision_context',{}).get('bomb_probability'),
                "action_timing_version": ACTION_TIMING_VERSION,
                "action": sample.action,
                "position": {"x": f.x, "y": f.y}, "movement": movement,
                "behavior": {"corner_exposure": feedback.corner_exposure, "loiter": feedback.loiter,
                             "item_potential": feedback.item_potential}, "visible_items": len(f.items),
                "visible_enemies": len(f.enemies),
                "enemy_damage": self.reward_tracker.enemy_damage, "boss_damage": self.reward_tracker.boss_damage,
                "graze_gained": feedback.graze_gained,
                "bullet_observation": coverage,
                "planning": decision_planning, "planning_next": planning,
                "decision_enemy_bodies": before.enemy_bodies,
                "next_enemy_bodies": f.enemy_bodies,
                "state_reader": dict(getattr(self.reader, "snapshot_diagnostics", {})),
                "input_latency": latency,
                "risk_estimate": {"min_clearance": feedback.min_clearance,
                                  "geometry_coverage": feedback.geometry_coverage},
                "resources": {"power": f.resources.power, "point_items": f.resources.point_items,
                              "score": f.resources.score, "bombs": f.resources.bombs, "lives": f.resources.lives,
                              "graze": f.resources.graze}
                             if f.resources is not None else {},
                "bombs_before": before.resources.bombs if before.resources is not None else None,
                "bombs_after": f.resources.bombs if f.resources is not None else None,
                "bomb_consumed": max(0, before.resources.bombs - f.resources.bombs)
                                 if not died and before.resources is not None and f.resources is not None else 0,
                "reward_version": REWARD_VERSION, "reward_components": feedback.components,
                "threat": feedback.threat}

    def _publish_transition(self, f, reward, info, diagnostics):
        self.status(phase=msg('backend.collecting_policy_samples'), message=msg('backend.the_policy_selects_direction_and_focus_speed'),
                    game_settings=info['game_settings'],
                    stage_id=info['stage_id'], stage_label=info['stage_label'],
                    stage_seconds=round(info['stage_seconds'], 2), stage_changes=info['stage_changes'],
                    game_phase=info['game_phase'], game_phase_evidence=info['game_phase_evidence'],
                    player_shots=info['player_shots'], visible_lasers=info['visible_lasers'],
                    env_steps=self.total_steps, survival_seconds=round(info['survival_seconds'], 2),
                    native_tick=info['native_tick'], native_tick_before=info['native_tick_before'],
                    native_tick_delta=info['native_tick_delta'], native_tick_dispatch=info['native_tick_dispatch'],
                    pre_dispatch_ticks=info['pre_dispatch_ticks'], action_hold_ticks=info['action_hold_ticks'],
                    action_timing_version=info['action_timing_version'],
                    bullets=len(f.bullets), game_pid=self.reader.window.pid,
                    bullet_observation=info['bullet_observation'], planning=info['planning'],
                    reward_version=info['reward_version'], reward=round(reward, 4),
                    reward_components=info['reward_components'], threat=round(info['threat'], 4),
                    movement=info['movement'], corner_exposure=round(info['behavior']['corner_exposure'], 4),
                    loiter_seconds=round(self.reward_tracker.loiter_ticks / 60, 3),
                    power_gained=self.reward_tracker.power_gained,
                    point_items_gained=self.reward_tracker.point_items_gained,
                    graze_gained=self.reward_tracker.graze_gained,
                    graze_with_risk=self.reward_tracker.graze_with_risk,
                    risk_estimate=info["risk_estimate"],
                    state_reader=info["state_reader"],
                    input_latency=info['input_latency'],
                    goal=info['goal'],episode_bombs_used=self.episode_bombs_used,
                    inference=info['inference'],diagnostics=diagnostics.status() if diagnostics is not None else None,
                    enemies=len(f.enemies),
                    boss_hp=int(sum(row[3] for row in f.enemies if row[7])),
                    boss_max_hp=int(sum(row[4] for row in f.enemies if row[7])),
                    enemy_damage=self.reward_tracker.enemy_damage, boss_damage=self.reward_tracker.boss_damage,
                    resources=info['resources'] or None,
                    wall_push_seconds=round(self.reward_tracker.wall_push_ticks / 60, 3))

    def close(self):
        try:
            self.release_input()
            if not self.reader.window.focused:
                return
            self.lease()
            frame = self.reader.read_control()
            if not frame.stage_active or frame.paused or phase_of(frame) in ('result', 'continue'):
                return
            # The session is over. Leave a running game paused if possible,
            # without regaining focus or waiting for operator input.
            self.keys.set({'esc'})
            try:
                deadline = time.monotonic() + .25
                while self.reader.window.focused and time.monotonic() < deadline:
                    self.lease()
                    frame = self.reader.read_control()
                    if frame.paused or not frame.stage_active:
                        break
                    time.sleep(.005)
            finally:
                self.keys.release()
        finally:
            self.reader.close()
            diagnostics = getattr(self,'diagnostics',None)
            if diagnostics is not None:
                diagnostics.close()
