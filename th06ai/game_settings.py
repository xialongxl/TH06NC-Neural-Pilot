"""Requested game setup and observed identity are deliberately separate."""
from __future__ import annotations
from .i18n import msg

from dataclasses import dataclass, asdict
import math
import numpy as np

DIFFICULTIES = ('easy', 'normal', 'hard', 'lunatic', 'extra')
CHARACTERS = ('reimu_a', 'reimu_b', 'marisa_a', 'marisa_b')
CHARACTER_LABELS = (msg('backend.reimu_a'), msg('backend.reimu_b'), msg('backend.marisa_a'), msg('backend.marisa_b'))
GAME_MODES = ('story', 'practice', 'challenge', 'spell_practice')
DIFFICULTY_RVA = 0x53D410
CHARACTER_RVA = 0x53CAD0  # two bytes: character, shot
PRACTICE_RVA = 0x53D404
CHALLENGE_RVA = 0x53D414
PRACTICE_UNLOCK_RVA = 0x549D55
CONTEXT_FIELDS = (*DIFFICULTIES, 'difficulty_known', *CHARACTERS, 'character_known',
                  *GAME_MODES, 'mode_known', 'fast_speed', 'slow_speed', 'speed_known')
CONTEXT_FEATURES = len(CONTEXT_FIELDS)
MENU_PHASES = ('title_prompt', 'title', 'difficulty', 'character', 'shot',
               'run_mode', 'practice_stage', 'start_confirm', 'menu_transition')


@dataclass(frozen=True)
class GameSetup:
    mode: str = 'story'
    difficulty: str = 'normal'
    character: str = 'reimu_a'
    stage: int = 1

    def __post_init__(self):
        if self.mode not in ('story', 'practice'):
            raise ValueError(msg('backend.game_mode_must_be_story_or_practice'))
        if self.difficulty not in DIFFICULTIES or self.character not in CHARACTERS:
            raise ValueError(msg('backend.unknown_difficulty_or_character'))
        if isinstance(self.stage, bool) or not isinstance(self.stage, int):
            raise ValueError(msg('backend.stage_must_be_an_integer'))
        if self.mode == 'story' and self.stage != 1:
            raise ValueError(msg('backend.story_starts_at_stage_1_choose_practice'))
        if self.mode == 'practice' and (self.difficulty == 'extra' or not 1 <= self.stage <= 6):
            raise ValueError(msg('backend.practice_supports_stages_1_6_on_easy'))
        if self.mode == 'practice' and self.difficulty == 'easy' and self.stage == 6:
            raise ValueError(msg('backend.easy_practice_is_limited_to_stage_5'))

    def to_dict(self):
        return asdict(self)

    def matches(self, frame, *, check_stage=True):
        return (getattr(frame, 'difficulty', None) == self.difficulty
                and getattr(frame, 'character', None) == self.character
                and getattr(frame, 'game_mode', None) == self.mode
                and (not check_stage or self.mode != 'practice' or frame.stage_id == self.stage))


def parse_setup(value):
    if value is None or isinstance(value, GameSetup):
        return value
    if not isinstance(value, dict):
        raise ValueError(msg('backend.game_settings_must_be_an_object'))
    try:
        return GameSetup(**value)
    except TypeError as error:
        raise ValueError(msg('backend.game_settings_contain_unknown_fields')) from error


def decode_identity(difficulty, character, shot, practice, spell_practice, challenge):
    diff = DIFFICULTIES[difficulty] if type(difficulty) is int and 0 <= difficulty < 5 else None
    char = (CHARACTERS[character*2+shot] if type(character) is int and type(shot) is int
            and character in (0, 1) and shot in (0, 1) else None)
    mode = None
    if all(type(v) is int and v in (0, 1) for v in (practice, spell_practice, challenge)):
        mode = ('spell_practice' if spell_practice else 'challenge' if challenge else
                'practice' if practice else 'story')
    return diff, char, mode


def movement_speeds(player):
    """Pinned player update uses 7860/7864 and the 7710/7714 axis scales."""
    import struct
    sx, sy = struct.unpack_from('<ff', player)
    fast, slow = struct.unpack_from('<ff', player, 0x150)
    if (all(math.isfinite(v) for v in (sx, sy, fast, slow)) and sx > 0
            and math.isclose(sx, sy, abs_tol=1e-5) and 0 < slow*sx <= fast*sx < 20):
        return fast*sx, slow*sx
    return None


def context_vector(frame):
    out = np.zeros(CONTEXT_FEATURES, np.float32)
    offset = 0
    for names, attr in ((DIFFICULTIES, 'difficulty'), (CHARACTERS, 'character'), (GAME_MODES, 'game_mode')):
        value = getattr(frame, attr, None)
        if value in names:
            out[offset+names.index(value)] = out[offset+len(names)] = 1
        offset += len(names)+1
    speeds = getattr(frame, 'movement_speeds', None)
    if speeds is not None:
        out[-3:] = (*np.clip(np.asarray(speeds)/20, 0, 1), 1)
    return out


def observed_settings(frame):
    selected = getattr(frame, 'phase_evidence', {}).get('menu_game_settings')
    if getattr(frame, 'phase', None) == 'practice_stage' and selected is not None:
        return dict(selected)
    return {key: getattr(frame, attr, None) for key, attr in
            (('difficulty', 'difficulty'), ('character', 'character'), ('mode', 'game_mode'),
             ('stage', 'stage_id'), ('movement_speeds', 'movement_speeds'))} | {'source':'game'}


def menu_key(frame, setup):
    """One edge for one positively identified menu; never an in-game action."""
    phase, selection = frame.phase, frame.menu_selection
    if phase == 'title_prompt':
        return 'z'
    if type(selection) is not int or not 0 <= selection <= 10:
        raise ValueError(msg('backend.menu_option_not_recognized'))
    targets = {'title': 2 if setup.mode == 'practice' else 1 if setup.difficulty == 'extra' else 0,
               'difficulty': DIFFICULTIES.index(setup.difficulty) if setup.difficulty != 'extra' else 0,
               'character': CHARACTERS.index(setup.character)//2,
               'shot': CHARACTERS.index(setup.character)%2, 'run_mode': 0,
               'practice_stage': setup.stage-1, 'start_confirm': 0}
    if phase not in targets:
        raise ValueError(msg('backend.no_actionable_menu_detected'))
    # A request may start on an already-open deep menu. Only earlier,
    # confirmed choices are compared; the current cursor is not a commit.
    evidence = frame.phase_evidence
    confirmed = evidence.get('menu_confirmed_settings', {})
    if phase not in ('title', 'title_prompt'):
        if ('practice_menu' in evidence
                and evidence['practice_menu'] != (setup.mode == 'practice')):
            return 'x'
        difficulty = confirmed.get('difficulty')
        if (phase == 'difficulty' and difficulty is not None
                and (difficulty == 'extra') != (setup.difficulty == 'extra')):
            return 'x'  # Extra and the regular difficulty list have distinct title entries.
        if (phase in ('character', 'shot', 'run_mode', 'practice_stage', 'start_confirm')
                and difficulty is not None and difficulty != setup.difficulty):
            return 'x'
        character = confirmed.get('character')
        if (phase in ('shot', 'run_mode', 'practice_stage', 'start_confirm')
                and character is not None and character != setup.character.split('_')[0]):
            return 'x'
        shot = confirmed.get('shot')
        if (phase in ('run_mode', 'practice_stage', 'start_confirm') and shot is not None
                and shot != setup.character.split('_')[1]):
            return 'x'
    if phase == 'practice_stage':
        count = frame.phase_evidence.get('practice_stage_count')
        if count is None or setup.stage > count:
            raise ValueError(msg('backend.the_selected_practice_stage_is_locked_or'))
    if phase == 'difficulty' and setup.difficulty == 'extra':
        # Extra has its own title entry; this screen is still confirmed with Z.
        return 'z'
    target = targets[phase]
    if selection == target:
        return 'z'
    if phase == 'character':
        return 'left' if selection > target else 'right'
    return 'up' if selection > target else 'down'
