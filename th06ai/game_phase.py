"""Build-pinned game phases. A frozen tick or empty bullet pool is not a menu."""
from __future__ import annotations
from .i18n import msg
from dataclasses import dataclass

PHASE_NAMES={'combat':msg('backend.stage_in_progress'),'boss':msg('backend.boss_fight'),'pause':msg('backend.game_paused'),'continue':msg('backend.continue_menu'),
             'dialogue':msg('backend.boss_dialogue'),'script_wait':msg('backend.cutscene'),'title':msg('backend.main_menu'),'difficulty':msg('backend.select_difficulty'),
             'character':msg('backend.select_character'),'shot':msg('backend.select_shot_type'),'menu_transition':msg('backend.menu_transition'),
             'result':msg('backend.results_screen'),'dead':msg('backend.player_hit'),'unknown':msg('backend.waiting_for_game_state'),
             'title_prompt':msg('backend.title_screen'),'run_mode':msg('backend.select_game_mode'),'start_confirm':msg('backend.confirm_game_start'),
             'practice_stage':msg('backend.practice_stage_select')}
COMBAT_PHASES={'combat','boss','dead'}

def phase_of(frame):
    return getattr(frame,'phase','') or ('pause' if frame.paused else 'combat' if frame.stage_active else 'unknown')

def combat_frame(frame):
    return frame.stage_active and not frame.paused and phase_of(frame) in COMBAT_PHASES

@dataclass(frozen=True)
class PhaseState:
    phase:str
    paused:bool
    stage_active:bool
    selection:int|None=None
    ready:bool=False

def classify(*,active,engine_state=None,pause_flag=0,continue_flag=0,result_flag=0,
             script_stop=False,dialogue_index=-1,nc_screen=None,selection=None,pause_selection=None,
             pause_animation=None,player_state=0,boss=False,menu_ticks=0):
    # NC's own pause UI at 0x9806/0xa850 uses 0x53d400. 0x53d409 is
    # an ECL script stop, and is deliberately not called a pause menu here.
    in_stage=bool(active and (engine_state is None or engine_state==2))
    if in_stage and pause_flag:
        return PhaseState('pause',True,True,pause_selection,pause_animation==1)
    if in_stage and continue_flag:return PhaseState('continue',False,True,selection,True)
    if result_flag:return PhaseState('result',False,in_stage)
    if not in_stage:
        if engine_state==1:
            menus={2:'title_prompt',3:'title',9:'difficulty',11:'character',13:'shot',14:'run_mode',22:'practice_stage',24:'start_confirm'}
            phase=menus.get(nc_screen,'menu_transition')
            ready_at=30 if nc_screen in (2,3) else 15
            return PhaseState(phase,False,False,selection,phase!='menu_transition' and menu_ticks>=ready_at)
        return PhaseState('unknown',False,False,selection,False)
    if dialogue_index>=0:return PhaseState('dialogue',False,True)
    if script_stop:return PhaseState('script_wait',False,True)
    return PhaseState('dead' if player_state==2 else 'boss' if boss else 'combat',False,True)
