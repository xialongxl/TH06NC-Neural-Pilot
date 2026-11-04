"""Explicit web-start window activation, without a game reader or GPU model."""
from .i18n import msg
from pathlib import Path


def activate_game(path):
    from .win32 import GameWindow, processes, process_path
    expected = Path(path).resolve()
    try:
        matches = [pid for pid, name in processes() if name.casefold() == expected.name.casefold()
                   and Path(process_path(pid)).resolve() == expected]
        if len(matches) != 1:
            return {'activated': False, 'message': msg('backend.open_exactly_one_game_window_found_0', len(matches))}
        window = GameWindow.find(matches[0])
        was_foreground = window.focused
        activated = window.focus_for_start()
        return {'activated': activated, 'target_pid': window.pid, 'was_foreground': was_foreground,
                'message': msg('backend.game_focused') if activated else msg('backend.windows_has_not_granted_focus_click_the'),
                'diagnostics': getattr(window, 'focus_diagnostics', {})}
    except (OSError, RuntimeError) as error:
        return {'activated': False, 'message': msg('backend.could_not_focus_the_game_window_0', error)}
