"""Explicit web-start window activation, without a game reader or GPU model."""
from pathlib import Path


def activate_game(path):
    from .win32 import GameWindow, processes, process_path
    expected = Path(path).resolve()
    try:
        matches = [pid for pid, name in processes() if name.casefold() == expected.name.casefold()
                   and Path(process_path(pid)).resolve() == expected]
        if len(matches) != 1:
            return {'activated': False, 'message': f'需要一个已打开的游戏窗口，目前找到 {len(matches)} 个'}
        window = GameWindow.find(matches[0])
        was_foreground = window.focused
        activated = window.focus_for_start()
        return {'activated': activated, 'target_pid': window.pid, 'was_foreground': was_foreground,
                'message': '已切回游戏' if activated else '系统暂未允许切换前台；可点击游戏窗口或重试',
                'diagnostics': getattr(window, 'focus_diagnostics', {})}
    except (OSError, RuntimeError) as error:
        return {'activated': False, 'message': f'窗口激活未完成：{error}'}
