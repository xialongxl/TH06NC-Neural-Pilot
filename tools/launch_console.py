"""Windowless Windows entry point for the packaged web console launcher."""

import argparse
import ctypes
from pathlib import Path
import sys

from th06ai.console_lifecycle import open_console, shutdown_console
from th06ai.i18n import localized, ui_text
from th06ai.paths import DEFAULT_CONSOLE_PORT, DEFAULT_PREVIEW_PORT, ROOT


class LauncherParser(argparse.ArgumentParser):
    def error(self, message):
        raise ValueError(message)


def main():
    parser = LauncherParser(prog="LaunchAI.exe", add_help=False)
    parser.add_argument("command", choices=("open", "shutdown"), nargs="?", default="open")
    parser.add_argument("--preview", action="store_true")
    parser.add_argument("--port", type=int)
    parser.add_argument("--no-browser", action="store_true")
    try:
        args = parser.parse_args()
        root = ROOT
        if getattr(sys, "frozen", False):
            root = Path(sys.executable).resolve().parent
            # The backend and browser use their own DLLs, outside the frozen bundle.
            if not ctypes.windll.kernel32.SetDllDirectoryW(None):
                raise ctypes.WinError()
        port = args.port if args.port is not None else (
            DEFAULT_PREVIEW_PORT if args.preview else DEFAULT_CONSOLE_PORT)
        if args.command == "shutdown":
            shutdown_console(root=root, port=port, preview=args.preview)
        else:
            open_console(root=root, port=port, preview=args.preview, open_browser=not args.no_browser)
    except (OSError, ValueError, RuntimeError) as error:
        message = ui_text("backend.operation_failed_0", localized(str(error)))
        ctypes.windll.user32.MessageBoxW(None, message, "TH06NC-Neural-Pilot", 0x10)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
