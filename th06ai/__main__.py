from __future__ import annotations
from .i18n import localized, ui_text

import argparse
import json
from pathlib import Path

from .config import DEFAULT_CONSOLE_PORT, DEFAULT_PREVIEW_PORT, ROOT, load_config


def main():
    parser = argparse.ArgumentParser(description=ui_text('backend.th06nc_learning_agent_f8_emergency_stop'))
    parser.add_argument("command", choices=["open", "gui", "web", "shutdown", "legacy-gui", "monitor", "doctor", "check", "calibrate", "train", "play", "evaluate", "profile"], nargs="?", default="gui")
    parser.add_argument("--minutes", type=float, default=None)
    parser.add_argument("--model", default=None)
    parser.add_argument('--goal',choices=['survival','nmnb'],help=ui_text('backend.survival_allows_emergency_bombs_nmnb_disables_bombs'))
    parser.add_argument('--game-setup',help=argparse.SUPPRESS)
    parser.add_argument('--difficulty',choices=['easy','normal','hard','lunatic','extra'])
    parser.add_argument('--character',choices=['reimu_a','reimu_b','marisa_a','marisa_b'])
    parser.add_argument('--game-mode',choices=['story','practice'])
    parser.add_argument('--stage',type=int,help=ui_text('backend.starting_stage_for_practice_1_6'))
    parser.add_argument('--profile-kind',choices=['inference','latency'],default='inference',help=ui_text('backend.offline_inference_or_read_only_control_profiling'))
    parser.add_argument('--profile-wait-combat',action='store_true',help=ui_text('backend.wait_for_the_game_to_be_manually'))
    parser.add_argument("--preview", action="store_true", help=ui_text('backend.preview_the_console_without_connecting_to_the'))
    parser.add_argument("--port", type=int, default=DEFAULT_CONSOLE_PORT, help=ui_text('backend.local_web_console_port'))
    parser.add_argument("--no-browser", action="store_true", help=ui_text('backend.start_the_local_backend_without_opening_the'))
    parser.add_argument("--run-dir", type=Path, default=None, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.command in ('open', 'shutdown'):
        from .console_lifecycle import open_console, shutdown_console
        try:
            port = DEFAULT_PREVIEW_PORT if args.preview and args.port == DEFAULT_CONSOLE_PORT else args.port
            if args.command == 'open':
                print(localized(open_console(port=port, preview=args.preview, open_browser=not args.no_browser)))
            else:
                print(localized(shutdown_console(port=port, preview=args.preview)))
        except (OSError, ValueError, RuntimeError) as error:
            print(ui_text('backend.operation_failed_0', localized(str(error))))
            raise SystemExit(1)
        return
    config = load_config()
    from dataclasses import replace
    from .game_settings import GameSetup, parse_setup
    try:
        setup = parse_setup(json.loads(args.game_setup)) if args.game_setup is not None else config.game_setup
        if any(value is not None for value in (args.difficulty,args.character,args.game_mode,args.stage)):
            values = (setup or GameSetup()).to_dict()
            values.update({key:value for key,value in dict(difficulty=args.difficulty,character=args.character,
                          mode=args.game_mode,stage=args.stage).items() if value is not None})
            setup = parse_setup(values)
        config = replace(config, game_setup=setup)
    except (ValueError,TypeError) as error:
        parser.error(localized(str(error)))
    if args.command in ("gui", "web"):
        from .web_console import launch
        launch(config, preview=args.preview, port=args.port, open_browser=not args.no_browser)
        return
    if args.command == "legacy-gui":
        from .dashboard import launch
        launch(config, preview=args.preview)
        return
    if args.command == "doctor":
        from .native import inspect_game
        from .gpu import NvmlReader, SafetyPolicy
        gpu = NvmlReader(config.gpu_index)
        try:
            sample = gpu.sample()
            SafetyPolicy(config.safety).check(sample)
            print(json.dumps({"game": inspect_game(config.game_path), "gpu": sample.to_dict(),
                              "sensors": gpu.capabilities}, ensure_ascii=False, indent=2))
        finally:
            gpu.close()
        return
    from .supervisor import run_guard
    if args.run_dir is not None:
        target = args.run_dir.resolve()
        # An immediate UI stop can arrive before this process finishes
        # starting. Keep that cancellation marker; it is not an old run.
        used = target.exists() and (not target.is_dir() or any(p.name != "STOP" or not p.is_file() for p in target.iterdir()))
        if not target.is_relative_to((ROOT / "runs").resolve()) or used:
            parser.error(ui_text('backend.the_run_directory_must_be_unused_and'))
    result = run_guard(config, args.command, args.minutes, args.model, run_dir=args.run_dir,
                       goal=args.goal or ('nmnb' if args.command=='evaluate' else config.goal),
                       profile_kind=args.profile_kind if args.command=='profile' else None,
                       profile_wait_combat=args.profile_wait_combat)
    raise SystemExit(result.code)


if __name__ == "__main__":
    main()
