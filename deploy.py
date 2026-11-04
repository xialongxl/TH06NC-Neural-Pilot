"""Install and build the Windows checkout without starting a session."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import shutil
import subprocess

from th06ai.paths import ROOT
from tools.fetch_torch_cuda import DEST as WHEEL_DIR, NAME as WHEEL_NAME


PYTHON_DEPENDENCIES = (
    "numpy==2.2.6", "pillow==12.3.0", "mss==10.2.0", "pefile==2024.8.26",
    "capstone==5.0.9", "nvidia-ml-py==13.615.71", "gymnasium==1.4.0",
    "stable-baselines3==2.9.2", "tensorboard==2.21.0", "pytest==9.1.1",
    "fastapi==0.162.0", "uvicorn==0.54.0", "httpx2==2.14.0",
)


def require_tool(name, hint):
    path = shutil.which(name)
    if not path:
        raise RuntimeError(hint)
    return path


def install_backend(uv, env):
    venv = ROOT / ".venv"
    python = venv / "Scripts" / "python.exe"
    if not python.is_file():
        if venv.exists():
            raise RuntimeError("The existing .venv has no Windows Python executable. Check that environment first.")
        subprocess.run([uv, "venv", "--python", "3.10", str(venv)], cwd=ROOT, env=env, check=True)

    probe = "import sys, sysconfig; print(sys.implementation.name, *sys.version_info[:2], sysconfig.get_platform())"
    abi = subprocess.check_output([str(python), "-c", probe], cwd=ROOT, env=env, text=True).strip()
    if abi != "cpython 3 10 win-amd64":
        raise RuntimeError("The existing .venv must use 64-bit CPython 3.10 for the pinned CUDA wheel.")

    subprocess.run([str(python), "tools/fetch_torch_cuda.py"], cwd=ROOT, env=env, check=True)
    subprocess.run(
        [uv, "pip", "install", "--python", str(python), str(WHEEL_DIR / WHEEL_NAME), *PYTHON_DEPENDENCIES],
        cwd=ROOT, env=env, check=True,
    )
    config = ROOT / "config.json"
    if not config.exists():
        template = (ROOT / "config.example.json").read_bytes()
        try:
            with config.open("xb") as stream:
                stream.write(template)
        except FileExistsError:
            pass
        else:
            print("Created config.json. Set game_exe to the full path to your th06nc.exe.", flush=True)


def build_ui(npm, env, skip_npm_ci):
    frontend = ROOT / "frontend"
    npm_env = dict(env, NPM_CONFIG_CACHE=str(ROOT / ".cache" / "npm"))
    if not skip_npm_ci:
        if not (frontend / "node_modules").resolve().is_relative_to(frontend):
            raise RuntimeError("frontend/node_modules resolves outside the frontend directory.")
        subprocess.run([npm, "ci", "--no-fund", "--no-audit"], cwd=frontend, env=npm_env, check=True)
    subprocess.run([npm, "run", "build"], cwd=frontend, env=npm_env, check=True)


def build_launcher(uv, env, pyinstaller_python):
    cache = ROOT / ".cache" / "launcher"
    cache.mkdir(parents=True, exist_ok=True)
    dist = cache / "dist"
    if pyinstaller_python is None:
        command = [uv, "tool", "run", "--python", "3.10", "--from", "pyinstaller==6.22.2", "pyinstaller"]
    else:
        command = [str(pyinstaller_python.resolve()), "-m", "PyInstaller"]
    command += [
        "--noconfirm", "--onefile", "--windowed", "--noupx", "--name", "LaunchAI",
        "--paths", str(ROOT), "--add-data", f"{ROOT / 'th06ai' / 'messages.json'}:th06ai",
        "--distpath", str(dist), "--workpath", str(cache / "build"),
        "--specpath", str(cache), "tools/launch_console.py",
    ]
    launcher_env = dict(env, PYINSTALLER_CONFIG_DIR=str(cache / "config"))
    subprocess.run(command, cwd=ROOT, env=launcher_env, check=True)
    (dist / "LaunchAI.exe").replace(ROOT / "LaunchAI.exe")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("step", choices=("all", "install", "ui", "launcher"), nargs="?", default="all",
                        help="default: install dependencies, build the web UI, then package LaunchAI.exe")
    parser.add_argument("--skip-npm-ci", action="store_true", help="build the UI with existing node_modules")
    parser.add_argument("--pyinstaller-python", type=Path, metavar="EXE",
                        help="use an existing Python with PyInstaller for the launcher build")
    args = parser.parse_args(argv)
    install = args.step in ("all", "install")
    ui = args.step in ("all", "ui")
    launcher = args.step in ("all", "launcher")
    if (args.skip_npm_ci and not ui) or (args.pyinstaller_python and not launcher):
        parser.error("build options require their corresponding step or all")
    if os.name != "nt":
        parser.error("this deployment targets Windows")

    env = dict(os.environ, PYTHONUTF8="1", UV_CACHE_DIR=str(ROOT / ".cache" / "uv"))
    try:
        uv = require_tool("uv", "Install uv and add it to PATH first.") if (
            install or (launcher and args.pyinstaller_python is None)) else None
        npm = require_tool("npm.cmd", "Install Node.js 22.12+ or 24 LTS with npm.") if ui else None
        if install:
            print("Installing Python dependencies...", flush=True)
            install_backend(uv, env)
        if ui:
            print("Building the web console...", flush=True)
            build_ui(npm, env, args.skip_npm_ci)
        if launcher:
            print("Packaging LaunchAI.exe...", flush=True)
            build_launcher(uv, env, args.pyinstaller_python)
            print("LaunchAI.exe is ready in the project root.", flush=True)
    except (OSError, RuntimeError, subprocess.CalledProcessError) as error:
        parser.exit(1, f"Deployment failed: {error}\n")


if __name__ == "__main__":
    main()
