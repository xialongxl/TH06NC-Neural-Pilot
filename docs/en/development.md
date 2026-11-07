# Development, installation, and local deployment

English · [简体中文](../development.md)

Run commands from the repository root unless indicated otherwise. The calibration and session examples identify when they use the GPU, send game input, or update weights.

## Requirements and installation

| Component | Supported or tested setup |
| --- | --- |
| OS | Windows 10/11 x64 for live play |
| Python | CPython 3.10 x64 for the installer and native ABI |
| GPU | NVIDIA with NVML support and a driver compatible with the pinned CUDA build of PyTorch |
| Package installer | `uv` on PATH |
| Frontend | Node.js 22.12+ or 24 LTS and npm to build the UI |
| Native extension | Optional Rust and MSVC Build Tools, with PyO3 and Cargo dependencies pinned |
| Game | A legitimately installed copy of the supported Steam build. [Game requirements](../../README.md#requirements) |

```powershell
python deploy.py
notepad .\config.json
```

The Python entry point locates the checkout from its own file and stores build caches inside the project. With no arguments it runs these steps in order. Use the corresponding command to repeat an individual step.

| Command | Work |
| --- | --- |
| `python deploy.py install` | Create or reuse a Windows CPython 3.10 x64 environment. Verify the exact CUDA wheel against the official PyTorch SHA-256 and install the pinned dependencies. Create `config.json` if missing. |
| `python deploy.py ui` | Run `npm ci` with the lockfile and build `frontend/dist`. Add `--skip-npm-ci` to build with existing dependencies. |
| `python deploy.py launcher` | Build the windowless `LaunchAI.exe` with an isolated PyInstaller environment. Pass `--pyinstaller-python EXE` to use a Python installation that already has PyInstaller. |

Set `game_exe` in `config.json` to the full executable path, then double-click `LaunchAI.exe`. Deployment keeps existing configuration, checkpoints, and calibration records. Optional Rust builds and first-run calibration are separate steps. A CPython ABI change requires a new native build.

`device=cpu` selects the network device. Live sessions still use the independent NVIDIA watchdog. Offline CPU tests use fixtures.

## Configuration

`load_config()` reads `config.json` from the project root derived from the source location. Local paths and calibration files are ignored by Git. See the sample for a portable starting point. [Sample configuration](../../config.example.json)

| Field | Meaning and current defaults |
| --- | --- |
| `game_exe` | Full local path to `th06nc.exe` |
| `gpu_index` | NVML index, default 0. The guard binds the selected GPU by UUID. |
| `minutes` | Session budget, default 60 and greater than 0 through 60 |
| `episode_seconds` | Per-round combat cap. Default 0 disables the cap. A nonzero value is 10 through 1800. |
| `device` | `cuda` or `cpu`. An unavailable CUDA selection fails explicitly. |
| `input_backend`, `planning.collision_backend` | Python, Rust, or Auto. Dataclass default is Python and the sample selects Auto. |
| `capture_backend` | Auto batches small fields in Rust when available and reuses large Python buffers. Python uses Python capture. Rust selects the explicit owned large-block path. |
| `inference_backend` | `auto`, `eager`, or `cuda_graph`. Auto can fall back for graph-capture limitations. |
| `inference_readback` | `auto` uses blocking-event readback with CUDA Graph and synchronous readback with CUDA eager or CPU. Set `synchronous` to restore synchronous readback on the next session. |
| `goal` | `survival` or `nmnb`. NMNB masks bomb actions. |
| `game_setup` | Optional mode, difficulty, character, and starting stage. An explicit launch request can set the session values. |
| `planning.enabled`, `planning.beam_width` | Planning enabled by default. Beam width defaults to 2 and accepts 1–4. The current move set still has 18 entries. |
| `diagnostics` | Optional rollout detail, recording and save intervals, and compression. Defaults retain a full trace and a save each rollout. |

Read the effective config and session parameters when reproducing an issue. The sample file, dataclass defaults, and an existing local config can differ. [Game setup](game-settings.md) · [Diagnostics configuration](performance-tuning.md)

### Safety settings

| `safety` field | Default | Accepted range |
| --- | --- | --- |
| `temperature_limit_c` | 70°C | 30–70 |
| `memory_temperature_limit_c` | 80°C | 40–80, when the sensor is available |
| `vram_fraction_limit` | 0.90 | 0.10–0.90 |
| `power_limit_ratio` | 1.03 | 1.00–1.03 |
| `fan_required_at_c` | 60°C | 40–60 |
| `poll_seconds` | 0.25 s | 0.05–0.25 s |
| `telemetry_timeout_seconds` | 60 s | 0.1–60 s |
| `worker_heartbeat_seconds` | 2 s | 0.5–2 s |

These are validation limits. The console backend-selection API exposes only the relevant acceleration settings. The independent guard owns protection.

### Environment variables

Set environment variables in the launching shell. The application does not automatically load `.env`. The sample documents process variables such as `PYTHONUTF8`, `PYTHONUNBUFFERED`, and `UV_CACHE_DIR`. Use `config.json` for the game path and safety settings. The guard sets `CUDA_VISIBLE_DEVICES` for its own worker.

```powershell
$env:PYTHONUTF8 = '1'
$env:PYTHONUNBUFFERED = '1'
$env:TH06AI_LANG = 'en'
```

Use `zh-CN` for Simplified Chinese CLI and desktop text. The web console has its own browser language selector. [Localization](localization.md)

## First-run calibration

A new checkout needs three local checks. Each check records separate evidence. Use the GPU and game steps only when the operator is ready.

### 1. Check the GPU

Use **GPU Check** in the console or run the command below. It uses the protected CUDA path and writes `training_backend.json` after a backend check. It sends no game input and updates no user model.

```powershell
.\.venv\Scripts\python.exe -m th06ai check --minutes 1
```

### 2. Calibrate movement and restart

Open a controllable Stage 1 on Normal with Reimu A. Keep the game foregrounded and use an English input method. Start the console calibration or use the command below. It sends live movement and restart inputs, verifies focused and unfocused movement, observes bullets, and confirms that the stage counter resets and advances after R. A successful calibration pauses the game and records `native_calibration.json`. Ordinary training accepts empty bullet pools.

```powershell
.\.venv\Scripts\python.exe -m th06ai calibrate --minutes 1
```

### 3. Confirm Power, bombs, and lives

Resource confirmation is a separate manual step. The console does not yet create `native_resources_calibration.json`. With the game displaying its resource counters, run this read-only check. It reads the running game without keyboard input or CUDA.

```powershell
.\.venv\Scripts\python.exe -c "from th06ai.config import load_config; from th06ai.native import GameReader; from dataclasses import asdict; import json; r=GameReader(load_config().game_path); f=r.read(); print(json.dumps({'game_sha256':r.identity['sha256'],'resources':asdict(f.resources) if f.resources else None},ensure_ascii=False)); r.close()"
```

Compare Power, bombs, and remaining lives against the in-game HUD. Save the following structure as UTF-8 in `native_resources_calibration.json` at the repository root. Replace the placeholder with the full SHA-256 printed by the read-only check. Set `operator_confirmed` to `true` only after verifying the readings. Use the exact filename and keep this local record out of Git.

```json
{
  "game_sha256": "REPLACE_WITH_THE_SHA256_FROM_THE_READ_ONLY_CHECK",
  "operator_confirmed": false
}
```

## Local service and development mode

For daily use, double-click the generated `LaunchAI.exe` in the project root. The build script uses an isolated PyInstaller environment and packages the lifecycle code and bilingual messages. The backend still runs through the project's `pythonw.exe`, so install the project dependencies and build the web console first. Use **Shut Down** in the console or run `.\LaunchAI.exe shutdown`.

```powershell
.\.venv\Scripts\python.exe -m th06ai open
.\.venv\Scripts\python.exe -m th06ai shutdown
```

The production backend serves the build directly on loopback port `18765`. For development, run `python -m th06ai web --no-browser` using the project interpreter, then run `npm.cmd ci` and `npm.cmd run dev` from `frontend`. Keep the Vite proxy target in sync with a changed backend port. Diagnose origin, host, and token on a 403 while preserving the same-origin check.

`python -m th06ai open --preview` defaults to port `18766`. The preview can show saved state without attaching to the game, opening a GPU session, or starting a worker. The legacy Tk console remains available through `legacy-gui`.

## CLI operations

| Command | Game access | GPU path | Learns |
| --- | --- | --- | --- |
| `open`, `gui`, `web` | No input for ordinary display. Explicit focus and probe have separate behavior. | The HTTP service does not own CUDA. | No |
| `doctor` | Inspects the configured executable | One NVML check | No |
| `monitor` | No game input | Independent monitoring | No |
| `check` | No game input | Protected CUDA check | No |
| `calibrate` | Moves, pauses, and restarts | Independent guard | No |
| `train` | Controls the game | Guard and learning worker | Yes |
| `play`, `evaluate` | Control the game | Guard and inference worker | No |
| `profile` | Offline by default. The latency profile can read the running game. | Guard and inference worker | No |

Choose one operation. The following examples include live game control.

```powershell
.\.venv\Scripts\python.exe -m th06ai train --minutes 5 --goal survival
.\.venv\Scripts\python.exe -m th06ai train --minutes 5 --model models/latest.zip
.\.venv\Scripts\python.exe -m th06ai play --minutes 1 --model models/latest.zip --goal nmnb
.\.venv\Scripts\python.exe -m th06ai evaluate --minutes 60 --model models/latest.zip --goal nmnb
.\.venv\Scripts\python.exe -m th06ai profile --profile-kind latency --profile-wait-combat --minutes 1 --model models/latest.zip
```

Evaluation requests ten independent rounds by default. Check `evaluation_progress.json` for the actual completed count after a timeout or interruption.

## Control API

| Endpoint | Contract |
| --- | --- |
| `GET /api/health` | Service identity, checkout, preview flag, and PID |
| `GET /api/bootstrap?state=false` | Token and protocol. Omitting the query can include initial state |
| `GET /api/state`, `GET /api/events?protocol=2&hidden=false`, `GET /api/logs` | Full diagnostics, shared display deltas, or bounded log tail |
| `GET /api/performance` | System CPU readings on demand, shared across tabs and sampled at most once per second |
| `POST /api/models/select` | Select an existing `model_id` |
| `POST /api/sessions` | Explicit mode, minutes, optional model, goal, and game setup |
| `POST /api/session/pause` | Boolean `paused` for a training, play, or evaluation session |
| `POST /api/session/finish`, `POST /api/session/stop` | Save and finish training, or stop the owned session immediately |
| `POST /api/session/focus`, `POST /api/game/focus`, `POST /api/game/probe` | Retry active focus, explicitly activate the configured game, or probe read-only while idle |
| `POST /api/settings/acceleration` | Set validated `collision` and `input` backends while idle |
| `POST /api/server/shutdown` | Coordinate session finish and local service shutdown |

POST requests require the bootstrap `X-Console-Token` and same-origin validation. Request models reject unknown fields. The service is designed for local operation, with no public authentication layer. [Web console protocol](../../frontend/README.md)

## Native build and source deployment

From an MSVC/Rust development shell run `python tools/build_native.py` using the project interpreter. Use `--test` for native tests or `--planner-only` to build the standalone planner. Custom toolchains use `--toolchain` and `--vcvars`. Finish the session before replacing a loaded `.pyd`. [Native build guide](../../native/README.md)

Supported deployment is a Windows source checkout with the locally configured game, installed dependencies, and built frontend. The single-file executable provides the launcher. Bundled wheel deployment, Docker, Linux live play, and public multi-user hosting remain outside the verified scope.

For a source release, review tracked and staged contents from this checkout before pushing the actual remote. Git ignore rules exclude local runtime artifacts from new additions. Audit the index and history separately. A source ZIP can be generated with `git archive` from a reviewed commit. [Release checklist](open-source-readiness.md)
