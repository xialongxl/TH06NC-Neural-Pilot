# Code map

English · [简体中文](../code-map.md)

The product is TH06NC-Neural-Pilot. The Python import and CLI package is `th06ai`. Retain serialized class paths and service identities when changing display text. Use the [generated Python index](../code-index.json) for signatures, imports, normalized source hashes, and line numbers. Rebuild it with `python tools/check_docs.py --update-index` after source changes.

## Where to start

| Task | Reading order |
| --- | --- |
| Run locally | [Development guide](development.md) → config → CLI → console lifecycle |
| Understand training | Environment → observations → planning → policy → pausable PPO → learning |
| Diagnose late input | Reader → input timing → inference → environment refresh/step → planner |
| Inspect a death or resources | Diagnostics and rows → rewards and objectives → planner → policy |
| Change the UI | useConsole → stream and types → App → console stream and API |
| Modify safety | Config → GPU → lease → supervisor → heartbeat and Job |
| Adapt a game build | Reader → combat, lasers, stage, phase → offline fixtures and read-only verification |

## Repository layout

```text
.
├── README.md / README.zh-CN.md / CONTRIBUTING.md / DISCLAIMER.md / LICENSE
├── THIRD_PARTY_NOTICES.md / LICENSES/      upstream notices
├── AGENTS.md                               repository instructions
├── config.example.json / .env.example      portable examples
├── install.ps1 / build-ui.ps1               setup and production build
├── StartAI.cmd / CloseAI.cmd                local service launchers
├── .github/                                bilingual collaboration templates
├── docs/                                   Chinese guides and shared indexes
│   └── en/                                 English guides
├── th06ai/                                 Python runtime and message catalog
│   ├── native_runtime/                     optional ABI v3
│   └── native_planner/                     optional standalone search
├── frontend/
│   ├── src/                               React, transport, display logic, tests
│   ├── public/                            local static assets
│   ├── package*.json                      dependencies and npm lockfile
│   └── dist/                              generated, ignored production assets
├── native/
│   ├── src/lib.rs                         PyO3, collision, capture, and input
│   ├── src/planner.rs                     owned native search
│   └── Cargo.toml / Cargo.lock
├── tests/                                  offline tests and minimal fixtures
└── tools/                                  install, build, and release checks
```

`models/`, `runs/`, `.venv/`, `.cache/`, `research/`, and `tmp/` are local. The [Stage 5 switch-window fixture](../../tests/fixtures/stage5-switch-window.npz) is a public geometry fixture for a regression.

## Python modules

### Entry points, localization, and service lifecycle

| Module | Role |
| --- | --- |
| [__init__.py](../../th06ai/__init__.py) | Package entry with no game or GPU initialization. |
| [__main__.py](../../th06ai/__main__.py) | CLI parsing and dispatch to open, shutdown, web, or the guard. |
| [i18n.py](../../th06ai/i18n.py) | Shared message catalog, process locale, and archived-text localization. |
| [config.py](../../th06ai/config.py) | Repository root, ports, typed runtime and safety settings, and validation. |
| [console_lifecycle.py](../../th06ai/console_lifecycle.py) | Service identity, port and startup lock, readiness, shutdown, and late-start cancellation. |
| [web_console.py](../../th06ai/web_console.py) | Loopback routes, schemas, Origin and token checks, SSE, and static files. |
| [console_service.py](../../th06ai/console_service.py) | Explicit session operations, checkpoint IDs, local state, and logs. |
| [console_stream.py](../../th06ai/console_stream.py) | Shared sampling, display DTO, deltas, backpressure, and curve cursors. |
| [dashboard.py](../../th06ai/dashboard.py) | Legacy Tk console using the same CLI and protection path. |

### Independent protection and Windows boundaries

| Module | Role |
| --- | --- |
| [gpu.py](../../th06ai/gpu.py) | GPU samples, NVML, asynchronous telemetry, and latched safety policy. |
| [driver_events.py](../../th06ai/driver_events.py) | Windows driver event collection and error reporting. |
| [lease.py](../../th06ai/lease.py) | Owned shared lease, original sample timestamps, and mutex handling. |
| [supervisor.py](../../th06ai/supervisor.py) | Guard loop, preflight, budget, Job, F7/F8, and final status. |
| [hotkeys.py](../../th06ai/hotkeys.py) | Edge-based dispatch and fallback polling, including while Shift is held. |
| [win32.py](../../th06ai/win32.py) | Read-only process handles, foreground window, input, process enumeration, Job, and guard lock. |
| [focus.py](../../th06ai/focus.py) | Bounded foreground activation and focus recovery. |
| [worker.py](../../th06ai/worker.py) | Lease-gated Torch startup, mode dispatch, heartbeat, and calibration. |
| [backend.py](../../th06ai/backend.py) | Protected backend forward/backward self-check and readiness evidence. |

### Game state and geometry

| Module | Role |
| --- | --- |
| [native.py](../../th06ai/native.py) | Executable identity, read-only capture, and native Frame and ControlSample decoding. |
| [game_phase.py](../../th06ai/game_phase.py) | Pause, dialogue, menu, continue, and results classification. |
| [stage.py](../../th06ai/stage.py) | Stage IDs, labels, one-hot features, and context boundaries. |
| [combat.py](../../th06ai/combat.py) | Separate attack and contact decoding, credible damage, and alignment. |
| [lasers.py](../../th06ai/lasers.py) | Straight laser pool layout, visible phases, and collision width. |
| [bullet_state.py](../../th06ai/bullet_state.py) | Slot matching, motion validity, radii, and closest approach. |
| [body_state.py](../../th06ai/body_state.py) | Enemy-body and laser motion, orientation, and uncertainty. |
| [motion.py](../../th06ai/motion.py) | Input-confirmed stable speed samples and calibration. |
| [risk.py](../../th06ai/risk.py) | Observed risk from bullets and damaging lasers. |

### Decision, reward, and native acceleration

| Module | Role |
| --- | --- |
| [observations.py](../../th06ai/observations.py) | Versioned feature encoding, control history, and joint action decoding. |
| [game_settings.py](../../th06ai/game_settings.py) | Validated setup requests, observed identity, menu state, and context encoding. |
| [objectives.py](../../th06ai/objectives.py) | Inventory-aware item weights and Power potential. |
| [rewards.py](../../th06ai/rewards.py) | Actual transition rewards, budgets, camping, and wall pressure. |
| [planner_spec.py](../../th06ai/planner_spec.py) | Lightweight planning constants and configuration. |
| [navigation.py](../../th06ai/navigation.py) | Navigation and decision features, exits, and conditional bomb prior constants. |
| [planning.py](../../th06ai/planning.py) | Continuous geometry, beam search, switch-window checks, and fallback. |
| [acceleration.py](../../th06ai/acceleration.py) | Extension discovery, owned serialization, and Rust keyboard wrapper. |
| [native_planner/__init__.py](../../th06ai/native_planner/__init__.py) | Independent optional search extension entry. |
| [native_runtime/__init__.py](../../th06ai/native_runtime/__init__.py) | Optional ABI v3 capture, geometry, and input entry. |
| [policy_spec.py](../../th06ai/policy_spec.py) | Lightweight policy version and base bomb prior. |
| [policy.py](../../th06ai/policy.py) | Set encoders, ResourcePolicy, and the joint distribution. |
| [inference.py](../../th06ai/inference.py) | Reusable policy buffers, CUDA Graphs, readback, and timing. |

### Sampling, learning, and persistence

| Module | Role |
| --- | --- |
| [environment.py](../../th06ai/environment.py) | Game state machine, reset/step, actions, timing, rewards, and boundaries. |
| [control.py](../../th06ai/control.py) | Cooperative PAUSE/FOCUS/FINISH and sampling interruption. |
| [input_timing.py](../../th06ai/input_timing.py) | Host timeline, observed movement-bit match, and recent percentiles. |
| [pausable_ppo.py](../../th06ai/pausable_ppo.py) | Single-environment rollout gate, real transitions, and boundary bootstrap. |
| [ppo_update.py](../../th06ai/ppo_update.py) | ResourcePolicy PPO update and exploration objective. |
| [learning.py](../../th06ai/learning.py) | Build, load, migrate, save, train, and evaluate a policy. |
| [model_store.py](../../th06ai/model_store.py) | Compatibility, catalog, and immutable ZIP/JSON checkpoint retention. |
| [diagnostic_writer.py](../../th06ai/diagnostic_writer.py) | Bounded CPU helper queue, private transport, compression, and cleanup. |
| [diagnostics.py](../../th06ai/diagnostics.py) | Frame and control rings, death archives, and asynchronous diagnostics. |
| [log_rows.py](../../th06ai/log_rows.py) | Shared action and episode row mapping. |
| [files.py](../../th06ai/files.py) | Atomic JSON, Windows shared-delete reads, timestamps, and run paths. |
| [profile_inference.py](../../th06ai/profile_inference.py) | Protected offline policy inference benchmarks. |
| [profile_latency.py](../../th06ai/profile_latency.py) | Read-only live capture, planning, and inference timing with fixed-geometry comparisons. |

## Frontend, Rust, and build tooling

| Entry | Role |
| --- | --- |
| `frontend/src/main.tsx`, `App.tsx` | React entry, pages, session operations, dialogs |
| `GameSetupFields.tsx`, `gameSetup.ts` | Setup form, supported combinations, and stored preferences |
| `useConsole.ts`, `stream.ts`, `types.ts` | Bootstrap, controls, SSE, version/cursor reconciliation, and display DTO |
| `logic.ts`, `i18n.tsx`, `style.css` | Presentation, shared catalog, and accessible responsive layout |
| `renderProfile.tsx`, `vite.config.ts` | Benchmark-only profiler, API proxy, build aliases |
| `native/src/lib.rs`, `native/src/planner.rs` | Python ABI, owned snapshots, continuous geometry, and bounded search |
| `tools/fetch_torch_cuda.py`, `tools/build_native.py` | Verified wheel assembly and local native build |
| `tools/check_docs.py`, `tools/check_source.py` | Offline documentation/index validation and Git release checks |

For a new UI metric, preserve raw evidence first and then decide what belongs in the display DTO. A new observation input also needs migration, optimizer, inference buffers, and trace coverage. A new attack needs version evidence, explicit geometry validity, Python/Rust parity, and separately authorized live verification. New control actions require explicit user intent and a session state transition. [Testing matrix](testing-and-performance.md)
