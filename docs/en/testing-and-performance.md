# Tests, live verification, and performance

English · [简体中文](../testing-and-performance.md)

## Choose evidence for the question

| Method | What it verifies |
| --- | --- |
| Offline tests | Geometry, formats, timing semantics, probabilities, persistence, and error handling under fixtures |
| Paired component benchmark | Cost of one component for the same inputs and timing assumptions |
| Read-only live profile | Capture, observation, planning, and inference from a real scene without dispatch |
| Authorized live input and independent evaluation | Actual commands and results for a specific game setup, objective, and machine |

Log the checkpoint hash, steps, objective, and setup for fixed-weight evaluation. Use the actual number of completed rounds. The read-only path omits key dispatch and full training callbacks.

## Run checks

Run the Python examples from the repository root. Use the npm commands in `frontend`.

```powershell
.\.venv\Scripts\python.exe tools\check_docs.py
.\.venv\Scripts\python.exe -m pytest -q --basetemp .cache\pytest-local
.\.venv\Scripts\python.exe -m pytest -q tests/test_native_search.py tests/test_planning.py tests/test_planner_timing_recovery.py tests/test_enemy_contact.py tests/test_laser_geometry.py
.\.venv\Scripts\python.exe tools\build_native.py --test
```

```powershell
npm.cmd test
npm.cmd run build
```

Use your own MSVC/Rust toolchain for native tests. An unavailable optional extension can cause a skip. Report it separately. Windows-specific imports limit the verified scope of the full suite. Offline tests use fixtures or dry run at hardware and input boundaries. CPU PPO tests update temporary models.

## Regression matrix

| Area | Representative tests | Risk |
| --- | --- | --- |
| Guard | `test_safety.py`, `test_supervisor.py`, `test_lease.py` | Fault latching, independent stop, original lease age, owned process tree |
| Lifecycle | `test_console_lifecycle.py`, `test_files.py` | Service duplication, port identity, late launch, atomic replace |
| Input and focus | `test_focus_hotkeys.py`, `test_input_latency.py`, `test_motion_calibration.py` | Shift held, key edges, foreground activation, timing |
| Capture and resources | `test_state_capture.py`, `test_resources.py`, `test_combat.py` | Empty and cross-tick reads, resource and body decoding |
| Scene and Practice | `test_game_phases.py`, `test_stage_awareness.py`, `test_game_settings.py` | Pause evidence, stages, setup mismatches, unlocks, new shot speeds, GAE boundary |
| Observation | `test_observations.py`, `test_control_observation.py`, `test_full_bullet_observation.py`, `test_observation_traces.py` | Layout, valid masks, threats beyond the near field, real rollout data |
| Geometry and native | `test_planning.py`, `test_planner_timing_recovery.py`, `test_enemy_contact.py`, `test_laser_geometry.py`, `test_rust_backend.py`, `test_native_search.py` | Window fallback, hazards, ABI shapes, concurrency, Python parity |
| Rewards | `test_rewards.py`, `test_reward_balance.py`, `test_behavior_objective.py`, `test_power_progress.py`, `test_graze_principle.py` | Camping, recollection, death settlement, grazing |
| Policy and migration | `test_policy.py`, `test_bomb_policy.py`, `test_learning.py`, `test_stage5_upgrade.py` | Joint probabilities, bomb exploration, retained weights and optimizer, graph state |
| Environment | `test_environment.py`, `test_training_controls.py`, `test_sampling_timing.py` | Resume, real transition, Miss after respawn, bounded refresh |
| Model library and evaluation | `test_model_store.py`, `test_evaluation_progress.py` | Latest integrity, immutable history, partial reports |
| API and display | `test_dashboard.py`, `test_web_console.py`, `test_console_stream.py`, frontend `stream.test.ts`, `logic.test.ts`, and `i18n.test.tsx` | Authorization, fan-out, delta recovery, reference stability, localization |
| Performance work | `test_latency_profiler.py`, `test_optimization_contracts.py` | Stable outputs, timing semantics, async diagnostics, catalog and capture parity |

The acquisition tests that read real memory target the test process itself. Q-key tests use a fake OS or native dry run. [Optimization contracts](performance-tuning.md)

## Live acceptance sequence

Obtain the operator's explicit approval for the relevant live operations.

1. Read identity, resources, stage, and scene, then compare against the display.
2. Calibrate movement, focus movement, and restart with a short input test.
3. Play fixed weights briefly under the selected objective.
4. Evaluate independent rounds with the same setup, objective, and truncation, writing each completed result.
5. Train against a stated hypothesis and compare actual resources, attacks, camping, bombs, and Misses alongside reward.

## Measure latency

Capture monotonic timestamps for one action across full observation, planning, policy, send, and the first observed input match. Derive nonoverlapping intervals first, then compute total and stage percentiles. CUDA stream events may include transfer, kernels, and device scheduling. Host readback can also include queueing and host scheduling. A local Python CPU timing probe previously showed about 15.625 ms quantization.

For paired comparisons, freeze inputs, observation age, and timing history, alternate execution order, and verify output and candidate parity. Report workload, backend, CPU/GPU, sample count, P50/P95/P99, and whether input was sent. Preserve real training evidence separately from synthetic tests and profiling.

## Historical reference results

These measurements are from the maintainer's machine and the listed scenarios. Private models and raw runs are excluded from the source release. The Stage 5 timing fixture is published for the specific regression.

The 2026-11-02 baseline had 446 offline Python tests and four Rust unit tests passing, plus nine focused checks after diagnostic changes. Seven death recordings supplied 140 parity scenes. Twenty-eight scenes using a 74,318-step local model had identical deterministic actions and maximum probability difference `5.96e-8`. A later 2026-11-04 optimization snapshot passed 528 Python, six Rust, and 26 frontend tests and completed the production build. These counts refer to those snapshots. [Later component comparisons](performance-tuning.md#paired-component-results)

Paired offline comparison with 128 pairs per scenario. Both paths used the Rust sweep. The comparison isolates the outer search.

| Scenario | Python outer loop P95 | Full Rust search P95 |
| --- | ---: | ---: |
| Recorded planning with 504 visible bullets | 16.25 ms | 7.21 ms |
| Synthetic 640 bullets plus items, enemies, and lasers | 32.12 ms | 7.88 ms |
| Full observation construction | 23.09 ms | 8.46 ms |

Later read-only live profiling captured 226 samples, 205 in combat, with 0–480 ordinary bullets. Combat percentiles follow.

| Stage | P50 | P95 | P99 |
| --- | ---: | ---: | ---: |
| Full capture | 4.45 ms | 12.69 ms | 16.32 ms |
| Observation construction | 5.29 ms | 18.14 ms | 20.71 ms |
| Planning within observation | 3.05 ms | 10.16 ms | 12.79 ms |
| Inference and readback | 5.88 ms | 13.17 ms | 20.04 ms |
| GPU stream span within inference | 2.63 ms | 7.34 ms | 13.57 ms |
| Capture to result ready | 16.93 ms | 39.91 ms | 48.70 ms |

The live profile stopped at policy output. An earlier planning P95 of 33.09 ms came from a different scene. Use the paired component results for matched comparisons. A claim about complete live input latency requires a trace that includes dispatch and observed game response.
