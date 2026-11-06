# Five core execution paths

English · [简体中文](../core-logic.md)

Use the [symbol index](../code-index.json) to locate definitions and the [data contracts](data-contracts.md) for exact array and protocol formats.

## 1. Capture and observation

[`GameReader`](../../th06ai/native.py) verifies the executable SHA-256, PE layout, and signatures, then confirms there is one matching running process with the configured image path. It checks live signatures and resolves RVAs against the actual module base.

- `read() -> Frame` batches the large regions and decodes visible bullets, items, targets, contact bodies, lasers, resources, native ticks, phase, and input bits. The returned frame owns its read-only arrays.
- `read_control() -> ControlSample` reads the smaller player, timing, input, and scene fields before a dispatch and while holding an action. Keep Miss evidence from these samples even if the next full frame already shows a respawn.
- `close()` releases the reader. Preserve actual capture timestamps and record ordinary cross-tick reads in diagnostics. `game_phase.classify()` distinguishes a pause menu from dialogue, script freeze, continue, results, and transitions.

[`ObservationBuilder.build(frame)`](../../th06ai/observations.py) encodes twelve float32 arrays. It estimates motion using stable pool slots and the last valid distinct tick, preserves known flags, encodes the near 64 and full 640 bullets separately, and adds inventory, targets, control history, stage, and game context. It caches the planner result and exposes bomb availability for the selected objective. An input change must also update fusion, migration, optimizer handling, saved trajectories, and inference buffers.

## 2. Plan and sample one joint action

[`RollingPlanner.plan(frame, previous=None, control=None, state=None)`](../../th06ai/planning.py) can reuse computed bullet state. Its `PlanResult` contains features `(18,8)`, per-move polyline paths `(18,P,2)`, raw clearances and collision times, navigation `(18,9)`, decision context `(8,)`, diagnostics, and start/end timestamps.

It estimates a switch window from capture age, planning cost, and paired measurements from planning completion to observed input matching. It evaluates the old-key prefix, expands all 18 current movements, and searches a bounded set of later turns. Continuous collision checks account for circles, enemy bodies, oriented lasers, and wall contacts. Geometric reachability can rule out distant objects within the same finite prediction. A nearest-bullet count would discard relevant threats.

Among viable routes, scores reflect clearance, wall pressure, mobility, reachable items, and attack alignment. The normal switch window is checked tick by tick. An abnormally wide window samples at most 12 times and carries an incomplete flag. Fallback ranks remaining risk and escape choices using the tested window. Keep the fallback distinct from robust safety. A route found under the model remains subject to new spawns, turns, unknown geometry, and other unobserved changes.

The Python path is a reference. `rust_sweep()` can accelerate only collision, while `plan_bytes()` handles a complete search on an owned snapshot with the GIL released. Inspect `search_backend` and `collision_backend` separately. [Native runtime](../../native/README.md)

Both paths apply planner v8's final fallback ranking. If the contact probe with laser telegraphs has no candidate and the switch window was fully checked, the planner prioritizes moves that clear the short confirmed-damage probe at every switch tick and pass a separate check for enemy bodies and damaging lasers. The full forecast continues to inform risk ranking. Action logs identify this adjustment with `planner_active_threat_priority`.

[`ResourcePolicy`](../../th06ai/policy.py) applies masked mean and max pooling to per-entity encodings and fuses the player, inventory, control, planning, navigation, and context inputs. It combines learned logits with planner masks and biases. It masks bombs in NMNB and with empty stock. Survival's conditional prior affects exploration in strict rescue windows.

[`PolicyRunner.predict()`](../../th06ai/inference.py) returns `actions`, `values`, `log_probs`, `bomb_probability`, and timing. It packs the twelve inputs into reusable buffers, transfers one continuous input on CUDA, and can replay a fixed-shape CUDA Graph using the same parameter storage as training. Auto fallback handles graph capture limitations. Device loss, illegal access, and OOM remain errors. The sampled, executed, and replayed joint distribution must agree.

## 3. Execute, reward, update, and save

[`TouhouEnv.step(action)`](../../th06ai/environment.py) accepts action 0–35 and returns the Gymnasium tuple `(observation, reward, terminated, truncated, info)`. `_prepare_action()` checks authorization, the session budget, pause requests, and the scene. `_capture_action()` reconciles held shooting state, sends the ordered batch, and validates captures. A normal hold lasts at least two ticks from the first post-dispatch control sample. A Miss detected by a lightweight capture remains part of the transition even if the full frame shows a respawn. `_record_transition()` updates rewards and episode state, builds the next observation, and publishes diagnostics.

`SamplingInterrupted` requests a fresh observation for recoverable pauses and interruptions. Near hazards, the pre-send stale-frame check may refresh once. It records a warning if the retry is still over budget. Record committed transitions before a budget callback can terminate the run.

[`RewardTracker.step(frame, action, died)`](../../th06ai/rewards.py) computes observed feedback for combat time, first Miss, wall and camping behavior, inventory, pickups, Power potential, target potential, shot alignment, and credible HP changes. It closes relevant potentials on death to reduce recollection exploits. Item reachability uses the verified unfocused speed, with the historical 4 px/tick fallback when unavailable. Score and grazing remain diagnostic or contextual.

[`PausablePPO.collect_rollouts()`](../../th06ai/pausable_ppo.py) and [`TrainingCallback`](../../th06ai/learning.py) gate controls, refresh if needed, sample and record real transitions, pause at the rollout boundary, queue traces, update PPO, save, and resume from a fresh frame. Live defaults use a single environment, `n_steps=512`, `batch_size=64`, `n_epochs=4`, `learning_rate=3e-4`, `gamma=.999`, and `gae_lambda=.95`. A stage boundary found during resume bootstraps from the valid previous observation, cuts GAE, and then refreshes the observation. A Practice results boundary ends the episode without synthetic combat input.

`TrainingControl` reads active time from the watchdog snapshot cached by the worker heartbeat thread. It reports `overtime_stage` when the target is reached during an identified stage. It finishes at a confirmed Story handoff, native results screen, or end of the attempt. If a Miss ends the round, the sampler records the terminal transition and skips the usual automatic restart. Overtime is capped at 600 active seconds. A partial rollout is retained for diagnostics, and the checkpoint contains the last completed parameter update.

`save_model()` saves an updated model and preserves a verified prior latest as immutable history. The ZIP and its sidecar are separately written, so a hash mismatch must block use or replacement. `load_model()` checks project ownership scope and metadata, preserves source files, and migrates compatible weights and optimizer history.

At normal training completion, the worker releases keys and checks the watchdog authorization before saving. Cleanup then makes a brief attempt to pause a game that is still in the foreground. A read or input failure during cleanup is recorded separately from the checkpoint result as `cleanup_error` in `worker.json`.

## 4. Authorize independently

[`run_guard(config, mode, minutes=None, model=None, ...)`](../../th06ai/supervisor.py) returns `code`, `reason`, `run_dir`, and `child_pid`. Factory injection is for fault tests. A fault is latched until a new protected session.

`SafetyPolicy.check()` checks freshness, identity, finite telemetry, temperature, VRAM, power, throttling, fans, and available error counters. An unavailable optional sensor and a supported sensor failing mid-session have different meanings. `SharedLease.publish()` is owner-only. The last complete record can cover a busy read only with its original timestamp. Worker heartbeat and telemetry expiry use separate thresholds. F8 and hardware faults can stop the Job without waiting for PPO or model I/O.

After successful work and cleanup, the worker writes `exit_ready` in its final heartbeat. The watchdog freezes session timing when it receives that fresh record and gives the process up to five seconds to exit. It records completion when the process exits with code 0. GPU checks and F8 continue during this interval.

`GuardedSessionClock` owns live session time. Startup has its own deadline. The watchdog starts session accounting when it observes `clock_started`. It excludes manual waits reported through `clock_paused` and up to 60 seconds of automatic waiting per accepted sample reported through `scene_wait_step`. It publishes the snapshot in `status.json`. The console displays it, and the worker reads it on the heartbeat thread for deadline checks on the control path.

The training target, stage overtime, and hard execution deadline use the same active duration. At the target, a valid `overtime_stage` grants one fixed allowance. The hard deadline includes 120 seconds for cooperative shutdown and saving. GPU protection and F8 remain effective throughout.

## 5. Manage sessions and display state

[`create_app()`](../../th06ai/web_console.py) serves loopback routes. [`ConsoleService`](../../th06ai/console_service.py) resolves model IDs and verifies checkpoints outside the session lock. The final launch rechecks the session, shutdown state, and checkpoint files before creating and registering the guarded process under the lock. PAUSE/FOCUS/FINISH/STOP marker writes use the same lock. A `FINISH` received before worker creation cancels startup. A separate game lock serializes window activation and read-only probes with launch. Writes use same-origin validation and the current bootstrap token.

[`SnapshotBroker`](../../th06ai/console_stream.py) shares low-rate reads. A snapshot copies session and configuration references under the session lock, then releases it before reading status and telemetry. Log reads and cache misses do their I/O outside the session lock. The catalog prepares rows in the background and publishes them under a short lock. [`applyFrame()`](../../frontend/src/stream.ts) applies a snapshot or revisioned delta, reconciles curve cursors, and preserves unchanged references. A gap triggers a fresh snapshot and slow clients converge on the latest revision.

```json
{
  "capture": {"entry": "GameReader.read", "consumer": "ObservationBuilder.build", "type": "Frame"},
  "decision": {"entry": "RollingPlanner.plan + ResourcePolicy", "runner": "PolicyRunner.predict", "type": "Decision"},
  "learning": {"entry": "PausablePPO.collect_rollouts", "environment": "TouhouEnv", "reward": "RewardTracker", "storage": "save_model"},
  "protection": {"entry": "run_guard", "policy": "SafetyPolicy.check", "authorization": "SharedLease", "owner": "OwnedJob"},
  "console": {"entry": "create_app", "service": "ConsoleService", "display": "SnapshotBroker -> applyFrame"}
}
```

The [component map](../ai-context.json) adds dependencies and regression groups.
