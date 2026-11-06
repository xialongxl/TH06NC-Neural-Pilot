# Data and version contracts

English · [简体中文](../data-contracts.md)

The code defines exact dtypes, parsing, and serialization. Review readers, writers, migration, and tests whenever you change one of these contracts.

## Independent versions

| Contract | Value | Source |
| --- | --- | --- |
| Game identity | Steam build 25306795 and fixed SHA-256 | `native.py` build, hash, and signature constants |
| Observation layout | 9 | `observations.py:OBSERVATION_VERSION` |
| Reward semantics | 9 | `rewards.py:REWARD_VERSION` |
| Policy and bomb head | 2 | `policy_spec.py:POLICY_VERSION` |
| Planning | 8 | `planner_spec.py:PLANNER_VERSION` |
| Action timing | 3 | `observations.py:ACTION_TIMING_VERSION` |
| Input timing diagnostics | 3 | `input_timing.py:InputTiming.begin` |
| Native collision, input, and capture ABI | 3 | `native/src/lib.rs:API_VERSION` |
| Standalone search ABI | 1 | `native/src/lib.rs:SEARCH_API_VERSION` |
| Display protocol | 2, with legacy protocol 1 | `console_stream.py` and `frontend/src/stream.ts` |

## Captured state

[`Frame`](../../th06ai/native.py) is an owned read-only snapshot. `ControlSample` contains only player, tick, phase, stage, input bits, and timing needed for the lightweight input path. A complete policy observation still needs a full frame.

| Group | Contract |
| --- | --- |
| Player | `tick,x,y,state,focused`. `state==2` is Miss evidence. Player focus is the slow-movement flag. Window focus is separate. |
| Scene | `paused,stage_active,phase,phase_evidence,menu_selection,menu_ready` |
| Bounds | `(low_x,low_y,width,height)`. Movement maxima add the corresponding low bound. The displayed field dimensions use width/height plus twice the low bounds. |
| Time | `observed_at` is monotonic seconds and capture/input timestamps are perf-counter nanoseconds. Cross-tick reads carry diagnostic evidence. |
| Stage and setup | `stage_id` is 1–7 or None, with 7 for EX. Raw stage is retained. Difficulty, shot type, mode, and unfocused/focused speeds can be unknown. |
| Bullets | `bullets (N,3)` uses slot,x,y with width and height in `(N,2)` and an optional state vector `(N,)` |
| Items | `(N,6)` slot,x,y,type,vx,vy from the 1024-slot pool |
| Targets | `(N,8)` slot,x,y,HP,maxHP,phaseHP reference,combat flags,Boss |
| Contact bodies | `(N,6)` slot,x,y,halfWidth,halfHeight,geometryKnown |
| Lasers | `(N,8)` slot,originX,originY,startDistance,endDistance,angle,collisionHalfWidth,state |
| Inputs and shots | `input_bits` and `player_shots` can be None. `input_observed_ns=0` marks an unavailable time. |

An empty pool keeps its second dimension. Laser state 0 is telegraph, 1 is damaging, and 2 is fade-out. Capture retains visible telegraph and damaging lasers. Planning can account for telegraphs, while current damage and emergency-bomb evidence require the appropriate phase.

`GameResources(score, power, bombs, lives, point_items, graze=None)` is optional. Score is unsigned 64-bit, Power 0–128, bombs 0–8, lives −1–8, point items unsigned 16-bit, and graze either unknown or 0–999999. Point counter wrap is handled. Item IDs 0–6 represent Power, point, large Power, bomb, full Power, life, and a star/score item in this build.

## Observation v9

Each array is float32 without a leading batch dimension. Single-environment PPO and the policy runner add the batch dimension.

| Key | Shape | Range | Contents |
| --- | --- | --- | --- |
| `bullets` | `(64,5)` | −2 to 2 | Near-field dx,dy,vx,vy,valid |
| `player` | `(4,)` | 0 to 1 | Normalized position, focus flag, bullet fraction |
| `items` | `(32,12)` | −2 to 2 | Position and velocity pairs, seven type bits, valid |
| `resources` | `(5,)` | 0 to 1 | Power/128, bombs/8, lives/8, log score, point items/1000, clipped |
| `enemies` | `(32,8)` | −2 to 2 | Position and velocity pairs, HP ratio, Boss, vulnerable, valid |
| `control` | `(28,)` | −2 to 2 | Previous request, measured motion, and feedback history |
| `full_bullets` | `(640,10)` | −2 to 2 | Full visible bullet pool with geometry, motion, and closest approach |
| `planning` | `(18,8)` | −1 to 1 | Per-move planner evaluation |
| `stage` | `(8,)` | 0 to 1 | Stage 1–6 and EX one-hot plus known |
| `navigation` | `(18,9)` | −1 to 1 | Robust safety, fallback, and exit information |
| `decision` | `(8,)` | 0 to 1 | Decision context and objective availability |
| `game_context` | `(19,)` | 0 to 1 | Difficulty 5+known, shot type 4+known, mode 4+known, two speeds/20 and known |

Padding sets `valid=0`. Unknown motion can have a numeric zero paired with a false known bit. Preserve the last distinct tick across a repeated build.

`full_bullets` uses `dx,dy,vx,vy,motion_known,collision_radius,geometry_known,closest_time,predicted_clearance,valid`. Divide positions by field dimensions, velocities by 16, radius by 128, time by 24 ticks, and clearance by 512. Known merged radius is `min(width,height)*0.5 + player_radius`. Closest approach assumes historical velocity and a stationary player. The near and full encoders are parallel paths. Lasers enter through collision and planner features.

| `control` columns | Meaning |
| --- | --- |
| 0:18, 18, 19 | Previous move one-hot, bomb request, known request |
| 20:22, 22, 23 | Measured velocity/16, clipped tick interval/60, motion validity |
| 24:26, 26, 27 | Axis blockage history/180, camping indicator, resource diminishing-return state |

`planning` columns are `enabled,allowed,clearance,collision_time,blocked,goal,utility,logit_bias`. Raw unbounded clearance and collision time are encoded as finite values for the network. `navigation` uses `robust_safe,fallback,prefix_risk,action_clearance,action_collision,wall_space,exit_fraction,exit_measured,nominal_collision`. `decision` uses `known,fallback,late_control_risk,rescue_opportunity,bomb_allowed,time_to_contact,observation_age,safe_exit_fraction`.

## Actions and timing

The joint action space has 36 entries. Movement is `action % 18` and direction is `action % 9`. Directions are still, up, down, left, right, up-left, up-right, down-left, down-right.

| IDs | Movement | Bomb request |
| --- | --- | --- |
| 0–8 | Unfocused | No |
| 9–17 | Focused with Shift | No |
| 18–26 | Unfocused | Yes |
| 27–35 | Focused with Shift | Yes |

Normal actions hold Z. Menu confirm, pause, restart, and guard hotkeys are outside the policy space. Movement input matching uses up `0x10`, down `0x20`, left `0x40`, right `0x80`, focus `0x04`, and mask `0xF4`.

The game tick is 60 Hz. Action timing v3 holds for at least two native ticks after sending. Keep capture, planning, policy, SendInput, and first observed matching input as distinct monotonic times. The first matching sample gives an observation bound on input response. A held action may have zero new events. Input diagnostics use up to 64 valid recent samples. Version 3 adds capture and pre-policy work breakdowns tied to the same observation.

## Models, migration, and run evidence

`models/latest.zip` contains the SB3 model, optimizer, and counters. The JSON sidecar records game and model hashes, steps, `gradient_updates`, format and domain versions, provenance, migration, and training goal. `gradient_updates` mirrors SB3's `_n_updates`. Immutable checkpoint history is stored in `models/checkpoints/`. Verify model paths, identities, and hashes before load or replacement.

Observation v1–v8 migration keeps compatible weights, optimizer state, and original files. The v8 fusion grows from 743 to 762 columns. The 19 added columns and corresponding Adam history start at zero. The initial prior behavior and v2 bomb head are retained. Request setup is logged separately from observed settings. The Practice result ends the current sample segment, cuts GAE without bootstrap, and creates no synthetic combat action.

| File | Evidence |
| --- | --- |
| `runs/current.json` | Latest public run pointer. Profiling is separate. |
| `status.json`, `result.json`, `worker.json`, `worker.log`, `gpu.jsonl` | Guard state, result, worker heartbeat and errors, actual telemetry |
| `actions.csv`, `episodes.csv`, `sampling_events.jsonl` | Executed actions and timing, round results, refreshes and boundaries |
| `observations-*.npz` | Full or summary rollout with explicit trajectory level. Absent when disabled. |
| `death-*.npz` | Recent native snapshots and lightweight control evidence |
| `evaluation_progress.json`, `evaluation.json` | Per-round progress, then final report once the requested count completes |
| `training_summary.json` | Normal training summary |

In `worker.json`, `clock_started` signals readiness and `clock_paused` indicates a manual wait. `scene_wait_step` identifies the accepted sample count at the start of an automatic wait. It is `null` after the wait ends. These fields report state.

The watchdog publishes `training_seconds`, `paused_seconds`, and `budget_seconds` in the `session_time` field of `status.json` and `result.json`. The first two values are `null` until session timing starts. `paused_seconds` includes manual waits and credited automatic waits. `guarded_scene_wait_seconds` reports the automatic portion separately. For training sessions, the watchdog projects the three clock fields into `worker` for the existing console. `training_summary.json` captures the most recent snapshot read by the worker at save time. `result.json` records the watchdog's final observation.

The last CSV row can precede the actual last action after an emergency stop. A full PPO NPZ stores `obs_<key>` as steps × env × observation shape and retains the sampled action, value, and log probability. `metadata.gae_ready=false` identifies a rollout whose final advantages and returns were unavailable. Summary traces omit the observation arrays.

### Fixed-checkpoint evaluation report

`report_version=1` retains the checkpoint SHA-256, requested setup, and domain versions. Aggregates cover completed rounds. Each round records its observed starting setup, whether it matched the request, the furthest stage observed, misses, and bombs consumed. `stage_progress` separates the first stage observation, the first combat sample, and the first matching input observation using native ticks. The current round's milestones appear in `worker.json` as `evaluation_round`.

`confirmed_stage_transitions` requires two consecutive native stage identities in the same Story round with the same known difficulty and shot type. Only these observed transitions contribute to `rounds_clearing_stage`. The report keeps Practice menu returns, native results, and time limits as distinct endings. A clear of the final stage requires separate verification.

Boss Power comes from the first frame with a visible Boss in each observation window. A new window requires at least 120 native ticks of continuous Boss-free combat observations. Appearance numbers support comparisons under the same setup. The identity of a midboss or final boss still requires separate evidence. `boss_power_summary` gives the sample count and median Power for each stage and window. Missing readings remain `null`.

Death NPZ `frames` rows include tick, stage, x/y, state, paused, focused, radius, input bits, Power, bombs, and lives. Variable-length pools use offsets into concatenated arrays. Bullet rows use slot,x,y,width,height,state. Contact bodies and items have six columns and lasers eight. The snapshot excludes the full enemy HP target list. A nearby object is evidence for analysis. A native damage-source ID remains unknown where the game did not supply one.

## Native binary bridge

| API | Input | Output |
| --- | --- | --- |
| `sweep_bytes` | Little-endian float32 starts and velocities 2N each, durations N, ticks N, bounds 4, circles 6M, bodies 8K | `(N,5)` float32 end x/y, clearance, collision tick, blocked fraction |
| `plan_bytes` | Owned little-endian float64 snapshot of `88+7M+9K+5Q` entries, Q ≤ 32, plus search timing inputs | Float32 feature, path, clearance, collision time, navigation, and decision bytes plus diagnostic metadata |
| `NativeInput.set(mask)` | Supported key mask under the instance mutex | Held mask and event count |

Planner path length is `len(durations)+2`. Use the encoder in `acceleration.rust_plan` and definitions in `planner.rs` and `plan_bytes` for exact ordering. Validate shapes, numeric values, and budget before detaching the GIL, and pass owned memory.

## Web display protocol

Bootstrap returns a token and protocol. Full diagnostics use `/api/state`. The frontend explicitly subscribes to v2, while the endpoint defaults to legacy v1. [Control API](development.md#control-api)

```json
{"protocol":2,"kind":"snapshot","revision":1,"state":{"active":false}}
```

```json
{"protocol":2,"kind":"delta","base":1,"revision":2,"set":[[["active"],true]],"remove":[]}
```

```json
{"protocol":2,"kind":"heartbeat","revision":2}
```

Paths for set/remove are arrays. The client rejects prototype mutation paths. Temperature history supports append with an `after` cursor or reset, capped at 180 points. A missing base or cursor forces resynchronization. The DTO excludes high-frequency internal clocks and full bullet pools. Keep persisted raw evidence independently of the display protocol.
