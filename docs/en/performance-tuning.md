# Control latency, diagnostics, and checkpoint tuning

English · [简体中文](../performance-tuning.md)

The control path keeps optional diagnostics off the input-critical thread. The comparisons below are component measurements with the stated inputs. Live capture, input, and training verification are separate.

## Current implementation

| Area | Design |
| --- | --- |
| Read-only capture | Auto reuses large-block buffers and batches small control, phase, and HUD fields through Rust where available. Explicit Rust selects owned large-block capture. Capture retains full pools, enemies, lasers, and cross-tick evidence. |
| Bullet and item decoding | Identify active slots first, then gather their fields. Preserve pool state validation, slot order, visibility bounds, and unknown geometry. |
| Bullet features | Reward and observation share `BulletStateCache` keyed by snapshots and geometry. Owned native arrays are read-only. Mutable external fixtures require content checks. |
| Diagnostics | Bounded tasks offload action row encoding, list conversion, death archives, and compression to a CPU-only helper. |
| Inference | Twelve arrays are packed into continuous pinned host/device storage for one CUDA H2D input copy. Small outputs also reuse buffers. |
| Model library | An incremental background catalog caches unchanged metadata and publishes under a short lock. Selection and loading revalidate paths and hashes. Pages contain up to 25 entries, with filtering across the catalog. |
| Input and reward | Both input tables include Q. Item reachability uses the verified active shot's unfocused speed. Reward v9 retains existing compatible weights and optimizer state. |

The guard still owns telemetry, F7/F8, the lease, and the Job. Normal empty pools, cross-tick capture, and brief focus loss retain their respective handling.

## Diagnostics helper

`SessionDiagnostics` retains recent raw frames. `DiagnosticWriter` sends owned internal tasks through a private pipe to a CPU helper. Pickle is restricted to this private protocol. The helper owns no game handle, CUDA context, or model update.

The control thread enqueues work without blocking on I/O. Dedicated threads transmit tasks and collect results. At most two complete trace tasks can be pending. Backpressure increments `dropped_records`. A missed death record also produces a critical diagnostic error. Normal finish drains within a deadline. Emergency stop terminates the helper with the owned Job.

Check `diagnostic_worker.log` for startup or unhandled errors and `diagnostics_summary.json` for counts, losses, and write failures. An incomplete asynchronous write carries no success status.

## Configuration

Merge the optional fragment into local `config.json`. Changes apply to the next session.

```json
{
  "diagnostics": {
    "trajectory": "full",
    "trajectory_every_rollouts": 1,
    "compression_level": 1,
    "checkpoint_every_rollouts": 1,
    "thread_profile_steps": 0,
    "readback_ab": false
  }
}
```

| Field | Default and range | Trade-off |
| --- | --- | --- |
| `trajectory` | `full`, with `summary` and `off` available | Full contains actual PPO inputs. Summary retains scalars and action/target arrays. Off skips rollout NPZ. All retain death snapshots. |
| `trajectory_every_rollouts` | 1, integers 1–100 | Keep every Nth rollout. Skipped intervals lack complete rollout replay. |
| `compression_level` | 1, integers 0–9 | 0 stores without compression. Higher levels can trade CPU time for disk space. |
| `checkpoint_every_rollouts` | 1, integers 1–100 | Space out regular saves. Normal finish still saves the latest completed update. A longer interval exposes more updates to emergency loss. |
| `thread_profile_steps` | 0, integers 0–4096 | A positive value captures control-thread counters for up to that many completed actions in the next session. |
| `readback_ab` | Boolean, default false | The next CUDA Graph training session compares up to 512 actions in its first rollout and collects control-thread counters for that window. |

Trace metadata identifies `trajectory_level`. When the weights are newer than disk, `policy_checkpoint` is null, `last_saved_checkpoint` identifies the latest disk copy, and `policy_gradient_epochs` gives the actual update count. External offline fixtures can use the synchronous path.

Top-level `capture_backend` is `auto`, `python`, or `rust`. Collision and keyboard backends are selected independently.

## Timing fields

- `input_read_breakdown` separates capture, decode, phase, and input fields. Native fields distinguish `native_compute_ms` and `native_handoff_ms`, which can include GIL handoff and scheduling.
- `input_pre_policy_work` separates environment diagnostics, rollout buffer insertion, and callbacks. It refers to the observation used by that decision.
- Worker rollout fields separate sampling, pause, update and callbacks, checkpoint save, resume and observation, and trace preparation.
- Helper fields such as `last_trace_ms` and `last_action_info_ms` report writing work on the helper side.

Keep same-action raw timestamps. SendInput return, first observed matching game input, first displacement, and rendered response are different endpoints. Nested intervals and independent percentiles cannot be added to estimate total latency.

### Synchronized control-thread counters

Set `diagnostics.thread_profile_steps` to `512` in the local config, then start a session as usual. A matched sample covers capture, preparation, planning, post-planning work, inference, pre-dispatch checks, and dispatch. Each row belongs to the observation and plan used by the completed action. The `input_thread_timing` column is flushed with the normal rollout action log.

```powershell
.\.venv\Scripts\python.exe -m th06ai.thread_profile .\runs\<session-directory>\actions.csv
```

The report compares elapsed time and thread cycles by phase and uses the same slowest 5% of actions across phases. It compares cumulative CPU time over same-thread windows of at least one second, including gaps between actions. Windows CPU time may be coarsely quantized. Cycles retain their native units. Off-CPU intervals can include I/O, GPU waits, sleeping, and scheduling delays. Separating these causes requires a scheduling trace. [Windows thread CPU time](https://learn.microsoft.com/en-us/windows/win32/api/processthreadsapi/nf-processthreadsapi-getthreadtimes) · [Windows thread cycles](https://learn.microsoft.com/en-us/windows/win32/api/realtimeapiset/nf-realtimeapiset-querythreadcycletime)

### Compare CUDA result readbacks

Run this command when training is idle. The independent watchdog supervises inference on a saved checkpoint and complete recorded observations. The new run directory contains `readback_profile.json`.

```powershell
.\.venv\Scripts\python.exe -m th06ai profile --profile-kind readback --minutes 1 --model models/latest.zip
```

Both methods use the same CUDA graph. Sampled pairs restore the same CUDA generator state, alternate execution order, and compare the action, value, action log probability, and bomb probability. Device events cover input transfer and the forward pass. Host readback time and thread cycles are reported separately. The report also verifies the configured readback method and its outputs on deterministic observations.

Regular sessions default to `inference_readback: "auto"`. With CUDA Graph, the worker copies results asynchronously into pinned host memory and waits on a blocking event in the same stream before reading them. CUDA eager and CPU use synchronous readback. Set `inference_readback` to `"synchronous"` in local `config.json` to restore synchronous readback on the next session.

Two offline runs on 2026-11-07 each used 64 recorded observations and 512 sampled pairs. Async readback with a blocking event cut mean readback thread cycles by 54% to 58% and mean inference thread cycles by 14% to 17%. Inference P95 increased by about 0.03 ms in each run.

### Compare readback during training

Set `diagnostics.readback_ab` to `true` in local `config.json`, then resume training as usual. The first rollout pairs up to 512 accepted actions and randomizes the order of synchronous and blocking-event readback within each pair. Both methods use the current policy and each action is inferred and executed once. The configured readback method resumes after the sample window. A worker without CUDA Graph reports `readback_ab: unavailable`.

```powershell
.\.venv\Scripts\python.exe -m th06ai.thread_profile '.\runs\<session-directory>\actions.csv' --readback-ab
```

The report matches adjacent valid actions in the same stage and combat phase. It includes excluded pairs and the available sample count for each metric. The first rollout holds model parameters fixed while the game continues to advance. Restore `diagnostics.readback_ab` to `false` after the comparison to use the default on later sessions.

On 2026-11-07, one rollout yielded 256 live pairs in Lunatic Stage 1 Practice with Reimu A and NMNB. Blocking-event readback cut mean inference thread cycles by 84.2% and capture-to-dispatch thread cycles by 45.1%. Capture-to-dispatch P95 was 17.94 ms with synchronous readback and 17.93 ms with a blocking event. The inference cycle reduction was about 84% in both execution orders.

## Paired component results

These historical comparisons alternated order and checked identical outputs. The two memory reads used the test process itself.

| Component | Pairs | Previous P95 | Revised or alternate P95 | Result |
| --- | ---: | ---: | ---: | --- |
| Full vs incremental catalog scan across 295 models | 12 | 691.84 ms | 25.81 ms | Same metadata and order |
| Reward and observation with 640 synthetic bullets | 96 | 6.17 ms | 4.99 ms | Same reward and observation arrays |
| 2.51 MiB reused buffer vs owned Rust reads | 96 | 1.32 ms | 5.08 ms | Auto keeps the reused large buffer |
| Four small control regions, 449 bytes | 256 | 0.0116 ms | 0.0051 ms | Same bytes, batched in Auto |

The revised CUDA packing had CPU numerical parity checks. A protected live check is still required to measure revised CUDA Graph and complete game-input latency.

On 2026-11-07, 171 snapshots from the latest recorded run were replayed in 3,420 alternating pairs. The Rust planner uses fixed-size candidate paths and reuses rectangle rotations within each collision sweep. The planner outputs matched byte for byte. Mean thread cycles per `plan_bytes` call fell from 1.45 million to 1.02 million, and P95 elapsed time fell from 1.77 ms to 1.24 ms. Live core utilization and capture-to-input latency require separate measurements. [Windows thread cycle counter](https://learn.microsoft.com/en-us/windows/win32/api/realtimeapiset/nf-realtimeapiset-querythreadcycletime)

Capture decoding filters enemy slots before processing their fields and returns early when the laser pool contains no warning or damaging beams. The near and full-field bullet observations share normalization, with in-place clipping of full-field rows after the near rows are copied. Across 768 alternating pairs on a simulated stage snapshot, median `GameReader.read()` time fell from 0.92 ms to 0.80 ms. Decoded fields and observations matched byte for byte on 64 simulated snapshots, and observations matched on 140 recorded frames. The simulated capture includes buffer copying and decoding. Live capture-to-input latency requires a separate measurement.

## Apply changes and verify

Planner v9 supports existing checkpoints. The current versions are observation v9, policy v2, action timing v3, reward v9, diagnostics v3, native ABI v3, and UI SSE v2. Build Rust using the native instructions and end the session before replacing a loaded DLL. Existing workers keep their loaded source until a new session starts. [Native build](../../native/README.md)

Use `test_optimization_contracts.py`, `test_state_capture.py`, `test_observation_traces.py`, `test_latency_profiler.py`, `test_native_search.py`, `test_rust_backend.py`, `test_model_store.py`, and the console concurrency checks. [Test and benchmark methodology](testing-and-performance.md)
