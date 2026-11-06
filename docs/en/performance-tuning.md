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
    "checkpoint_every_rollouts": 1
  }
}
```

| Field | Default and range | Trade-off |
| --- | --- | --- |
| `trajectory` | `full`, with `summary` and `off` available | Full contains actual PPO inputs. Summary retains scalars and action/target arrays. Off skips rollout NPZ. All retain death snapshots. |
| `trajectory_every_rollouts` | 1, integers 1–100 | Keep every Nth rollout. Skipped intervals lack complete rollout replay. |
| `compression_level` | 1, integers 0–9 | 0 stores without compression. Higher levels can trade CPU time for disk space. |
| `checkpoint_every_rollouts` | 1, integers 1–100 | Space out regular saves. Normal finish still saves the latest completed update. A longer interval exposes more updates to emergency loss. |

Trace metadata identifies `trajectory_level`. When the weights are newer than disk, `policy_checkpoint` is null, `last_saved_checkpoint` identifies the latest disk copy, and `policy_gradient_epochs` gives the actual update count. External offline fixtures can use the synchronous path.

Top-level `capture_backend` is `auto`, `python`, or `rust`. Collision and keyboard backends are selected independently.

## Timing fields

- `input_read_breakdown` separates capture, decode, phase, and input fields. Native fields distinguish `native_compute_ms` and `native_handoff_ms`, which can include GIL handoff and scheduling.
- `input_pre_policy_work` separates environment diagnostics, rollout buffer insertion, and callbacks. It refers to the observation used by that decision.
- Worker rollout fields separate sampling, pause, update and callbacks, checkpoint save, resume and observation, and trace preparation.
- Helper fields such as `last_trace_ms` and `last_action_info_ms` report writing work on the helper side.

Keep same-action raw timestamps. SendInput return, first observed matching game input, first displacement, and rendered response are different endpoints. Nested intervals and independent percentiles cannot be added to estimate total latency.

## Paired component results

These historical comparisons alternated order and checked identical outputs. The two memory reads used the test process itself.

| Component | Pairs | Previous P95 | Revised or alternate P95 | Result |
| --- | ---: | ---: | ---: | --- |
| Full vs incremental catalog scan across 295 models | 12 | 691.84 ms | 25.81 ms | Same metadata and order |
| Reward and observation with 640 synthetic bullets | 96 | 6.17 ms | 4.99 ms | Same reward and observation arrays |
| 2.51 MiB reused buffer vs owned Rust reads | 96 | 1.32 ms | 5.08 ms | Auto keeps the reused large buffer |
| Four small control regions, 449 bytes | 256 | 0.0116 ms | 0.0051 ms | Same bytes, batched in Auto |

The revised CUDA packing had CPU numerical parity checks. A protected live check is still required to measure revised CUDA Graph and complete game-input latency.

## Apply changes and verify

Planner v8 supports existing checkpoints. The current versions are observation v9, policy v2, action timing v3, reward v9, diagnostics v3, native ABI v3, and UI SSE v2. Build Rust using the native instructions and end the session before replacing a loaded DLL. Existing workers keep their loaded source until a new session starts. [Native build](../../native/README.md)

Use `test_optimization_contracts.py`, `test_state_capture.py`, `test_observation_traces.py`, `test_latency_profiler.py`, `test_native_search.py`, `test_rust_backend.py`, `test_model_store.py`, and the console concurrency checks. [Test and benchmark methodology](testing-and-performance.md)
