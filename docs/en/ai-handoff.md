# Maintainer handoff

English · [简体中文](../ai-handoff.md)

TH06NC-Neural-Pilot is an experimental real-time reinforcement learning system maintained by **xialongxl** under MIT. Its boundaries reflect concrete failures observed during development. Start by identifying the invariant a piece of code protects and the test or observation that exercises it.

## Design intent

The PPO policy learns a distribution over joint movement and bomb actions. Rolling planning turns observed geometry and input timing into a route evaluation that the policy can use and PPO can replay. A session state machine handles non-combat inputs. A separate guard can stop the owned process tree even if the worker or driver stalls. To change behavior, specify an observable state, reward, action distribution, boundary, or evaluation that would demonstrate the change.

## First working session

1. Read the [repository instructions](maintainer-instructions.md), [architecture](architecture.md), and [pitfalls](gotchas.md).
2. Use the [component map](../ai-context.json) for dependencies and regression groups, and the [generated index](../code-index.json) for real Python symbols.
3. Inspect the current source, version constants, effective configuration, and session state before interpreting an archived result.
4. Separate offline inspection, game attachment, keyboard input, GPU access, and model updates. Collect the smallest evidence needed to reproduce the behavior.

The live system uses React and TypeScript, loopback REST, versioned JSON SSE, a session service, an independent NVML/F8/Job/lease supervisor, and a worker that captures state, plans, and runs PPO or inference. State and evidence live in local files. `config.json` is the application configuration. `th06ai` remains the Python package and CLI name, including in serialized class paths. The optional native runtime uses ABI v3, with a separate legacy planner path. Stage Practice uses operator-selected settings and cuts GAE at its results boundary. [Game setup](game-settings.md)

## Regressions to keep in view

| Area | What to protect |
| --- | --- |
| Reward and resources | Corner camping can exploit survival shaping. A death and subsequent recollection can exploit raw pickup rewards. Compare actual resource use, mobility, and results. |
| Bomb policy | Joint entropy can reward unnecessary bombs. Keep the conditional bomb prior, objective mask, and compatible learned head. |
| Geometry and timing | Include enemy bodies and oriented lasers. Check earlier and later switch ticks before calling a route robust. |
| Capture and pause | Empty pools and ordinary cross-tick reads are valid. A frozen counter can mean pause, dialogue, script state, or loss of focus. A lightweight input sample can contain the only Miss evidence. |
| Session semantics | Refreshing the browser preserves the session. F7 resumes in memory. Resuming a checkpoint starts a new session. |
| Measurement | Kernel latency, policy latency, observation-to-send time, input matching, and visual response measure different events. |

## When changing a contract

| Change | Review together |
| --- | --- |
| Observation shape or column order | `observations`, policy fusion, `learning` migration, inference buffers, and saved rollout arrays |
| Planner mask or bias | Python and Rust, policy distribution, PPO replay, and death fixtures |
| Reward, objective, or bomb behavior | `objectives`, control history, boundary handling, conditional entropy/prior, and independent evaluation |
| Pause, restart, or stage transition | `environment`, `TrainingControl`, phase decoding, and `PausablePPO` |
| Input or hotkeys | `GlobalHotkeys`, focus, Win32/Rust held state, and independent guard Job |
| Console protocol | Display DTO, SSE base/revision and curve cursors, reference reconciliation, and concurrency tests |
| Model persistence | ZIP and sidecar, SHA-256, immutable history, catalog revisions, and migration state |
| Protection | `SafetyPolicy`, `AsyncTelemetry`, `SharedLease`, supervisor, and heartbeat |

Prefer small modules, typed boundaries, dataclasses, explicit units, and tests for concrete failures. Keep the Python reference path when changing Rust. Compare actions, values, and log probabilities when changing inference. Keep game access, learning, and protection in the backend. Preserve private models, paths, calibrations, and logs in the operator's checkout.

The [test matrix](testing-and-performance.md) explains the scope of existing evidence. Refresh the [code index](../code-index.json) after source changes, then run the documentation checker. Open areas include self-service resource confirmation, additional game versions and attacks, full-clear acceptance criteria, dependency locking, cross-machine verification, and more live tail-latency measurements.
