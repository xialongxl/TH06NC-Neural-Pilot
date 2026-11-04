# Frequently asked questions

English · [简体中文](../faq.md)

## How does the agent choose actions?

A trainable PPO network and a rolling planner jointly choose movement and bomb actions from observed state. Training updates the network from real transitions. Play and evaluation use fixed weights. A session state machine handles menus, dialogue, pause, and restart outside combat sampling. [Core execution paths](core-logic.md)

## Where can I get a trained model?

The source repository ships no checkpoint archives. Start with **New model** or select a compatible checkpoint from your own model library. Use a short initial session to check input, shooting, resource reads, pause, and logs. Measure fixed-weight runs before drawing conclusions about learning, clears, or NMNB. [Testing and performance](testing-and-performance.md)

## Why is training unavailable after installation?

First-time setup requires a GPU check, live movement calibration, and a manual comparison of Power, bombs, and lives with the HUD. The three checks create `training_backend.json`, `native_calibration.json`, and `native_resources_calibration.json` respectively. Resource confirmation is still a manual step tied to the current executable. Use the preview to inspect the UI without live sessions. [First-run calibration](development.md#first-run-calibration)

## Why does the CPU stay busy with CUDA enabled?

CUDA runs the policy's training and inference work. Game capture, session logic, input, planning, recording, and system scheduling still use the CPU. The Rust extension optionally accelerates specific components. Check the actual backend in diagnostics. Live sessions still require the NVIDIA watchdog even with `device=cpu`. [Native build and scope](../../native/README.md)

## What happens when I pause, close the browser, or resume?

F7 pauses or resumes the same process and keeps the current round and partial rollout in memory. Saving and resuming from a checkpoint starts a new session with saved weights and compatible optimizer state. F8 stops immediately. The latest successful checkpoint is the recovery point. F7 and F8 are standalone keys.

Closing or refreshing the browser leaves the session running. Use **Save and Finish** for a normal training finish. To close the backend as well, use **Shut Down** in the console or run `.\LaunchAI.exe shutdown`. Use F8 when immediate stop is needed.

## What do survival and NMNB measure?

`survival` permits learned emergency bomb use. `nmnb` masks bombs. Survival with bomb use and NMNB attempts are separate evaluation conditions. Report Misses, actual bomb consumption, stage, and completed rounds. The planner and policy can still make mistakes in survival mode. [Data contracts](data-contracts.md)

## Does the agent get rewarded for resources, attacks, and grazing?

Pickups, Power, and credible damage feedback affect training or planning. Reward shaping considers diminishing returns and death/recollection cycles. Grazing and total score are recorded for analysis and do not directly add to reward. Verify actual pickups and attack results before concluding the policy learned them. [Reward and sampling pitfalls](gotchas.md#reward-and-sampling)

## Why do I see a 180-second setting with a 60-minute session?

The session budget and per-round combat cap use separate clocks. The shipped sample uses `episode_seconds=0` to disable the cap. An existing local config can still contain 180. User pause and controlled waits have their own accounting. PPO updates while the game is paused are still training work. [Configuration](development.md#configuration)

## Does the console affect training performance?

The browser uses REST and JSON SSE. Display updates are capped at 2 Hz for visible subscribers and 0.2 Hz when all are hidden. The backend shares state sampling, and the GPU guard independently polls at 250 ms. Control does not pass through the browser. Logs and rendering still use some system resources. [Web console protocol](../../frontend/README.md)

## What should I do before updating the game or project?

Save and finish before loading new project code. Keep original checkpoints when migrating formats. A game update can invalidate the executable hash, layout, and code signatures. Use an adapter verified for that build before running live sessions. [Version contracts](data-contracts.md)

## What information belongs in a bug report?

Start with the exact status reason, project commit, game scene, and minimal reproducer. For worker errors, include a sanitized stack trace and relevant status fields. For behavior, add stage, objective, and action or death IDs. For latency, identify endpoints, workload, sample count, and percentiles. [Contribution guide](../../CONTRIBUTING.md#report-a-bug)

## Can I run it as a single executable or server?

The supported setup is a Windows source checkout with a locally built UI served by Python over loopback. `LaunchAI.exe` provides a single-file entry point to that installation. A bundled wheel, Docker, Linux live play, and public multi-user hosting are outside the verified deployment scope. [Development and deployment](development.md)

## How do MIT and the game's rights interact?

The project uses standard MIT for its own code and documentation and encourages non-commercial research. The game, assets, trademarks, and third-party code have their own terms. [Disclaimer](../../DISCLAIMER.md) · [Third-party notices](../../THIRD_PARTY_NOTICES.md)
