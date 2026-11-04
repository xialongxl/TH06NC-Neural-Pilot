# Repository instructions

English · [简体中文](../../AGENTS.md)

This is the English companion to the repository instructions in [AGENTS.md](../../AGENTS.md). Explicit operator instructions take priority. Reading these instructions provides no authorization to start training, calibrate, send keys, or take control of the game.

## Before editing

1. Read the [README](../../README.md), [handoff](ai-handoff.md), and [pitfalls](gotchas.md).
2. For the affected area, consult the [code map](code-map.md), [execution paths](core-logic.md), and [data contracts](data-contracts.md).
3. Locate components and tests with the [component map](../ai-context.json) and [symbol index](../code-index.json). Resolve discrepancies against the current implementation.
4. Inspect the actual configuration and relevant tests. Treat historical results as evidence from the stated date, machine, and scenario.

## Invariants

- Keep React as the primary console with a separate loopback API. Keep Tk as a compatibility entry point.
- Capture game state read-only and act through ordinary keyboard input. Preserve game files and memory, and use only currently observed attacks for prediction.
- Require explicit permission for live training, calibration, takeover, or manual key injection. In read-only checks, disclose whether game attachment or GPU access is involved.
- Route CUDA through the independent guard. Temperature monitoring, F8, shared authorization, and termination of the owned Job belong to the guard.
- Preserve safety bounds and accept ordinary empty bullet pools, cross-tick capture, and brief focus loss through their existing paths.
- Preserve local caches, checkpoints, logs, and snapshots unless the task explicitly calls for cleanup. Migrations keep original files and learned parameters, including compatible optimizer history.
- PPO must sample, execute, and train on the same joint action and log probability. A fallback action and its risk must remain distinguishable from a path verified over the timing window.
- F7 and F8 are standalone user keys. Shift is the game's focus movement key.

## Implementation and verification

Measure the real control path before changing it. Keep low-rate UI work and log encoding off the hot path, and represent unknown values explicitly. Check independent version boundaries for observations, models, rewards, planning, action timing, the native ABI, and the UI protocol before changing a contract. Rust must own snapshots before releasing the GIL and serialize held-key state with input transactions.

Select verification from the [test matrix](testing-and-performance.md). Offline tests, component timing, read-only game checks, and real takeover provide different evidence. Run `python tools/check_docs.py` for documentation. After source changes, run it with `--update-index` and review the index diff. Keep documentation checks offline and use durable product language for visible controls.

Report what changed, what was verified, whether the game, GPU, or model was accessed, and what still needs evidence. Label offline timing and live performance separately.
