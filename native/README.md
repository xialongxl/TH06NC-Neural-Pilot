# Optional native runtime

English · [简体中文](README.zh-CN.md)

The Web console's **Safety & Diagnostics** page lets you select Python, Rust, or Auto for collision and input. Auto selects a supported extension when available. The next session uses the change and keeps the same observations and learned parameters.

## Build

The normal CPython 3.10 x64 Windows build produces `th06ai/native_runtime/_native_backend.cp310-win_amd64.pyd`. A compatible built extension runs without a Rust installation. Changing the interpreter ABI or platform requires a rebuild. Use the Rust and MSVC Build Tools on PATH, or supply your own toolchain directory with `--toolchain` and the path to `vcvars64.bat` with `--vcvars`. Cargo dependencies are pinned by `Cargo.lock`.

```powershell
.\.venv\Scripts\python.exe tools\build_native.py
.\.venv\Scripts\python.exe tools\build_native.py --test
.\.venv\Scripts\python.exe tools\build_native.py --planner-only
```

Build environment changes are scoped to the child process. Cargo caches live under `.cache/cargo` and `.cache/rust-target`. End a session or backend that has loaded the target `.pyd` before rebuilding it. Windows holds the DLL open while it is mapped.

## Runtime paths

- **Capture.** Python passes a read-only process handle and validated regions. Rust owns the request and returned data while it batches `ReadProcessMemory` calls with the GIL released. Auto retains reusable buffers for large reads and batches small control and phase fields. `capture_backend=rust` explicitly opts in to owned large-block reads. Local component timings favor the reused buffer for the measured large block. [Capture benchmarks](../docs/en/performance-tuning.md)
- **Geometry.** Collision works from owned float32 snapshots. It checks continuous motion against round bullets, enemy-body rectangles, and oriented straight lasers, and splits paths at wall contacts.
- **Search.** ABI v3 or the independent planner extension can evaluate candidate expansion, switch timing, ranking, fallback, and escape checks in one call. The Python reference path provides fallback and a parity oracle. `search_backend` identifies complete native search independently of `collision_backend`.
- **Input.** A Rust mutex keeps held state and the SendInput batch in the same transaction. The Python wrapper also uses a reentrant lock. The independent F8 path can release input without acquiring the Rust worker's lock.

The default ABI v3 runtime includes search, collision, the Q menu key, and batched read-only capture. It installs under `native_runtime`, separate from the old ABI v2 input binary. `--planner-only` updates the independent search extension. Running workers load their native path when a session starts.

Offline input tests use dry-run mode. Component timings exclude game response, which depends on the game tick, capture, operating system scheduling, and model inference. SendInput reports acceptance by the system queue. [Test and timing scope](../docs/en/testing-and-performance.md)
