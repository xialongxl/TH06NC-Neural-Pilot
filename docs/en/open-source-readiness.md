# Source release checklist

English · [简体中文](../open-source-readiness.md)

**TH06NC-Neural-Pilot** is maintained by **xialongxl** under standard MIT. The primary release is the existing Git checkout pushed to the reviewed remote. A separate clean working directory or ZIP is optional. [Public repository](https://github.com/xialongxl/TH06NC-Neural-Pilot)

## What belongs in Git

| Include | Keep local |
| --- | --- |
| Python, Rust, React source, owned static assets, and build scripts | `.venv`, `node_modules`, `target`, `frontend/dist`, `.cache`, and binaries |
| Lockfiles, portable examples, launchers, and metadata | Actual `config.json`, `.env`, UI state, and calibration or backend records |
| Tests and minimal approved fixtures | Private `runs`, `models` including nested ZIPs, `research`, and scratch files |
| Documentation, licenses, attribution, and collaboration templates | Game files and downloaded dependencies |

The published Stage 5 NPZ is a geometry regression fixture. Runtime archives and models remain local.

## Review the actual staged content

```powershell
git remote -v
git status --short
git status --short --ignored
git ls-files
git diff --cached --name-only
git diff --cached --check
.\.venv\Scripts\python.exe tools\check_docs.py
.\.venv\Scripts\python.exe tools\check_source.py --staged
```

Use `tools/check_source.py` without `--staged` for tracked and unignored candidate files. It checks known artifacts and path patterns, including fixture metadata. Separately audit secrets, rights, and history. Ignore rules do not remove tracked files. After review, `git rm --cached` can remove a path from the index while preserving its local copy.

```powershell
git check-ignore -v .venv/Scripts/python.exe frontend/node_modules/react/package.json native/target/release/example.dll runs/current.json .cache/example.bin models/latest.zip models/checkpoints/example.zip config.json
git check-ignore config.example.json .env.example frontend/package-lock.json native/Cargo.lock tests/fixtures/stage5-switch-window.npz
```

The second command should have no matches. Inspect staged changes and remote ownership before committing or pushing. A source archive from a reviewed commit can be created with `git archive` and only contains files in that commit.

## Portability and rights

`deploy.py` and the locally generated `LaunchAI.exe` resolve the project from their own locations. The operator supplies `game_exe` locally and can provide a portable Rust/MSVC toolchain. Loopback ports and the Vite proxy must agree. Confirm that installation creates only missing local configuration and that calibration requires real evidence.

Inspect absolute paths, user names, GPU UUIDs, tokens, screenshots, log fields, archives, and generated assets before sharing. Keep upstream notices for derived SB3 code. Review the actual contents of any future bundled binary separately. [Disclaimer](../../DISCLAIMER.md) · [Third-party notices](../../THIRD_PARTY_NOTICES.md)

## Release scope

The repository includes source, instructions, collaboration templates, and offline regression evidence. Self-service resource confirmation, a complete Python transitive lock and CI setup, clean-machine validation, broad full-clear and NMNB evidence, and binary or remote deployment remain separate work. Publish screenshots or videos only with sanitized content and a clear indication of preview versus live operation.
