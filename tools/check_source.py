"""Inspect Git's actual candidate/staged files; never deletes or stages files."""
from __future__ import annotations

import argparse
from collections import Counter
import io
import json
from pathlib import Path
import subprocess
import zipfile

ROOT = Path(__file__).resolve().parents[1]
FORBIDDEN_PARTS = {".venv", "node_modules", "target", "runs", ".cache", "models", "research", "tmp",
                   ".git", "__pycache__", ".pytest_cache", ".codex", ".agents", ".idea", ".vscode"}
FORBIDDEN_SUFFIXES = {".zip", ".pyd", ".dll", ".exe", ".whl", ".log", ".tmp", ".pstats", ".pyc"}
LOCAL_FILES = {"config.json", "native_calibration.json", "native_resources_calibration.json",
               "native_combat_calibration.json", "training_backend.json", "ui_state.json", ".env"}


def git(*args):
    result = subprocess.run(["git", *args], cwd=ROOT, capture_output=True)
    if result.returncode:
        raise RuntimeError(result.stderr.decode("utf-8", errors="replace").strip())
    return result.stdout


def private_markers():
    values = {str(ROOT.resolve()), str(Path.home())}
    try:
        game = json.loads((ROOT / "config.json").read_text(encoding="utf-8"))["game_exe"]
        values.add(str(Path(game).parent))
    except (OSError, ValueError, KeyError):
        pass
    variants = set()
    for value in values:
        if len(value) < 8:
            continue
        variants.update((value.casefold(), value.replace("\\", "/").casefold(),
                         value.replace("\\", "\\\\").casefold()))
    return variants


def check(staged=False):
    raw = (git("diff", "--cached", "--name-only", "--diff-filter=ACMR", "-z") if staged else
           git("ls-files", "--cached", "--others", "--exclude-standard", "-z"))
    names = sorted(set(n.decode("utf-8") for n in raw.split(b"\0") if n))
    markers = private_markers(); errors = []; sizes = {}
    for name in names:
        relative = Path(name); path = ROOT / relative
        parts = {p.casefold() for p in relative.parts}
        if (parts & FORBIDDEN_PARTS or relative.suffix.lower() in FORBIDDEN_SUFFIXES
                or name in LOCAL_FILES or name.startswith(".env.") and name != ".env.example"
                or name.startswith("frontend/dist/")):
            errors.append({"path": name, "reason": "local/generated/private artifact in Git scope"})
            continue
        if not staged and (not path.is_file() or not path.resolve().is_relative_to(ROOT)):
            errors.append({"path": name, "reason": "missing file or external symlink"})
            continue
        # Validate index bytes, not a potentially cleaner unstaged replacement.
        data = git("show", ":" + name) if staged else path.read_bytes()
        sizes[name] = len(data)
        if len(data) > 5 * 1024 * 1024:
            errors.append({"path": name, "reason": "large source artifact requires explicit review"})
        chunks = [data]
        if relative.suffix.lower() == ".npz":
            with zipfile.ZipFile(io.BytesIO(data)) as archive:
                chunks = []
                for info in archive.infolist():
                    if info.file_size > 32 * 1024 * 1024:
                        errors.append({"path": name, "reason": "oversized compressed fixture member"})
                    else:
                        chunks.append(archive.read(info))
        text = "\n".join(block.decode("utf-8", errors="ignore") for block in chunks).casefold()
        if any(marker in text for marker in markers):
            errors.append({"path": name, "reason": "contains this machine's project/home/game directory"})
    return {"ok": not errors, "scope": "staged" if staged else "tracked-and-unignored-candidates",
            "files": len(names), "bytes": sum(sizes.values()),
            "top_level_counts": dict(sorted(Counter(Path(n).parts[0] for n in names).items())),
            "errors": errors,
            "limits": "Checks known artifact/path patterns, not a full secret, copyright or history audit."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--staged", action="store_true")
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    try:
        result = check(args.staged)
    except (OSError, RuntimeError, zipfile.BadZipFile) as error:
        result = {"ok": False, "errors": [str(error)]}
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    raise SystemExit(0 if result["ok"] else 1)


if __name__ == "__main__":
    main()
