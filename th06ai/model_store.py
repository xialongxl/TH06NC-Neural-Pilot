"""Checkpoint discovery and preservation; never delete or overwrite history."""
from __future__ import annotations
from .i18n import msg

import hashlib
import os
import shutil
import threading
from dataclasses import dataclass
from pathlib import Path

from .config import ROOT
from .files import atomic_json, read_json
from .native import SHA256
from .observations import OBSERVATION_VERSION
from .policy_spec import POLICY_VERSION


@dataclass(frozen=True)
class Checkpoint:
    path: Path
    metadata: dict
    compatibility: str

    @property
    def usable(self):
        return self.compatibility in (msg('backend.ready_to_resume'), msg('backend.legacy_model_can_be_migrated'), msg('backend.bomb_policy_upgraded_on_load'))

    @property
    def label(self):
        return msg('backend.0_steps_1_2', f"{self.metadata.get('steps', 0):,}", self.path.parent.name, self.path.name)


def inspect_checkpoint(path: Path, *, verify=False) -> Checkpoint:
    meta = read_json(path.with_suffix(".json"))
    if not path.is_file() or not meta:
        state = msg('backend.model_or_metadata_missing')
    elif meta.get("game_sha256") != SHA256:
        state = msg('backend.game_build_mismatch')
    elif meta.get("format") == OBSERVATION_VERSION:
        version = meta.get("policy_version") or 1
        state = (msg('backend.bomb_policy_upgraded_on_load') if version < POLICY_VERSION else
                 msg('backend.ready_to_resume') if version == POLICY_VERSION else msg('backend.unsupported_policy_version'))
    elif meta.get("format") in (1, 2, 3, 4, 5, 6, 7, 8):
        state = msg('backend.unsupported_policy_version') if (meta.get('policy_version') or 1) > POLICY_VERSION else msg('backend.legacy_model_can_be_migrated')
    else:
        state = msg('backend.unsupported_observation_format')
    if verify and path.is_file() and hashlib.sha256(path.read_bytes()).hexdigest() != meta.get("model_sha256"):
        state = msg('backend.file_verification_failed')
    return Checkpoint(path.resolve(), meta, state)


def list_checkpoints(root: Path | None = None):
    directory = root or ROOT / "models"
    entries = [inspect_checkpoint(p) for p in directory.rglob("*.zip") if ".pending" not in p.name]
    return sorted(entries, key=lambda c: (c.metadata.get("time", ""), c.path.name == "latest.zip", str(c.path)), reverse=True)


class CheckpointCatalog:
    """Incremental display cache; loading still performs fresh path/hash checks."""
    def __init__(self, root: Path):
        self.root = root.resolve()
        self._entries = {}
        self._lock = threading.Lock()

    @staticmethod
    def _stamp(entry):
        stat = entry.stat()
        return stat.st_mtime_ns, stat.st_size, stat.st_ino

    def scan(self):
        # Only the scanner owns this cache. No console/control lock is held.
        with self._lock:
            updated, result = {}, []
            pending, visited = [self.root], set()
            while pending:
                folder = pending.pop()
                try:
                    folder = folder.resolve()
                    if folder in visited or not folder.is_relative_to(self.root):
                        continue
                    visited.add(folder)
                    with os.scandir(folder) as stream:
                        files = {}
                        for entry in stream:
                            if entry.is_dir(follow_symlinks=False):
                                pending.append(Path(entry.path))
                            elif entry.is_file() and entry.name.endswith(('.zip', '.json')):
                                files[entry.name] = entry
                        for name, entry in files.items():
                            if not name.endswith('.zip') or '.pending' in name:
                                continue
                            path = Path(entry.path)
                            meta = files.get(path.with_suffix('.json').name)
                            stamp = (self._stamp(entry), self._stamp(meta) if meta else None)
                            cached = self._entries.get(path)
                            value = cached[1] if cached and cached[0] == stamp else inspect_checkpoint(path)
                            if value.path.is_relative_to(self.root):
                                result.append(value)
                                if value.metadata:
                                    updated[path] = (stamp, value)
                except (FileNotFoundError, NotADirectoryError):
                    continue  # A concurrent atomic save can move a candidate.
            self._entries = updated
            return sorted(result, key=lambda c: (c.metadata.get('time', ''), c.path.name == 'latest.zip', str(c.path)), reverse=True)


def preserve_checkpoint(path: Path, root: Path | None = None):
    """Make a content-addressed immutable copy before replacing latest.zip."""
    if not path.is_file():
        return None
    checkpoint = inspect_checkpoint(path, verify=True)
    if not checkpoint.usable:
        raise RuntimeError(msg('backend.the_existing_checkpoint_failed_verification_and_cannot', checkpoint.compatibility))
    digest = checkpoint.metadata["model_sha256"]
    target = (root or ROOT / "models") / "checkpoints" / f"step-{checkpoint.metadata.get('steps', 0):09d}-{digest[:12]}.zip"
    target.parent.mkdir(parents=True, exist_ok=True)
    if not target.exists():
        shutil.copy2(path, target)
        atomic_json(target.with_suffix(".json"), checkpoint.metadata)
        atomic_json(target.parent.parent / 'catalog_revision.json', {'checkpoint': target.name, 'digest': digest})
    return target
