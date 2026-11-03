"""Checkpoint discovery and preservation; never delete or overwrite history."""
from __future__ import annotations

import hashlib
import shutil
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
        return self.compatibility in ("可直接续训", "旧版可迁移", "加载时修复炸弹策略")

    @property
    def label(self):
        return f"{self.metadata.get('steps', 0):,} 步 · {self.path.parent.name}/{self.path.name}"


def inspect_checkpoint(path: Path, *, verify=False) -> Checkpoint:
    meta = read_json(path.with_suffix(".json"))
    if not path.is_file() or not meta:
        state = "缺少模型或元数据"
    elif meta.get("game_sha256") != SHA256:
        state = "游戏版本不匹配"
    elif meta.get("format") == OBSERVATION_VERSION:
        version = meta.get("policy_version") or 1
        state = ("加载时修复炸弹策略" if version < POLICY_VERSION else
                 "可直接续训" if version == POLICY_VERSION else "不支持的策略版本")
    elif meta.get("format") in (1, 2, 3, 4, 5, 6, 7, 8):
        state = "不支持的策略版本" if (meta.get('policy_version') or 1) > POLICY_VERSION else "旧版可迁移"
    else:
        state = "不支持的观测格式"
    if verify and path.is_file() and hashlib.sha256(path.read_bytes()).hexdigest() != meta.get("model_sha256"):
        state = "文件校验失败"
    return Checkpoint(path.resolve(), meta, state)


def list_checkpoints(root: Path | None = None):
    directory = root or ROOT / "models"
    entries = [inspect_checkpoint(p) for p in directory.rglob("*.zip") if ".pending" not in p.name]
    return sorted(entries, key=lambda c: (c.metadata.get("time", ""), c.path.name == "latest.zip"), reverse=True)


def preserve_checkpoint(path: Path, root: Path | None = None):
    """Make a content-addressed immutable copy before replacing latest.zip."""
    if not path.is_file():
        return None
    checkpoint = inspect_checkpoint(path, verify=True)
    if not checkpoint.usable:
        raise RuntimeError(f"旧模型未通过校验，拒绝覆盖：{checkpoint.compatibility}")
    digest = checkpoint.metadata["model_sha256"]
    target = (root or ROOT / "models") / "checkpoints" / f"step-{checkpoint.metadata.get('steps', 0):09d}-{digest[:12]}.zip"
    target.parent.mkdir(parents=True, exist_ok=True)
    if not target.exists():
        shutil.copy2(path, target)
        atomic_json(target.with_suffix(".json"), checkpoint.metadata)
    return target
