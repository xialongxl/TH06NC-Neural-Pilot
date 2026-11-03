from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from .planner_spec import PlanningConfig
from .game_settings import GameSetup, parse_setup

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONSOLE_PORT = 18765
DEFAULT_PREVIEW_PORT = 18766


@dataclass(frozen=True)
class SafetyConfig:
    temperature_limit_c: float = 70
    memory_temperature_limit_c: float = 80
    vram_fraction_limit: float = 0.90
    power_limit_ratio: float = 1.03
    fan_required_at_c: float = 60
    poll_seconds: float = 0.25
    telemetry_timeout_seconds: float = 60.0
    worker_heartbeat_seconds: float = 2.0

    def __post_init__(self):
        limits = {
            "temperature_limit_c": (30, 70),
            "memory_temperature_limit_c": (40, 80),
            "vram_fraction_limit": (0.1, 0.90),
            "power_limit_ratio": (1, 1.03),
            "fan_required_at_c": (40, 60),
            "poll_seconds": (0.05, 0.25),
            "telemetry_timeout_seconds": (0.1, 60.0),
            "worker_heartbeat_seconds": (0.5, 2.0),
        }
        for name, (low, high) in limits.items():
            value = getattr(self, name)
            if not low <= value <= high:
                raise ValueError(f"安全参数 {name} 必须在 {low} 到 {high} 之间")


@dataclass(frozen=True)
class DiagnosticsConfig:
    trajectory: str = 'full'
    trajectory_every_rollouts: int = 1
    compression_level: int = 1
    checkpoint_every_rollouts: int = 1

    def __post_init__(self):
        if self.trajectory not in ('full', 'summary', 'off'):
            raise ValueError('trajectory 必须是 full、summary 或 off')
        for name in ('trajectory_every_rollouts', 'checkpoint_every_rollouts'):
            value = getattr(self, name)
            if type(value) is not int or not 1 <= value <= 100:
                raise ValueError(f'{name} 必须为 1 到 100 的整数')
        if type(self.compression_level) is not int or not 0 <= self.compression_level <= 9:
            raise ValueError('compression_level 必须为 0 到 9 的整数')


@dataclass(frozen=True)
class Config:
    game_exe: str
    gpu_index: int = 0
    minutes: float = 60
    episode_seconds: float = 0
    device: str = "cuda"
    input_backend: str = "python"
    inference_backend: str = 'auto'
    goal: str = 'survival'
    safety: SafetyConfig = field(default_factory=SafetyConfig)
    planning: PlanningConfig = field(default_factory=PlanningConfig)
    game_setup: GameSetup | None = None
    diagnostics: DiagnosticsConfig = field(default_factory=DiagnosticsConfig)
    capture_backend: str = 'auto'

    @property
    def game_path(self) -> Path:
        return Path(self.game_exe)


def load_config(path: Path | str | None = None) -> Config:
    data = json.loads(Path(path or ROOT / "config.json").read_text(encoding="utf-8"))
    data["safety"] = SafetyConfig(**data.get("safety", {}))
    data["planning"] = PlanningConfig(**data.get("planning", {}))
    data['game_setup'] = parse_setup(data.get('game_setup'))
    data['diagnostics'] = DiagnosticsConfig(**data.get('diagnostics', {}))
    result = Config(**data)
    if result.device not in ("cpu", "cuda"):
        raise ValueError("device 必须是 cpu 或 cuda")
    if result.input_backend not in ('python','rust','auto'):
        raise ValueError('input_backend 必须是 python、rust 或 auto')
    if result.inference_backend not in ('auto','eager','cuda_graph'):
        raise ValueError('inference_backend 必须是 auto、eager 或 cuda_graph')
    if result.capture_backend not in ('auto','python','rust'):
        raise ValueError('capture_backend 必须是 auto、python 或 rust')
    if result.goal not in ('survival','nmnb'):
        raise ValueError('goal 必须是 survival 或 nmnb')
    if not 0 < result.minutes <= 60:
        raise ValueError("首版每次运行时间必须在 0 到 60 分钟之间")
    if result.episode_seconds != 0 and not 10 <= result.episode_seconds <= 1800:
        raise ValueError("单局训练上限必须为 0（不中途截断），或 10 到 1800 个游戏秒")
    return result
