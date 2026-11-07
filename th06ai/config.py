from __future__ import annotations
from .i18n import msg

import json
from dataclasses import dataclass, field
from pathlib import Path
from .paths import DEFAULT_CONSOLE_PORT, DEFAULT_PREVIEW_PORT, ROOT
from .planner_spec import PlanningConfig
from .game_settings import GameSetup, parse_setup

PROFILE_KINDS = ('inference', 'latency', 'readback')
READBACK_AB_STEPS = 512
INFERENCE_READBACK_MODES = ('auto', 'synchronous')


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
                raise ValueError(msg('backend.safety_setting_0_must_be_between_1', name, low, high))


@dataclass(frozen=True)
class DiagnosticsConfig:
    trajectory: str = 'full'
    trajectory_every_rollouts: int = 1
    compression_level: int = 1
    checkpoint_every_rollouts: int = 1
    thread_profile_steps: int = 0
    readback_ab: bool = False

    def __post_init__(self):
        if self.trajectory not in ('full', 'summary', 'off'):
            raise ValueError(msg('backend.trajectory_must_be_full_summary_or_off'))
        for name in ('trajectory_every_rollouts', 'checkpoint_every_rollouts'):
            value = getattr(self, name)
            if type(value) is not int or not 1 <= value <= 100:
                raise ValueError(msg('backend.0_must_be_an_integer_from_1', name))
        if type(self.compression_level) is not int or not 0 <= self.compression_level <= 9:
            raise ValueError(msg('backend.compression_level_must_be_an_integer_from'))
        if type(self.thread_profile_steps) is not int or not 0 <= self.thread_profile_steps <= 4096:
            raise ValueError(msg('backend.thread_profile_steps_must_be_an_integer'))
        if type(self.readback_ab) is not bool:
            raise ValueError(msg('backend.readback_ab_must_be_boolean'))


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
    inference_readback: str = 'auto'

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
        raise ValueError(msg('backend.device_must_be_cpu_or_cuda'))
    if result.input_backend not in ('python','rust','auto'):
        raise ValueError(msg('backend.input_backend_must_be_python_rust_or'))
    if result.inference_backend not in ('auto','eager','cuda_graph'):
        raise ValueError(msg('backend.inference_backend_must_be_auto_eager_or'))
    if result.inference_readback not in INFERENCE_READBACK_MODES:
        raise ValueError(msg('backend.inference_readback_must_be_auto_or_synchronous'))
    if result.capture_backend not in ('auto','python','rust'):
        raise ValueError(msg('backend.capture_backend_must_be_auto_python_or'))
    if result.goal not in ('survival','nmnb'):
        raise ValueError(msg('backend.goal_must_be_survival_or_nmnb'))
    if not 0 < result.minutes <= 60:
        raise ValueError(msg('backend.the_session_duration_must_be_greater_than'))
    if result.episode_seconds != 0 and not 10 <= result.episode_seconds <= 1800:
        raise ValueError(msg('backend.the_episode_limit_must_be_0_unlimited'))
    return result
