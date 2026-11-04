"""Lightweight planning metadata; no NumPy/Torch in the hardware watchdog."""
from .i18n import msg
from dataclasses import dataclass

PLANNER_VERSION = 7
PLAN_MOVEMENTS = 18
PLAN_FIELDS = ("enabled", "allowed", "clearance", "collision_time", "blocked", "goal", "utility", "logit_bias")
PLAN_FEATURES = len(PLAN_FIELDS)
PLAN_LOGIT_SCALE = 16.0
HORIZON_TICKS = 32
DELAY_TICKS = 2
SEGMENT_TICKS = (2, 6, 6, 8)
MAX_SEARCH_NODES = 1024


@dataclass(frozen=True)
class PlanningConfig:
    enabled: bool = True
    beam_width: int = 2
    collision_backend: str = "python"

    def __post_init__(self):
        if not isinstance(self.enabled, bool) or not 1 <= self.beam_width <= 4:
            raise ValueError(msg('backend.planning_enabled_must_be_boolean_and_beam'))
        if self.collision_backend not in ('python','rust','auto'):
            raise ValueError(msg('backend.collision_backend_must_be_python_rust_or'))
