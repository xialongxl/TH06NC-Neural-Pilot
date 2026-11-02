"""Collision feedback from current visible geometry and observed past motion.

This estimates risk for learning; it never chooses an action or promises that
an observed trajectory will stay safe after a bullet turns or a laser appears.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .native import Frame
from .bullet_state import visible_bullet_state, PREDICTION_TICKS, UNKNOWN_COLLISION_RADIUS

SAFETY_CLEARANCE = 6.0  # Pixels beyond the verified combined collision radius.


@dataclass(frozen=True)
class RiskEstimate:
    threat: float
    min_clearance: float | None
    geometry_coverage: float


def collision_risk(frame: Frame, previous: Frame | None = None) -> RiskEstimate:
    def contact_risk(clearance):
        # Outside the collision envelope + margin, proximity has no cost.
        # Being inside the much wider graze ring is neither a bonus nor a fault.
        return np.clip(1 - np.maximum(clearance, 0) / SAFETY_CLEARANCE, 0, 1)**2

    threat,clearance,coverage=0.,None,1.
    if len(frame.bullets):
        state = visible_bullet_state(frame, previous)
        risk = np.maximum(contact_risk(state.current_clearance),
                          contact_risk(state.predicted_clearance) * np.exp(-state.closest_tick / PREDICTION_TICKS))
        clearance = float(state.predicted_clearance.min()) if state.known_geometry.all() and state.known_motion.all() else None
        threat,coverage=float(risk.max()),float(state.known_geometry.mean())
    lasers=frame.lasers
    if len(lasers):
        from .body_state import laser_centers
        live=lasers[lasers[:,7]==1]  # A warning is planned around, not a current hit.
        if len(live):
            relative=np.array([frame.x,frame.y])-laser_centers(live)
            c,s=np.cos(live[:,5]),np.sin(live[:,5])
            local=np.column_stack((relative[:,0]*c+relative[:,1]*s,-relative[:,0]*s+relative[:,1]*c))
            extent=np.column_stack(((live[:,4]-live[:,3])*.5,live[:,6]))
            gap=np.maximum(np.abs(local)-extent,0)
            distance=np.linalg.norm(gap,axis=1)-(frame.player_radius or UNKNOWN_COLLISION_RADIUS)
            value=float(distance.min())
            clearance=min(clearance,value) if clearance is not None else value
            threat=max(threat,float(contact_risk(distance).max()))
    return RiskEstimate(threat,clearance,coverage)


def visible_threat(frame: Frame, previous: Frame | None = None) -> float:
    return collision_risk(frame, previous).threat
