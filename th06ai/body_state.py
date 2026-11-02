"""Observed contact rectangles and motion, independent of attack targets."""
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class BodyState:
    positions: np.ndarray
    velocity: np.ndarray
    half_extents: np.ndarray
    known_motion: np.ndarray
    known_geometry: np.ndarray
    player_radius: float
    uncertainty: np.ndarray
    angles: np.ndarray | None = None

    def subset(self, mask):
        return BodyState(self.positions[mask], self.velocity[mask], self.half_extents[mask],
                         self.known_motion[mask], self.known_geometry[mask], self.player_radius,
                         self.uncertainty[mask],self.angles[mask] if self.angles is not None else None)


def visible_body_state(frame, previous=None):
    bodies = frame.enemy_bodies
    known_geometry = (bodies[:, 5] > .5) & np.all(np.isfinite(bodies[:, 3:5]), axis=1)
    half = np.where(known_geometry[:, None], bodies[:, 3:5],
                    [frame.bounds[2]+2*frame.bounds[0], frame.bounds[3]+2*frame.bounds[1]]).astype(np.float32)
    velocity = np.zeros((len(bodies), 2), np.float32)
    known_motion = np.zeros(len(bodies), bool)
    if previous is not None and 0 < frame.tick-previous.tick <= 15:
        old = {int(row[0]): row for row in previous.enemy_bodies}
        for index, row in enumerate(bodies):
            earlier = old.get(int(row[0]))
            if earlier is not None and np.array_equal(row[3:6], earlier[3:6]):
                estimate = (row[1:3]-earlier[1:3])/(frame.tick-previous.tick)
                if np.all(np.abs(estimate) < 20):
                    velocity[index], known_motion[index] = estimate, True
    radius = frame.player_radius
    radius = float(radius) if radius is not None and np.isfinite(radius) and 0 < radius <= 32 else 32.
    positions=bodies[:,1:3].astype(np.float32)
    uncertainty=(.25+np.where(known_motion,0.,4.)).astype(np.float32)
    angles=np.zeros(len(bodies),np.float32)
    lasers=getattr(frame,'lasers',np.empty((0,8),np.float32))
    if len(lasers):
        center=laser_centers(lasers)
        laser_half=np.column_stack(((lasers[:,4]-lasers[:,3])*.5,lasers[:,6]))
        motion=np.zeros_like(center)
        error=np.full(len(lasers),4.,np.float32)
        known=np.zeros(len(lasers),bool)
        if previous is not None and 0<frame.tick-previous.tick<=15:
            old={int(row[0]):row for row in previous.lasers}
            dt=frame.tick-previous.tick
            for i,row in enumerate(lasers):
                before=old.get(int(row[0]))
                if before is None:continue
                old_center=laser_centers(before[None])[0]
                speed=(center[i]-old_center)/dt
                angular=abs(np.arctan2(np.sin(row[5]-before[5]),np.cos(row[5]-before[5])))/dt
                if np.linalg.norm(speed)<=32:
                    motion[i]=speed
                    # Rotation/growth change the occupied segment. Inflate by
                    # observed tip speed rather than assuming a frozen beam.
                    growth=abs((row[4]-row[3])-(before[4]-before[3]))/(2*dt)
                    error[i]=.25+angular*float(laser_half[i,0])+growth
                    known[i]=True
        positions=np.concatenate((positions,center))
        velocity=np.concatenate((velocity,motion))
        half=np.concatenate((half,laser_half))
        known_motion=np.concatenate((known_motion,known))
        known_geometry=np.concatenate((known_geometry,np.ones(len(lasers),bool)))
        uncertainty=np.concatenate((uncertainty,error))
        angles=np.concatenate((angles,lasers[:,5]))
    return BodyState(positions,velocity,half,known_motion,known_geometry,radius,uncertainty,angles)

def laser_centers(lasers):
    distance=(lasers[:,3]+lasers[:,4])*.5
    return lasers[:,1:3]+np.column_stack((np.cos(lasers[:,5]),np.sin(lasers[:,5])))*distance[:,None]
