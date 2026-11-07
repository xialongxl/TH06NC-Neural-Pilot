"""Straight laser pool pinned to TH06NC's own 0x11cdc..0x12068 update loop."""
import numpy as np

LASER_CAPACITY=64
LASER_STRIDE=0x298
LASER_START=0x436EF0+0xF502C-0x10
LASER_DTYPE=np.dtype({'names':['width','start','end','angle','active','x','y','state'],
 'formats':['<f4','<f4','<f4','<f4','u1','<f4','<f4','u1'],
 'offsets':[0,0xC,0x10,0x25C,0x270,0x274,0x278,0x28C],'itemsize':LASER_STRIDE})

def decode_lasers(raw,width,height):
    """slot,origin x/y,start/end distance,angle,collision half-width,state.

    State 0 is a visible warning, 1 damages, 2 only fades (the game's fading
    branch does not call 0x6c2b0). Warning geometry remains available to plan.
    """
    if len(raw)!=LASER_CAPACITY*LASER_STRIDE:raise ValueError('laser pool shape mismatch')
    data=np.frombuffer(raw,LASER_DTYPE)
    valid=(data['active']!=0)&(data['state']<=1)
    if not valid.any():
        return np.empty((0, 8), np.float32)
    for name in ('x','y','start','end','angle','width'):valid &= np.isfinite(data[name])
    valid&=(data['end']>=data['start'])&(data['start']>=0)&(data['width']>0)&(data['width']<=512)
    # Native caller passes (length, nominal_width*0.5); the hit routine halves
    # both dimensions again. Do not mistake sprite width for collision width.
    half_width=data['width']*.25
    distance=(data['start']+data['end'])*.5
    cosine, sine = np.cos(data['angle']), np.sin(data['angle'])
    cx=data['x']+cosine*distance
    cy=data['y']+sine*distance
    half_len=(data['end']-data['start'])*.5
    abs_cosine, abs_sine = np.abs(cosine), np.abs(sine)
    hx=abs_cosine*half_len+abs_sine*half_width
    hy=abs_sine*half_len+abs_cosine*half_width
    valid&=(cx+hx>=0)&(cx-hx<=width)&(cy+hy>=0)&(cy-hy<=height)
    slots=np.flatnonzero(valid)
    return np.column_stack((slots,data['x'][valid],data['y'][valid],data['start'][valid],data['end'][valid],
                            data['angle'][valid],half_width[valid],data['state'][valid])).astype(np.float32)
