"""Task priorities for visible pickups; collision feasibility stays separate."""
import numpy as np

def item_weights(frame):
    r=frame.resources
    power=r.power if r is not None else 0
    bombs=r.bombs if r is not None else 3
    lives=r.lives if r is not None else 2
    needed=power<128
    # P is weapon capacity. Score/star items must not crowd it out before a
    # Boss. These are goal scores, never permission to cross a collision path.
    values=np.array([2.*needed,.5,2.5*needed,float(bombs<8),3.*needed,1.5*(lives<8),.01],np.float32)
    return values[frame.items[:,3].astype(int)]

def select_items(frame,count=32):
    distance=np.linalg.norm(frame.items[:,1:3]-[frame.x,frame.y],axis=1)
    weights=item_weights(frame)
    return np.argsort(-weights*np.exp(-distance/96),kind='stable')[:count]

def power_potential(frame):
    # Potential-based shaping: falling Power and terminal death repay the
    # potential. Repeated collect/die loops cannot farm this extra feedback.
    return 24.*frame.resources.power/128 if frame is not None and frame.resources is not None else 0.
