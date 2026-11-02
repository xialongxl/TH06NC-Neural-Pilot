"""Steady-input motion measurements; no input or process APIs."""
import math
from statistics import median

from .input_timing import INPUT_BITS, MOVEMENT_MASK


def held_speed(before, after, keys):
    """Reject key transitions and wall clipping instead of diluting speed.

    Both snapshots must already confirm the same input. Measuring from a
    snapshot taken before SendInput includes the acknowledgement delay and
    was the source of the old 12/7 focused-speed estimate.
    """
    ticks = after.tick-before.tick
    if not 0 < ticks <= 10 or before.state == 2 or after.state == 2:
        return None
    if before.paused or after.paused or not before.stage_active or not after.stage_active:
        return None
    expected = sum(INPUT_BITS.get(k, 0) for k in keys)
    if any(f.input_bits is None or (f.input_bits & MOVEMENT_MASK) != expected for f in (before, after)):
        return None
    direction = (int('right' in keys)-int('left' in keys), int('down' in keys)-int('up' in keys))
    length = math.hypot(*direction)
    if not length:
        return None
    direction = tuple(v/length for v in direction)
    for frame in (before, after):
        low = frame.bounds[:2]
        high = tuple(a+b for a,b in zip(low, frame.bounds[2:]))
        if any(v and not lo+.5 < p < hi-.5 for v,p,lo,hi in zip(direction, (frame.x, frame.y), low, high)):
            return None
    dx, dy = after.x-before.x, after.y-before.y
    along = dx*direction[0]+dy*direction[1]
    across = abs(dx*direction[1]-dy*direction[0])
    speed = along/ticks
    return float(speed) if 0 < speed < 20 and across < .1 else None


def measured_speed(samples):
    usable = [float(v) for v in samples if v is not None and math.isfinite(v) and 0 < v < 20]
    if len(usable) < 5:
        raise ValueError('缺少已确认按键下的稳定移动样本')
    center = median(usable)
    consistent = [v for v in usable if abs(v-center) <= max(.05, center*.05)]
    if len(consistent) < 5 or len(consistent) < len(usable)*.7:
        raise ValueError('稳定移动测量不一致，请重新校准')
    return float(median(consistent))
