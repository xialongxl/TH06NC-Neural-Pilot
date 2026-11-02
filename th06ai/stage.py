"""Observed stage identity, independent of Boss visibility or elapsed time."""
import numpy as np

STAGE_FEATURES = 8  # seven one-hot stages, then identity-known bit
STAGE_RVA = 0x53CAD4


def decode_stage(raw):
    return int(raw) if isinstance(raw, (int, np.integer)) and not isinstance(raw, bool) and 1 <= raw <= 7 else None


def stage_label(stage_id):
    stage_id = decode_stage(stage_id)
    return 'EX' if stage_id == 7 else f'Stage {stage_id}' if stage_id is not None else '关卡待识别'


def stage_vector(stage_id):
    result = np.zeros(STAGE_FEATURES, np.float32)
    stage_id = decode_stage(stage_id)
    if stage_id is not None:
        result[stage_id-1] = result[-1] = 1
    return result


def context_changed(before, after):
    """Unknown identity is a separate context, never silently treated as Stage 1."""
    if before is None or after is None:
        return False
    return decode_stage(getattr(before, 'stage_id', None)) != decode_stage(getattr(after, 'stage_id', None))
