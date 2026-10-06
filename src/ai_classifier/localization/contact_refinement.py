"""Conservative contact-frame refinement from shuttle and wrist tracks."""

from __future__ import annotations

import numpy as np


def refine_contact_index(
    shuttle: np.ndarray,
    wrists: np.ndarray,
    center: int,
    radius: int,
    *,
    temporal_penalty: float = 0.01,
    max_wrist_distance: float = 0.20,
) -> int:
    """Return a nearby frame where the shuttle is closest to a valid wrist.

    Coordinates must be normalized to the video width and height.  A temporal
    penalty keeps noisy tracks from moving an already reasonable detector peak.
    If no plausible shuttle/wrist pair exists, the original center is retained.
    """
    shuttle = np.asarray(shuttle, dtype=np.float32)
    wrists = np.asarray(wrists, dtype=np.float32)
    if radius <= 0 or len(shuttle) == 0 or shuttle.shape != wrists.shape:
        return int(center)
    if not 0 <= center < len(shuttle):
        raise ValueError("center must index the shuttle and wrist tracks")

    start = max(0, center - radius)
    stop = min(len(shuttle), center + radius + 1)
    valid = np.any(shuttle[start:stop] != 0.0, axis=1) & np.any(
        wrists[start:stop] != 0.0, axis=1
    )
    if not np.any(valid):
        return int(center)

    candidates = np.arange(start, stop, dtype=np.int64)[valid]
    distances = np.linalg.norm(shuttle[candidates] - wrists[candidates], axis=1)
    plausible = distances <= max_wrist_distance
    if not np.any(plausible):
        return int(center)

    candidates = candidates[plausible]
    distances = distances[plausible]
    scores = distances + temporal_penalty * np.abs(candidates - center)
    return int(candidates[int(np.argmin(scores))])
