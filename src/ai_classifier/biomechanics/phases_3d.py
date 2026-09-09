"""Phase detection adapted to long MultiSense annotation windows."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .features import GeometryFeatures, PhaseBoundaries, detect_stroke_phases


@dataclass(frozen=True)
class MultiSensePhaseResult:
    features: GeometryFeatures
    phases: PhaseBoundaries
    start_frame: int
    end_frame: int
    motion_peak_frame: int


def detect_multisense_stroke_phases(
    features: GeometryFeatures, *, before_peak_seconds: float = 1.2,
    after_peak_seconds: float = 0.6,
) -> MultiSensePhaseResult:
    """Trim launch/recovery context, then detect phases in the dominant stroke.

    MultiSense annotations typically span 4--5 seconds.  A compact window is
    selected around dominant racket-wrist motion before applying the existing
    phase semantics.  Contact remains an estimated candidate, not ground truth.
    """
    if before_peak_seconds <= 0 or after_peak_seconds <= 0:
        raise ValueError("phase window durations must be positive")
    frame_count = len(features.timestamps)
    if frame_count < 10:
        return MultiSensePhaseResult(
            features, detect_stroke_phases(features), 0, frame_count, 0
        )
    speed = _median_filter(features.column("wrist_speed"), window=5)
    search_start = int(round(frame_count * 0.15))
    search_end = max(search_start + 1, int(round(frame_count * 0.90)))
    candidate = speed[search_start:search_end]
    peak = (
        search_start + int(np.nanargmax(candidate))
        if np.isfinite(candidate).any() else frame_count // 2
    )
    peak_time = float(features.timestamps[peak])
    start = int(np.searchsorted(
        features.timestamps, peak_time - before_peak_seconds, side="left"
    ))
    end = int(np.searchsorted(
        features.timestamps, peak_time + after_peak_seconds, side="right"
    ))
    start, end = max(0, start), min(frame_count, end)
    trimmed = GeometryFeatures(
        features.names,
        features.values[start:end],
        features.confidence[start:end],
        features.timestamps[start:end] - features.timestamps[start],
    )
    return MultiSensePhaseResult(
        trimmed, detect_stroke_phases(trimmed), start, end, peak
    )


def _median_filter(values: np.ndarray, *, window: int) -> np.ndarray:
    result = np.full(np.asarray(values).shape, np.nan, dtype=np.float32)
    radius = window // 2
    for index in range(len(values)):
        chunk = np.asarray(values)[
            max(0, index-radius):min(len(values), index+radius+1)
        ]
        finite = chunk[np.isfinite(chunk)]
        if finite.size:
            result[index] = np.median(finite)
    return result
