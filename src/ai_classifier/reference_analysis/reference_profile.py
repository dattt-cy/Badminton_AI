"""Build reference ranges for rule-based checks from confirmed-good sample strokes."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from ai_classifier.biomechanics import KineticChainTiming
from ai_classifier.error_detection import RangeCheck, phase_feature_value

# Fewer samples than this make a percentile-based range unreliable (mục 10.1).
MIN_REFERENCE_SAMPLES = 3


def build_range_checks(
    samples: dict[str, Sequence[float]],
    *,
    low_percentile: float = 10.0,
    high_percentile: float = 90.0,
    buffer_ratio: float = 0.05,
) -> dict[str, RangeCheck]:
    """Turn per-feature sample values into percentile-based reference ranges.

    Each feature needs at least `MIN_REFERENCE_SAMPLES` finite values;
    features with too few valid samples are skipped rather than guessed.
    """
    checks: dict[str, RangeCheck] = {}
    for name, values in samples.items():
        finite = np.asarray([v for v in values if np.isfinite(v)], dtype=np.float64)
        if len(finite) < MIN_REFERENCE_SAMPLES:
            continue
        low = float(np.percentile(finite, low_percentile))
        high = float(np.percentile(finite, high_percentile))
        checks[name] = RangeCheck(low, high, buffer_ratio=buffer_ratio)
    return checks


def build_kinetic_chain_reference(
    timings: Sequence[KineticChainTiming],
    *,
    buffer_ratio: float = 0.05,
) -> dict[str, RangeCheck]:
    """Aggregate per-lag samples across confirmed-good strokes into `RangeCheck`s.

    Strokes with `timing.valid=False` (insufficient pose data) are skipped.
    """
    samples: dict[str, list[float]] = {}
    for timing in timings:
        if not timing.valid:
            continue
        for lag_name, value in timing.lags.items():
            samples.setdefault(lag_name, []).append(value)
    return build_range_checks(samples, buffer_ratio=buffer_ratio)
