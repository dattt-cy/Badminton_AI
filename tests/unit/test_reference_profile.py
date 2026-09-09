import math

import numpy as np
import pytest

from ai_classifier.biomechanics import GeometryFeatures, KineticChainTiming
from ai_classifier.error_detection import check_kinetic_chain
from ai_classifier.reference_analysis import (
    build_kinetic_chain_reference,
    build_range_checks,
    phase_feature_value,
)


def test_build_range_checks_uses_percentile_bounds() -> None:
    samples = {"hip_after_knee": [0.05, 0.06, 0.07, 0.08, 0.09, 0.10]}

    checks = build_range_checks(samples, low_percentile=10, high_percentile=90)

    check = checks["hip_after_knee"]
    assert check.low == pytest.approx(np.percentile(samples["hip_after_knee"], 10))
    assert check.high == pytest.approx(np.percentile(samples["hip_after_knee"], 90))


def test_build_range_checks_skips_features_with_too_few_samples() -> None:
    checks = build_range_checks({"hip_after_knee": [0.05, 0.06]})

    assert "hip_after_knee" not in checks


def test_build_range_checks_ignores_nan_samples() -> None:
    samples = {"hip_after_knee": [0.05, math.nan, 0.06, 0.07]}

    checks = build_range_checks(samples)

    assert "hip_after_knee" in checks


def _timing(lags: dict[str, float], *, valid: bool = True) -> KineticChainTiming:
    return KineticChainTiming(
        peak_frames={},
        lags=lags,
        order_correct=True,
        transfer_time=lags.get("wrist_after_elbow", math.nan),
        valid=valid,
    )


def test_build_kinetic_chain_reference_aggregates_valid_strokes_only() -> None:
    timings = [
        _timing({"hip_after_knee": 0.05, "wrist_after_elbow": 0.03}),
        _timing({"hip_after_knee": 0.06, "wrist_after_elbow": 0.04}),
        _timing({"hip_after_knee": 0.07, "wrist_after_elbow": 0.05}),
        _timing({"hip_after_knee": 999.0, "wrist_after_elbow": 999.0}, valid=False),
    ]

    reference = build_kinetic_chain_reference(timings)

    assert reference["hip_after_knee"].high < 900.0


def test_reference_ranges_feed_directly_into_check_kinetic_chain() -> None:
    timings = [
        _timing({"hip_after_knee": 0.05}),
        _timing({"hip_after_knee": 0.06}),
        _timing({"hip_after_knee": 0.07}),
    ]
    reference = build_kinetic_chain_reference(timings)
    user_timing = _timing({"hip_after_knee": 0.5})

    events = check_kinetic_chain(user_timing, reference)

    assert any(event.rule_name == "hip_after_knee" for event in events)


def test_phase_feature_value_takes_median_within_range() -> None:
    features = GeometryFeatures(
        ("elbow_angle",),
        np.array([[10.0], [20.0], [30.0], [999.0]], dtype=np.float32),
        np.ones((4, 1), dtype=np.float32),
        np.arange(4, dtype=np.float32) / 30,
    )

    value = phase_feature_value(features, (0, 3), "elbow_angle")

    assert value == pytest.approx(20.0)


def test_phase_feature_value_nan_for_empty_range() -> None:
    features = GeometryFeatures(
        ("elbow_angle",),
        np.array([[10.0]], dtype=np.float32),
        np.ones((1, 1), dtype=np.float32),
        np.zeros(1, dtype=np.float32),
    )

    assert math.isnan(phase_feature_value(features, (2, 2), "elbow_angle"))


def test_phase_feature_value_nan_when_all_values_invalid() -> None:
    features = GeometryFeatures(
        ("elbow_angle",),
        np.full((3, 1), np.nan, dtype=np.float32),
        np.zeros((3, 1), dtype=np.float32),
        np.arange(3, dtype=np.float32) / 30,
    )

    assert math.isnan(phase_feature_value(features, (0, 3), "elbow_angle"))
