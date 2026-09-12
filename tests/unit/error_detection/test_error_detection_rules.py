import math

import numpy as np
import pytest

from ai_classifier.biomechanics import GeometryFeatures, PhaseBoundaries, extract_kinetic_chain_timing
from ai_classifier.error_detection import (
    RangeCheck,
    RuleDefinition,
    check_kinetic_chain,
    check_phase_features,
    evaluate_range,
    evaluate_rule_definitions,
)


def test_evaluate_range_returns_none_when_inside_reference() -> None:
    assert evaluate_range("elbow_angle", 100.0, RangeCheck(90.0, 150.0)) is None


def test_evaluate_range_flags_review_inside_buffer() -> None:
    event = evaluate_range("elbow_angle", 88.0, RangeCheck(90.0, 150.0, buffer_ratio=0.05))

    assert event is not None
    assert event.severity == "review"


def test_evaluate_range_flags_deviation_outside_buffer() -> None:
    event = evaluate_range("elbow_angle", 40.0, RangeCheck(90.0, 150.0, buffer_ratio=0.05))

    assert event is not None
    assert event.severity == "deviation"
    assert event.observed == pytest.approx(40.0)


def test_evaluate_range_ignores_nan() -> None:
    assert evaluate_range("elbow_angle", math.nan, RangeCheck(90.0, 150.0)) is None


def test_evaluate_range_respects_one_sided_direction() -> None:
    check = RangeCheck(90.0, 150.0, direction="lower")
    assert evaluate_range("arm_not_extended", 170.0, check) is None
    assert evaluate_range("arm_not_extended", 80.0, check) is not None


def _chain_features(peak_frames: list[int]) -> GeometryFeatures:
    names = (
        "knee_angular_speed",
        "hip_angular_speed",
        "shoulder_angular_speed",
        "elbow_angular_speed",
        "wrist_speed",
    )
    values = np.zeros((6, len(names)), dtype=np.float32)
    for column, frame in enumerate(peak_frames):
        values[frame, column] = 5
    confidence = np.ones_like(values)
    timestamps = np.arange(6, dtype=np.float32) / 30
    return GeometryFeatures(names, values, confidence, timestamps)


def test_check_kinetic_chain_no_events_when_order_and_lags_ok() -> None:
    timing = extract_kinetic_chain_timing(_chain_features([1, 2, 3, 4, 5]))
    reference_lags = {
        "hip_after_knee": RangeCheck(0.0, 0.1),
        "wrist_after_elbow": RangeCheck(0.0, 0.1),
    }

    assert check_kinetic_chain(timing, reference_lags) == []


def test_check_kinetic_chain_flags_out_of_order() -> None:
    timing = extract_kinetic_chain_timing(_chain_features([3, 1, 2, 4, 5]))

    events = check_kinetic_chain(timing, {})

    assert any(event.rule_name == "kinetic_chain_out_of_order" for event in events)


def test_check_kinetic_chain_flags_lag_outside_reference() -> None:
    timing = extract_kinetic_chain_timing(_chain_features([1, 2, 3, 4, 5]))
    reference_lags = {"hip_after_knee": RangeCheck(0.5, 1.0)}

    events = check_kinetic_chain(timing, reference_lags)

    assert len(events) == 1
    assert events[0].rule_name == "hip_after_knee"


def test_check_kinetic_chain_skips_when_timing_invalid() -> None:
    names = (
        "knee_angular_speed",
        "hip_angular_speed",
        "shoulder_angular_speed",
        "elbow_angular_speed",
        "wrist_speed",
    )
    values = np.full((4, len(names)), np.nan, dtype=np.float32)
    confidence = np.zeros_like(values)
    timestamps = np.arange(4, dtype=np.float32) / 30
    timing = extract_kinetic_chain_timing(GeometryFeatures(names, values, confidence, timestamps))

    assert check_kinetic_chain(timing, {"hip_after_knee": RangeCheck(0.0, 0.1)}) == []


def test_check_phase_features_flags_value_outside_reference() -> None:
    features = GeometryFeatures(
        ("elbow_height", "elbow_angle", "stance_width"),
        np.array([[-2.0, 100.0, 1.0]] * 5, dtype=np.float32),
        np.ones((5, 3), dtype=np.float32),
        np.arange(5, dtype=np.float32) / 30,
    )
    phases = PhaseBoundaries(
        preparation=(0, 1),
        backswing=(1, 3),
        forward_swing=(3, 4),
        contact_estimated=(4, 5),
        follow_through=(5, 5),
        contact_frame=4,
        valid=True,
    )
    reference = {"elbow_too_low_in_backswing": RangeCheck(-1.0, -0.2)}

    events = check_phase_features(features, phases, "forehand_lift", reference)

    assert len(events) == 1
    assert events[0].rule_name == "elbow_too_low_in_backswing"


def test_check_phase_features_skips_when_phases_invalid() -> None:
    features = GeometryFeatures(
        ("elbow_height",),
        np.zeros((3, 1), dtype=np.float32),
        np.ones((3, 1), dtype=np.float32),
        np.arange(3, dtype=np.float32) / 30,
    )
    empty = (0, 0)
    phases = PhaseBoundaries(empty, empty, empty, empty, empty, 0, False)

    events = check_phase_features(
        features, phases, "forehand_lift", {"elbow_too_low_in_backswing": RangeCheck(-1.0, -0.2)}
    )

    assert events == []


def test_view_incompatible_rule_returns_insufficient_data() -> None:
    features = GeometryFeatures(
        ("stance_width",), np.ones((8, 1), dtype=np.float32),
        np.ones((8, 1), dtype=np.float32), np.arange(8, dtype=np.float32) / 30,
    )
    phases = PhaseBoundaries((0, 5), (5, 6), (6, 7), (7, 8), (8, 8), 7, True)
    rule = RuleDefinition(
        "stance_too_narrow", "stance_width", "preparation", "lower",
        RangeCheck(0.5, 0.8, direction="lower"),
        compatible_views=("front",),
    )

    result = evaluate_rule_definitions(features, phases, [rule], view="side")[0]

    assert result.status == "insufficient_data"
    assert result.reason == "incompatible_camera_view"


def test_low_contact_confidence_downgrades_contact_result_to_review() -> None:
    features = GeometryFeatures(
        ("elbow_angle",), np.full((8, 1), 120.0, dtype=np.float32),
        np.ones((8, 1), dtype=np.float32), np.arange(8, dtype=np.float32) / 30,
    )
    phases = PhaseBoundaries(
        (0, 1), (1, 2), (2, 3), (3, 8), (8, 8), 5, True,
        contact_confidence=0.3,
    )
    rule = RuleDefinition(
        "arm_not_extended", "elbow_angle", "contact_estimated", "lower",
        RangeCheck(90.0, 150.0, direction="lower"),
    )

    result = evaluate_rule_definitions(features, phases, [rule], view="front")[0]

    assert result.status == "review"
    assert result.reason == "low_contact_confidence"


def test_rule_requires_most_phase_frames_to_be_valid() -> None:
    values = np.array([[120.0], [120.0], [120.0]] + [[np.nan]] * 7, dtype=np.float32)
    features = GeometryFeatures(
        ("elbow_angle",), values, np.isfinite(values).astype(np.float32),
        np.arange(10, dtype=np.float32) / 30,
    )
    phases = PhaseBoundaries((0, 1), (0, 10), (0, 1), (0, 1), (0, 1), 0, True)
    rule = RuleDefinition(
        "elbow", "elbow_angle", "backswing", "two_sided",
        RangeCheck(90.0, 150.0), min_valid_frames=3, min_valid_ratio=0.6,
    )

    result = evaluate_rule_definitions(features, phases, [rule], view="front")[0]

    assert result.status == "insufficient_data"
    assert result.valid_frames == 3
