import numpy as np
import pytest

from ai_classifier.biomechanics import (
    GeometryFeatures,
    assess_geometry_quality,
    detect_stroke_phases,
    estimate_pose_scale_px,
    extract_geometry_features,
    extract_kinetic_chain_timing,
)
from ai_classifier.pose import PoseSequence


def test_extracts_right_elbow_angle_and_normalized_distance() -> None:
    keypoints = np.zeros((3, 17, 3), dtype=np.float32)
    keypoints[..., 2] = 0.9
    keypoints[:, 5, :2] = [0, 0]
    keypoints[:, 6, :2] = [2, 0]
    keypoints[:, 8, :2] = [2, 1]
    keypoints[:, 10, :2] = [3, 1]
    keypoints[:, 11, :2] = [0, 2]
    keypoints[:, 12, :2] = [2, 2]
    keypoints[:, 14, :2] = [2, 3]
    keypoints[:, 16, :2] = [2, 4]
    keypoints[:, 15, :2] = [0, 4]

    features = extract_geometry_features(PoseSequence(keypoints, 30, 100, 100))

    assert features.column("elbow_angle")[1] == pytest.approx(90)
    assert features.column("wrist_shoulder_distance")[1] == pytest.approx(np.sqrt(2) / 2)
    assert features.column("stance_width")[1] == pytest.approx(1.0)
    assert features.column("elbow_height")[1] == pytest.approx(-0.5)
    assert features.column("elbow_torso_distance")[1] == pytest.approx(0.5)


def test_invalid_joint_confidence_produces_nan() -> None:
    keypoints = np.ones((4, 17, 3), dtype=np.float32)
    keypoints[..., 2] = 0.9
    keypoints[:, 10, 2] = 0.1

    features = extract_geometry_features(PoseSequence(keypoints, 30, 100, 100))

    assert np.isnan(features.column("elbow_angle")).all()
    assert np.isnan(features.column("wrist_speed")).all()


def test_rejects_near_zero_projected_elbow_angle() -> None:
    keypoints = np.zeros((4, 17, 3), dtype=np.float32)
    keypoints[..., 2] = 0.9
    keypoints[:, 5, :2] = [0, 0]
    keypoints[:, 6, :2] = [2, 0]
    keypoints[:, 8, :2] = [3, 0]
    keypoints[:, 10, :2] = [1, 0]
    keypoints[:, 11, :2] = [0, 2]
    keypoints[:, 12, :2] = [2, 2]

    features = extract_geometry_features(PoseSequence(keypoints, 30, 100, 100))

    assert np.isnan(features.column("elbow_angle")).all()


def test_body_scale_falls_back_to_torso_length_when_shoulders_are_foreshortened() -> None:
    # Shoulders rotated near-edge-on (width 0.2) while torso length (~2.0) is
    # normal; using shoulder width directly would blow up normalized features.
    keypoints = np.zeros((1, 17, 3), dtype=np.float32)
    keypoints[..., 2] = 0.9
    keypoints[:, 5, :2] = [1.0, 0.0]
    keypoints[:, 6, :2] = [1.2, 0.0]
    keypoints[:, 11, :2] = [0.0, 2.0]
    keypoints[:, 12, :2] = [2.0, 2.0]
    keypoints[:, 15, :2] = [0.0, 4.0]
    keypoints[:, 16, :2] = [2.0, 4.0]

    features = extract_geometry_features(PoseSequence(keypoints, 30, 100, 100))

    assert features.column("stance_width")[0] == pytest.approx(0.9988, abs=1e-3)


def test_kinetic_chain_timing_detects_correct_order() -> None:
    names = (
        "knee_angular_speed",
        "hip_angular_speed",
        "shoulder_angular_speed",
        "elbow_angular_speed",
        "wrist_speed",
    )
    values = np.zeros((6, len(names)), dtype=np.float32)
    values[1, 0] = 5  # knee peak
    values[2, 1] = 5  # hip peak
    values[3, 2] = 5  # shoulder peak
    values[4, 3] = 5  # elbow peak
    values[5, 4] = 5  # wrist peak
    confidence = np.ones_like(values)
    timestamps = np.arange(6, dtype=np.float32) / 30
    features = GeometryFeatures(names, values, confidence, timestamps)

    timing = extract_kinetic_chain_timing(features)

    assert timing.valid
    assert timing.order_correct
    assert timing.lags["hip_after_knee"] == pytest.approx(1 / 30)
    assert timing.lags["wrist_after_elbow"] == pytest.approx(1 / 30)
    assert timing.transfer_time == pytest.approx(4 / 30)


def test_kinetic_chain_timing_flags_out_of_order_peaks() -> None:
    names = (
        "knee_angular_speed",
        "hip_angular_speed",
        "shoulder_angular_speed",
        "elbow_angular_speed",
        "wrist_speed",
    )
    values = np.zeros((6, len(names)), dtype=np.float32)
    values[3, 0] = 5  # knee peaks after hip: wrong order
    values[1, 1] = 5
    values[2, 2] = 5
    values[4, 3] = 5
    values[5, 4] = 5
    confidence = np.ones_like(values)
    timestamps = np.arange(6, dtype=np.float32) / 30
    features = GeometryFeatures(names, values, confidence, timestamps)

    timing = extract_kinetic_chain_timing(features)

    assert timing.valid
    assert not timing.order_correct


def test_kinetic_chain_timing_invalid_when_column_missing_data() -> None:
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
    features = GeometryFeatures(names, values, confidence, timestamps)

    timing = extract_kinetic_chain_timing(features)

    assert not timing.valid


def test_detect_stroke_phases_splits_prep_backswing_forward_swing_contact_follow_through() -> None:
    speed = np.array(
        [0.1, 0.1, 0.1, 0.1, 1.0, 1.2, 1.1, 0.05, 1.5, 3.0, 4.0, 4.5, 5.0, 2.0, 1.0, 0.3],
        dtype=np.float32,
    )
    features = GeometryFeatures(
        ("wrist_speed",),
        speed.reshape(-1, 1),
        np.ones((len(speed), 1), dtype=np.float32),
        np.arange(len(speed), dtype=np.float32) / 30,
    )

    phases = detect_stroke_phases(features)

    assert phases.valid
    assert phases.preparation == (0, 2)
    assert phases.backswing == (2, 4)
    assert phases.forward_swing == (4, 9)
    assert phases.contact_estimated == (9, 14)
    assert phases.follow_through == (14, 16)
    assert phases.contact_frame == 11
    assert phases.contact_confidence > 0.9
    assert phases.contact_candidates[0] == phases.contact_frame


def test_detect_stroke_phases_invalid_when_no_valid_speed() -> None:
    features = GeometryFeatures(
        ("wrist_speed",),
        np.full((5, 1), np.nan, dtype=np.float32),
        np.zeros((5, 1), dtype=np.float32),
        np.arange(5, dtype=np.float32) / 30,
    )

    phases = detect_stroke_phases(features)

    assert not phases.valid


def test_detect_stroke_phases_rejects_edge_peak_and_empty_phases() -> None:
    speed = np.array([10.0] + [0.1] * 11, dtype=np.float32)
    features = GeometryFeatures(
        ("wrist_speed",), speed[:, None], np.ones((12, 1), dtype=np.float32),
        np.arange(12, dtype=np.float32) / 30,
    )

    assert not detect_stroke_phases(features).valid


def test_contact_selector_prefers_coordinated_later_swing_over_early_wrist_burst() -> None:
    frame_count = 100
    wrist = np.full(frame_count, 0.1, dtype=np.float32)
    elbow = np.full(frame_count, 0.1, dtype=np.float32)
    wrist[25:30] = 10.0
    wrist[40:68] = 2.0
    wrist[68:73] = 9.0
    elbow[25:30] = 1.0
    elbow[40:68] = 2.0
    elbow[68:73] = 20.0
    features = GeometryFeatures(
        ("wrist_speed", "elbow_angular_speed"),
        np.column_stack((wrist, elbow)),
        np.ones((frame_count, 2), dtype=np.float32),
        np.arange(frame_count, dtype=np.float32) / 60,
    )

    phases = detect_stroke_phases(features)

    assert phases.valid
    assert 68 <= phases.contact_frame <= 72
    assert any(25 <= frame <= 30 for frame in phases.contact_candidates)


def test_estimate_pose_scale_px_returns_median_shoulder_width() -> None:
    keypoints = np.zeros((3, 17, 3), dtype=np.float32)
    keypoints[..., 2] = 0.9
    keypoints[:, 5, :2] = [0, 0]
    keypoints[:, 6, :2] = [100, 0]

    scale = estimate_pose_scale_px(PoseSequence(keypoints, 30, 200, 200))

    assert scale == pytest.approx(100.0)


def test_estimate_pose_scale_px_nan_when_no_frame_usable() -> None:
    keypoints = np.zeros((3, 17, 3), dtype=np.float32)

    scale = estimate_pose_scale_px(PoseSequence(keypoints, 30, 200, 200))

    assert np.isnan(scale)


def test_geometry_quality_rejects_incomplete_short_sequence() -> None:
    keypoints = np.zeros((5, 17, 3), dtype=np.float32)
    keypoints[..., 2] = 0.9
    quality = assess_geometry_quality(PoseSequence(keypoints, 30, 100, 100))

    assert not quality.phases_valid
    assert not quality.suitable_for_reference


def test_geometry_quality_scale_is_not_destabilized_by_shoulder_rotation() -> None:
    keypoints = np.zeros((20, 17, 3), dtype=np.float32)
    keypoints[..., 2] = 0.9
    widths = np.linspace(10, 80, len(keypoints))
    keypoints[:, 5, 0] = 100 - widths / 2
    keypoints[:, 6, 0] = 100 + widths / 2
    keypoints[:, [5, 6], 1] = 50
    keypoints[:, [11, 12], 0] = 100
    keypoints[:, [11, 12], 1] = 150
    keypoints[:, 8, :2] = [130, 100]
    keypoints[:, 10, :2] = [160, 100]
    keypoints[:, 14, :2] = [100, 200]
    keypoints[:, 16, :2] = [100, 250]
    keypoints[:, 15, :2] = [80, 250]

    quality = assess_geometry_quality(PoseSequence(keypoints, 30, 200, 300))

    assert quality.scale_cv == pytest.approx(0.0)
