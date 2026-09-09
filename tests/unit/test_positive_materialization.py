import numpy as np

from ai_classifier.biomechanics import GeometryFeatures
from scripts.data.materialize_3class_positives import stroke_motion_rejection_reason


def _features(wrist_speed, wrist_height, wrist_distance, elbow, shoulder):
    values = np.column_stack(
        (wrist_speed, wrist_height, wrist_distance, elbow, shoulder)
    ).astype(np.float32)
    return GeometryFeatures(
        (
            "wrist_speed",
            "wrist_height",
            "wrist_shoulder_distance",
            "elbow_angle",
            "shoulder_angle",
        ),
        values,
        np.ones_like(values),
        np.arange(len(values), dtype=np.float32) / 30,
    )


def test_rejects_stationary_pose_jitter():
    x = np.linspace(0, 1, 60)
    features = _features(
        0.2 + 0.1 * x,
        0.05 * x,
        0.04 * x,
        90 + 2 * x,
        60 + x,
    )

    assert stroke_motion_rejection_reason(features) == "insufficient_wrist_speed"


def test_accepts_substantial_coordinated_arm_motion():
    x = np.linspace(0, 1, 60)
    swing = np.sin(np.pi * x)
    features = _features(
        0.2 + 8 * swing,
        -0.5 + 1.5 * x,
        0.3 + 0.8 * swing,
        50 + 90 * x,
        30 + 75 * x,
    )

    assert stroke_motion_rejection_reason(features) is None


def test_rejects_fast_keypoint_jitter_without_arm_excursion():
    x = np.linspace(0, 1, 60)
    features = _features(
        0.2 + 8 * np.sin(np.pi * x),
        0.05 * x,
        0.04 * x,
        90 + 2 * x,
        60 + x,
    )

    assert stroke_motion_rejection_reason(features) == "insufficient_wrist_excursion"
