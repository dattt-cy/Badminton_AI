"""Safe scalar correction from observable 2D geometry to a 3D estimate."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


CORRECTION_FEATURES = (
    "elbow_2d_median",
    "projected_ratio_median",
    "pose_confidence_median",
)
FRAME_CORRECTION_FEATURES = CORRECTION_FEATURES + (
    "shoulder_angle_2d",
    "wrist_shoulder_distance_2d",
    "elbow_torso_distance_2d",
    "wrist_height_2d",
    "torso_lean_2d",
)


@dataclass(frozen=True)
class ElbowCorrectionResult:
    status: str
    estimated_3d_angle: float | None
    uncertainty_degrees: float | None
    reason: str | None = None


def predict_elbow_3d(
    model: dict,
    *,
    elbow_2d: float,
    projected_ratio: float,
    pose_confidence: float,
    peak_confidence: float,
) -> ElbowCorrectionResult:
    """Apply an accepted correction model, otherwise fail closed."""
    measurements = {
        "elbow_2d_median": elbow_2d,
        "projected_ratio_median": projected_ratio,
        "pose_confidence_median": pose_confidence,
    }
    return predict_elbow_3d_from_features(
        model, measurements, peak_confidence=peak_confidence
    )


def predict_elbow_3d_from_features(
    model: dict, measurements: dict[str, float], *, peak_confidence: float
) -> ElbowCorrectionResult:
    """Apply a model using its declared observable 2D feature vector."""
    if model.get("status") != "ready":
        return ElbowCorrectionResult(
            "unavailable", None, None, "correction_model_not_validated"
        )
    feature_names = tuple(model.get("feature_names", CORRECTION_FEATURES))
    try:
        values = np.asarray([measurements[name] for name in feature_names], dtype=np.float64)
    except KeyError:
        return ElbowCorrectionResult(
            "insufficient_data", None, None, "missing_correction_feature"
        )
    if not np.isfinite(values).all():
        return ElbowCorrectionResult("insufficient_data", None, None, "non_finite_input")

    gates = model["quality_gates"]
    elbow_2d = float(measurements["elbow_2d_median"])
    projected_ratio = float(measurements["projected_ratio_median"])
    pose_confidence = float(measurements["pose_confidence_median"])
    if not gates["elbow_2d_min"] <= elbow_2d <= gates["elbow_2d_max"]:
        return ElbowCorrectionResult("insufficient_data", None, None, "implausible_2d_angle")
    if not (
        gates["projected_ratio_min"]
        <= projected_ratio
        <= gates.get("projected_ratio_max", float("inf"))
    ):
        return ElbowCorrectionResult(
            "insufficient_data", None, None, "body_orientation_mismatch"
        )
    if pose_confidence < gates["pose_confidence_min"]:
        return ElbowCorrectionResult("insufficient_data", None, None, "low_pose_confidence")
    if peak_confidence < gates["peak_confidence_min"]:
        return ElbowCorrectionResult("insufficient_data", None, None, "low_contact_confidence")

    mean = np.asarray(model["feature_mean"], dtype=np.float64)
    scale = np.asarray(model["feature_scale"], dtype=np.float64)
    coefficients = np.asarray(model["coefficients"], dtype=np.float64)
    normalized = (values - mean) / scale
    if model.get("basis", "linear") == "quadratic":
        normalized = np.concatenate([
            normalized,
            normalized ** 2,
            np.asarray([
                normalized[first] * normalized[second]
                for first in range(len(normalized))
                for second in range(first + 1, len(normalized))
            ]),
        ])
    estimate = float(model["intercept"] + np.dot(normalized, coefficients))
    estimate = float(np.clip(estimate, 0.0, 180.0))
    uncertainty = float(model["validation_metrics"]["absolute_error_p90"])
    return ElbowCorrectionResult("estimated", estimate, uncertainty)
