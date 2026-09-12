from ai_classifier.biomechanics.correction import predict_elbow_3d


def _model(status="ready"):
    return {
        "status": status,
        "feature_mean": [100.0, 0.3, 0.8],
        "feature_scale": [10.0, 0.1, 0.1],
        "coefficients": [5.0, 1.0, 0.0],
        "intercept": 130.0,
        "quality_gates": {
            "elbow_2d_min": 30.0,
            "elbow_2d_max": 180.0,
            "projected_ratio_min": 0.1,
            "projected_ratio_max": 1.0,
            "pose_confidence_min": 0.3,
            "peak_confidence_min": 0.1,
        },
        "validation_metrics": {"absolute_error_p90": 12.0},
    }


def test_predict_elbow_3d_applies_validated_model():
    result = predict_elbow_3d(
        _model(), elbow_2d=110, projected_ratio=0.4,
        pose_confidence=0.8, peak_confidence=0.6,
    )
    assert result.status == "estimated"
    assert result.estimated_3d_angle == 136.0
    assert result.uncertainty_degrees == 12.0


def test_predict_elbow_3d_rejects_unvalidated_model():
    result = predict_elbow_3d(
        _model("rejected"), elbow_2d=110, projected_ratio=0.4,
        pose_confidence=0.8, peak_confidence=0.6,
    )
    assert result.status == "unavailable"
    assert result.reason == "correction_model_not_validated"


def test_predict_elbow_3d_fails_closed_on_collapsed_projection():
    result = predict_elbow_3d(
        _model(), elbow_2d=110, projected_ratio=0.05,
        pose_confidence=0.8, peak_confidence=0.6,
    )
    assert result.status == "insufficient_data"
    assert result.reason == "body_orientation_mismatch"


def test_predict_elbow_3d_supports_quadratic_basis():
    model = _model()
    model["basis"] = "quadratic"
    model["coefficients"] = [5.0, 1.0, 0.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0]

    result = predict_elbow_3d(
        model, elbow_2d=110, projected_ratio=0.4,
        pose_confidence=0.8, peak_confidence=0.6,
    )

    assert result.estimated_3d_angle == 138.0
