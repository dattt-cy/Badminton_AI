import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from ai_classifier.biomechanics import GeometryFeatures, StrokeGeometrySummary
from ai_classifier.error_detection import RangeCheck, RuleDefinition, RuleResult
from ai_classifier.pose import PoseSequence


SCRIPT = Path(__file__).parents[4] / "scripts/evaluation/check_technique_rules.py"
SPEC = importlib.util.spec_from_file_location("check_technique_rules", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def make_sequence(projected_ratio: float, *, frames: int = 30, fps: float = 30.0):
    keypoints = np.zeros((frames, 17, 3), dtype=np.float32)
    keypoints[..., 2] = 1.0
    # Torso length is 100 px. Shoulder width therefore directly controls the
    # projected shoulder/torso ratio used by view estimation.
    keypoints[:, 5, :2] = (-50.0 * projected_ratio, 0.0)
    keypoints[:, 6, :2] = (50.0 * projected_ratio, 0.0)
    keypoints[:, 11, :2] = (-10.0, 100.0)
    keypoints[:, 12, :2] = (10.0, 100.0)
    return PoseSequence(keypoints, fps, 640, 480)


def test_projected_view_has_an_oblique_uncertainty_band() -> None:
    assert MODULE.estimate_view_with_ratio(make_sequence(0.20))[0] == "side"
    assert MODULE.estimate_view_with_ratio(make_sequence(0.37))[0] == "oblique"
    assert MODULE.estimate_view_with_ratio(make_sequence(0.60))[0] == "front"


def test_auto_uses_full_short_single_stroke_clip() -> None:
    assert MODULE.should_use_full_sequence(make_sequence(0.6, frames=120), 1)
    assert not MODULE.should_use_full_sequence(make_sequence(0.6, frames=120), 2)
    assert not MODULE.should_use_full_sequence(make_sequence(0.6, frames=180), 1)


def evaluate_for_projected_orientation(projected_ratio: float, *, reliable: bool = True):
    sequence = make_sequence(projected_ratio)
    values = np.full((30, 1), 120.0, dtype=np.float32)
    features = GeometryFeatures(
        ("elbow_angle",), values, np.ones_like(values),
        np.arange(30, dtype=np.float32) / 30,
    )
    phases = SimpleNamespace(**{name: (0, 30) for name in MODULE.PHASE_NAMES})
    phases.valid = True
    phases.invalid_reason = None
    phases.contact_confidence = 1.0
    rule = RuleDefinition(
        "elbow", "elbow_angle", "preparation", "two_sided",
        RangeCheck(90.0, 150.0),
    )
    return MODULE.evaluate_rules_fixed_view(
        features, phases, sequence, [rule], "front",
        camera_view_reliable=reliable,
    )[0][0]


def test_fixed_front_reference_rejects_side_on_phase() -> None:
    result = evaluate_for_projected_orientation(0.20)

    assert result.status == "insufficient_data"
    assert result.reason == "out_of_plane_rotation"


def test_fixed_reference_marks_oblique_phase_for_review() -> None:
    result = evaluate_for_projected_orientation(0.37)

    assert result.status == "review"
    assert result.reason == "oblique_body_orientation"


def test_prefer_aligned_rule_reviews_orientation_mismatch() -> None:
    sequence = make_sequence(0.20)
    values = np.full((30, 1), 120.0, dtype=np.float32)
    features = GeometryFeatures(
        ("elbow_angle",), values, np.ones_like(values),
        np.arange(30, dtype=np.float32) / 30,
    )
    phases = SimpleNamespace(**{name: (0, 30) for name in MODULE.PHASE_NAMES})
    phases.valid = True
    phases.invalid_reason = None
    phases.contact_confidence = 1.0
    rule = RuleDefinition(
        "elbow", "elbow_angle", "preparation", "two_sided",
        RangeCheck(90.0, 150.0), orientation_requirement="prefer_aligned",
    )

    result = MODULE.evaluate_rules_fixed_view(
        features, phases, sequence, [rule], "front"
    )[0][0]

    assert result.status == "review"
    assert result.reason == "body_orientation_mismatch"


def test_unreliable_camera_view_is_insufficient() -> None:
    result = evaluate_for_projected_orientation(0.60, reliable=False)

    assert result.status == "insufficient_data"
    assert result.reason == "ambiguous_camera_view"


def test_rejected_2d3d_feature_is_not_scored() -> None:
    sequence = make_sequence(0.60)
    values = np.full((30, 1), 120.0, dtype=np.float32)
    features = GeometryFeatures(
        ("elbow_angle",), values, np.ones_like(values),
        np.arange(30, dtype=np.float32) / 30,
    )
    phases = SimpleNamespace(**{name: (0, 30) for name in MODULE.PHASE_NAMES})
    phases.valid = True
    phases.invalid_reason = None
    phases.contact_confidence = 1.0
    rule = RuleDefinition(
        "elbow", "elbow_angle", "preparation", "two_sided",
        RangeCheck(90.0, 150.0), orientation_requirement="any",
    )
    validation = {
        "techniques": {"forehand_clear": {"views": {"front": {"features": {
            "elbow_angle": {"status": "rejected", "mae": 30.0}
        }}}}}
    }

    result, contexts, _ = MODULE.evaluate_rules_fixed_view(
        features, phases, sequence, [rule], "front",
        feature_validation=validation, technique="forehand_clear",
    )

    assert result[0].status == "insufficient_data"
    assert result[0].reason == "feature_not_validated_2d_against_3d"
    assert contexts[0]["feature_validation"]["mae"] == 30.0


def test_review_2d3d_feature_forces_review_status() -> None:
    sequence = make_sequence(0.60)
    values = np.full((30, 1), 120.0, dtype=np.float32)
    features = GeometryFeatures(
        ("torso_lean",), values, np.ones_like(values),
        np.arange(30, dtype=np.float32) / 30,
    )
    phases = SimpleNamespace(**{name: (0, 30) for name in MODULE.PHASE_NAMES})
    phases.valid = True
    phases.invalid_reason = None
    phases.contact_confidence = 1.0
    rule = RuleDefinition(
        "torso", "torso_lean", "preparation", "two_sided",
        RangeCheck(90.0, 150.0), orientation_requirement="any",
    )
    validation = {
        "techniques": {"forehand_clear": {"views": {"front": {"features": {
            "torso_lean": {"status": "review", "mae": 10.0}
        }}}}}
    }

    result, _, _ = MODULE.evaluate_rules_fixed_view(
        features, phases, sequence, [rule], "front",
        feature_validation=validation, technique="forehand_clear",
    )

    assert result[0].status == "review"
    assert result[0].reason == "feature_requires_2d_3d_review"


def test_relative_2d_validated_indicator_can_pass_without_hard_deviation():
    sequence = make_sequence(0.2)
    values = np.column_stack([
        np.linspace(70, 150, 30),
        np.ones(30), np.linspace(-0.5, 1.0, 30), np.ones(30),
    ]).astype(np.float32)
    features = GeometryFeatures(
        ("elbow_angle", "wrist_shoulder_distance", "wrist_height", "wrist_speed"),
        values, np.ones_like(values), np.arange(30, dtype=np.float32) / 30,
    )
    phases = SimpleNamespace(
        preparation=(0, 5), backswing=(5, 10), forward_swing=(10, 20),
        contact_estimated=(20, 25), follow_through=(25, 30), valid=True,
    )
    summary = StrokeGeometrySummary(50.0, 1.0, 1.5, 0.1, {})
    registry = {"techniques": {"forehand_clear": {"views": {"side": {"rules": {
        "elbow_extension_delta": {
            "status": "validated", "threshold": 20.0, "direction": "higher",
            "test_metrics": {"balanced_accuracy": 0.7},
        }
    }}}}}}

    result = MODULE.evaluate_relative_2d_indicators(
        sequence, phases, summary, "forehand_clear", "side", registry
    )

    assert result[0]["status"] == "pass"
    assert result[0]["observed"] == 50.0


def test_practical_feedback_has_no_score_and_does_not_praise_review_feature():
    rule = RuleDefinition(
        "reach", "wrist_shoulder_distance", "contact_estimated", "lower",
        RangeCheck(0.5, 1.0, direction="lower"),
    )
    result = RuleResult("reach", "review", 0.8, 0.5, 1.0, 5)
    context = {
        "rules": {"side": rule},
        "feature_validation": {"status": "review", "mae": 0.2},
    }

    feedback = MODULE._practical_user_feedback([result], [context], [])

    assert feedback["score"] is None
    assert feedback["good_signals"] == []
    assert feedback["overall"] == "review_available"
    assert feedback["observations_for_review"][0]["observed"] == 0.8


def test_summary_comparison_uses_view_specific_reference_ranges() -> None:
    summary = StrokeGeometrySummary(
        elbow_extension_delta=60.0,
        wrist_path_length=4.0,
        wrist_vertical_excursion=1.2,
        balance_offset_preparation=0.1,
        phase_duration_ratios={"preparation": 0.2},
    )
    document = {
        "stroke_summary_ranges": {
            "elbow_extension_delta": {"low": 40.0, "high": 70.0, "sample_count": 12},
            "phase_duration_ratio.preparation": {
                "low": 0.1, "high": 0.3, "sample_count": 12,
            },
        }
    }

    comparisons = {item["feature"]: item for item in MODULE._compare_summary(summary, document)}

    assert comparisons["elbow_extension_delta"]["status"] == "within_reference"
    assert comparisons["phase_duration_ratio.preparation"]["status"] == "within_reference"
    assert comparisons["wrist_path_length"]["status"] == "unavailable"


def test_summary_elbow_delta_is_not_judged_across_out_of_plane_phase() -> None:
    summary = StrokeGeometrySummary(60.0, 4.0, 1.2, 0.1, {"preparation": 0.2})
    document = {
        "stroke_summary_ranges": {
            "elbow_extension_delta": {"low": 40.0, "high": 70.0, "sample_count": 12}
        }
    }
    orientations = {
        name: {"view": "side", "projected_ratio": 0.2}
        for name in MODULE.PHASE_NAMES
    }
    orientations["backswing"]["view"] = "front"

    comparison = MODULE._compare_summary(
        summary, document, orientations, "side"
    )[0]

    assert comparison["status"] == "unavailable"
    assert comparison["reason"] == "out_of_plane_rotation"
