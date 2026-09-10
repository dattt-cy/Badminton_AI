import importlib.util
from pathlib import Path

import numpy as np

from ai_classifier.error_detection import RuleResult
from ai_classifier.pose import PoseSequence


SCRIPT = Path(__file__).parents[2] / "scripts/evaluation/check_technique_rules.py"
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


def test_oblique_disagreement_is_review_not_deviation() -> None:
    front = RuleResult("elbow", "pass", 120.0, 90.0, 150.0, 5)
    side = RuleResult("elbow", "deviation", 120.0, 130.0, 160.0, 5)

    result = MODULE._combine_oblique_results(front, side)

    assert result.status == "review"
    assert result.reason == "view_disagreement"


def test_oblique_agreement_keeps_status() -> None:
    front = RuleResult("elbow", "deviation", 70.0, 90.0, 150.0, 5)
    side = RuleResult("elbow", "deviation", 70.0, 95.0, 155.0, 5)

    assert MODULE._combine_oblique_results(front, side).status == "deviation"
