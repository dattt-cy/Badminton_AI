"""Unit tests for boundary_refinement.py."""

import numpy as np
import pytest

from ai_classifier.biomechanics import GeometryFeatures
from ai_classifier.pose import PoseSequence
from ai_classifier.segmentation import MotionProposal, RefinedStroke, refine_boundary


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_features(wrist_speed: list[float]) -> GeometryFeatures:
    """Build a minimal GeometryFeatures with only wrist_speed populated."""
    n = len(wrist_speed)
    values = np.zeros((n, 1), dtype=np.float32)
    values[:, 0] = wrist_speed
    confidence = np.ones((n, 1), dtype=np.float32)
    timestamps = np.arange(n, dtype=np.float32) / 30.0
    return GeometryFeatures(("wrist_speed",), values, confidence, timestamps)


def _make_proposal(start: int, end: int, peak: int, score: float = 1.0) -> MotionProposal:
    return MotionProposal(start, end, peak, score)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_refine_boundary_tightens_around_clear_peak() -> None:
    # Speed: flat low, then burst, then flat low again
    # proposal covers the whole sequence; refined should shrink to burst region
    speed = [0.0] * 10 + [0.0, 0.5, 1.0, 2.0, 3.0, 2.0, 1.0, 0.5, 0.0] + [0.0] * 10
    # peak is at index 14 (value 3.0)
    features = _make_features(speed)
    proposal = _make_proposal(0, len(speed) - 1, 14)

    stroke = refine_boundary(proposal, features, fps=30, stable_ratio=0.20, stable_window=2, fine_padding=0)

    # refined boundaries should be inside the low-speed zones
    assert stroke.peak_frame == 14
    assert stroke.start_frame > 0       # not at the very beginning
    assert stroke.end_frame < len(speed) - 1  # not at the very end
    assert stroke.start_frame <= stroke.peak_frame
    assert stroke.peak_frame <= stroke.end_frame


def test_refine_boundary_fallback_when_speed_never_drops() -> None:
    # Speed stays high throughout — never falls below stable_ratio * peak
    # peak=19 (value=20), stable_threshold=0.20*20=4.0
    # All values: 10..20 (minimum=10 > 4.0) so no stable region exists before peak
    speed = list(range(10, 21))  # 10..20, all above 20% of peak (20*0.20=4)
    features = _make_features(speed)
    proposal = _make_proposal(0, len(speed) - 1, len(speed) - 1)

    stroke = refine_boundary(proposal, features, fps=30, stable_ratio=0.20, stable_window=3, fine_padding=0)

    # Should fall back to proposal boundaries since no stable region found
    assert stroke.start_frame == proposal.start_frame
    assert stroke.end_frame == proposal.end_frame


def test_refine_boundary_fine_padding_respected() -> None:
    # Simple burst in the middle; padding=5 should add 5 frames each side
    speed = [0.0] * 5 + [0.0, 1.0, 3.0, 1.0, 0.0] + [0.0] * 5
    features = _make_features(speed)
    n = len(speed)
    proposal = _make_proposal(0, n - 1, 7)

    stroke_no_pad = refine_boundary(
        proposal, features, fps=30, stable_ratio=0.20, stable_window=1, fine_padding=0
    )
    stroke_padded = refine_boundary(
        proposal, features, fps=30, stable_ratio=0.20, stable_window=1, fine_padding=5
    )

    assert stroke_padded.start_frame <= stroke_no_pad.start_frame
    assert stroke_padded.end_frame >= stroke_no_pad.end_frame
    # But never out of video bounds
    assert stroke_padded.start_frame >= 0
    assert stroke_padded.end_frame < n


def test_refine_boundary_clip_returns_correct_frame_count() -> None:
    speed = [0.0] * 5 + [1.0, 2.0, 3.0, 2.0, 1.0] + [0.0] * 5
    n = len(speed)
    features = _make_features(speed)
    proposal = _make_proposal(0, n - 1, 7)

    keypoints = np.zeros((n, 17, 3), dtype=np.float32)
    keypoints[..., 2] = 0.9
    sequence = PoseSequence(keypoints, fps=30, frame_width=640, frame_height=480)

    stroke = refine_boundary(proposal, features, fps=30, fine_padding=0)
    clipped = stroke.clip(sequence)

    expected_frames = stroke.end_frame - stroke.start_frame + 1
    assert len(clipped.keypoints) == expected_frames


def test_refine_boundary_all_nan_speed_falls_back() -> None:
    speed = [float("nan")] * 20
    features = _make_features(speed)
    proposal = _make_proposal(3, 16, 10)

    stroke = refine_boundary(proposal, features, fps=30, fine_padding=0)

    # Falls back to proposal bounds (peak comes from proposal.peak_frame)
    assert stroke.start_frame == proposal.start_frame
    assert stroke.end_frame == proposal.end_frame
    assert stroke.peak_frame == proposal.peak_frame


def test_refined_stroke_with_technique() -> None:
    speed = [0.0] * 5 + [1.0, 3.0, 1.0] + [0.0] * 5
    features = _make_features(speed)
    proposal = _make_proposal(0, len(speed) - 1, 6)

    stroke = refine_boundary(proposal, features, fps=30)
    assert stroke.technique is None

    labelled = stroke.with_technique("forehand_lift")
    assert labelled.technique == "forehand_lift"
    # Original is unchanged (frozen dataclass)
    assert stroke.technique is None


def test_refined_stroke_local_frame() -> None:
    speed = [0.0] * 5 + [1.0, 3.0, 1.0] + [0.0] * 5
    features = _make_features(speed)
    proposal = _make_proposal(0, len(speed) - 1, 6)
    stroke = refine_boundary(proposal, features, fps=30, fine_padding=0)

    assert stroke.local_frame(stroke.start_frame) == 0
    assert stroke.local_frame(stroke.peak_frame) == stroke.peak_frame - stroke.start_frame


def test_refine_boundary_invalid_fps_raises() -> None:
    features = _make_features([1.0] * 10)
    proposal = _make_proposal(0, 9, 5)
    with pytest.raises(ValueError, match="fps"):
        refine_boundary(proposal, features, fps=0)


def test_refine_boundary_invalid_stable_ratio_raises() -> None:
    features = _make_features([1.0] * 10)
    proposal = _make_proposal(0, 9, 5)
    with pytest.raises(ValueError, match="stable_ratio"):
        refine_boundary(proposal, features, fps=30, stable_ratio=1.5)
