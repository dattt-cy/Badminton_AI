import numpy as np

from ai_classifier.pose import PoseSequence
from ai_classifier.preprocessing import KeypointSmoother


def make_sequence(x_values: list[float], confidences: list[float]) -> PoseSequence:
    keypoints = np.zeros((len(x_values), 17, 3), dtype=np.float32)
    keypoints[:, :, 0] = np.asarray(x_values)[:, None]
    keypoints[:, :, 1] = 50.0
    keypoints[:, :, 2] = np.asarray(confidences)[:, None]
    return PoseSequence(keypoints, 60.0, 1080, 1920)


def test_smoothing_reduces_frame_to_frame_jitter() -> None:
    sequence = make_sequence(
        [10, 13, 9, 14, 11, 15, 13],
        [0.9] * 7,
    )

    smoothed = KeypointSmoother(alpha=0.35).smooth(sequence)

    raw_changes = np.diff(sequence.keypoints[:, 0, 0])
    smooth_changes = np.diff(smoothed.keypoints[:, 0, 0])
    assert smooth_changes.std() < raw_changes.std()


def test_smoothing_interpolates_only_short_gaps() -> None:
    sequence = make_sequence(
        [10, 0, 30, 0, 0, 0, 50],
        [0.9, 0.0, 0.9, 0.0, 0.0, 0.0, 0.9],
    )

    smoothed = KeypointSmoother(alpha=1.0, max_gap=1).smooth(sequence)

    assert smoothed.keypoints[1, 0, 0] == 20.0
    assert np.all(smoothed.keypoints[3:6, 0] == 0.0)

