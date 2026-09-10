import numpy as np
import pytest

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


def test_hampel_rejects_and_interpolates_isolated_localization_spike() -> None:
    sequence = make_sequence([10, 11, 12, 100, 14, 15, 16], [0.9] * 7)

    smoothed = KeypointSmoother(
        filter_type="none", reject_outliers=True, max_gap=1,
        hampel_window_seconds=0.12,
    ).smooth(sequence)

    assert smoothed.keypoints[3, 0, 0] == 13.0
    assert smoothed.keypoints[3, 0, 2] == pytest.approx(0.9)


def test_butterworth_reduces_high_frequency_noise_without_phase_shift() -> None:
    fps = 60.0
    time = np.arange(120, dtype=np.float32) / fps
    slow = 10.0 * np.sin(2 * np.pi * time)
    noisy = slow + 3.0 * np.sin(2 * np.pi * 15 * time)
    sequence = make_sequence(noisy.tolist(), [0.9] * len(noisy))

    smoothed = KeypointSmoother(
        filter_type="butterworth", cutoff_hz=6.0, filter_order=2
    ).smooth(sequence)

    filtered = smoothed.keypoints[:, 0, 0]
    assert np.std(filtered - slow) < np.std(noisy - slow) * 0.25
    # Compare one period: the two equal sinusoid peaks can differ by tiny
    # floating-point amounts across the full two-second signal.
    assert abs(int(np.argmax(filtered[:60])) - int(np.argmax(slow[:60]))) <= 1
