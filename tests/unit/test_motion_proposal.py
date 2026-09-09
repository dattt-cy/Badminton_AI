import numpy as np

from ai_classifier.biomechanics import GeometryFeatures
from ai_classifier.segmentation import MotionProposal, find_motion_proposals, merge_motion_proposals


def test_finds_padded_high_motion_interval() -> None:
    names = ("wrist_speed", "elbow_angular_speed")
    values = np.zeros((100, 2), dtype=np.float32)
    values[:, 0] = np.linspace(0, 0.2, 100)
    values[40:51, 0] = np.linspace(2, 5, 11)
    values[40:51, 1] = np.linspace(100, 300, 11)
    features = GeometryFeatures(
        names,
        values,
        np.ones_like(values),
        np.arange(100, dtype=np.float32) / 30,
    )

    proposals = find_motion_proposals(
        features,
        fps=30,
        percentile=85,
        padding_seconds=0.2,
    )

    assert len(proposals) == 1
    assert proposals[0].start_frame <= 40
    assert proposals[0].end_frame >= 50
    assert 40 <= proposals[0].peak_frame <= 50


def test_merges_motion_bursts_separated_by_short_pause() -> None:
    features = GeometryFeatures(
        ("wrist_speed",), np.zeros((90, 1), dtype=np.float32),
        np.ones((90, 1), dtype=np.float32), np.arange(90, dtype=np.float32) / 30,
    )
    proposals = [
        MotionProposal(0, 24, 8, 2.0),
        MotionProposal(53, 78, 65, 3.0),
    ]

    merged = merge_motion_proposals(proposals, features, fps=30)

    assert merged == [MotionProposal(0, 78, 65, 3.0, source_count=2)]


def test_does_not_merge_distinct_long_strokes() -> None:
    features = GeometryFeatures(
        ("wrist_speed",), np.zeros((200, 1), dtype=np.float32),
        np.ones((200, 1), dtype=np.float32), np.arange(200, dtype=np.float32) / 30,
    )
    proposals = [MotionProposal(0, 30, 10, 2.0), MotionProposal(80, 130, 100, 3.0)]

    assert len(merge_motion_proposals(proposals, features, fps=30)) == 2
