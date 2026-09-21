import numpy as np

from ai_classifier.localization.shuttle_events import direction_change_events


def test_detects_and_suppresses_trajectory_reversal() -> None:
    frames = np.arange(30)
    y = np.concatenate([np.arange(15), np.arange(15, 0, -1)]).astype(np.float32)
    points = np.column_stack([np.full(30, 50, dtype=np.float32), y * 5 + 100])

    events = direction_change_events(
        frames, points, width=100, height=300, window=3,
        min_score=0.005, nms_radius=5,
    )

    assert len(events) == 1
    assert 13 <= events[0].frame <= 16
