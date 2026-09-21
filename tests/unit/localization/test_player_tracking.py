import numpy as np

from ai_classifier.localization.player_tracking import (
    PoseDetection,
    build_tracks,
    select_active_track,
)


def detection(frame, x, y, wrist_x):
    return PoseDetection(
        frame=frame,
        box=np.asarray([x - 10, y - 20, x + 10, y + 20], dtype=float),
        center=np.asarray([x, y], dtype=float),
        wrists=np.asarray([[wrist_x, y], [wrist_x + 2, y]], dtype=float),
    )


def test_active_track_prefers_moving_top_player_over_static_official():
    frames = [
        [detection(0, 50, 35, 45), detection(0, 85, 30, 85)],
        [detection(1, 52, 36, 65), detection(1, 85, 30, 85)],
        [detection(2, 54, 35, 42), detection(2, 85, 30, 85)],
    ]
    tracks = build_tracks(frames, max_center_distance=20)
    selected = select_active_track(tracks, player_side="top", width=100, height=100)

    assert selected is not None
    assert selected.detections[0].center[0] == 50
