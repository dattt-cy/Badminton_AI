import numpy as np

from ai_classifier.pose.visualization import draw_pose


def test_draw_pose_changes_pixels_for_visible_keypoints() -> None:
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    keypoints = np.zeros((17, 3), dtype=np.float32)
    keypoints[5] = (25, 25, 0.9)
    keypoints[6] = (75, 75, 0.9)

    rendered = draw_pose(frame, keypoints)

    assert rendered.any()


def test_draw_pose_ignores_low_confidence_keypoints() -> None:
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    keypoints = np.full((17, 3), (50, 50, 0.1), dtype=np.float32)

    rendered = draw_pose(frame, keypoints, confidence=0.5)

    assert not rendered.any()

