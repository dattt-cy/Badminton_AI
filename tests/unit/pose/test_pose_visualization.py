import numpy as np

from ai_classifier.pose.visualization import (
    draw_pose,
    draw_skeleton_canvas,
    render_skeleton_video,
)


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


def test_draw_skeleton_canvas_adds_analysis_visuals() -> None:
    keypoints = np.zeros((17, 3), dtype=np.float32)
    keypoints[5] = (30, 30, 0.9)
    keypoints[6] = (70, 30, 0.9)
    keypoints[11] = (35, 70, 0.9)
    keypoints[12] = (65, 70, 0.9)

    rendered = draw_skeleton_canvas(keypoints, width=100, height=100)

    assert rendered.shape == (100, 100, 3)
    assert np.unique(rendered.reshape(-1, 3), axis=0).shape[0] > 4


def test_render_skeleton_video_uses_a_blank_background(
    tmp_path, monkeypatch
) -> None:
    source = tmp_path / "source.mp4"
    pose_path = tmp_path / "pose.npz"
    output = tmp_path / "skeleton.mp4"
    source.write_bytes(b"placeholder")
    keypoints = np.zeros((1, 17, 3), dtype=np.float32)
    keypoints[0, 5] = (25, 25, 0.9)
    keypoints[0, 6] = (75, 75, 0.9)
    np.savez(pose_path, keypoints=keypoints)

    source_frame = np.full((100, 100, 3), 180, dtype=np.uint8)

    class Capture:
        def isOpened(self):
            return True

        def get(self, prop):
            return {3: 100, 4: 100, 5: 30.0}[prop]

        def read(self):
            return True, source_frame.copy()

        def release(self):
            pass

    written = []

    class Writer:
        def isOpened(self):
            return True

        def write(self, frame):
            written.append(frame.copy())

        def release(self):
            pass

    monkeypatch.setattr("cv2.VideoCapture", lambda _: Capture())
    monkeypatch.setattr("cv2.VideoWriter", lambda *args: Writer())
    monkeypatch.setattr("cv2.VideoWriter_fourcc", lambda *args: 0)
    monkeypatch.setattr(
        "ai_classifier.pose.visualization._convert_to_h264",
        lambda source, destination: None,
    )

    assert render_skeleton_video(source, pose_path, output) == 1
    assert len(written) == 1
    assert np.all(written[0][0, 0] < 50)
    assert not np.all(written[0] == source_frame)
    assert written[0].any()
