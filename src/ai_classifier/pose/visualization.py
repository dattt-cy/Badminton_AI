"""Utilities for rendering pose keypoints on videos."""

from __future__ import annotations

from pathlib import Path
import shutil
import subprocess

import cv2
import numpy as np
from numpy.typing import NDArray


# COCO-17 skeleton. Each pair contains indices into the keypoint array.
COCO_SKELETON = (
    (0, 1), (0, 2), (1, 3), (2, 4),
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
    (5, 11), (6, 12), (11, 12),
    (11, 13), (13, 15), (12, 14), (14, 16),
)


def draw_pose(
    frame: NDArray[np.uint8],
    keypoints: NDArray[np.float32],
    *,
    confidence: float = 0.5,
) -> NDArray[np.uint8]:
    """Draw one COCO-17 skeleton on a video frame in place."""
    visible = keypoints[:, 2] >= confidence

    for start, end in COCO_SKELETON:
        if visible[start] and visible[end]:
            point_a = tuple(np.rint(keypoints[start, :2]).astype(int))
            point_b = tuple(np.rint(keypoints[end, :2]).astype(int))
            cv2.line(frame, point_a, point_b, (0, 220, 255), 4, cv2.LINE_AA)

    for x, y, score in keypoints:
        if score >= confidence:
            cv2.circle(
                frame,
                (int(round(float(x))), int(round(float(y)))),
                6,
                (0, 80, 255),
                -1,
                cv2.LINE_AA,
            )
    return frame


def render_pose_video(
    video_path: str | Path,
    pose_path: str | Path,
    output_path: str | Path,
    *,
    confidence: float = 0.5,
) -> int:
    """Render a saved pose sequence over its source video."""
    video_path = Path(video_path)
    pose_path = Path(pose_path)
    output_path = Path(output_path)

    pose_data = np.load(pose_path)
    keypoints = np.asarray(pose_data["keypoints"], dtype=np.float32)

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    if fps <= 0:
        capture.release()
        raise ValueError(f"Invalid video FPS: {fps}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_name(
        f".{output_path.stem}.mp4v{output_path.suffix}"
    )
    writer = cv2.VideoWriter(
        str(temporary_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        capture.release()
        raise ValueError(f"Cannot create output video: {temporary_path}")

    rendered_frames = 0
    try:
        for frame_index, pose in enumerate(keypoints):
            success, frame = capture.read()
            if not success:
                break
            writer.write(draw_pose(frame, pose, confidence=confidence))
            rendered_frames = frame_index + 1
    finally:
        capture.release()
        writer.release()

    _convert_to_h264(temporary_path, output_path)
    return rendered_frames


def _convert_to_h264(source: Path, destination: Path) -> None:
    """Convert OpenCV MPEG-4 output to a broadly playable H.264 MP4."""
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        source.replace(destination)
        return

    try:
        subprocess.run(
            [
                ffmpeg,
                "-y",
                "-loglevel",
                "error",
                "-i",
                str(source),
                "-c:v",
                "libx264",
                "-preset",
                "fast",
                "-crf",
                "20",
                "-pix_fmt",
                "yuv420p",
                "-movflags",
                "+faststart",
                str(destination),
            ],
            check=True,
        )
    finally:
        source.unlink(missing_ok=True)
