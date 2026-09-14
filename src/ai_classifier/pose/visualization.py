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

LEFT_LIMBS = frozenset({(5, 7), (7, 9), (11, 13), (13, 15)})
RIGHT_LIMBS = frozenset({(6, 8), (8, 10), (12, 14), (14, 16)})
ANGLE_JOINTS = (
    (5, 7, 9),
    (6, 8, 10),
    (11, 13, 15),
    (12, 14, 16),
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


def draw_skeleton_canvas(
    keypoints: NDArray[np.float32],
    *,
    width: int,
    height: int,
    confidence: float = 0.5,
    frame_index: int | None = None,
    fps: float | None = None,
) -> NDArray[np.uint8]:
    """Draw a presentation-focused skeleton on a dark analysis canvas."""
    canvas = np.full((height, width, 3), (13, 18, 17), dtype=np.uint8)
    grid_color = (26, 35, 32)
    grid_step = max(64, min(width, height) // 8)
    for x in range(0, width, grid_step):
        cv2.line(canvas, (x, 0), (x, height), grid_color, 1, cv2.LINE_AA)
    for y in range(0, height, grid_step):
        cv2.line(canvas, (0, y), (width, y), grid_color, 1, cv2.LINE_AA)

    visible = keypoints[:, 2] >= confidence
    line_width = max(5, round(min(width, height) / 135))
    joint_radius = max(7, round(min(width, height) / 105))
    left_color = (255, 220, 55)
    right_color = (145, 80, 255)
    center_color = (65, 245, 200)

    if all(visible[index] for index in (5, 6, 11, 12)):
        torso = np.rint(keypoints[[5, 6, 12, 11], :2]).astype(np.int32)
        overlay = canvas.copy()
        cv2.fillConvexPoly(overlay, torso, (48, 105, 89), cv2.LINE_AA)
        cv2.addWeighted(overlay, 0.28, canvas, 0.72, 0, canvas)

    # Add a neck so the head does not float above the torso as in raw COCO-17.
    if visible[0] and visible[5] and visible[6]:
        neck = np.mean(keypoints[[5, 6], :2], axis=0)
        head = tuple(np.rint(keypoints[0, :2]).astype(int))
        cv2.line(
            canvas,
            head,
            tuple(np.rint(neck).astype(int)),
            center_color,
            line_width,
            cv2.LINE_AA,
        )
        cv2.circle(
            canvas,
            head,
            joint_radius * 2,
            center_color,
            max(2, line_width // 2),
            cv2.LINE_AA,
        )

    for start, end in COCO_SKELETON[4:]:
        if not (visible[start] and visible[end]):
            continue
        edge = (start, end)
        color = (
            left_color if edge in LEFT_LIMBS
            else right_color if edge in RIGHT_LIMBS
            else center_color
        )
        point_a = tuple(np.rint(keypoints[start, :2]).astype(int))
        point_b = tuple(np.rint(keypoints[end, :2]).astype(int))
        cv2.line(canvas, point_a, point_b, color, line_width + 5, cv2.LINE_AA)
        cv2.line(canvas, point_a, point_b, (245, 255, 252), line_width, cv2.LINE_AA)

    # Facial points are noisy at video scale; keep a clean head marker instead.
    body_indices = (0, *range(5, 17))
    for index in body_indices:
        x, y, score = keypoints[index]
        if score < confidence:
            continue
        point = (int(round(float(x))), int(round(float(y))))
        color = (
            left_color if index in {5, 7, 9, 11, 13, 15}
            else right_color if index in {6, 8, 10, 12, 14, 16}
            else center_color
        )
        cv2.circle(canvas, point, joint_radius + 7, color, -1, cv2.LINE_AA)
        cv2.circle(canvas, point, joint_radius, (250, 255, 253), -1, cv2.LINE_AA)
        cv2.circle(canvas, point, max(2, joint_radius // 3), color, -1, cv2.LINE_AA)

    font_scale = max(0.48, min(width, height) / 1250)
    for start, vertex, end in ANGLE_JOINTS:
        if not (visible[start] and visible[vertex] and visible[end]):
            continue
        angle = _joint_angle(
            keypoints[start, :2], keypoints[vertex, :2], keypoints[end, :2]
        )
        x, y = np.rint(keypoints[vertex, :2]).astype(int)
        label = f"{angle:.0f} deg"
        offset = joint_radius * 2
        origin = (x + offset, y - offset)
        (text_width, text_height), _ = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_DUPLEX, font_scale, 1
        )
        cv2.rectangle(
            canvas,
            (origin[0] - 5, origin[1] - text_height - 5),
            (origin[0] + text_width + 5, origin[1] + 5),
            (27, 39, 35),
            -1,
            cv2.LINE_AA,
        )
        cv2.putText(
            canvas,
            label,
            origin,
            cv2.FONT_HERSHEY_DUPLEX,
            font_scale,
            (224, 238, 232),
            1,
            cv2.LINE_AA,
        )

    margin = max(18, round(min(width, height) * 0.03))
    cv2.putText(
        canvas,
        "AI MOTION  /  COCO-17",
        (margin, margin + 18),
        cv2.FONT_HERSHEY_DUPLEX,
        max(0.55, min(width, height) / 1050),
        (185, 215, 202),
        1,
        cv2.LINE_AA,
    )
    if frame_index is not None and fps and fps > 0:
        time_label = f"{frame_index / fps:05.2f}s"
        (text_width, _), _ = cv2.getTextSize(
            time_label, cv2.FONT_HERSHEY_DUPLEX, font_scale, 1
        )
        cv2.putText(
            canvas,
            time_label,
            (width - margin - text_width, margin + 18),
            cv2.FONT_HERSHEY_DUPLEX,
            font_scale,
            (185, 215, 202),
            1,
            cv2.LINE_AA,
        )
    return canvas


def _joint_angle(
    start: NDArray[np.float32],
    vertex: NDArray[np.float32],
    end: NDArray[np.float32],
) -> float:
    first = start - vertex
    second = end - vertex
    denominator = float(np.linalg.norm(first) * np.linalg.norm(second))
    if denominator <= 1e-6:
        return 0.0
    cosine = float(np.clip(np.dot(first, second) / denominator, -1.0, 1.0))
    return float(np.degrees(np.arccos(cosine)))


def _fit_pose_sequence(
    keypoints: NDArray[np.float32],
    *,
    width: int,
    height: int,
    confidence: float,
) -> NDArray[np.float32]:
    """Apply one stable crop transform so the athlete fills the canvas."""
    fitted = keypoints.copy()
    visible = fitted[..., 2] >= confidence
    coordinates = fitted[..., :2][visible]
    if len(coordinates) < 2:
        return fitted
    lower = np.percentile(coordinates, 1, axis=0)
    upper = np.percentile(coordinates, 99, axis=0)
    span = np.maximum(upper - lower, 1.0)
    center = (lower + upper) / 2.0
    scale = min(width * 0.72 / span[0], height * 0.78 / span[1])
    scale = max(1.0, float(scale))
    fitted[..., :2] = (
        (fitted[..., :2] - center) * scale
        + np.asarray((width / 2.0, height / 2.0), dtype=np.float32)
    )
    return fitted


def render_pose_video(
    video_path: str | Path,
    pose_path: str | Path,
    output_path: str | Path,
    *,
    confidence: float = 0.5,
    skeleton_only: bool = False,
) -> int:
    """Render a saved pose sequence over its source video or a blank canvas."""
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
    display_keypoints = (
        _fit_pose_sequence(
            keypoints,
            width=width,
            height=height,
            confidence=confidence,
        )
        if skeleton_only else keypoints
    )
    try:
        for frame_index, pose in enumerate(display_keypoints):
            success, frame = capture.read()
            if not success:
                break
            if skeleton_only:
                frame = draw_skeleton_canvas(
                    pose,
                    width=width,
                    height=height,
                    confidence=confidence,
                    frame_index=frame_index,
                    fps=fps,
                )
            else:
                frame = draw_pose(frame, pose, confidence=confidence)
            writer.write(frame)
            rendered_frames = frame_index + 1
    finally:
        capture.release()
        writer.release()

    _convert_to_h264(temporary_path, output_path)
    return rendered_frames


def render_skeleton_video(
    video_path: str | Path,
    pose_path: str | Path,
    output_path: str | Path,
    *,
    confidence: float = 0.5,
) -> int:
    """Render only the COCO-17 skeleton, preserving source size and timing."""
    return render_pose_video(
        video_path,
        pose_path,
        output_path,
        confidence=confidence,
        skeleton_only=True,
    )


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
