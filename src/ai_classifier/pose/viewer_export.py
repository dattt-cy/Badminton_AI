"""Export compact, normalized pose data for the interactive web viewer."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from ai_classifier.pose.estimator import PoseSequence
from ai_classifier.preprocessing.keypoint_smoothing import KeypointSmoother


LEFT_JOINTS = frozenset({1, 3, 5, 7, 9, 11, 13, 15})
RIGHT_JOINTS = frozenset({2, 4, 6, 8, 10, 12, 14, 16})


def build_viewer_payload(
    keypoints: np.ndarray,
    *,
    fps: float,
    frame_width: int,
    frame_height: int,
) -> dict:
    sequence = PoseSequence(
        np.asarray(keypoints, dtype=np.float32), fps, frame_width, frame_height
    )
    smoothed = KeypointSmoother(
        alpha=0.5, min_confidence=0.2, max_gap=6
    ).smooth(sequence).keypoints
    visible = smoothed[..., 2] >= 0.2
    coordinates = smoothed[..., :2][visible]
    if len(coordinates) < 2:
        center = np.asarray((frame_width / 2.0, frame_height / 2.0))
        scale = float(max(frame_width, frame_height, 1))
    else:
        lower = np.percentile(coordinates, 1, axis=0)
        upper = np.percentile(coordinates, 99, axis=0)
        center = (lower + upper) / 2.0
        scale = float(max(*(upper - lower), 1.0))

    normalized = np.zeros((len(smoothed), 17, 4), dtype=np.float32)
    normalized[..., 0] = (smoothed[..., 0] - center[0]) / scale
    normalized[..., 1] = (center[1] - smoothed[..., 1]) / scale
    for joint in range(17):
        if joint in LEFT_JOINTS:
            normalized[:, joint, 2] = 0.055
        elif joint in RIGHT_JOINTS:
            normalized[:, joint, 2] = -0.055
        elif joint == 0:
            normalized[:, joint, 2] = 0.025
    normalized[..., 3] = smoothed[..., 2]

    return {
        "version": 1,
        "layout": "coco17",
        "fps": round(float(fps), 4),
        "frame_count": int(len(normalized)),
        "source_size": [int(frame_width), int(frame_height)],
        "frames": np.round(normalized.astype(np.float64), 5).tolist(),
    }


def export_pose_viewer(pose_path: str | Path, output_path: str | Path) -> Path:
    pose_path = Path(pose_path)
    output_path = Path(output_path)
    data = np.load(pose_path)
    payload = build_viewer_payload(
        data["keypoints"],
        fps=float(data["fps"]),
        frame_width=int(data["frame_width"]),
        frame_height=int(data["frame_height"]),
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, separators=(",", ":")), encoding="utf-8"
    )
    return output_path
