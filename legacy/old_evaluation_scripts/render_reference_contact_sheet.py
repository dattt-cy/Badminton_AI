"""Render estimated-contact frames for visual reference-set auditing."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import cv2
import numpy as np

from ai_classifier.biomechanics import detect_stroke_phases, extract_geometry_features
from ai_classifier.pose import PoseSequence
from ai_classifier.pose.visualization import draw_pose


def load_pose(path: Path) -> PoseSequence:
    with np.load(path) as data:
        return PoseSequence(
            np.asarray(data["keypoints"], dtype=np.float32), float(data["fps"]),
            int(data["frame_width"]), int(data["frame_height"]),
        )


def render_tile(pose_path: Path, *, width: int = 320, height: int = 240) -> np.ndarray:
    sequence = load_pose(pose_path)
    phases = detect_stroke_phases(extract_geometry_features(sequence))
    frame_index = phases.contact_frame if phases.valid else len(sequence.keypoints) // 2
    video_path = pose_path.with_name(pose_path.name.removesuffix("_pose.npz") + ".mp4")
    capture = cv2.VideoCapture(str(video_path))
    try:
        capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ok, frame = capture.read()
    finally:
        capture.release()
    if not ok:
        frame = np.zeros((height, width, 3), dtype=np.uint8)
    pose = sequence.keypoints[min(frame_index, len(sequence.keypoints) - 1)]
    draw_pose(frame, pose, confidence=0.4)
    frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
    label = f"{pose_path.stem.removesuffix('_pose')} f={frame_index} c={phases.contact_confidence:.2f}"
    cv2.rectangle(frame, (0, height - 28), (width, height), (0, 0, 0), -1)
    cv2.putText(frame, label, (6, height - 9), cv2.FONT_HERSHEY_SIMPLEX, 0.42,
                (255, 255, 255), 1, cv2.LINE_AA)
    return frame


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pose_root", type=Path)
    parser.add_argument("clip_list", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--columns", type=int, default=4)
    args = parser.parse_args()
    entries = [
        line.strip() for line in args.clip_list.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    tiles = [render_tile(args.pose_root / entry) for entry in entries]
    if not tiles:
        raise ValueError("Clip list contains no pose entries")
    columns = min(args.columns, len(tiles))
    rows = math.ceil(len(tiles) / columns)
    blank = np.zeros_like(tiles[0])
    tiles.extend([blank] * (rows * columns - len(tiles)))
    sheet = np.vstack([
        np.hstack(tiles[row * columns:(row + 1) * columns]) for row in range(rows)
    ])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(args.output), sheet):
        raise RuntimeError(f"Could not write contact sheet: {args.output}")
    print(f"Rendered {len(entries)} contact frames to {args.output}")


if __name__ == "__main__":
    main()
