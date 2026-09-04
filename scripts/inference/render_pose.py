"""Render extracted pose keypoints over their source video."""

from __future__ import annotations

import argparse
from pathlib import Path

from ai_classifier.pose.visualization import render_pose_video


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video", type=Path, help="Input video path")
    parser.add_argument("pose", type=Path, help="Input pose .npz path")
    parser.add_argument("output", type=Path, help="Output preview .mp4 path")
    parser.add_argument(
        "--confidence",
        type=float,
        default=0.5,
        help="Minimum keypoint confidence to draw (default: 0.5)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not 0.0 <= args.confidence <= 1.0:
        raise SystemExit("--confidence must be between 0 and 1")
    frame_count = render_pose_video(
        args.video,
        args.pose,
        args.output,
        confidence=args.confidence,
    )
    print(f"Rendered {frame_count} frames to {args.output}")


if __name__ == "__main__":
    main()

