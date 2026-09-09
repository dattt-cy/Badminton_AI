"""Classify a badminton action from a video or extracted pose NPZ file."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import yaml

from ai_classifier.action_recognition import STGCNPPClassifier
from ai_classifier.pose import PoseSequence, YOLOv8PoseEstimator
from ai_classifier.preprocessing import KeypointSmoother


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="Input video or pose .npz file")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path(
            "models/checkpoints/action_recognition/best_top1_acc_3class.pth"
        ),
    )
    parser.add_argument(
        "--action-config",
        type=Path,
        default=Path("configs/action_recognition/stgcnpp_multisense_3class.py"),
    )
    parser.add_argument(
        "--pose-config",
        type=Path,
        default=Path("configs/pose/yolov8.yaml"),
    )
    parser.add_argument(
        "--target",
        choices=("any", "single", "far", "near"),
        default="far",
        help="Player to track when the input is a video",
    )
    parser.add_argument("--device", help="ST-GCN++ device, e.g. cuda:0 or cpu")
    parser.add_argument(
        "--no-motion-align",
        action="store_true",
        help="Classify the whole pose sequence instead of a swing-aligned window",
    )
    parser.add_argument(
        "--window-seconds",
        type=float,
        default=3.5,
        help="Motion-aligned window duration in seconds (default: 3.5)",
    )
    parser.add_argument(
        "--window-frames",
        type=int,
        help="Override the motion-aligned duration with an exact frame count",
    )
    parser.add_argument(
        "--peak-position",
        type=float,
        default=0.55,
        help="Target motion-peak position inside the aligned window (default: 0.55)",
    )
    parser.add_argument(
        "--handedness",
        choices=("left", "right"),
        default="right",
        help="Racket hand used to locate the swing",
    )
    parser.add_argument(
        "--min-action-confidence",
        type=float,
        default=0.75,
        help="Return an uncertain decision below this confidence (default: 0.75)",
    )
    parser.add_argument("--pose-output", type=Path, help="Optionally save pose NPZ")
    parser.add_argument("--json-output", type=Path, help="Optionally save prediction JSON")
    return parser.parse_args()


def load_pose(path: Path) -> PoseSequence:
    with np.load(path) as data:
        return PoseSequence(
            keypoints=np.asarray(data["keypoints"], dtype=np.float32),
            fps=float(data["fps"]),
            frame_width=int(data["frame_width"]),
            frame_height=int(data["frame_height"]),
        )


def extract_pose(video: Path, config_path: Path, target: str) -> PoseSequence:
    with config_path.open("r", encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file) or {}
    estimator = YOLOv8PoseEstimator(
        model_path=config.get("model", "yolov8n-pose.pt"),
        confidence=float(config.get("confidence", 0.25)),
        image_size=int(config.get("image_size", 640)),
        device=config.get("device"),
        tracking_distance_weight=float(config.get("tracking_distance_weight", 1.0)),
        max_tracking_distance=float(config.get("max_tracking_distance", 0.15)),
        max_tracking_gap=int(config.get("max_tracking_gap", 5)),
        target_region=target,
    )
    sequence = estimator.extract(video)
    smoothing = config.get("smoothing", {})
    if smoothing.get("enabled", True):
        sequence = KeypointSmoother(
            alpha=float(smoothing.get("alpha", 0.35)),
            min_confidence=float(smoothing.get("min_confidence", 0.3)),
            max_gap=int(smoothing.get("max_gap", 4)),
        ).smooth(sequence)
    return sequence


def main() -> None:
    args = parse_args()
    if not args.input.is_file():
        raise FileNotFoundError(f"Input not found: {args.input}")
    if not 0.0 <= args.min_action_confidence <= 1.0:
        raise ValueError("--min-action-confidence must be between 0 and 1")

    sequence = (
        load_pose(args.input)
        if args.input.suffix.lower() == ".npz"
        else extract_pose(args.input, args.pose_config, args.target)
    )
    if args.pose_output:
        sequence.save(args.pose_output)

    classifier = STGCNPPClassifier(
        args.action_config,
        args.checkpoint,
        device=args.device,
    )
    prediction = (
        classifier.predict(sequence)
        if args.no_motion_align
        else classifier.predict_aligned(
            sequence,
            window_seconds=args.window_seconds,
            window_frames=args.window_frames,
            peak_position=args.peak_position,
            handedness=args.handedness,
        )
    )
    result = asdict(prediction)
    result["decision"] = (
        prediction.label
        if prediction.confidence >= args.min_action_confidence
        else "uncertain"
    )
    result["accepted"] = prediction.confidence >= args.min_action_confidence
    result["min_action_confidence"] = args.min_action_confidence
    rendered = json.dumps(result, ensure_ascii=False, indent=2)
    print(rendered)
    if args.json_output:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
