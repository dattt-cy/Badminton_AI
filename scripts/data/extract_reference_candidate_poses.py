"""Extract and audit 2D poses for short reference-candidate videos."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import yaml

from ai_classifier.biomechanics import assess_geometry_quality, extract_geometry_features
from ai_classifier.pose import PoseSequence, YOLOv8PoseEstimator
from ai_classifier.preprocessing import KeypointSmoother


def robust_range(values: np.ndarray) -> float:
    finite = values[np.isfinite(values)]
    if not len(finite):
        return 0.0
    low, high = np.percentile(finite, [10, 90])
    return float(high - low)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--config", type=Path, default=Path("configs/pose/yolov8.yaml"))
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--report", type=Path)
    parser.add_argument(
        "--output-root", type=Path,
        help="Optional separate pose directory preserving paths relative to root.",
    )
    args = parser.parse_args()

    config = yaml.safe_load(args.config.read_text(encoding="utf-8")) or {}
    estimator = YOLOv8PoseEstimator(
        model_path=config.get("model", "yolov8n-pose.pt"),
        confidence=float(config.get("confidence", 0.25)),
        image_size=int(config.get("image_size", 640)),
        device=config.get("device"),
        target_region="single",
    )
    smoothing = config.get("smoothing", {})
    smoother = KeypointSmoother(
        alpha=float(smoothing.get("alpha", 0.35)),
        min_confidence=float(smoothing.get("min_confidence", 0.3)),
        max_gap=int(smoothing.get("max_gap", 4)),
    )
    videos = sorted(args.root.rglob("*.mp4"))
    if not videos:
        raise FileNotFoundError(f"No MP4 candidates found under {args.root}")

    rows = []
    for index, video in enumerate(videos, start=1):
        pose_path = (
            args.output_root / video.relative_to(args.root).with_suffix(".npz")
            if args.output_root else video.with_name(f"{video.stem}_pose.npz")
        )
        pose_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            if pose_path.exists() and not args.overwrite:
                with np.load(pose_path) as data:
                    sequence = PoseSequence(
                        np.asarray(data["keypoints"], dtype=np.float32), float(data["fps"]),
                        int(data["frame_width"]), int(data["frame_height"]),
                    )
            else:
                sequence = smoother.smooth(estimator.extract(video))
                sequence.save(pose_path)
        except Exception as exc:
            rows.append({
                "video": str(video), "pose": str(pose_path), "frame_count": 0,
                "valid_ratio": 0.0, "median_torso_scale_px": 0.0,
                "scale_cv": float("inf"), "wrist_speed_spike_ratio": float("inf"),
                "wrist_speed_range": 0.0, "wrist_excursion": 0.0,
                "arm_angle_range": 0.0,
                "phases_valid": False, "suitable_for_reference": False,
                "error": str(exc),
            })
            print(f"[{index}/{len(videos)}] FAILED {video.name}: {exc}")
            continue
        quality = assess_geometry_quality(sequence)
        features = extract_geometry_features(sequence)
        wrist_excursion = max(
            robust_range(features.column("wrist_height")),
            robust_range(features.column("wrist_shoulder_distance")),
        )
        arm_angle_range = max(
            robust_range(features.column("elbow_angle")),
            robust_range(features.column("shoulder_angle")),
        )
        rows.append({
            "video": str(video), "pose": str(pose_path),
            "frame_count": len(sequence.keypoints), "valid_ratio": quality.valid_ratio,
            "median_torso_scale_px": quality.scale_px, "scale_cv": quality.scale_cv,
            "wrist_speed_spike_ratio": quality.speed_spike_ratio,
            "wrist_speed_range": robust_range(features.column("wrist_speed")),
            "wrist_excursion": wrist_excursion,
            "arm_angle_range": arm_angle_range,
            "phases_valid": quality.phases_valid,
            "suitable_for_reference": quality.suitable_for_reference,
            "error": "",
        })
        print(
            f"[{index}/{len(videos)}] {video.name}: valid={quality.valid_ratio:.1%} "
            f"scale={quality.scale_px:.1f}px phases={quality.phases_valid} "
            f"reference={quality.suitable_for_reference}"
        )

    report = args.report or args.root / "pose_quality.csv"
    with report.open("w", newline="", encoding="utf-8-sig") as output:
        writer = csv.DictWriter(output, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    usable = sum(row["suitable_for_reference"] for row in rows)
    print(f"Usable candidates: {usable}/{len(rows)}; report: {report}")


if __name__ == "__main__":
    main()
