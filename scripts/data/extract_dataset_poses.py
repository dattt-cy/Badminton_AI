"""Extract target-player poses for every configured dataset video."""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml
from ultralytics import YOLO

from ai_classifier.pose import YOLOv8PoseEstimator
from ai_classifier.preprocessing import KeypointSmoother


VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/action_recognition/dataset.yaml"),
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--match",
        default=None,
        help="Only process videos whose POSIX-style relative path contains this text",
    )
    args = parser.parse_args()

    with args.config.open("r", encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file)

    source_root = Path(config["dataset_root"])
    output_root = Path(config["pose_output_root"])
    model = YOLO(config["model"])
    smoothing = config["smoothing"]
    smoother = KeypointSmoother(
        alpha=float(smoothing["alpha"]),
        min_confidence=float(smoothing["min_confidence"]),
        max_gap=int(smoothing["max_gap"]),
    )

    jobs: list[tuple[Path, Path, dict]] = []
    for action in config["classes"]:
        for recording_type, settings in config["recording_types"].items():
            folder = source_root / action / recording_type
            for video_path in sorted(folder.glob("*")):
                if video_path.suffix.lower() not in VIDEO_EXTENSIONS:
                    continue
                output_path = (
                    output_root / action / recording_type / f"{video_path.stem}.npz"
                )
                if args.overwrite or not output_path.exists():
                    jobs.append((video_path, output_path, settings))

    if args.match:
        jobs = [
            job for job in jobs
            if args.match in job[0].relative_to(source_root).as_posix()
        ]
    if args.limit is not None:
        jobs = jobs[:args.limit]
    print(f"Pose extraction jobs: {len(jobs)}")

    failures: list[tuple[Path, str]] = []
    for index, (video_path, output_path, settings) in enumerate(jobs, start=1):
        try:
            estimator = YOLOv8PoseEstimator(
                model=model,
                image_size=int(settings["image_size"]),
                device=config.get("device"),
                max_tracking_distance=float(config.get("max_tracking_distance", 0.15)),
                max_tracking_gap=int(config.get("max_tracking_gap", 5)),
                target_region=settings["target"],
            )
            sequence = smoother.smooth(estimator.extract(video_path))
            sequence.save(output_path)
            detected = float((sequence.keypoints[..., 2].max(axis=1) > 0).mean())
            print(
                f"[{index}/{len(jobs)}] {video_path}: "
                f"{len(sequence.keypoints)} frames, detected={detected:.1%}"
            )
        except Exception as exc:
            failures.append((video_path, str(exc)))
            print(f"[{index}/{len(jobs)}] FAILED {video_path}: {exc}")

    print(f"Completed={len(jobs) - len(failures)}, failed={len(failures)}")
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
