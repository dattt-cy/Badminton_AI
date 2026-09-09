"""Extract one tracked COCO-17 pose sequence from each manual video clip."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import yaml
from ultralytics import YOLO

from ai_classifier.pose import YOLOv8PoseEstimator
from ai_classifier.preprocessing import KeypointSmoother


VIDEO_EXTENSIONS = {".avi", ".mkv", ".mov", ".mp4"}


def discover_jobs(
    source_root: Path,
    output_root: Path,
    *,
    splits: list[str],
    classes: list[str],
    views: list[str],
    overwrite: bool,
) -> list[tuple[Path, Path, str]]:
    """Return videos and mirrored NPZ destinations in deterministic order."""
    jobs: list[tuple[Path, Path, str]] = []
    for split in splits:
        for action in classes:
            for view in views:
                folder = source_root / split / action / view
                if not folder.is_dir():
                    raise FileNotFoundError(f"Dataset folder not found: {folder}")
                for video_path in sorted(folder.rglob("*")):
                    if (
                        not video_path.is_file()
                        or video_path.suffix.lower() not in VIDEO_EXTENSIONS
                    ):
                        continue
                    relative = video_path.relative_to(source_root)
                    output_path = (output_root / relative).with_suffix(".npz")
                    if overwrite or not output_path.exists():
                        jobs.append((video_path, output_path, view))
    return jobs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/action_recognition/dataset_manual_clips_2class.yaml"),
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--limit", type=int)
    parser.add_argument(
        "--match",
        help="Only process source paths containing this POSIX-style substring.",
    )
    parser.add_argument(
        "--device",
        help="Ultralytics device override, for example cuda:0, 0, or cpu.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.limit is not None and args.limit < 0:
        raise ValueError("--limit cannot be negative")

    config = yaml.safe_load(args.config.read_text(encoding="utf-8")) or {}
    source_root = Path(config["dataset_root"])
    output_root = Path(config["pose_output_root"])
    classes = list(config["classes"])
    splits = [str(value) for value in config.get("splits", ["train", "val"])]
    view_config = config["views"]
    views = list(view_config)

    jobs = discover_jobs(
        source_root,
        output_root,
        splits=splits,
        classes=classes,
        views=views,
        overwrite=args.overwrite,
    )
    if args.match:
        jobs = [
            job
            for job in jobs
            if args.match in job[0].relative_to(source_root).as_posix()
        ]
    if args.limit is not None:
        jobs = jobs[: args.limit]

    print(f"Pose extraction jobs: {len(jobs)}", flush=True)
    if not jobs:
        return

    model = YOLO(str(config.get("model", "yolov8n-pose.pt")))
    smoothing = config.get("smoothing", {})
    smoother = KeypointSmoother(
        alpha=float(smoothing.get("alpha", 0.35)),
        min_confidence=float(smoothing.get("min_confidence", 0.30)),
        max_gap=int(smoothing.get("max_gap", 4)),
    )
    device = args.device if args.device is not None else config.get("device")

    failures: list[tuple[Path, str]] = []
    for index, (video_path, output_path, view) in enumerate(jobs, start=1):
        settings = view_config[view]
        try:
            estimator = YOLOv8PoseEstimator(
                model=model,
                confidence=float(config.get("confidence", 0.25)),
                image_size=int(settings.get("image_size", 640)),
                device=device,
                max_tracking_distance=float(
                    config.get("max_tracking_distance", 0.15)
                ),
                max_tracking_gap=int(config.get("max_tracking_gap", 5)),
                target_region=str(settings.get("target", "single")),
            )
            sequence = smoother.smooth(estimator.extract(video_path))
            sequence.save(output_path)
            confidence = np.clip(sequence.keypoints[..., 2], 0.0, 1.0)
            detected_ratio = float((confidence.max(axis=1) > 0).mean())
            mean_confidence = float(confidence.mean())
            print(
                f"[{index}/{len(jobs)}] {video_path}: "
                f"frames={len(sequence.keypoints)}, detected={detected_ratio:.1%}, "
                f"mean_conf={mean_confidence:.3f}",
                flush=True,
            )
        except Exception as exc:  # Continue so a single corrupt clip is auditable.
            failures.append((video_path, str(exc)))
            print(
                f"[{index}/{len(jobs)}] FAILED {video_path}: {exc}",
                flush=True,
            )

    print(
        f"Completed={len(jobs) - len(failures)}, failed={len(failures)}",
        flush=True,
    )
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
