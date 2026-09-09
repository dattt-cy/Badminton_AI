"""Export manual pose sequences as a two-class, subject-separated PySKL PKL."""

from __future__ import annotations

import argparse
import csv
import pickle
from collections import Counter
from pathlib import Path

import numpy as np
import yaml


def pose_quality(keypoints: np.ndarray) -> tuple[float, float]:
    confidence = np.clip(keypoints[..., 2], 0.0, 1.0)
    detected_ratio = float((confidence.max(axis=1) > 0).mean())
    mean_confidence = float(confidence.mean())
    return detected_ratio, mean_confidence


def export_dataset(config: dict, output: Path | None = None) -> dict[str, int]:
    pose_root = Path(config["pose_output_root"])
    destination = output or Path(config["annotation_output"])
    classes = {str(name): int(label) for name, label in config["classes"].items()}
    source_splits = [
        str(value) for value in config.get("splits", ["train", "val"])
    ]
    quality = config.get("quality", {})
    min_frames = int(quality.get("min_frames", 1))
    min_detected_ratio = float(quality.get("min_detected_ratio", 0.0))
    min_mean_confidence = float(quality.get("min_mean_confidence", 0.0))

    split: dict[str, list[str]] = {"train": [], "val": [], "test": []}
    annotations: list[dict] = []
    manifest: list[dict] = []
    rejected: Counter[str] = Counter()

    for split_name in source_splits:
        if split_name not in split:
            raise ValueError(f"Unsupported split: {split_name}")
        for class_name, label in classes.items():
            folder = pose_root / split_name / class_name
            if not folder.is_dir():
                raise FileNotFoundError(f"Pose folder not found: {folder}")
            for path in sorted(folder.rglob("*.npz")):
                relative = path.relative_to(pose_root)
                if len(relative.parts) < 5:
                    rejected["invalid_path"] += 1
                    continue
                _, _, view, subject = relative.parts[:4]
                with np.load(path) as data:
                    keypoints = np.asarray(data["keypoints"], dtype=np.float32)
                    height = int(data["frame_height"])
                    width = int(data["frame_width"])
                if keypoints.ndim != 3 or keypoints.shape[1:] != (17, 3):
                    rejected["invalid_shape"] += 1
                    continue
                detected_ratio, mean_confidence = pose_quality(keypoints)
                if len(keypoints) < min_frames:
                    rejected["too_short"] += 1
                    continue
                if detected_ratio < min_detected_ratio:
                    rejected["low_detection"] += 1
                    continue
                if mean_confidence < min_mean_confidence:
                    rejected["low_confidence"] += 1
                    continue

                identifier = relative.with_suffix("").as_posix()
                split[split_name].append(identifier)
                annotations.append(
                    {
                        "frame_dir": identifier,
                        "total_frames": len(keypoints),
                        "img_shape": (height, width),
                        "original_shape": (height, width),
                        "label": label,
                        "keypoint": keypoints[None, ..., :2],
                        "keypoint_score": keypoints[None, ..., 2],
                    }
                )
                manifest.append(
                    {
                        "frame_dir": identifier,
                        "class": class_name,
                        "label": label,
                        "view": view,
                        "subject": subject,
                        "split": split_name,
                        "total_frames": len(keypoints),
                        "detected_ratio": f"{detected_ratio:.6f}",
                        "mean_confidence": f"{mean_confidence:.6f}",
                    }
                )

    if not annotations:
        raise RuntimeError("No pose sequences passed the configured quality filters")
    for split_name in source_splits:
        present = {
            item["label"]
            for item in annotations
            if item["frame_dir"] in split[split_name]
        }
        if present != set(classes.values()):
            raise RuntimeError(f"{split_name} is missing one or more classes")

    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("wb") as target:
        pickle.dump(
            {"split": split, "annotations": annotations},
            target,
            protocol=pickle.HIGHEST_PROTOCOL,
        )
    csv_path = destination.with_suffix(".csv")
    with csv_path.open("w", newline="", encoding="utf-8") as target:
        writer = csv.DictWriter(target, fieldnames=list(manifest[0]))
        writer.writeheader()
        writer.writerows(manifest)

    class_counts = Counter(item["class"] for item in manifest)
    split_counts = {name: len(items) for name, items in split.items()}
    print(f"Exported {len(annotations)} poses: {dict(class_counts)}")
    print(f"Splits: {split_counts}")
    print(f"Rejected: {dict(rejected)}")
    return {
        "exported": len(annotations),
        **{f"split_{name}": count for name, count in split_counts.items()},
        **{f"rejected_{name}": count for name, count in rejected.items()},
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/action_recognition/dataset_manual_clips_2class.yaml"),
    )
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8")) or {}
    export_dataset(config, args.output)


if __name__ == "__main__":
    main()
