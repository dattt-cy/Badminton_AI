"""Export subject-separated MultiSense poses to a three-class PySKL dataset."""

from __future__ import annotations

import argparse
import csv
import pickle
from collections import Counter
from pathlib import Path

import numpy as np
import yaml


CLASSES = {"background": 0, "backhand_drive": 1, "forehand_clear": 2}
DEFAULT_SUBJECT_SPLIT = {
    "Sub11": "train",
    "Sub19": "train",
    "Sub20": "val",
    "Sub24": "test",
    "LegacyTrain": "train",
    "LegacyClearTrain": "train",
}


def load_subject_split(config_path: Path | None) -> dict[str, str]:
    """Load a subject-exclusive split while retaining the legacy default."""
    if config_path is None:
        return dict(DEFAULT_SUBJECT_SPLIT)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    configured = config.get("subject_split")
    if not isinstance(configured, dict):
        raise ValueError("split config must contain a subject_split mapping")
    result: dict[str, str] = {}
    for split_name in ("train", "val", "test"):
        subjects = configured.get(split_name)
        if not isinstance(subjects, list) or not subjects:
            raise ValueError(f"subject_split.{split_name} must be a non-empty list")
        for subject in subjects:
            subject = str(subject)
            if subject in result:
                raise ValueError(f"Subject appears in multiple splits: {subject}")
            result[subject] = split_name
    return result


def spread(paths: list[Path], limit: int) -> list[Path]:
    """Select deterministic samples distributed through a recording."""
    if len(paths) <= limit:
        return paths
    indices = np.linspace(0, len(paths) - 1, limit, dtype=int)
    return [paths[index] for index in indices]


def pose_quality(path: Path) -> tuple[float, float]:
    with np.load(path) as data:
        keypoints = np.asarray(data["keypoints"], dtype=np.float32)
    confidence = np.clip(keypoints[..., 2], 0.0, 1.0)
    detected_ratio = float((confidence.max(axis=1) > 0).mean())
    return detected_ratio, float(confidence.mean())


def selected_pose_paths(
    pose_root: Path,
    subject_split: dict[str, str],
    *,
    balance_per_subject: bool,
    max_per_subject_class: int | None,
    min_detected_ratio: float,
    min_mean_confidence: float,
) -> list[tuple[str, Path]]:
    grouped: dict[tuple[str, str], list[Path]] = {}
    for label_name in CLASSES:
        for path in sorted((pose_root / label_name).rglob("*.npz")):
            subject = path.parent.name
            if subject not in subject_split:
                continue
            detected_ratio, mean_confidence = pose_quality(path)
            if (
                detected_ratio >= min_detected_ratio
                and mean_confidence >= min_mean_confidence
            ):
                grouped.setdefault((subject, label_name), []).append(path)

    selected: list[tuple[str, Path]] = []
    for subject in subject_split:
        counts = {
            label_name: len(grouped.get((subject, label_name), []))
            for label_name in CLASSES
        }
        if not any(counts.values()):
            continue
        missing = [name for name, count in counts.items() if count == 0]
        if balance_per_subject and missing:
            raise RuntimeError(
                f"Cannot balance {subject}; missing classes: {', '.join(missing)}"
            )
        limit = min(counts.values()) if balance_per_subject else None
        if max_per_subject_class is not None:
            limit = (
                max_per_subject_class
                if limit is None
                else min(limit, max_per_subject_class)
            )
        for label_name in CLASSES:
            paths = grouped.get((subject, label_name), [])
            if limit is not None:
                paths = spread(paths, limit)
            selected.extend((label_name, path) for path in paths)
    return selected


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pose_root", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument(
        "--split-config", type=Path,
        help="YAML containing subject_split train/val/test lists.",
    )
    parser.add_argument(
        "--balance-per-subject", action="store_true",
        help="Use the same number of background/backhand/forehand poses per subject.",
    )
    parser.add_argument("--max-per-subject-class", type=int)
    parser.add_argument("--min-detected-ratio", type=float, default=0.0)
    parser.add_argument("--min-mean-confidence", type=float, default=0.0)
    args = parser.parse_args()
    if args.max_per_subject_class is not None and args.max_per_subject_class <= 0:
        raise ValueError("max-per-subject-class must be positive")
    for name in ("min_detected_ratio", "min_mean_confidence"):
        value = getattr(args, name)
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"{name.replace('_', '-')} must be between 0 and 1")

    subject_split = load_subject_split(args.split_config)
    paths = selected_pose_paths(
        args.pose_root,
        subject_split,
        balance_per_subject=args.balance_per_subject,
        max_per_subject_class=args.max_per_subject_class,
        min_detected_ratio=args.min_detected_ratio,
        min_mean_confidence=args.min_mean_confidence,
    )
    annotations, split, manifest = [], {"train": [], "val": [], "test": []}, []
    for label_name, path in sorted(paths, key=lambda item: item[1].as_posix()):
        label = CLASSES[label_name]
        subject = path.parent.name
        with np.load(path) as data:
            keypoints = np.asarray(data["keypoints"], dtype=np.float32)
            height, width = int(data["frame_height"]), int(data["frame_width"])
        identifier = path.relative_to(args.pose_root).with_suffix("").as_posix()
        split_name = subject_split[subject]
        split[split_name].append(identifier)
        annotations.append({"frame_dir": identifier, "total_frames": len(keypoints),
                            "img_shape": (height, width), "original_shape": (height, width),
                            "label": label, "keypoint": keypoints[None, ..., :2],
                            "keypoint_score": keypoints[None, ..., 2]})
        manifest.append({"frame_dir": identifier, "class": label_name, "label": label,
                         "subject": subject, "split": split_name})
    if not annotations:
        raise RuntimeError("No poses matched the configured subjects and quality thresholds")
    present = {item["label"] for item in annotations}
    if present != set(CLASSES.values()):
        raise RuntimeError("Dataset is missing one or more classes")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("wb") as target:
        pickle.dump({"split": split, "annotations": annotations}, target, pickle.HIGHEST_PROTOCOL)
    csv_path = args.output.with_suffix(".csv")
    with csv_path.open("w", newline="", encoding="utf-8") as target:
        writer = csv.DictWriter(target, fieldnames=list(manifest[0])); writer.writeheader(); writer.writerows(manifest)
    print(f"Exported {len(annotations)} poses: {Counter(item['class'] for item in manifest)}")
    print(f"Splits: {', '.join(f'{name}={len(items)}' for name, items in split.items())}")


if __name__ == "__main__":
    main()
