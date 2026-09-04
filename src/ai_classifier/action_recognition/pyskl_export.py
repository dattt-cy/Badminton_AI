"""Export extracted pose sequences to the PySKL PoseDataset format."""

from __future__ import annotations

import pickle
import re
from pathlib import Path

import numpy as np


def build_pyskl_dataset(
    pose_root: str | Path,
    output_path: str | Path,
    classes: dict[str, int],
    *,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    seed: int = 42,
    min_detected_ratio: float = 0.8,
    min_mean_confidence: float = 0.6,
) -> dict:
    """Build a stratified PySKL annotation dictionary from pose NPZ files."""
    if train_ratio <= 0 or val_ratio < 0 or train_ratio + val_ratio >= 1:
        raise ValueError("split ratios must leave a positive test split")

    pose_root = Path(pose_root)
    annotations: list[dict] = []
    del seed  # Kept in the API for backward-compatible configuration files.
    identifiers_by_stratum: dict[tuple[str, str], list[str]] = {}

    for action, label in sorted(classes.items(), key=lambda item: item[1]):
        action_files = sorted(
            (pose_root / action).rglob("*.npz"), key=_natural_path_key
        )
        for pose_path in action_files:
            data = np.load(pose_path)
            keypoints = np.asarray(data["keypoints"], dtype=np.float32)
            if keypoints.ndim != 3 or keypoints.shape[1:] != (17, 3):
                raise ValueError(f"Invalid keypoint shape in {pose_path}: {keypoints.shape}")

            scores = keypoints[..., 2]
            detected_ratio = float((scores.max(axis=1) > 0).mean())
            mean_confidence = float(scores.mean())
            if (
                detected_ratio < min_detected_ratio
                or mean_confidence < min_mean_confidence
            ):
                print(
                    f"Skipping low-quality pose {pose_path}: "
                    f"detected={detected_ratio:.1%}, confidence={mean_confidence:.3f}"
                )
                continue

            identifier = pose_path.relative_to(pose_root).with_suffix("").as_posix()
            relative_parts = pose_path.relative_to(pose_root).parts
            recording_type = relative_parts[1]
            identifiers_by_stratum.setdefault((action, recording_type), []).append(
                identifier
            )
            height = int(data["frame_height"])
            width = int(data["frame_width"])
            annotations.append({
                "frame_dir": identifier,
                "total_frames": int(keypoints.shape[0]),
                "img_shape": (height, width),
                "original_shape": (height, width),
                "label": int(label),
                "recording_type": recording_type,
                "source_group": (
                    "match_01"
                    if recording_type == "match"
                    else _single_player_source_group(identifier)
                ),
                "keypoint": keypoints[None, ..., :2],
                "keypoint_score": keypoints[None, ..., 2],
            })

    split = {"train": [], "val": [], "test": []}
    # Keep adjacent clips together in ordered blocks. Match clips still share
    # one source, so this is only an exploratory baseline split.
    for identifiers in identifiers_by_stratum.values():
        count = len(identifiers)
        train_end = round(count * train_ratio)
        val_end = train_end + round(count * val_ratio)
        split["train"].extend(identifiers[:train_end])
        split["val"].extend(identifiers[train_end:val_end])
        split["test"].extend(identifiers[val_end:])

    dataset = {"split": split, "annotations": annotations}
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("wb") as output_file:
        pickle.dump(dataset, output_file, protocol=pickle.HIGHEST_PROTOCOL)
    return dataset


def _single_player_source_group(identifier: str) -> str:
    """Group numbered clips cut from the same named single-player source."""
    path = Path(identifier)
    match = re.match(r"(.+)_\d+$", path.name)
    if match and not match.group(1).isdigit():
        return path.with_name(match.group(1)).as_posix()
    return identifier


def _natural_path_key(path: Path) -> list[int | str]:
    return [
        int(part) if part.isdigit() else part.lower()
        for part in re.split(r"(\d+)", path.as_posix())
    ]
