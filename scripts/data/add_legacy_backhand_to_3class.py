"""Add quality-screened legacy backhand poses to the three-class train split."""

from __future__ import annotations

import argparse
import re
import shutil
from pathlib import Path

import numpy as np

from ai_classifier.biomechanics import assess_geometry_quality, extract_geometry_features
from ai_classifier.pose import PoseSequence


def accepted(path: Path) -> bool:
    if path.name.startswith("expert01_"):
        return False
    if path.stem.startswith("005_backhand_dive5_"):
        number = int(path.stem.rsplit("_", 1)[1])
        if number >= 5:  # Receiving-shuttle demonstrations, not reference drives.
            return False
    with np.load(path) as data:
        sequence = PoseSequence(
            np.asarray(data["keypoints"], dtype=np.float32), float(data["fps"]),
            int(data["frame_width"]), int(data["frame_height"]),
        )
    if len(sequence.keypoints) < 30:
        return False
    quality = assess_geometry_quality(sequence)
    speed = extract_geometry_features(sequence).column("wrist_speed")
    if not np.isfinite(speed).any():
        return False
    peak = int(np.nanargmax(speed))
    return (
        quality.valid_ratio >= 0.90 and quality.scale_px >= 30
        and quality.scale_cv <= 0.25 and quality.speed_spike_ratio <= 3.0
        and quality.phases_valid and int(0.15 * len(speed)) <= peak < int(0.90 * len(speed))
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("dataset_root", type=Path)
    args = parser.parse_args()
    output = args.dataset_root / "backhand_drive" / "front" / "LegacyTrain"
    output.mkdir(parents=True, exist_ok=True)
    selected = []
    for path in sorted(args.source.glob("*.npz")):
        if accepted(path):
            destination = output / f"legacy_{path.name}"
            shutil.copy2(path, destination)
            selected.append(destination.name)
    (output / "selected.txt").write_text("\n".join(selected) + "\n", encoding="utf-8")
    print(f"Added {len(selected)} legacy backhand poses to {output}")


if __name__ == "__main__":
    main()
