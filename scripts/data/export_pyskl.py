"""Export processed NPZ poses to a PySKL annotation pickle."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import yaml

from ai_classifier.action_recognition.pyskl_export import build_pyskl_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/action_recognition/dataset.yaml"),
    )
    args = parser.parse_args()
    with args.config.open("r", encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file)

    split_config = config["split"]
    quality_config = config.get("quality", {})
    dataset = build_pyskl_dataset(
        config["pose_output_root"],
        config["annotation_output"],
        config["classes"],
        train_ratio=float(split_config["train"]),
        val_ratio=float(split_config["val"]),
        seed=int(split_config["seed"]),
        min_detected_ratio=float(quality_config.get("min_detected_ratio", 0.8)),
        min_mean_confidence=float(quality_config.get("min_mean_confidence", 0.6)),
    )
    split = dataset["split"]
    split_by_identifier = {
        identifier: split_name
        for split_name, identifiers in split.items()
        for identifier in identifiers
    }
    manifest_path = Path(config["annotation_output"]).with_suffix(".csv")
    with manifest_path.open("w", newline="", encoding="utf-8") as manifest_file:
        writer = csv.DictWriter(
            manifest_file,
            fieldnames=(
                "frame_dir", "label", "recording_type", "source_group", "split"
            ),
        )
        writer.writeheader()
        for annotation in dataset["annotations"]:
            identifier = annotation["frame_dir"]
            writer.writerow({
                "frame_dir": identifier,
                "label": annotation["label"],
                "recording_type": annotation["recording_type"],
                "source_group": annotation["source_group"],
                "split": split_by_identifier[identifier],
            })
    source_splits: dict[str, set[str]] = {}
    for annotation in dataset["annotations"]:
        source_splits.setdefault(annotation["source_group"], set()).add(
            split_by_identifier[annotation["frame_dir"]]
        )
    leaking_sources = {
        source: splits for source, splits in source_splits.items() if len(splits) > 1
    }
    print(
        f"Exported {len(dataset['annotations'])} clips: "
        f"train={len(split['train'])}, val={len(split['val'])}, "
        f"test={len(split['test'])}"
    )
    print(f"Manifest: {manifest_path}")
    if leaking_sources:
        print(
            "WARNING: source groups span multiple splits; metrics are baseline-only: "
            + ", ".join(sorted(leaking_sources))
        )


if __name__ == "__main__":
    main()
