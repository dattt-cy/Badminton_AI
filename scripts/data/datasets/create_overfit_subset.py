"""Create a tiny intentionally leaked dataset for pipeline sanity checks."""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--per-class", type=int, default=5)
    args = parser.parse_args()

    with args.source.open("rb") as source_file:
        dataset = pickle.load(source_file)

    train_ids = set(dataset["split"]["train"])
    selected = []
    for label in sorted({item["label"] for item in dataset["annotations"]}):
        candidates = [
            item
            for item in dataset["annotations"]
            if item["label"] == label and item["frame_dir"] in train_ids
        ]
        # Prefer separate source recordings so the sanity check is not five
        # adjacent cuts from the same original video.
        seen_sources: set[str] = set()
        class_items = []
        for item in candidates:
            if item["source_group"] in seen_sources:
                continue
            seen_sources.add(item["source_group"])
            class_items.append(item)
            if len(class_items) == args.per_class:
                break
        if len(class_items) != args.per_class:
            raise ValueError(
                f"Label {label} has only {len(class_items)} independent train sources"
            )
        selected.extend(class_items)

    identifiers = [item["frame_dir"] for item in selected]
    sanity_dataset = {
        # Intentional leakage: the goal is to prove that the complete training
        # and inference pipeline can memorize these exact ten samples.
        "split": {"train": identifiers, "val": identifiers, "test": identifiers},
        "annotations": selected,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("wb") as output_file:
        pickle.dump(sanity_dataset, output_file, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"Wrote {len(selected)} samples to {args.output}")
    for item in selected:
        print(item["label"], item["source_group"], item["frame_dir"])


if __name__ == "__main__":
    main()
