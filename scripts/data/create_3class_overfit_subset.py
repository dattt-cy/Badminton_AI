"""Create a balanced three-class sanity subset from a PySKL annotation file."""

from __future__ import annotations

import argparse
import pickle
from collections import defaultdict
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--per-class", type=int, default=3)
    args = parser.parse_args()
    with args.input.open("rb") as source:
        dataset = pickle.load(source)
    grouped = defaultdict(list)
    for item in dataset["annotations"]:
        if item["frame_dir"] in dataset["split"]["train"]:
            grouped[item["label"]].append(item)
    selected = [item for label in sorted(grouped) for item in grouped[label][:args.per_class]]
    identifiers = [item["frame_dir"] for item in selected]
    subset = {"split": {name: identifiers for name in ("train", "val", "test")},
              "annotations": selected}
    with args.output.open("wb") as target:
        pickle.dump(subset, target, pickle.HIGHEST_PROTOCOL)
    print(f"Wrote {len(selected)} samples ({args.per_class} per class) to {args.output}")


if __name__ == "__main__":
    main()
