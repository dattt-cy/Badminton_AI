"""Add quality-screened legacy forehand-clear poses to the train split."""

from __future__ import annotations

import argparse
import csv
import shutil
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("quality_csv", type=Path)
    parser.add_argument("dataset_root", type=Path)
    args = parser.parse_args()
    output = args.dataset_root / "forehand_clear" / "front" / "LegacyClearTrain"
    output.mkdir(parents=True, exist_ok=True)
    with args.quality_csv.open(newline="", encoding="utf-8-sig") as source:
        rows = list(csv.DictReader(source))
    selected = []
    for row in rows:
        if row["suitable_for_reference"].lower() != "true":
            continue
        source_pose = Path(row["pose"])
        destination = output / f"legacy_clear_{source_pose.name}"
        shutil.copy2(source_pose, destination)
        selected.append(destination.name)
    (output / "selected.txt").write_text("\n".join(selected) + "\n", encoding="utf-8")
    print(f"Added {len(selected)} legacy forehand-clear poses to {output}")


if __name__ == "__main__":
    main()
