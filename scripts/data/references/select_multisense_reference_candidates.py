"""Create balanced pose lists from a MultiSense pose-quality report."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path)
    parser.add_argument("pose_root", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--max-per-subject", type=int, default=8)
    parser.add_argument("--max-total", type=int, default=20)
    args = parser.parse_args()
    with args.report.open(newline="", encoding="utf-8-sig") as source:
        rows = [row for row in csv.DictReader(source) if row["suitable_for_reference"].lower() == "true"]

    groups: dict[tuple[str, str], dict[str, list[dict[str, str]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for row in rows:
        path = Path(row["pose"])
        relative = path.relative_to(args.pose_root)
        technique, view, subject = relative.parts[:3]
        groups[(technique, view)][subject].append(row)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for (technique, view), by_subject in sorted(groups.items()):
        selected = []
        for subject in sorted(by_subject):
            ranked = sorted(
                by_subject[subject],
                key=lambda row: (
                    -float(row["valid_ratio"]), float(row["scale_cv"]),
                    float(row["wrist_speed_spike_ratio"]),
                ),
            )
            selected.extend(ranked[: args.max_per_subject])
        selected = selected[: args.max_total]
        output = args.output_dir / f"{technique}_{view}_curated_clips.txt"
        lines = [
            "# Automatically quality-filtered; pending visual expert review.",
            *[
                Path(row["pose"]).relative_to(args.pose_root / technique).as_posix()
                for row in selected
            ],
        ]
        output.write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(f"{technique}/{view}: {len(selected)} -> {output}")


if __name__ == "__main__":
    main()
