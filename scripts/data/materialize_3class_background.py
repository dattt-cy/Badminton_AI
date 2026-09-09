"""Copy quality-screened no-stroke gaps into the three-class dataset."""

from __future__ import annotations

import argparse
import csv
import shutil
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("quality_csv", type=Path)
    parser.add_argument("output_root", type=Path)
    args = parser.parse_args()
    with args.quality_csv.open(newline="", encoding="utf-8-sig") as source:
        rows = list(csv.DictReader(source))
    accepted = []
    for row in rows:
        if not (
            float(row["valid_ratio"]) >= 0.8
            and float(row["median_torso_scale_px"]) >= 30
            and float(row["scale_cv"]) <= 0.25
            and float(row["wrist_speed_spike_ratio"]) <= 3
            and float(row["wrist_speed_range"]) <= 1.5
            and float(row["wrist_excursion"]) <= 0.30
            and float(row["arm_angle_range"]) <= 20.0
            and row["phases_valid"].lower() == "false"
        ):
            continue
        source_pose, source_video = Path(row["pose"]), Path(row["video"])
        subject = source_pose.parent.name
        output = args.output_root / "background" / "front" / subject
        output.mkdir(parents=True, exist_ok=True)
        stem = source_video.stem
        pose = output / f"{stem}_background.npz"
        video = output / f"{stem}_background.mp4"
        shutil.copy2(source_pose, pose)
        shutil.copy2(source_video, video)
        accepted.append({"pose": str(pose), "video": str(video), "label": "background",
                         "subject": subject, "source": str(source_video)})
    manifest = args.output_root / "background_manifest.csv"
    with manifest.open("w", newline="", encoding="utf-8-sig") as target:
        fields = ("pose", "video", "label", "subject", "source")
        writer = csv.DictWriter(target, fieldnames=fields)
        writer.writeheader()
        writer.writerows(accepted)
    print(f"Materialized {len(accepted)} background clips to {args.output_root}")


if __name__ == "__main__":
    main()
