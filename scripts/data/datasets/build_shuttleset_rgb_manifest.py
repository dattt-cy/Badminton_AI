"""Build virtual RGB clips around ShuttleSet hit frames without copying videos."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--npy-manifest",
        type=Path,
        default=Path("data/manifests/shuttleset_npy.csv"),
    )
    parser.add_argument(
        "--video-audit",
        type=Path,
        default=Path("data/manifests/shuttleset_video_audit.csv"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/manifests/shuttleset_rgb.csv"),
    )
    parser.add_argument(
        "--window-seconds",
        type=float,
        default=1.0,
        help="Seconds retained before and after each annotated hit.",
    )
    return parser.parse_args()


def load_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def main() -> None:
    args = parse_args()
    if args.window_seconds <= 0:
        raise ValueError("window-seconds must be positive")
    videos = {int(row["match_id"]): row for row in load_csv(args.video_audit)}
    records: list[dict[str, object]] = []
    skipped: Counter[str] = Counter()
    for source in load_csv(args.npy_manifest):
        if source["valid_stroke_side"].lower() != "true":
            skipped["invalid_stroke_side"] += 1
            continue
        if source["valid_coarse_label"].lower() != "true":
            skipped["invalid_coarse_label"] += 1
            continue
        match_id = int(source["match_id"])
        video = videos.get(match_id)
        if video is None or not video["local_video_path"]:
            skipped["missing_video"] += 1
            continue
        fps = float(video["fps"])
        frame_count = int(round(float(video["duration_seconds"]) * fps))
        hit_frame = int(source["hit_frame"])
        radius = max(1, int(round(args.window_seconds * fps)))
        start_frame = max(0, hit_frame - radius)
        end_frame = min(frame_count - 1, hit_frame + radius)
        if not start_frame <= hit_frame <= end_frame:
            skipped["hit_out_of_bounds"] += 1
            continue
        records.append(
            {
                "sample_id": source["sample_id"],
                "dataset": "shuttleset",
                "split": source["split"],
                "match_id": match_id,
                "set_id": source["set_id"],
                "rally": source["rally"],
                "ball_round": source["ball_round"],
                "video_path": video["local_video_path"],
                "start_frame": start_frame,
                "hit_frame": hit_frame,
                "end_frame": end_frame,
                "fps": video["fps"],
                "player_side": source["player_side"],
                "raw_label": source["raw_label"],
                "coarse_label": source["coarse_label"],
                "stroke_side": source["stroke_side"],
                "annotation_path": source["annotation_path"],
            }
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    summary = {
        "samples": len(records),
        "split_counts": dict(Counter(str(row["split"]) for row in records)),
        "stroke_side_counts": dict(
            Counter(str(row["stroke_side"]) for row in records)
        ),
        "coarse_label_counts": dict(
            Counter(str(row["coarse_label"]) for row in records)
        ),
        "skipped": dict(skipped),
        "window_seconds_each_side": args.window_seconds,
        "output": str(args.output.resolve()),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
