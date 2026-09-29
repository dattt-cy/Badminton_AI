"""Extract leakage-safe online Robust KAP meta-features from validation videos."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import cv2
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ai_classifier.models import STROKE_CLASSES
from scripts.inference.aligned_video_pipeline import AlignedVideoPipeline
from scripts.inference.auto_court_detection import detect_court_corners


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--match-id", required=True)
    parser.add_argument("--start-frame", type=int, required=True)
    parser.add_argument(
        "--video-offset", type=int, default=0,
        help="Absolute source frame represented by frame 0 of --video.",
    )
    parser.add_argument("--frames", type=int, default=9000)
    parser.add_argument("--manifest", type=Path, default=Path("data/manifests/shuttleset_rgb.csv"))
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    end_frame = args.start_frame + args.frames
    rows = []
    with args.manifest.open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            hit = int(row["hit_frame"])
            if row["match_id"] == args.match_id and args.start_frame <= hit < end_frame:
                rows.append(row)
    rows.sort(key=lambda row: int(row["hit_frame"]))
    corners = detect_court_corners(str(args.video))
    cap = cv2.VideoCapture(str(args.video))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    pipeline = AlignedVideoPipeline(corners=corners, orig_w=width, orig_h=height)
    records = []
    for index, row in enumerate(rows, 1):
        hit = int(row["hit_frame"])
        side = "upper" if row["player_side"] == "top" else "lower"
        local_hit = hit - args.video_offset
        feature = pipeline.extract_features(cap, local_hit, total, hitter_side=side)
        if feature is None:
            continue
        detail = pipeline.predict_detailed(feature)
        records.append({
            "feature": detail["reranker_feature"],
            "physics": feature["specialist_physics"],
            "top2": detail["base_top2_indices"],
            "target": STROKE_CLASSES.index(row["coarse_label"]),
            "match_id": args.match_id, "hit_frame": hit,
        })
        print(f"[{index}/{len(rows)}] match={args.match_id} frame={hit}", flush=True)
    cap.release()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(records, args.output)
    print(f"saved={len(records)} output={args.output}")


if __name__ == "__main__":
    main()
