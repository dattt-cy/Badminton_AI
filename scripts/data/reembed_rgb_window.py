"""Recompute only the RGB embedding of cached KAP features with a different RGB window.

The visual backbone was trained on 16 hitter crops spanning hit-30..hit+30, while
the pipeline used hit-15..hit+30. This rewrites ``emb`` of the online fine-tuning
records (matches 13-17, validation 41) and of the 8-clip benchmark feature cache;
pose, court, shuttle and contact streams are left untouched.

    python scripts/data/reembed_rgb_window.py --rgb-before 30
    python scripts/data/reembed_rgb_window.py --rgb-before 15 --verify   # must reproduce the old emb
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO_ROOT), str(REPO_ROOT / "src")]

from scripts.evaluation.benchmark_8_matches_10min import CLIPS_CONFIG  # noqa: E402
from scripts.inference.auto_court_detection import detect_court_corners  # noqa: E402
from scripts.inference.robust_aligned_video_pipeline import AlignedVideoPipeline  # noqa: E402

WORK = REPO_ROOT / "work_dirs"
CKPTS = REPO_ROOT / "src/ai_classifier/models/checkpoints"
ONLINE = [
    ("match13_5min_clip.mp4", "match13_5min_features.pt"),
    ("match14_5min_clip.mp4", "online_train_match14_features.pt"),
    ("match15_5min_clip.mp4", "online_train_match15_features.pt"),
    ("match16_5min_clip.mp4", "online_train_match16_features.pt"),
    ("match17_5min_clip.mp4", "online_train_match17_features.pt"),
    ("match41_5min_clip.mp4", "match41_5min_aligned_features.pt"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rgb-before", type=int, default=30)
    parser.add_argument("--contact-refine-radius", type=int, default=6,
                        help="Radius used when the online records were extracted.")
    parser.add_argument("--feature-cache", type=Path, default=WORK / "kap_ab_8clips/feature_cache")
    parser.add_argument("--verify", action="store_true",
                        help="Only compare the first records of each file with the stored emb.")
    return parser.parse_args()


def make_pipeline(video: Path, rgb_before: int) -> tuple[AlignedVideoPipeline, cv2.VideoCapture, int]:
    cap = cv2.VideoCapture(str(video))
    pipeline = AlignedVideoPipeline(
        ckpt_path=CKPTS / "kap_classifier.pth", corners=detect_court_corners(str(video)),
        orig_w=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), orig_h=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        pose_model_path=CKPTS / "yolov8n-pose.pt", rgb_before=rgb_before,
    )
    return pipeline, cap, int(cap.get(cv2.CAP_PROP_FRAME_COUNT))


def main() -> None:
    args = parse_args()
    suffix = f"_rgbw{args.rgb_before}"
    for video, name in ONLINE:
        records = torch.load(WORK / name, map_location="cpu", weights_only=False)
        out = WORK / name.replace(".pt", f"{suffix}.pt")
        if out.exists() and not args.verify:
            print(f"[skip] {out.name} exists")
            continue
        pipeline, cap, total = make_pipeline(WORK / video, args.rgb_before)
        todo = records[:5] if args.verify else records
        for i, record in enumerate(todo):
            feat = pipeline.extract_features(cap, int(record["hit_frame"]), total, hitter_side=record["side"],
                                             contact_refine_radius=args.contact_refine_radius)
            if feat is None:
                continue
            if args.verify:
                diff = (feat["emb"] - record["emb"]).abs().max().item()
                print(f"  {name} #{i}: max |emb diff| = {diff:.2e}")
            else:
                record["emb"] = feat["emb"].cpu()
        if not args.verify:
            torch.save(records, out)
            print(f"[+] {out.name}: {len(records)} records", flush=True)
        cap.release()

    if args.verify:
        return
    out_dir = args.feature_cache.parent / f"{args.feature_cache.name}{suffix}"
    out_dir.mkdir(exist_ok=True)
    for cfg in CLIPS_CONFIG:
        out = out_dir / f"{cfg['video'].stem}.pt"
        if out.exists():
            continue
        cached = torch.load(args.feature_cache / f"{cfg['video'].stem}.pt", map_location="cpu", weights_only=False)
        pipeline, cap, total = make_pipeline(cfg["video"], args.rgb_before)
        for item in cached["features"]:
            f = item["feature"]
            f["emb"] = pipeline.rgb_embedding(cap, int(f["hit_frame"]), f["crop_box"], total).cpu()
        torch.save(cached, out)
        cap.release()
        print(f"[+] {out_dir.name}/{out.name}: {len(cached['features'])} hits", flush=True)


if __name__ == "__main__":
    main()
