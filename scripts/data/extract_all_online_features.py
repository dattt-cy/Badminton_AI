"""Extract real online video features for matches 13-17 and 41."""

from __future__ import annotations

import csv
import json
import sys
import time
from pathlib import Path

import cv2
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO_ROOT), str(REPO_ROOT / "src")]

from ai_classifier.localization import two_stage_temporal_nms
from ai_classifier.models import STROKE_CLASSES
from scripts.inference.auto_court_detection import detect_court_corners
from scripts.inference.robust_aligned_video_pipeline import AlignedVideoPipeline
from scripts.inference.scan_video_hit_rgb import load_model, scan_video


def extract_features_for_clip(
    video_path: Path,
    hit_model: torch.nn.Module,
    pose_model_path: Path,
    kap_ckpt_path: Path,
    device: torch.device,
    threshold: float = 0.75,
    stride: int = 4,
    contact_refine_radius: int = 6,
) -> list[dict]:
    cap = cv2.VideoCapture(str(video_path))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # 1. Hit scan
    print(f"[*] Scanning {video_path.name} ({total_frames} frames)...", flush=True)
    t0 = time.time()
    raw_events = scan_video(
        video_path,
        hit_model,
        16,
        stride=stride,
        batch_size=8,
        device=device,
        use_amp=True,
    )
    events = two_stage_temporal_nms(
        raw_events, threshold=threshold, radius=8, same_side_radius=18
    )
    print(f"[+] Found {len(events)} candidate hits in {time.time()-t0:.1f}s", flush=True)

    # 2. Pipeline setup
    corners = detect_court_corners(str(video_path))
    pipeline = AlignedVideoPipeline(
        ckpt_path=kap_ckpt_path,
        corners=corners,
        orig_w=width,
        orig_h=height,
        pose_model_path=pose_model_path,
    )

    # 3. Cache & extract features
    pipeline.prepare_event_cache(
        cap,
        [int(e.frame) for e in events],
        total_frames,
        contact_refine_radius=contact_refine_radius,
    )

    records = []
    for idx, e in enumerate(events, 1):
        feat = pipeline.extract_features(
            cap,
            int(e.frame),
            total_frames,
            hitter_side=e.side,
            contact_refine_radius=contact_refine_radius,
        )
        if feat is not None:
            records.append({
                "hit_frame": int(e.frame),
                "side": e.side,
                **{k: feat[k].detach().cpu() for k in ("emb", "s_j", "s_p", "s_s", "s_c", "s_q")},
            })
        if idx % 20 == 0 or idx == len(events):
            print(f"  [{idx}/{len(events)}] extracted {len(records)} features", flush=True)

    cap.release()
    return records


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    hit_ckpt = REPO_ROOT / "src/ai_classifier/models/checkpoints/hit_detector.pth"
    hit_model, _ = load_model(hit_ckpt)
    hit_model.to(device).eval()

    pose_model = REPO_ROOT / "models/checkpoints/pose/yolov8n-pose.pt"
    kap_ckpt = REPO_ROOT / "work_dirs/kap_robust_seed20260926/kap_best.pth"

    # Targets to process
    tasks = [
        ("13", REPO_ROOT / "work_dirs/match13_5min_clip.mp4", REPO_ROOT / "work_dirs/match13_5min_features.pt", 16300),
        ("14", REPO_ROOT / "work_dirs/match14_5min_clip.mp4", REPO_ROOT / "work_dirs/online_train_match14_features.pt", 89539),
        ("15", REPO_ROOT / "work_dirs/match15_5min_clip.mp4", REPO_ROOT / "work_dirs/online_train_match15_features.pt", 42240),
        ("16", REPO_ROOT / "work_dirs/match16_5min_clip.mp4", REPO_ROOT / "work_dirs/online_train_match16_features.pt", 13792),
        ("17", REPO_ROOT / "work_dirs/match17_5min_clip.mp4", REPO_ROOT / "work_dirs/online_train_match17_features.pt", 68475),
    ]

    manifest = REPO_ROOT / "data/manifests/shuttleset_rgb.csv"
    with manifest.open(encoding="utf-8-sig") as f:
        all_gt = list(csv.DictReader(f))

    for m_id, vid_path, out_path, offset in tasks:
        if out_path.exists():
            print(f"[!] {out_path} already exists, skipping.")
            continue
        print(f"\n========================================================")
        print(f"EXTRACTING ONLINE FEATURES: MATCH {m_id}")
        print(f"========================================================")
        records = extract_features_for_clip(vid_path, hit_model, pose_model, kap_ckpt, device)
        torch.save(records, out_path)
        print(f"[+] Saved {len(records)} records to {out_path}")

    # Process validation Match 41
    val_out = REPO_ROOT / "work_dirs/match41_5min_aligned_features.pt"
    if not val_out.exists():
        print(f"\n========================================================")
        print(f"EXTRACTING ONLINE VALIDATION: MATCH 41")
        print(f"========================================================")
        vid_41 = REPO_ROOT / "work_dirs/match41_5min_clip.mp4"
        offset_41 = 13500
        records_41 = extract_features_for_clip(vid_41, hit_model, pose_model, kap_ckpt, device)

        # Match with ground truth
        truth_41 = [r for r in all_gt if r["match_id"] == "41"]
        available = set(range(len(records_41)))
        aligned_val = []
        for row in sorted(truth_41, key=lambda x: int(x["hit_frame"])):
            local = int(row["hit_frame"]) - offset_41
            choices = [(abs(int(records_41[i]["hit_frame"]) - local), i)
                       for i in available if abs(int(records_41[i]["hit_frame"]) - local) <= 12]
            if not choices:
                continue
            _dist, idx = min(choices)
            available.remove(idx)
            rec = records_41[idx]
            rec["gt_stroke_idx"] = STROKE_CLASSES.index(row["coarse_label"])
            aligned_val.append(rec)

        torch.save(aligned_val, val_out)
        print(f"[+] Saved {len(aligned_val)} aligned validation records to {val_out}")


if __name__ == "__main__":
    main()
