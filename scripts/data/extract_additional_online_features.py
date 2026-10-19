"""Extract real online video features for additional training matches 1, 2, 3, 4, 7, 11."""

from __future__ import annotations

import csv
import sys
import time
from pathlib import Path

import cv2
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO_ROOT), str(REPO_ROOT / "src")]

from ai_classifier.localization import two_stage_temporal_nms
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

    corners = detect_court_corners(str(video_path))
    pipeline = AlignedVideoPipeline(
        ckpt_path=kap_ckpt_path,
        corners=corners,
        orig_w=width,
        orig_h=height,
        pose_model_path=pose_model_path,
    )

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

    tasks = [
        ("1", REPO_ROOT / "work_dirs/online_train_match1_5min_clip.mp4", REPO_ROOT / "work_dirs/online_train_match1_features.pt"),
        ("2", REPO_ROOT / "work_dirs/online_train_match2_5min_clip.mp4", REPO_ROOT / "work_dirs/online_train_match2_features.pt"),
        ("3", REPO_ROOT / "work_dirs/online_train_match3_5min_clip.mp4", REPO_ROOT / "work_dirs/online_train_match3_features.pt"),
        ("4", REPO_ROOT / "work_dirs/online_train_match4_5min_clip.mp4", REPO_ROOT / "work_dirs/online_train_match4_features.pt"),
        ("7", REPO_ROOT / "work_dirs/online_train_match7_5min_clip.mp4", REPO_ROOT / "work_dirs/online_train_match7_features.pt"),
        ("11", REPO_ROOT / "work_dirs/online_train_match11_5min_clip.mp4", REPO_ROOT / "work_dirs/online_train_match11_features.pt"),
    ]

    for m_id, vid_path, out_path in tasks:
        if out_path.exists():
            print(f"[!] {out_path} already exists, skipping.")
            continue
        print(f"\n========================================================")
        print(f"EXTRACTING ONLINE FEATURES: MATCH {m_id}")
        print(f"========================================================")
        records = extract_features_for_clip(vid_path, hit_model, pose_model, kap_ckpt, device)
        torch.save(records, out_path)
        print(f"[+] Saved {len(records)} records to {out_path}")


if __name__ == "__main__":
    main()
