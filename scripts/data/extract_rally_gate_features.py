"""Extract compact visual + scene features for the Rally-State Gate.

Features extracted per window (16 frames):
  - 512-d pooled spatiotemporal embedding from frozen R(2+1)D-18
  - 4-d scene & motion dynamics:
      1. court_ratio (green badminton court mask)
      2. motion_mean (frame difference intensity)
      3. motion_std  (motion variance across frame)
      4. frame_contrast (scene visual contrast)
Total feature dimension: 516-d.

Processes videos in sequential batches to maximize decoding throughput.
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision.models.video import r2plus1d_18
from torchvision.transforms import functional as VF

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_MEAN = torch.tensor([0.43216, 0.394666, 0.37645], device=DEVICE).view(1, 3, 1, 1, 1)
_STD = torch.tensor([0.22803, 0.22145, 0.216989], device=DEVICE).view(1, 3, 1, 1, 1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True, help="Path to rally_state_train.csv or val.csv")
    parser.add_argument("--output", type=Path, required=True, help="Output .pt file")
    parser.add_argument("--backbone-ckpt", type=Path, default=Path("work_dirs/r2plus1d18_hit_full/best.pth"))
    parser.add_argument("--max-samples-per-match", type=int, default=200, help="Cap samples per match for fast extraction")
    parser.add_argument("--max-matches", type=int, default=None, help="Cap number of matches to process")
    parser.add_argument("--batch-size", type=int, default=16, help="GPU forward batch size")
    return parser.parse_args()


def load_backbone(ckpt_path: Path) -> nn.Module:
    model = r2plus1d_18(weights=None)
    model.fc = nn.Linear(512, 3)
    if ckpt_path.exists():
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt["model"])
        print(f"[*] Loaded R(2+1)D backbone weights from {ckpt_path.name}")
    else:
        print(f"[WARN] Backbone checkpoint not found at {ckpt_path}, using uninitialized weights!")
    feature_extractor = nn.Sequential(*list(model.children())[:-1], nn.Flatten()).to(DEVICE)
    feature_extractor.eval()
    return feature_extractor


def extract_scene_metrics(frames: list[np.ndarray]) -> np.ndarray:
    """Compute 4-d scene and motion metrics from decoded BGR frames."""
    if not frames:
        return np.zeros(4, dtype=np.float32)
    # 1. Court ratio of center frame
    mid = len(frames) // 2
    hsv = cv2.cvtColor(frames[mid], cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array([30, 30, 30]), np.array([100, 255, 255]))
    court_ratio = float(np.count_nonzero(mask) / mask.size)

    # 2 & 3. Motion mean and std (temporal difference between first and second half)
    half = len(frames) // 2
    f_first = frames[0].astype(np.float32)
    f_last = frames[-1].astype(np.float32)
    diff = np.abs(f_last - f_first)
    motion_mean = float(diff.mean() / 255.0)
    motion_std = float(diff.std() / 255.0)

    # 4. Scene contrast (standard deviation of grayscale values)
    gray = cv2.cvtColor(frames[mid], cv2.COLOR_BGR2GRAY)
    contrast = float(gray.std() / 255.0)

    return np.array([court_ratio, motion_mean, motion_std, contrast], dtype=np.float32)


def main() -> None:
    args = parse_args()
    feature_extractor = load_backbone(args.backbone_ckpt)

    print(f"[*] Reading manifest: {args.manifest}...")
    samples_by_match = defaultdict(list)
    with args.manifest.open("r", encoding="utf-8-sig", newline="") as f:
        for r in csv.DictReader(f):
            samples_by_match[r["match_id"]].append(r)

    match_keys = sorted(samples_by_match.keys(), key=lambda k: int(k) if k.isdigit() else k)
    if args.max_matches is not None:
        match_keys = match_keys[:args.max_matches]
    print(f"[*] Processing {len(match_keys)} matches.")

    all_features = []
    all_labels = []
    all_match_ids = []
    all_center_frames = []

    total_processed = 0
    t_start = time.time()

    for mid in match_keys:
        rows = samples_by_match[mid]
        # Subsample to cap per match, balanced between live (1) and nonlive (0)
        live = [r for r in rows if r["label"] == "1"]
        nonlive = [r for r in rows if r["label"] == "0"]
        cap_half = args.max_samples_per_match // 2
        selected = live[:cap_half] + nonlive[:cap_half]
        selected.sort(key=lambda r: int(r["start_frame"]))

        vpath = selected[0]["video_path"]
        cap = cv2.VideoCapture(vpath)
        if not cap.isOpened():
            print(f"  [ERROR] Cannot open {vpath}", file=sys.stderr)
            continue

        match_clips = []
        match_scene = []
        match_meta = []

        for row in selected:
            start_f = int(row["start_frame"])
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_f)
            frames = []
            for _ in range(32):
                ok, f = cap.read()
                if not ok:
                    break
                frames.append(f)

            if len(frames) < 16:
                continue

            scene_vec = extract_scene_metrics(frames)
            idx = np.linspace(0, len(frames) - 1, 16, dtype=int)
            rgb_arr = np.stack([cv2.cvtColor(frames[i], cv2.COLOR_BGR2RGB) for i in idx])

            t = torch.from_numpy(rgb_arr).permute(0, 3, 1, 2).float().div_(255.0)  # T,C,H,W
            t = VF.resize(t, [128, 171], antialias=False)
            t = VF.center_crop(t, [112, 112])
            clip = t.permute(1, 0, 2, 3)  # C,T,H,W

            match_clips.append(clip)
            match_scene.append(scene_vec)
            match_meta.append((int(row["label"]), mid, int(row["center_frame"])))

        cap.release()

        # Batch forward through visual backbone on GPU
        for i in range(0, len(match_clips), args.batch_size):
            batch_tensors = torch.stack(match_clips[i : i + args.batch_size]).to(DEVICE)
            batch_tensors = (batch_tensors - _MEAN) / _STD
            with torch.inference_mode():
                with torch.autocast(device_type=DEVICE.type, dtype=torch.float16):
                    embs = feature_extractor(batch_tensors).cpu().float()

            for j, emb in enumerate(embs):
                scene_t = torch.from_numpy(match_scene[i + j])
                full_feat = torch.cat([emb, scene_t], dim=0)  # 516-d
                label, m_id, c_frame = match_meta[i + j]

                all_features.append(full_feat)
                all_labels.append(label)
                all_match_ids.append(m_id)
                all_center_frames.append(c_frame)

        total_processed += len(match_clips)
        elapsed = time.time() - t_start
        print(f"  Match {mid:>2}: {len(match_clips):>3} windows extracted | Total: {total_processed:>4} ({elapsed:.1f}s)", flush=True)

    if not all_features:
        raise RuntimeError("No features extracted!")

    features_tensor = torch.stack(all_features)
    labels_tensor = torch.tensor(all_labels, dtype=torch.long)
    center_frames_tensor = torch.tensor(all_center_frames, dtype=torch.long)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "features": features_tensor,
        "labels": labels_tensor,
        "match_ids": all_match_ids,
        "center_frames": center_frames_tensor,
        "feature_dim": features_tensor.shape[1],
        "samples_count": len(all_labels),
        "live_count": int((labels_tensor == 1).sum()),
        "nonlive_count": int((labels_tensor == 0).sum()),
    }
    torch.save(payload, args.output)

    print("\n========================================================")
    print(f" [DONE] Features saved to: {args.output}")
    print(f"   Tensor Shape: {features_tensor.shape} (dim={features_tensor.shape[1]})")
    print(f"   Live: {payload['live_count']}, Non-live: {payload['nonlive_count']}")
    print(f"   Total time: {time.time() - t_start:.1f}s")
    print("========================================================")


if __name__ == "__main__":
    main()
