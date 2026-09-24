"""Test 6 sample clips (1 per class) from C:\\Users\\ADMIN\\Downloads\\archive."""

import os
import sys
from pathlib import Path
import cv2
import numpy as np
import torch
from ultralytics import YOLO

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ai_classifier.models import MultiTaskR2Plus1D, STROKE_CLASSES, SIDE_CLASSES
from scripts.inference.classify_shuttleset_rgb_multitask import read_hitter_clip, detect_hitter_crop, Record, decode_clip

ARCHIVE_DIR = Path(r"C:\Users\ADMIN\Downloads\archive")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def scan_hit(video_path: Path, hit_model, frames: int = 16, stride: int = 4):
    cap = cv2.VideoCapture(str(video_path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    if total < frames:
        return total // 2, "upper"

    centers = list(range(frames // 2, total - frames // 2, stride))
    if not centers:
        return total // 2, "upper"

    best_score = -1.0
    best_center = total // 2
    best_side = "upper"

    # Fast sampling
    cap = cv2.VideoCapture(str(video_path))
    half = frames // 2
    for c in centers:
        cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, c - half))
        decoded = []
        for _ in range(frames):
            ok, frame = cap.read()
            if not ok:
                break
            decoded.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        if len(decoded) < frames:
            continue
        tensor = torch.from_numpy(np.stack(decoded)).permute(0, 3, 1, 2)
        tensor = torch.nn.functional.interpolate(tensor.float(), size=(112, 112), mode="bilinear", align_corners=False)
        tensor = tensor.div_(255.0)
        mean = tensor.new_tensor([0.43216, 0.394666, 0.37645]).view(1, 3, 1, 1)
        std = tensor.new_tensor([0.22803, 0.22145, 0.216989]).view(1, 3, 1, 1)
        tensor = ((tensor - mean) / std).permute(1, 0, 2, 3).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            logits = hit_model(tensor)
            probs = torch.softmax(logits, dim=-1)[0].cpu().numpy()
            score = float(probs[1] + probs[2])  # upper_hit + lower_hit
            side = "upper" if probs[1] >= probs[2] else "lower"
            if score > best_score:
                best_score = score
                best_center = c
                best_side = side
    cap.release()
    return best_center, best_side

def main():
    print(f"Device: {DEVICE}")
    folders = [f for f in sorted(os.listdir(ARCHIVE_DIR)) if (ARCHIVE_DIR / f).is_dir()]
    print(f"Found {len(folders)} classes: {folders}")

    # Load Hit detector
    from torchvision.models.video import r2plus1d_18
    hit_model = r2plus1d_18(weights=None)
    hit_model.fc = torch.nn.Linear(hit_model.fc.in_features, 3)
    hit_ckpt = torch.load("work_dirs/r2plus1d18_hit_full/best.pth", map_location="cpu", weights_only=False)
    hit_model.load_state_dict(hit_ckpt.get("model", hit_ckpt))
    hit_model.to(DEVICE).eval()
    print("Loaded Hit Model.")

    # Load Pose model
    pose_model = YOLO("models/checkpoints/pose/yolov8n-pose.pt")
    print("Loaded YOLO Pose Model.")

    # Load RGB Multitask Classifier
    rgb_model = MultiTaskR2Plus1D()
    rgb_ckpt = torch.load("work_dirs/r2plus1d18_mixed_shuttleset_finebadminton/best.pth", map_location="cpu", weights_only=False)
    rgb_model.load_state_dict(rgb_ckpt["model"])
    rgb_model.to(DEVICE).eval()
    print("Loaded RGB Multitask Model.")

    results = []
    for f in folders:
        clips = sorted([c for c in os.listdir(ARCHIVE_DIR / f) if c.endswith(".mp4") and not c.startswith(".")])
        if not clips:
            continue
        clip_name = clips[0]
        clip_path = ARCHIVE_DIR / f / clip_name

        # Parse ground truth
        parts = f.split("_")
        gt_side = parts[0]
        gt_stroke = "_".join(parts[1:])

        # Stage 1: Hit scan
        hit_frame, hit_side = scan_hit(clip_path, hit_model)

        # Stage 2: Hitter crop around hit_frame
        cap = cv2.VideoCapture(str(clip_path))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()

        start_f = max(0, hit_frame - 10)
        end_f = min(total_frames - 1, hit_frame + 10)
        p_side_str = "top" if hit_side == "upper" else "bottom"

        rec = Record(
            sample_id=clip_path.stem, video_path=clip_path, start_frame=start_f,
            end_frame=end_f, coarse_label="lift", stroke_side="forehand",
            player_side=p_side_str, hit_frame=hit_frame,
        )
        crop_box = detect_hitter_crop(rec, pose_model, 0.55)
        tensor = decode_clip(rec, 16, crop_box, 112).to(torch.float32).div_(255.0)
        mean = tensor.new_tensor([0.43216, 0.394666, 0.37645]).view(1, 3, 1, 1)
        std = tensor.new_tensor([0.22803, 0.22145, 0.216989]).view(1, 3, 1, 1)
        tensor = ((tensor - mean) / std).permute(1, 0, 2, 3).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            stroke_logits, side_logits = rgb_model(tensor)
            pred_stroke_idx = stroke_logits.argmax(dim=-1).item()
            pred_side_idx = side_logits.argmax(dim=-1).item()

            pred_stroke = STROKE_CLASSES[pred_stroke_idx]
            pred_side = SIDE_CLASSES[pred_side_idx]

        is_stroke_ok = (pred_stroke == gt_stroke)
        is_side_ok = (pred_side == gt_side)
        is_both_ok = is_stroke_ok and is_side_ok

        res_str = f"[{f}/{clip_name}] Hit@{hit_frame} ({hit_side}) | GT: {gt_side} {gt_stroke} | Pred: {pred_side} {pred_stroke} | Stroke: {'OK' if is_stroke_ok else 'FAIL'} | Side: {'OK' if is_side_ok else 'FAIL'}"
        print(res_str)
        results.append({
            "folder": f, "clip": clip_name, "gt_side": gt_side, "gt_stroke": gt_stroke,
            "pred_side": pred_side, "pred_stroke": pred_stroke,
            "stroke_ok": is_stroke_ok, "side_ok": is_side_ok, "both_ok": is_both_ok,
        })

    stroke_acc = sum(r["stroke_ok"] for r in results) / len(results)
    side_acc = sum(r["side_ok"] for r in results) / len(results)
    both_acc = sum(r["both_ok"] for r in results) / len(results)
    print(f"\nPilot 6 clips: Stroke Acc={stroke_acc*100:.1f}%, Side Acc={side_acc*100:.1f}%, Compound Acc={both_acc*100:.1f}%")

if __name__ == "__main__":
    main()

