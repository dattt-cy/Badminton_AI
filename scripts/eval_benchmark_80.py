"""Benchmark evaluation script for the 80 standardized ShuttleSet benchmark clips."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
import sys

import numpy as np
import torch
from ultralytics import YOLO

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ai_classifier.localization.fast_tracknet import FastTrackNet
from scripts.inference.auto_court_detection import detect_court_corners
from scripts.inference.analyze_long_video import default_corners
from scripts.inference.classify_shuttleset_fusion_video import (
    classify_fusion_event,
    STROKE_CLASSES,
    SIDE_CLASSES
)
from ai_classifier.models import FusionHead, MultiTaskR2Plus1D
import cv2

def benchmark_all():
    bench_dir = Path(r"C:\Users\ADMIN\Downloads\shuttleset_benchmark_clips")
    if not bench_dir.exists():
        print(f"Error: {bench_dir} not found")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load models
    pose_model = YOLO("models/checkpoints/pose/yolov8n-pose.pt")
    
    rgb_ckpt_path = REPO_ROOT / "work_dirs" / "r2plus1d18_mixed_shuttleset_finebadminton" / "best.pth"
    rgb_ckpt = torch.load(rgb_ckpt_path, map_location="cpu", weights_only=False)
    rgb_model = MultiTaskR2Plus1D()
    rgb_model.load_state_dict(rgb_ckpt["model"])
    rgb_model.to(device).eval()

    fusion_ckpt_path = REPO_ROOT / "work_dirs" / "shuttleset_fusion_epoch20_full_b4" / "best.pth"
    fusion_ckpt = torch.load(fusion_ckpt_path, map_location="cpu", weights_only=False)
    structured_dim = fusion_ckpt["model"]["structured.1.weight"].shape[1]
    fusion_model = FusionHead(512, structured_dim)
    fusion_model.load_state_dict(fusion_ckpt["model"])
    fusion_model.to(device).eval()

    tracker = FastTrackNet(device=device)

    classes = ["clear", "smash", "drop", "net_shot", "lift", "drive", "net_attack", "serve"]
    
    results = []
    
    for cls_name in classes:
        cls_dir = bench_dir / cls_name
        clips = sorted(cls_dir.glob("*.mp4"))
        print(f"\n--- Testing Class: {cls_name} ({len(clips)} clips) ---")
        
        for clip in clips:
            cap = cv2.VideoCapture(str(clip))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = float(cap.get(cv2.CAP_PROP_FPS))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            cap.release()
            
            event_frame = total_frames // 2
            
            try:
                corners = detect_court_corners(clip)
            except Exception:
                corners = default_corners(width, height)
                
            fname = clip.name.lower()
            if "_xa_" in fname or "_top_" in fname:
                p_side = "upper"
            elif "_gan_" in fname or "_bottom_" in fname:
                p_side = "lower"
            else:
                p_side = "upper"

            trajectory = tracker.predict_window(clip, event_frame, window_before=30, window_after=30)
            
            res = classify_fusion_event(
                clip, trajectory, event_frame, p_side, corners,
                pose_model, rgb_model, fusion_model, rgb_ckpt, fusion_ckpt, device,
            )
            
            gt_stroke = cls_name
            gt_side = "backhand" if "backhand" in fname else ("aroundhead" if "aroundhead" in fname else "forehand")
            
            pred_stroke = res["stroke"]["label"]
            pred_side = res["stroke_side"]["label"]
            raw_pred_stroke = res["raw_stroke"]["label"]
            
            is_ok = (pred_stroke == gt_stroke)
            is_raw_ok = (raw_pred_stroke == gt_stroke)
            is_side_ok = (pred_side == gt_side)
            
            results.append({
                "clip": clip.name,
                "gt_stroke": gt_stroke,
                "gt_side": gt_side,
                "player_side": p_side,
                "pred_stroke": pred_stroke,
                "raw_pred_stroke": raw_pred_stroke,
                "pred_side": pred_side,
                "is_ok": is_ok,
                "is_raw_ok": is_raw_ok,
                "is_side_ok": is_side_ok,
                "physics_applied": res.get("physics_adjustment", {}).get("applied", False),
                "physics_rule": res.get("physics_adjustment", {}).get("rule", None),
            })
            
            status_str = "OK" if is_ok else f"FAIL -> {pred_stroke} (raw: {raw_pred_stroke})"
            print(f"  [{clip.name}] GT: {gt_stroke} ({gt_side}) | Result: {status_str}")

    # Summary
    print("\n" + "=" * 60)
    print("BENCHMARK SUMMARY (80 CLIPS)")
    print("=" * 60)
    
    per_class_cal = defaultdict(lambda: {"total": 0, "correct": 0})
    per_class_raw = defaultdict(lambda: {"total": 0, "correct": 0})
    side_correct = sum(1 for r in results if r["is_side_ok"])
    
    for r in results:
        gt = r["gt_stroke"]
        per_class_cal[gt]["total"] += 1
        per_class_raw[gt]["total"] += 1
        if r["is_ok"]:
            per_class_cal[gt]["correct"] += 1
        if r["is_raw_ok"]:
            per_class_raw[gt]["correct"] += 1
            
    print(f"{'Class':<15} | {'Calibrated Acc':<15} | {'Raw Model Acc':<15}")
    print("-" * 50)
    for c in classes:
        cal_acc = per_class_cal[c]["correct"] / per_class_cal[c]["total"] * 100
        raw_acc = per_class_raw[c]["correct"] / per_class_raw[c]["total"] * 100
        print(f"{c:<15} | {cal_acc:5.1f}% ({per_class_cal[c]['correct']}/{per_class_cal[c]['total']})   | {raw_acc:5.1f}% ({per_class_raw[c]['correct']}/{per_class_raw[c]['total']})")
        
    total_cal = sum(r["is_ok"] for r in results) / len(results) * 100
    total_raw = sum(r["is_raw_ok"] for r in results) / len(results) * 100
    total_side = side_correct / len(results) * 100
    
    print("-" * 50)
    print(f"OVERALL STROKE ACC (Calibrated): {total_cal:.1f}% ({sum(r['is_ok'] for r in results)}/80)")
    print(f"OVERALL STROKE ACC (Raw Fusion): {total_raw:.1f}% ({sum(r['is_raw_ok'] for r in results)}/80)")
    print(f"OVERALL SIDE ACC:                {total_side:.1f}% ({side_correct}/80)")
    
    # Save results to json
    out_file = REPO_ROOT / "work_dirs" / "benchmark_80_results.json"
    out_file.parent.mkdir(parents=True, exist_ok=True)
    out_file.write_text(json.dumps({
        "total_clips": len(results),
        "overall_calibrated_acc": total_cal,
        "overall_raw_acc": total_raw,
        "overall_side_acc": total_side,
        "per_class": {c: {"calibrated": per_class_cal[c], "raw": per_class_raw[c]} for c in classes},
        "details": results
    }, indent=2), encoding="utf-8")
    print(f"\nSaved detailed results to {out_file}")

if __name__ == "__main__":
    benchmark_all()

