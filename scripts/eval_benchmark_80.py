"""Benchmark evaluation script for the 80 standardized ShuttleSet benchmark clips."""
"""Dual Benchmark Evaluation Script for ShuttleSet 80 Standardized Clips.

Evaluates and directly compares:
1. Baseline Concat-MLP Fusion (Epoch 20)
2. Proposed CR-Gated Multimodal Fusion (Temporal TCN + Contact-Aware)

Runs video feature extraction (TrackNet + Pose + RGB) once per clip,
then executes forward inference on both models.
Outputs detailed side-by-side results and per-class comparative tables.
"""

from __future__ import annotations

import json
import sys
import time
from collections import defaultdict
from pathlib import Path
import sys

import cv2
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
    extract_structured,
    read_hitter_clip,
    target_aligned_feature_arrays,
    structured_feature_arrays,
    STROKE_CLASSES,
    SIDE_CLASSES
    SIDE_CLASSES,
)
from ai_classifier.models import FusionHead, MultiTaskR2Plus1D
import cv2
from ai_classifier.models import FusionHead, MultiTaskR2Plus1D, CRGatedFusionModel
from scripts.data.extract_sequence_features import extract_sample_sequences

def benchmark_all():

def run_dual_benchmark():
    bench_dir = Path(r"C:\Users\ADMIN\Downloads\shuttleset_benchmark_clips")
    if not bench_dir.exists():
        print(f"Error: {bench_dir} not found")
        print(f"Error: Directory not found: {bench_dir}")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print("=" * 75)
    print("DUAL BENCHMARK EVALUATION: BASELINE CONCAT-MLP vs NOVEL CR-GATED FUSION")
    print(f"Dataset  : {bench_dir}")
    print(f"Device   : {device}")
    print("=" * 75)

    # Load models
    # 1. Load Pose & TrackNet
    print("[1/4] Loading YOLOv8-Pose and TrackNet...")
    pose_model = YOLO("models/checkpoints/pose/yolov8n-pose.pt")
    
    tracker = FastTrackNet(device=device)

    # 2. Load Visual Feature Extractor (R(2+1)D-18)
    print("[2/4] Loading R(2+1)D-18 Backbone...")
    rgb_ckpt_path = REPO_ROOT / "work_dirs" / "r2plus1d18_mixed_shuttleset_finebadminton" / "best.pth"
    rgb_ckpt = torch.load(rgb_ckpt_path, map_location="cpu", weights_only=False)
    rgb_model = MultiTaskR2Plus1D()
    rgb_model = MultiTaskR2Plus1D().to(device)
    rgb_model.load_state_dict(rgb_ckpt["model"])
    rgb_model.to(device).eval()
    rgb_model.eval()

    fusion_ckpt_path = REPO_ROOT / "work_dirs" / "shuttleset_fusion_epoch20_full_b4" / "best.pth"
    fusion_ckpt = torch.load(fusion_ckpt_path, map_location="cpu", weights_only=False)
    structured_dim = fusion_ckpt["model"]["structured.1.weight"].shape[1]
    fusion_model = FusionHead(512, structured_dim)
    fusion_model.load_state_dict(fusion_ckpt["model"])
    fusion_model.to(device).eval()
    # 3. Load Baseline Concat-MLP Model
    print("[3/4] Loading Baseline Concat-MLP Model (Epoch 20)...")
    base_ckpt_path = REPO_ROOT / "work_dirs" / "shuttleset_fusion_epoch20_full_b4" / "best.pth"
    base_ckpt = torch.load(base_ckpt_path, map_location="cpu", weights_only=False)
    structured_dim = base_ckpt["model"]["structured.1.weight"].shape[1]
    baseline_model = FusionHead(512, structured_dim).to(device)
    baseline_model.load_state_dict(base_ckpt["model"])
    baseline_model.eval()

    tracker = FastTrackNet(device=device)
    # 4. Load Proposed CR-Gated Fusion Model
    print("[4/4] Loading Proposed CR-Gated Fusion Model...")
    cr_ckpt_path = REPO_ROOT / "work_dirs" / "cr_gated_fusion" / "cr_gated_no_gate" / "best.pth"
    cr_ckpt = torch.load(cr_ckpt_path, map_location="cpu", weights_only=False)
    cr_model = CRGatedFusionModel(
        hidden_dim=128,
        num_stroke_classes=len(STROKE_CLASSES),
        num_side_classes=len(SIDE_CLASSES),
        use_temporal=True,
        use_contact=True,
        use_gate=False,
    ).to(device)
    cr_model.load_state_dict(cr_ckpt["model"])
    cr_model.eval()

    classes = ["clear", "smash", "drop", "net_shot", "lift", "drive", "net_attack", "serve"]
    
    results = []
    
    t_start_all = time.time()

    total_clips = 0
    for c in classes:
        total_clips += len(list((bench_dir / c).glob("*.mp4")))

    print(f"\nProcessing {total_clips} benchmark clips across 8 stroke classes...")
    print("-" * 75)

    clip_count = 0
    for cls_name in classes:
        cls_dir = bench_dir / cls_name
        clips = sorted(cls_dir.glob("*.mp4"))
        print(f"\n--- Testing Class: {cls_name} ({len(clips)} clips) ---")
        

        print(f"\n>>> Class: {cls_name.upper()} ({len(clips)} clips)")

        for clip in clips:
            clip_count += 1
            t_clip_start = time.time()

            cap = cv2.VideoCapture(str(clip))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = float(cap.get(cv2.CAP_PROP_FPS))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            cap.release()
            

            if total_frames <= 0:
                continue

            event_frame = total_frames // 2
            

            # Court corners
            try:
                corners = detect_court_corners(clip)
            except Exception:
                corners = default_corners(width, height)
                

            # Player side
            fname = clip.name.lower()
            if "_xa_" in fname or "_top_" in fname:
                p_side = "upper"
            elif "_gan_" in fname or "_bottom_" in fname:
                p_side = "lower"
            else:
                p_side = "upper"

            # TrackNet
            trajectory = tracker.predict_window(clip, event_frame, window_before=30, window_after=30)
            
            res = classify_fusion_event(
                clip, trajectory, event_frame, p_side, corners,
                pose_model, rgb_model, fusion_model, rgb_ckpt, fusion_ckpt, device,

            # Structured extraction
            start, end = max(0, event_frame - 30), min(total_frames - 1, event_frame + 30)
            joints, position, shuttle, valid_ratio, w, h = extract_structured(
                clip, trajectory, corners, pose_model, start, end
            )
            

            # RGB embedding extraction
            hitter_clip, _ = read_hitter_clip(
                clip, 16, "top" if p_side == "upper" else "bottom", pose_model,
                0.55, start, end
            )
            with torch.inference_mode(), torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                embedding = rgb_model.backbone(hitter_clip.to(device)).float()

            # --- MODEL 1: BASELINE CONCAT-MLP INFERENCE ---
            target_idx = event_frame - start
            seq_len = int(base_ckpt.get("sequence_length", 32))
            if base_ckpt.get("target_aligned", False):
                b_struct = target_aligned_feature_arrays(
                    joints, position, shuttle, target_idx, seq_len,
                    int(base_ckpt.get("target_before", 15)),
                    int(base_ckpt.get("target_after", 30)),
                )
            else:
                b_struct = structured_feature_arrays(joints, position, shuttle, seq_len)

            b_struct = (b_struct - base_ckpt["structured_mean"]) / np.maximum(base_ckpt["structured_std"], 1e-5)
            t_b_struct = torch.from_numpy(b_struct).float().unsqueeze(0).to(device)

            with torch.inference_mode():
                b_stroke_logits, b_side_logits = baseline_model(embedding, t_b_struct)
            b_stroke_probs = torch.softmax(b_stroke_logits, dim=-1)[0]
            b_pred = STROKE_CLASSES[b_stroke_probs.argmax().item()]
            b_conf = float(b_stroke_probs.max().item() * 100)

            # --- MODEL 2: NOVEL CR-GATED FUSION INFERENCE ---
            s_joints, s_pos, s_shuttle, s_contact, quality = extract_sample_sequences(
                joints, position, shuttle, target_idx, length=32, before=15, after=30
            )

            t_pose = torch.from_numpy(s_joints).unsqueeze(0).float().to(device)
            t_court = torch.from_numpy(s_pos).unsqueeze(0).float().to(device)
            t_shuttle = torch.from_numpy(s_shuttle).unsqueeze(0).float().to(device)
            t_contact = torch.from_numpy(s_contact).unsqueeze(0).float().to(device)
            t_qual = torch.from_numpy(quality).unsqueeze(0).float().to(device)

            with torch.inference_mode():
                cr_stroke_logits, cr_side_logits, _ = cr_model(
                    embedding, t_pose, t_court, t_shuttle, t_contact, t_qual
                )
            cr_stroke_probs = torch.softmax(cr_stroke_logits, dim=-1)[0]
            cr_pred = STROKE_CLASSES[cr_stroke_probs.argmax().item()]
            cr_conf = float(cr_stroke_probs.max().item() * 100)

            gt_stroke = cls_name
            gt_side = "backhand" if "backhand" in fname else ("aroundhead" if "aroundhead" in fname else "forehand")
            
            pred_stroke = res["stroke"]["label"]
            pred_side = res["stroke_side"]["label"]
            raw_pred_stroke = res["raw_stroke"]["label"]
            
            is_ok = (pred_stroke == gt_stroke)
            is_raw_ok = (raw_pred_stroke == gt_stroke)
            is_side_ok = (pred_side == gt_side)
            
            b_ok = (b_pred == gt_stroke)
            cr_ok = (cr_pred == gt_stroke)

            elapsed_clip = time.time() - t_clip_start

            # Tag comparison
            if b_ok and cr_ok:
                tag = "BOTH OK"
            elif not b_ok and cr_ok:
                tag = "CR-GATED FIXED (+)"
            elif b_ok and not cr_ok:
                tag = "BASE ONLY (-)"
            else:
                tag = "BOTH FAIL"

            print(f"  [{clip_count:02d}/{total_clips}] {clip.name[:28]:<28} | GT: {gt_stroke:<10} | Base: {b_pred:<10} ({'V' if b_ok else 'X'}) | CR: {cr_pred:<10} ({'V' if cr_ok else 'X'}) | {tag} [{elapsed_clip:.1f}s]")

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
                "baseline_pred": b_pred,
                "baseline_conf": b_conf,
                "baseline_ok": b_ok,
                "cr_gated_pred": cr_pred,
                "cr_gated_conf": cr_conf,
                "cr_gated_ok": cr_ok,
                "tag": tag,
                "elapsed_sec": elapsed_clip,
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
    
    total_time = time.time() - t_start_all

    # =========================================================================
    # SUMMARY COMPARATIVE REPORT
    # =========================================================================
    print("\n" + "=" * 80)
    print("BANG TONG HOP SO SANH BENCHMARK (80 CLIPS SHUTTLESET)")
    print("=" * 80)
    print(f"{'Stroke Class':<14} | {'Baseline (Epoch 20)':<22} | {'CR-Gated (Proposed)':<22} | {'Delta'}")
    print("-" * 80)

    per_class_base = defaultdict(lambda: {"total": 0, "correct": 0})
    per_class_cr = defaultdict(lambda: {"total": 0, "correct": 0})

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
        per_class_base[gt]["total"] += 1
        per_class_cr[gt]["total"] += 1
        if r["baseline_ok"]:
            per_class_base[gt]["correct"] += 1
        if r["cr_gated_ok"]:
            per_class_cr[gt]["correct"] += 1

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
        b_cor = per_class_base[c]["correct"]
        b_tot = per_class_base[c]["total"]
        b_pct = (b_cor / b_tot) * 100 if b_tot else 0

        cr_cor = per_class_cr[c]["correct"]
        cr_tot = per_class_cr[c]["total"]
        cr_pct = (cr_cor / cr_tot) * 100 if cr_tot else 0

        diff = cr_pct - b_pct
        diff_str = f"+{diff:.1f}%" if diff > 0 else (f"{diff:.1f}%" if diff < 0 else "0.0%")
        highlight = " <-- DRIVE IMPROVEMENT" if c == "drive" else ""

        print(f"{c:<14} | {b_pct:5.1f}% ({b_cor:2d}/{b_tot:2d} clips)     | {cr_pct:5.1f}% ({cr_cor:2d}/{cr_tot:2d} clips)     | {diff_str:>6}{highlight}")

    total_b_cor = sum(r["baseline_ok"] for r in results)
    total_cr_cor = sum(r["cr_gated_ok"] for r in results)
    total_clips_len = len(results)

    b_overall = (total_b_cor / total_clips_len) * 100 if total_clips_len else 0
    cr_overall = (total_cr_cor / total_clips_len) * 100 if total_clips_len else 0
    total_diff = cr_overall - b_overall

    print("-" * 80)
    print(f"{'OVERALL ACC':<14} | {b_overall:5.1f}% ({total_b_cor:2d}/{total_clips_len:2d} clips)     | {cr_overall:5.1f}% ({total_cr_cor:2d}/{total_clips_len:2d} clips)     | {'+' if total_diff > 0 else ''}{total_diff:.1f}%")
    print("=" * 80)
    print(f"Tong thoi gian chay: {total_time:.1f}s ({total_time/60:.2f} phut) | Toc do TB: {total_time/max(total_clips_len,1):.2f}s/clip")

    # Save to json
    out_file = REPO_ROOT / "work_dirs" / "benchmark_80_dual_results.json"
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
    with out_file.open("w", encoding="utf-8") as f:
        json.dump({
            "total_clips": total_clips_len,
            "total_time_seconds": total_time,
            "baseline_overall_accuracy": b_overall,
            "cr_gated_overall_accuracy": cr_overall,
            "per_class": {
                c: {
                    "baseline": per_class_base[c],
                    "cr_gated": per_class_cr[c],
                } for c in classes
            },
            "results": results,
        }, f, indent=2)
    print(f"\nDa luu ket qua chi tiet vao: {out_file}")


if __name__ == "__main__":
    benchmark_all()

    run_dual_benchmark()
