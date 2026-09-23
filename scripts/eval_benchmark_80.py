"""Tri-Model Benchmark Evaluation Script for ShuttleSet 80 Standardized Clips.

Directly evaluates and compares:
1. Baseline Concat-MLP Fusion (Epoch 20)
2. CR-Gated Fusion (no_gate variant - Ablation Winner)
3. CR-Gated Fusion (bst_enhanced variant - BST-Inspired PPF + AimPlayer + Cross-Shuttle)

Extracts video/spatial features once per clip (with automatic disk caching)
and evaluates all models simultaneously.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import sys
import time

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
    extract_structured,
    read_hitter_clip,
    target_aligned_feature_arrays,
    structured_feature_arrays,
    STROKE_CLASSES,
    SIDE_CLASSES,
)
from ai_classifier.models import FusionHead, MultiTaskR2Plus1D, CRGatedFusionModel
from scripts.data.extract_sequence_features import extract_sample_sequences


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate ShuttleSet 80 Benchmark Clips")
    parser.add_argument(
        "--bench-dir",
        type=Path,
        default=Path(r"C:\Users\ADMIN\Downloads\shuttleset_benchmark_clips"),
        help="Path to 80 benchmark clips directory",
    )
    parser.add_argument(
        "--cache-file",
        type=Path,
        default=REPO_ROOT / "work_dirs" / "benchmark_80_features_cache.pt",
        help="Path to feature cache file (.pt)",
    )
    parser.add_argument(
        "--force-extract",
        action="store_true",
        help="Force re-extraction of video features instead of using cache",
    )
    return parser.parse_args()


def load_all_models(device: torch.device):
    """Load Baseline, CR-Gated (no_gate), and CR-Gated (bst_enhanced)."""
    # 1. Baseline Concat-MLP
    base_ckpt_path = REPO_ROOT / "work_dirs" / "shuttleset_fusion_epoch20_full_b4" / "best.pth"
    base_ckpt = torch.load(base_ckpt_path, map_location="cpu", weights_only=False)
    structured_dim = base_ckpt["model"]["structured.1.weight"].shape[1]
    baseline_model = FusionHead(512, structured_dim).to(device)
    baseline_model.load_state_dict(base_ckpt["model"])
    baseline_model.eval()

    # 2. CR-Gated (no_gate)
    nogate_ckpt_path = REPO_ROOT / "work_dirs" / "cr_gated_fusion" / "cr_gated_no_gate" / "best.pth"
    nogate_ckpt = torch.load(nogate_ckpt_path, map_location="cpu", weights_only=False)
    nogate_model = CRGatedFusionModel(
        hidden_dim=128,
        num_stroke_classes=len(STROKE_CLASSES),
        num_side_classes=len(SIDE_CLASSES),
        use_temporal=True,
        use_contact=True,
        use_gate=False,
    ).to(device)
    nogate_model.load_state_dict(nogate_ckpt["model"])
    nogate_model.eval()

    # 3. CR-Gated (bst_enhanced)
    bst_ckpt_path = REPO_ROOT / "work_dirs" / "cr_gated_fusion" / "cr_gated_bst_enhanced" / "best.pth"
    bst_ckpt = torch.load(bst_ckpt_path, map_location="cpu", weights_only=False)
    bst_model = CRGatedFusionModel(
        hidden_dim=128,
        num_stroke_classes=len(STROKE_CLASSES),
        num_side_classes=len(SIDE_CLASSES),
        use_temporal=True,
        use_contact=True,
        use_gate=True,
        use_cross_attention=True,
        use_ppf=True,
        use_aim_player=True,
        use_cross_shuttle=True,
    ).to(device)
    bst_model.load_state_dict(bst_ckpt["model"])
    bst_model.eval()

    return {
        "baseline": (baseline_model, base_ckpt),
        "cr_nogate": (nogate_model, nogate_ckpt),
        "cr_bst": (bst_model, bst_ckpt),
    }


def main():
    args = parse_args()
    bench_dir = args.bench_dir
    if not bench_dir.exists():
        print(f"Error: Directory not found: {bench_dir}")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 82)
    print("TRI-MODEL BENCHMARK: BASELINE vs CR-GATED (NO-GATE) vs CR-GATED (BST-ENHANCED)")
    print(f"Dataset  : {bench_dir}")
    print(f"Device   : {device}")
    print("=" * 82)

    # Load classification models
    models = load_all_models(device)
    baseline_model, base_ckpt = models["baseline"]
    nogate_model, _ = models["cr_nogate"]
    bst_model, _ = models["cr_bst"]

    classes = ["clear", "smash", "drop", "net_shot", "lift", "drive", "net_attack", "serve"]

    # Check cache
    cache_data = None
    if args.cache_file.exists() and not args.force_extract:
        print(f"[*] Found feature cache at: {args.cache_file}")
        try:
            cache_data = torch.load(args.cache_file, map_location="cpu", weights_only=False)
            print(f"[*] Successfully loaded {len(cache_data)} clips from cache! Instant inference enabled.")
        except Exception as e:
            print(f"[!] Warning: Failed to load cache ({e}). Re-extracting...")
            cache_data = None

    extract_needed = cache_data is None
    new_cache = {}

    if extract_needed:
        print("[*] Initializing feature extractors (YOLOv8-Pose, TrackNet, R(2+1)D-18)...")
        pose_model = YOLO("models/checkpoints/pose/yolov8n-pose.pt")
        tracker = FastTrackNet(device=device)

        rgb_ckpt_path = REPO_ROOT / "work_dirs" / "r2plus1d18_mixed_shuttleset_finebadminton" / "best.pth"
        rgb_ckpt = torch.load(rgb_ckpt_path, map_location="cpu", weights_only=False)
        rgb_model = MultiTaskR2Plus1D().to(device)
        rgb_model.load_state_dict(rgb_ckpt["model"])
        rgb_model.eval()

    total_clips = sum(len(list((bench_dir / c).glob("*.mp4"))) for c in classes)
    print(f"\nEvaluating {total_clips} clips across 8 stroke classes...")
    print("-" * 82)

    results = []
    clip_count = 0
    t_start_all = time.time()

    for cls_name in classes:
        cls_dir = bench_dir / cls_name
        clips = sorted(cls_dir.glob("*.mp4"))

        for clip in clips:
            clip_count += 1
            t_clip_start = time.time()

            fname = clip.name.lower()
            if "_xa_" in fname or "_top_" in fname:
                p_side = "upper"
            elif "_gan_" in fname or "_bottom_" in fname:
                p_side = "lower"
            else:
                p_side = "upper"

            gt_stroke = cls_name
            gt_side = "backhand" if "backhand" in fname else ("aroundhead" if "aroundhead" in fname else "forehand")

            if cache_data and clip.name in cache_data:
                cached = cache_data[clip.name]
                embedding = cached["embedding"].to(device)
                t_b_struct = cached["t_b_struct"].to(device)
                t_pose = cached["t_pose"].to(device)
                t_court = cached["t_court"].to(device)
                t_shuttle = cached["t_shuttle"].to(device)
                t_contact = cached["t_contact"].to(device)
                t_qual = cached["t_qual"].to(device)
            else:
                # Video feature extraction
                cap = cv2.VideoCapture(str(clip))
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                cap.release()

                if total_frames <= 0:
                    continue

                event_frame = total_frames // 2
                try:
                    corners = detect_court_corners(clip)
                except Exception:
                    corners = default_corners(width, height)

                trajectory = tracker.predict_window(clip, event_frame, window_before=30, window_after=30)
                start, end = max(0, event_frame - 30), min(total_frames - 1, event_frame + 30)

                joints, position, shuttle, valid_ratio, w, h = extract_structured(
                    clip, trajectory, corners, pose_model, start, end
                )

                hitter_clip, _ = read_hitter_clip(
                    clip, 16, "top" if p_side == "upper" else "bottom", pose_model,
                    0.55, start, end
                )
                with torch.inference_mode(), torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                    embedding = rgb_model.backbone(hitter_clip.to(device)).float()

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

                s_joints, s_pos, s_shuttle, s_contact, quality = extract_sample_sequences(
                    joints, position, shuttle, target_idx, length=32, before=15, after=30
                )
                t_pose = torch.from_numpy(s_joints).unsqueeze(0).float().to(device)
                t_court = torch.from_numpy(s_pos).unsqueeze(0).float().to(device)
                t_shuttle = torch.from_numpy(s_shuttle).unsqueeze(0).float().to(device)
                t_contact = torch.from_numpy(s_contact).unsqueeze(0).float().to(device)
                t_qual = torch.from_numpy(quality).unsqueeze(0).float().to(device)

                new_cache[clip.name] = {
                    "embedding": embedding.cpu(),
                    "t_b_struct": t_b_struct.cpu(),
                    "t_pose": t_pose.cpu(),
                    "t_court": t_court.cpu(),
                    "t_shuttle": t_shuttle.cpu(),
                    "t_contact": t_contact.cpu(),
                    "t_qual": t_qual.cpu(),
                }

            # Inference 1: Baseline
            with torch.inference_mode():
                b_logits, _ = baseline_model(embedding, t_b_struct)
            b_pred = STROKE_CLASSES[b_logits[0].argmax().item()]
            b_conf = float(torch.softmax(b_logits[0], dim=-1).max().item() * 100)
            b_ok = (b_pred == gt_stroke)

            # Inference 2: CR-Gated (no_gate)
            with torch.inference_mode():
                nogate_logits, _, _ = nogate_model(embedding, t_pose, t_court, t_shuttle, t_contact, t_qual)
            nogate_pred = STROKE_CLASSES[nogate_logits[0].argmax().item()]
            nogate_conf = float(torch.softmax(nogate_logits[0], dim=-1).max().item() * 100)
            nogate_ok = (nogate_pred == gt_stroke)

            # Inference 3: CR-Gated (bst_enhanced)
            with torch.inference_mode():
                bst_logits, _, _ = bst_model(embedding, t_pose, t_court, t_shuttle, t_contact, t_qual)
            bst_pred = STROKE_CLASSES[bst_logits[0].argmax().item()]
            bst_conf = float(torch.softmax(bst_logits[0], dim=-1).max().item() * 100)
            bst_ok = (bst_pred == gt_stroke)

            elapsed_clip = time.time() - t_clip_start

            # Status mark
            b_mark = "V" if b_ok else "X"
            ng_mark = "V" if nogate_ok else "X"
            bst_mark = "V" if bst_ok else "X"

            print(
                f"  [{clip_count:02d}/{total_clips}] {clip.name[:25]:<25} | GT: {gt_stroke:<10} "
                f"| Base: {b_pred:<10} ({b_mark}) | NoGate: {nogate_pred:<10} ({ng_mark}) | BST: {bst_pred:<10} ({bst_mark}) "
                f"[{elapsed_clip:.1f}s]"
            )

            results.append({
                "clip": clip.name,
                "gt_stroke": gt_stroke,
                "gt_side": gt_side,
                "player_side": p_side,
                "baseline_pred": b_pred,
                "baseline_conf": b_conf,
                "baseline_ok": b_ok,
                "nogate_pred": nogate_pred,
                "nogate_conf": nogate_conf,
                "nogate_ok": nogate_ok,
                "bst_pred": bst_pred,
                "bst_conf": bst_conf,
                "bst_ok": bst_ok,
                "elapsed_sec": elapsed_clip,
            })

    total_time = time.time() - t_start_all

    # Save feature cache if newly extracted
    if new_cache:
        args.cache_file.parent.mkdir(parents=True, exist_ok=True)
        torch.save(new_cache, args.cache_file)
        print(f"\n[+] Saved {len(new_cache)} extracted clip features to cache: {args.cache_file}")

    # =========================================================================
    # SUMMARY COMPARATIVE REPORT
    # =========================================================================
    print("\n" + "=" * 92)
    print("BANG TONG HOP SO SANH BENCHMARK 80 CLIPS: BASELINE vs NO-GATE vs BST-ENHANCED")
    print("=" * 92)
    print(f"{'Stroke Class':<13} | {'Baseline':<18} | {'CR-Gated (NoGate)':<18} | {'CR-Gated (BST)':<18} | {'Best Model'}")
    print("-" * 92)

    per_class_b = defaultdict(lambda: {"total": 0, "correct": 0})
    per_class_ng = defaultdict(lambda: {"total": 0, "correct": 0})
    per_class_bst = defaultdict(lambda: {"total": 0, "correct": 0})

    for r in results:
        gt = r["gt_stroke"]
        per_class_b[gt]["total"] += 1
        per_class_ng[gt]["total"] += 1
        per_class_bst[gt]["total"] += 1
        if r["baseline_ok"]:
            per_class_b[gt]["correct"] += 1
        if r["nogate_ok"]:
            per_class_ng[gt]["correct"] += 1
        if r["bst_ok"]:
            per_class_bst[gt]["correct"] += 1

    for c in classes:
        b_acc = per_class_b[c]["correct"] / max(per_class_b[c]["total"], 1) * 100
        ng_acc = per_class_ng[c]["correct"] / max(per_class_ng[c]["total"], 1) * 100
        bst_acc = per_class_bst[c]["correct"] / max(per_class_bst[c]["total"], 1) * 100

        best_score = max(b_acc, ng_acc, bst_acc)
        winners = []
        if bst_acc == best_score:
            winners.append("BST")
        if ng_acc == best_score:
            winners.append("NoGate")
        if b_acc == best_score:
            winners.append("Base")

        winner_str = "/".join(winners)
        print(
            f"{c:<13} | {b_acc:5.1f}% ({per_class_b[c]['correct']:2d}/{per_class_b[c]['total']:2d})       "
            f"| {ng_acc:5.1f}% ({per_class_ng[c]['correct']:2d}/{per_class_ng[c]['total']:2d})       "
            f"| {bst_acc:5.1f}% ({per_class_bst[c]['correct']:2d}/{per_class_bst[c]['total']:2d})       "
            f"| {winner_str}"
        )

    b_total = sum(r["baseline_ok"] for r in results) / len(results) * 100
    ng_total = sum(r["nogate_ok"] for r in results) / len(results) * 100
    bst_total = sum(r["bst_ok"] for r in results) / len(results) * 100

    print("-" * 92)
    print(
        f"{'OVERALL ACC':<13} | {b_total:5.1f}% ({sum(r['baseline_ok'] for r in results)}/80)       "
        f"| {ng_total:5.1f}% ({sum(r['nogate_ok'] for r in results)}/80)       "
        f"| {bst_total:5.1f}% ({sum(r['bst_ok'] for r in results)}/80)       "
        f"| {'BST' if bst_total >= max(b_total, ng_total) else 'NoGate'}"
    )
    print("=" * 92)
    print(f"Tong thoi gian chay: {total_time:.1f}s | Toc do TB: {total_time/len(results):.2f}s/clip")

    # Save to json
    out_file = REPO_ROOT / "work_dirs" / "benchmark_80_tri_results.json"
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with out_file.open("w", encoding="utf-8") as f:
        json.dump({
            "total_clips": len(results),
            "total_time_seconds": total_time,
            "overall_accuracy": {
                "baseline": b_total,
                "cr_gated_nogate": ng_total,
                "cr_gated_bst_enhanced": bst_total,
            },
            "per_class": {
                c: {
                    "baseline": per_class_b[c],
                    "cr_gated_nogate": per_class_ng[c],
                    "cr_gated_bst_enhanced": per_class_bst[c],
                } for c in classes
            },
            "results": results,
        }, f, indent=2)
    print(f"\nDa luu ket qua vao: {out_file}")


if __name__ == "__main__":
    main()
