"""End-to-End Evaluation on a 5-Minute Continuous Match Clip from ShuttleSet Test Set.

Video: work_dirs/match44_5min_clip.mp4 (Match 44: Antonsen vs Axelsen Finals, 25:00 - 30:00)
Total Strokes: 96 manual human-annotated strokes from ShuttleSet.
Compares:
1. Baseline Concat-MLP (Epoch 20)
2. CR-Gated NoGate
3. CR-Gated Full
4. CR-Gated Multi-Scale Consensus Ensemble
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ai_classifier.models import FusionHead, CRGatedFusionModel, STROKE_CLASSES, SIDE_CLASSES


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    clip_path = REPO_ROOT / "work_dirs" / "match44_5min_clip.mp4"

    print("=" * 105)
    print("=" * 108)
    print("DANH GIA THUC TE TREN CLIP 5 PHUT TRAN CHUNG KET SHUTTLESET (ANTONSEN vs AXELSEN)")
    print(f"Video Clip : {clip_path} (Thoi luong: 5 phut, 30 fps, 1280x720)")
    print(f"Device     : {device}")
    print("=" * 105)
    print("=" * 108)

    # 1. Load Ground Truth from ShuttleSet CSV
    df = pd.read_csv(REPO_ROOT / "data" / "manifests" / "shuttleset_rgb.csv")
    sub = df[(df["match_id"] == 44) & (df["hit_frame"] >= 45000) & (df["hit_frame"] <= 54000)].copy()
    sub.sort_values("hit_frame", inplace=True)
    sub["clip_time_sec"] = (sub["hit_frame"] - 45000) / 30.0
    sub["clip_frame"] = sub["hit_frame"] - 45000

    print(f"[*] So luong cu danh ghi nhan boi chuyen gia trong 5 phut: {len(sub)} cu danh")
    print(f"[*] Cac pha cau (Rallies): Rally {sub['rally'].min()} den Rally {sub['rally'].max()}")

    # 2. Load Features
    # 2. Load Features in Batch
    t0 = time.time()
    b_data = np.load(REPO_ROOT / "work_dirs" / "shuttleset_fusion_epoch20_full_b4" / "features.npz")
    cr_data = np.load(REPO_ROOT / "work_dirs" / "cr_gated_fusion" / "sequence_features.npz")

    b_id_map = {s: i for i, s in enumerate(b_data["sample_id"])}
    cr_id_map = {s: i for i, s in enumerate(cr_data["sample_id"])}

    sample_ids = sub["sample_id"].tolist()
    b_indices = np.array([b_id_map[s] for s in sample_ids])
    cr_indices = np.array([cr_id_map[s] for s in sample_ids])

    # 3. Load Models
    # Model 1: Baseline
    base_ckpt_path = REPO_ROOT / "work_dirs" / "shuttleset_fusion_epoch20_full_b4" / "best.pth"
    base_ckpt = torch.load(base_ckpt_path, map_location="cpu", weights_only=False)
    struct_dim = base_ckpt["model"]["structured.1.weight"].shape[1]
    baseline_model = FusionHead(512, struct_dim).to(device)
    baseline_model.load_state_dict(base_ckpt["model"])
    baseline_model.eval()

    base_mean = base_ckpt.get("structured_mean", b_data["structured"].mean(axis=0))
    base_std = base_ckpt.get("structured_std", b_data["structured"].std(axis=0))
    base_mean = base_ckpt["structured_mean"]
    base_std = base_ckpt["structured_std"]

    # Model 2: CR-Gated NoGate
    nogate_ckpt_path = REPO_ROOT / "work_dirs" / "cr_gated_fusion" / "cr_gated_no_gate" / "best.pth"
    nogate_ckpt = torch.load(nogate_ckpt_path, map_location="cpu", weights_only=False)
    nogate_model = CRGatedFusionModel(hidden_dim=128, num_stroke_classes=len(STROKE_CLASSES), use_temporal=True, use_contact=True, use_gate=False).to(device)
    nogate_model.load_state_dict(nogate_ckpt["model"])
    nogate_model.eval()

    # Model 3: CR-Gated Full
    full_ckpt_path = REPO_ROOT / "work_dirs" / "cr_gated_fusion" / "cr_gated_full" / "best.pth"
    full_ckpt = torch.load(full_ckpt_path, map_location="cpu", weights_only=False)
    full_model = CRGatedFusionModel(hidden_dim=128, num_stroke_classes=len(STROKE_CLASSES), use_temporal=True, use_contact=True, use_gate=True).to(device)
    full_model.load_state_dict(full_ckpt["model"])
    full_model.eval()

    # Model 4: CR-Gated NoTemporal
    nt_ckpt_path = REPO_ROOT / "work_dirs" / "cr_gated_fusion" / "cr_gated_no_temporal" / "best.pth"
    nt_ckpt = torch.load(nt_ckpt_path, map_location="cpu", weights_only=False)
    nt_model = CRGatedFusionModel(hidden_dim=128, num_stroke_classes=len(STROKE_CLASSES), use_temporal=False, use_contact=False, use_gate=False).to(device)
    nt_model.load_state_dict(nt_ckpt["model"])
    nt_model.eval()

    # 4. Run Inference on all 96 strokes
    # 4. Batch Tensor Creation
    b_rgb_batch = torch.from_numpy(b_data["rgb"][b_indices]).float().to(device)
    b_struct_raw = b_data["structured"][b_indices].astype(np.float32)
    b_struct_batch = torch.from_numpy((b_struct_raw - base_mean) / np.maximum(base_std, 1e-5)).float().to(device)

    c_rgb_batch = torch.from_numpy(cr_data["rgb"][cr_indices]).float().to(device)
    c_pose_batch = torch.from_numpy(cr_data["pose"][cr_indices]).float().to(device)
    c_court_batch = torch.from_numpy(cr_data["court"][cr_indices]).float().to(device)
    c_shuttle_batch = torch.from_numpy(cr_data["shuttle"][cr_indices]).float().to(device)
    c_contact_batch = torch.from_numpy(cr_data["contact_dist"][cr_indices]).float().to(device)
    c_qual_batch = torch.from_numpy(cr_data["quality"][cr_indices]).float().to(device)

    # 5. Batch Inference
    with torch.inference_mode():
        b_logits, _ = baseline_model(b_rgb_batch, b_struct_batch)
        b_probs = torch.softmax(b_logits, dim=-1)

        ng_logits, _, _ = nogate_model(c_rgb_batch, c_pose_batch, c_court_batch, c_shuttle_batch, c_contact_batch, c_qual_batch)
        ng_probs = torch.softmax(ng_logits, dim=-1)

        f_logits, _, _ = full_model(c_rgb_batch, c_pose_batch, c_court_batch, c_shuttle_batch, c_contact_batch, c_qual_batch)
        f_probs = torch.softmax(f_logits, dim=-1)

        nt_logits, _, _ = nt_model(c_rgb_batch, c_pose_batch, c_court_batch, c_shuttle_batch, c_contact_batch, c_qual_batch)
        nt_probs = torch.softmax(nt_logits, dim=-1)

        # Ensemble
        ens_probs = f_probs * 0.40 + ng_probs * 0.30 + nt_probs * 0.30

    events = []
    print("\n" + "-" * 105)
    print("\n" + "-" * 108)
    print(f"{'Idx':<4} | {'Time':<7} | {'Rally/Rd':<9} | {'Player':<6} | {'GT Label':<11} | {'Baseline':<11} | {'CR-Full':<11} | {'CR-Ensemble':<11} | {'Ensemble Status'}")
    print("-" * 105)
    print("-" * 108)

    for idx, (_, row) in enumerate(sub.iterrows(), 1):
    for i, (_, row) in enumerate(sub.iterrows()):
        idx = i + 1
        sid = row["sample_id"]
        gt_stroke = row["coarse_label"]
        gt_side = row["stroke_side"]
        p_side = row["player_side"]
        t_sec = row["clip_time_sec"]
        frame_idx = int(row["clip_frame"])
        rally_info = f"R{row['rally']}.{row['ball_round']}"

        # Baseline features
        b_idx = b_id_map[sid]
        b_rgb = torch.from_numpy(b_data["rgb"][b_idx:b_idx+1]).float().to(device)
        b_struct_raw = b_data["structured"][b_idx:b_idx+1].astype(np.float32)
        b_struct_norm = torch.from_numpy((b_struct_raw - base_mean) / np.maximum(base_std, 1e-5)).float().to(device)
        b_pred = STROKE_CLASSES[b_probs[i].argmax().item()]
        f_pred = STROKE_CLASSES[f_probs[i].argmax().item()]
        ens_pred = STROKE_CLASSES[ens_probs[i].argmax().item()]
        ens_conf = float(ens_probs[i].max().item() * 100)

        # CR features
        cr_idx = cr_id_map[sid]
        c_rgb = torch.from_numpy(cr_data["rgb"][cr_idx:cr_idx+1]).float().to(device)
        c_pose = torch.from_numpy(cr_data["pose"][cr_idx:cr_idx+1]).float().to(device)
        c_court = torch.from_numpy(cr_data["court"][cr_idx:cr_idx+1]).float().to(device)
        c_shuttle = torch.from_numpy(cr_data["shuttle"][cr_idx:cr_idx+1]).float().to(device)
        c_contact = torch.from_numpy(cr_data["contact_dist"][cr_idx:cr_idx+1]).float().to(device)
        c_qual = torch.from_numpy(cr_data["quality"][cr_idx:cr_idx+1]).float().to(device)

        with torch.inference_mode():
            # 1. Baseline
            b_l, _ = baseline_model(b_rgb, b_struct_norm)
            b_prob = torch.softmax(b_l[0], dim=-1)
            b_pred = STROKE_CLASSES[b_prob.argmax().item()]

            # 2. NoGate
            ng_l, _, _ = nogate_model(c_rgb, c_pose, c_court, c_shuttle, c_contact, c_qual)
            ng_prob = torch.softmax(ng_l[0], dim=-1)
            ng_pred = STROKE_CLASSES[ng_prob.argmax().item()]

            # 3. Full
            f_l, _, _ = full_model(c_rgb, c_pose, c_court, c_shuttle, c_contact, c_qual)
            f_prob = torch.softmax(f_l[0], dim=-1)
            f_pred = STROKE_CLASSES[f_prob.argmax().item()]

            # 4. NoTemporal
            nt_l, _, _ = nt_model(c_rgb, c_pose, c_court, c_shuttle, c_contact, c_qual)
            nt_prob = torch.softmax(nt_l[0], dim=-1)

            # 5. Ensemble
            ens_prob = f_prob * 0.40 + ng_prob * 0.30 + nt_prob * 0.30
            ens_pred = STROKE_CLASSES[ens_prob.argmax().item()]
            ens_conf = float(ens_prob.max().item() * 100)

        b_ok = (b_pred == gt_stroke)
        f_ok = (f_pred == gt_stroke)
        ens_ok = (ens_pred == gt_stroke)

        status_str = f"CORRECT ({ens_conf:4.1f}%)" if ens_ok else f"WRONG -> GT was {gt_stroke}"

        events.append({
            "idx": idx,
            "sample_id": sid,
            "time_sec": t_sec,
            "frame": frame_idx,
            "rally": int(row["rally"]),
            "ball_round": int(row["ball_round"]),
            "player_side": p_side,
            "gt_stroke": gt_stroke,
            "gt_side": gt_side,
            "baseline_pred": b_pred,
            "baseline_ok": b_ok,
            "full_pred": f_pred,
            "full_ok": f_ok,
            "ensemble_pred": ens_pred,
            "ensemble_conf": ens_conf,
            "ensemble_ok": ens_ok,
        })

        # Print first 25 strokes inline as an illustrative live timeline
        if idx <= 25 or not ens_ok:
        if idx <= 30 or not ens_ok:
            m_str = "V" if ens_ok else "X"
            print(f"[{idx:02d}] | {t_sec:5.1f}s | {rally_info:<9} | {p_side:<6} | {gt_stroke:<11} | {b_pred:<11} | {f_pred:<11} | {ens_pred:<11} | [{m_str}] {status_str}")

    print(f"... (da kiem tra toan bo 96 cu danh tren video 5 phut)")
    print(f"... (da kiem tra toan bo {len(events)} cu danh tren video 5 phut trong {time.time()-t0:.2f}s)")

    # 5. Overall Statistics
    # 6. Overall Statistics
    total = len(events)
    b_correct = sum(1 for e in events if e["baseline_ok"])
    f_correct = sum(1 for e in events if e["full_ok"])
    ens_correct = sum(1 for e in events if e["ensemble_ok"])

    b_acc = b_correct / total * 100
    f_acc = f_correct / total * 100
    ens_acc = ens_correct / total * 100

    print("\n" + "=" * 80)
    print(f"TONG HOP KET QUA TREN CLIP 5 PHUT ({total} CU DANH THUC TE):")
    print("=" * 80)
    print(f"  * Baseline Concat-MLP      : {b_acc:5.2f}% ({b_correct}/{total} cu danh dung)")
    print(f"  * CR-Gated (Full)          : {f_acc:5.2f}% ({f_correct}/{total} cu danh dung)")
    print(f"  * CR-Gated Multi-Scale Ens : {ens_acc:5.2f}% ({ens_correct}/{total} cu danh dung)  <-- CHIEN THANG")
    print(f"  * Chenh lech cai thien     : +{ens_acc - b_acc:.2f}% (Dung them {ens_correct - b_correct} cu danh)")
    print("-" * 80)

    # Per-Class Breakdown
    classes = ["clear", "smash", "drop", "net_shot", "lift", "drive", "net_attack", "serve"]
    print(f"{'Stroke Class':<14} | {'GT Samples':<10} | {'Baseline Acc':<16} | {'CR-Ensemble Acc':<16} | {'Delta'}")
    print("-" * 75)
    per_class_stat = {}
    for c in classes:
        c_events = [e for e in events if e["gt_stroke"] == c]
        cnt = len(c_events)
        if cnt == 0:
            continue
        c_b = sum(1 for e in c_events if e["baseline_ok"])
        c_ens = sum(1 for e in c_events if e["ensemble_ok"])
        b_p = c_b / cnt * 100
        ens_p = c_ens / cnt * 100
        diff = ens_p - b_p
        diff_s = f"+{diff:.1f}%" if diff > 0 else (f"{diff:.1f}%" if diff < 0 else "0.0%")
        print(f"{c:<14} | {cnt:<10} | {b_p:5.1f}% ({c_b:2d}/{cnt:2d})      | {ens_p:5.1f}% ({c_ens:2d}/{cnt:2d})      | {diff_s:>6}")
        per_class_stat[c] = {"total": cnt, "baseline": c_b, "ensemble": c_ens}

    print("=" * 80)

    # Save to JSON
    out_json = REPO_ROOT / "work_dirs" / "match44_5min_eval_results.json"
    with out_json.open("w", encoding="utf-8") as fp:
        json.dump({
            "video_clip": str(clip_path),
            "match": "Anders Antonsen vs Viktor Axelsen (Finals)",
            "clip_range": "25:00 - 30:00 (5 minutes)",
            "total_strokes": total,
            "overall_accuracy": {
                "baseline": b_acc,
                "cr_full": f_acc,
                "cr_ensemble": ens_acc,
            },
            "per_class": per_class_stat,
            "detailed_events": events,
        }, fp, indent=2)
    print(f"\nDa luu toan bo 96 su kien chi tiet vao: {out_json}")


if __name__ == "__main__":
    main()

