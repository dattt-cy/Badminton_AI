"""Evaluate 30 clips from C:\\Users\\ADMIN\\Downloads\\archive using CR-Ensemble.

Compares:
  - Previous RGB Baseline (66.7% Stroke Acc)
  vs
  - 4-Model Balanced CR-Ensemble (Full 15% + NoGate 15% + NoTemp 35% + NoCont 35%)
"""

from __future__ import annotations

import json
import os
import sys
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torchvision.transforms import functional as VF
from ultralytics import YOLO

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ai_classifier.models import CRGatedFusionModel, MultiTaskR2Plus1D, SIDE_CLASSES, STROKE_CLASSES
from ai_classifier.localization.fast_tracknet import FastTrackNet
from scripts.inference.auto_court_detection import detect_court_corners
from scripts.inference.analyze_long_video import default_corners
from scripts.inference.classify_shuttleset_fusion_video import extract_structured
from scripts.inference.classify_shuttleset_rgb_multitask import Record, detect_hitter_crop, decode_clip
from scripts.data.extract_sequence_features import extract_sample_sequences

ARCHIVE_DIR = Path(r"C:\Users\ADMIN\Downloads\archive")
OUTPUT_JSON = Path("work_dirs/archive_30clips_cr_ensemble_results.json")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

_MEAN = torch.tensor([0.43216, 0.394666, 0.37645], device=DEVICE).view(1, 3, 1, 1)
_STD = torch.tensor([0.22803, 0.22145, 0.216989], device=DEVICE).view(1, 3, 1, 1)


def scan_hits_in_clip(video_path: Path, hit_model: torch.nn.Module, stride: int = 4, threshold: float = 0.35, nms_radius: int = 10) -> list[dict]:
    cap = cv2.VideoCapture(str(video_path))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames < 16:
        cap.release()
        return [{"frame": total_frames // 2, "side": "upper", "score": 1.0}]

    half = 8
    frame_buffer = deque(maxlen=16)
    pending_clips, pending_centers = [], []
    first_center = half
    raw_events = []

    def flush():
        if not pending_clips: return
        batch = torch.stack(pending_clips).to(DEVICE)
        with torch.inference_mode(), torch.autocast(device_type=DEVICE.type, dtype=torch.float16):
            logits = hit_model(batch)
            probs = F.softmax(logits.float(), dim=-1).cpu().numpy()
        for center, p in zip(pending_centers, probs):
            score = float(p[1] + p[2])
            side = "upper" if p[1] >= p[2] else "lower"
            raw_events.append({"frame": center, "side": side, "score": score})
        pending_clips.clear()
        pending_centers.clear()

    frame_idx = 0
    while True:
        ok, frame = cap.read()
        if not ok: break
        frame_buffer.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        if len(frame_buffer) == 16:
            center = frame_idx - 16 + 1 + half
            if center >= first_center and (center - first_center) % stride == 0:
                arr = np.stack(list(frame_buffer))
                t = torch.from_numpy(arr).permute(0, 3, 1, 2).float().div_(255.0)
                t = VF.resize(t, [128, 171], antialias=False)
                t = VF.center_crop(t, [112, 112]).to(DEVICE)
                t = (t - _MEAN) / _STD
                pending_clips.append(t.permute(1, 0, 2, 3))
                pending_centers.append(center)
                if len(pending_clips) >= 16:
                    flush()
        frame_idx += 1
    flush()
    cap.release()

    raw_events.sort(key=lambda x: x["score"], reverse=True)
    suppressed = set()
    nms_events = []
    for ev in raw_events:
        f = ev["frame"]
        if f in suppressed or ev["score"] < threshold:
            continue
        nms_events.append(ev)
        for other in range(max(0, f - nms_radius), f + nms_radius + 1):
            suppressed.add(other)

    nms_events.sort(key=lambda x: x["frame"])
    if not nms_events and raw_events:
        nms_events = [max(raw_events, key=lambda x: x["score"])]
    return nms_events


def load_cr_models():
    def load_m(path, use_temp, use_cont, use_gt):
        ckpt = torch.load(f"work_dirs/cr_gated_fusion/{path}/best.pth", map_location="cpu", weights_only=False)
        m = CRGatedFusionModel(hidden_dim=128, num_stroke_classes=len(STROKE_CLASSES), num_side_classes=len(SIDE_CLASSES),
                               use_temporal=use_temp, use_contact=use_cont, use_gate=use_gt, use_cross_attention=True).to(DEVICE)
        m.load_state_dict(ckpt["model"])
        m.eval()
        return m

    m_full = load_m("cr_gated_full", True, True, True)
    m_ng = load_m("cr_gated_no_gate", True, True, False)
    m_nt = load_m("cr_gated_no_temporal", False, False, False)
    m_nc = load_m("cr_gated_no_contact", True, False, True)
    return m_full, m_ng, m_nt, m_nc


def main():
    print("=" * 80)
    print("EVALUATING ARCHIVE 30 CLIPS WITH 4-MODEL BALANCED CR-ENSEMBLE")
    print(f"Device: {DEVICE}")
    print("=" * 80)

    start_time = time.time()

    # 1. Hit model
    from torchvision.models.video import r2plus1d_18
    hit_model = r2plus1d_18(weights=None)
    hit_model.fc = torch.nn.Linear(hit_model.fc.in_features, 3)
    hit_ckpt = torch.load("work_dirs/r2plus1d18_hit_full/best.pth", map_location="cpu", weights_only=False)
    hit_model.load_state_dict(hit_ckpt.get("model", hit_ckpt))
    hit_model.to(DEVICE).eval()
    print("[*] Loaded Hit Model")

    # 2. Pose & TrackNet
    pose_model = YOLO("models/checkpoints/pose/yolov8n-pose.pt")
    tracker = FastTrackNet(device=DEVICE)
    print("[*] Loaded Pose Model & FastTrackNet")

    # 3. RGB Backbone
    rgb_model = MultiTaskR2Plus1D().to(DEVICE)
    rgb_ckpt = torch.load("work_dirs/r2plus1d18_mixed_shuttleset_finebadminton/best.pth", map_location="cpu", weights_only=False)
    rgb_model.load_state_dict(rgb_ckpt["model"])
    rgb_model.eval()
    print("[*] Loaded RGB Multitask Backbone")

    # 4. CR-Gated models
    m_full, m_ng, m_nt, m_nc = load_cr_models()
    print("[*] Loaded 4 CR-Gated Models (Full, NoGate, NoTemp, NoCont)")

    folders = [f for f in sorted(os.listdir(ARCHIVE_DIR)) if (ARCHIVE_DIR / f).is_dir()]
    selected_clips = []
    for f in folders:
        clips = sorted([c for c in os.listdir(ARCHIVE_DIR / f) if c.endswith(".mp4") and not c.startswith(".")])
        for c in clips[:5]:
            selected_clips.append((f, c, ARCHIVE_DIR / f / c))

    print(f"\n[*] Evaluating {len(selected_clips)} clips across {len(folders)} classes...")
    print("-" * 80)

    # 4-Model Ensemble weights
    w = [0.15, 0.15, 0.35, 0.35]

    results = []
    for idx, (folder, clip_name, clip_path) in enumerate(selected_clips, 1):
        parts = folder.split("_")
        gt_side = parts[0]
        gt_stroke = "_".join(parts[1:])
        gt_compound = f"{gt_side}_{gt_stroke}"

        cap = cv2.VideoCapture(str(clip_path))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()

        try:
            corners = detect_court_corners(clip_path)
        except Exception:
            corners = default_corners(width, height)

        # 1. Hit Detection
        events = scan_hits_in_clip(clip_path, hit_model, stride=4, threshold=0.35)

        # 2. TrackNet trajectory for whole clip
        trajectory = tracker.predict_window(clip_path, total_frames // 2, window_before=total_frames, window_after=total_frames)

        classified_events = []
        for ev in events:
            hf = ev["frame"]
            hside = ev["side"]
            p_side_str = "top" if hside == "upper" else "bottom"
            start_f = max(0, hf - 15)
            end_f = min(total_frames - 1, hf + 30)

            # Structured features
            joints, position, shuttle, valid_ratio, w_val, h_val = extract_structured(
                clip_path, trajectory, corners, pose_model, start_f, end_f
            )
            joints_flat = joints.reshape(len(joints), 68)
            pos_flat = position.reshape(len(position), 4)

            s_j, s_p, s_s, s_c, s_q = extract_sample_sequences(
                joints_flat, pos_flat, shuttle, target_index=min(hf - start_f, len(joints) - 1), length=32
            )

            # RGB crop
            rec = Record(
                sample_id=f"{clip_path.stem}_{hf}", video_path=clip_path,
                start_frame=max(0, hf - 10), end_frame=min(total_frames - 1, hf + 10),
                coarse_label="lift", stroke_side="forehand",
                player_side=p_side_str, hit_frame=hf,
            )
            crop_box = detect_hitter_crop(rec, pose_model, 0.55)
            tensor = decode_clip(rec, 16, crop_box, 112).to(torch.float32).div_(255.0).to(DEVICE)
            tensor = (tensor - _MEAN) / _STD
            tensor = tensor.permute(1, 0, 2, 3).unsqueeze(0)

            with torch.inference_mode():
                # RGB embedding
                emb = rgb_model.backbone(tensor)

                t_pose = torch.from_numpy(s_j).unsqueeze(0).to(DEVICE)
                t_court = torch.from_numpy(s_p).unsqueeze(0).to(DEVICE)
                t_shuttle = torch.from_numpy(s_s).unsqueeze(0).to(DEVICE)
                t_contact = torch.from_numpy(s_c).unsqueeze(0).to(DEVICE)
                t_qual = torch.from_numpy(s_q).unsqueeze(0).to(DEVICE)

                # Forward 4 models
                f_p, f_side_logits, _ = m_full(emb, t_pose, t_court, t_shuttle, t_contact, t_qual)
                ng_p, ng_side_logits, _ = m_ng(emb, t_pose, t_court, t_shuttle, t_contact, t_qual)
                nt_p, nt_side_logits, _ = m_nt(emb, t_pose, t_court, t_shuttle, t_contact, t_qual)
                nc_p, nc_side_logits, _ = m_nc(emb, t_pose, t_court, t_shuttle, t_contact, t_qual)

                f_p = torch.softmax(f_p[0], dim=-1)
                ng_p = torch.softmax(ng_p[0], dim=-1)
                nt_p = torch.softmax(nt_p[0], dim=-1)
                nc_p = torch.softmax(nc_p[0], dim=-1)

                ens_stroke_p = w[0] * f_p + w[1] * ng_p + w[2] * nt_p + w[3] * nc_p
                pred_stroke = STROKE_CLASSES[ens_stroke_p.argmax().item()]
                s_conf = float(ens_stroke_p.max().item())

                # Side prediction from full model
                side_p = torch.softmax(f_side_logits[0], dim=-1)
                pred_side = SIDE_CLASSES[side_p.argmax().item()]
                side_conf = float(side_p.max().item())

            classified_events.append({
                "hit_frame": hf,
                "hit_side": hside,
                "pred_stroke": pred_stroke,
                "pred_side": pred_side,
                "stroke_conf": s_conf,
                "side_conf": side_conf,
                "compound": f"{pred_side}_{pred_stroke}",
                "is_stroke_match": (pred_stroke == gt_stroke),
                "is_side_match": (pred_side == gt_side),
                "is_compound_match": (pred_stroke == gt_stroke and pred_side == gt_side),
            })

        stroke_hit_matches = [e for e in classified_events if e["is_stroke_match"]]
        compound_hit_matches = [e for e in classified_events if e["is_compound_match"]]
        clip_has_stroke = len(stroke_hit_matches) > 0
        clip_has_compound = len(compound_hit_matches) > 0

        status_str = "COMPOUND MATCH" if clip_has_compound else ("STROKE MATCH" if clip_has_stroke else "MISMATCH")
        print(
            f"[{idx:2d}/30] {folder}/{clip_name:<8} ({total_frames:2d}f) | "
            f"Hits: {len(classified_events)} | "
            f"GT: {gt_compound} | "
            f"Status: {status_str}",
            flush=True,
        )
        for ev in classified_events:
            mark = "★" if ev["is_compound_match"] else ("✓" if ev["is_stroke_match"] else " ")
            print(
                f"      [{mark}] Frame {ev['hit_frame']:2d}: "
                f"{ev['pred_side']:9s} {ev['pred_stroke']:10s} (conf: {ev['stroke_conf']:.2f}, {ev['side_conf']:.2f})",
                flush=True,
            )

        results.append({
            "folder": folder,
            "clip_name": clip_name,
            "gt_side": gt_side,
            "gt_stroke": gt_stroke,
            "gt_compound": gt_compound,
            "detected_events": classified_events,
            "clip_has_stroke_match": clip_has_stroke,
            "clip_has_compound_match": clip_has_compound,
        })

    elapsed = time.time() - start_time
    total_clips = len(results)
    stroke_matches = sum(1 for r in results if r["clip_has_stroke_match"])
    compound_matches = sum(1 for r in results if r["clip_has_compound_match"])

    class_stats = {}
    for r in results:
        c = r["folder"]
        if c not in class_stats:
            class_stats[c] = {"total": 0, "stroke_ok": 0, "compound_ok": 0}
        class_stats[c]["total"] += 1
        if r["clip_has_stroke_match"]:
            class_stats[c]["stroke_ok"] += 1
        if r["clip_has_compound_match"]:
            class_stats[c]["compound_ok"] += 1

    summary = {
        "total_clips": total_clips,
        "stroke_accuracy": stroke_matches / total_clips,
        "stroke_matches": stroke_matches,
        "compound_accuracy": compound_matches / total_clips,
        "compound_matches": compound_matches,
        "elapsed_seconds": round(elapsed, 2),
        "per_class_stats": class_stats,
    }

    OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 80)
    print(f"EVALUATION COMPLETE ON 30 CLIPS IN {elapsed:.1f}s ({elapsed/total_clips:.2f}s/clip)")
    print(f"CR-ENSEMBLE Stroke Match Rate:   {stroke_matches}/{total_clips} ({summary['stroke_accuracy']*100:.1f}%)")
    print(f"CR-ENSEMBLE Compound Match Rate: {compound_matches}/{total_clips} ({summary['compound_accuracy']*100:.1f}%)")
    print(f"PREVIOUS RGB BASELINE:           20/30 (66.7%) Stroke | 11/30 (36.7%) Compound")
    print("-" * 80)
    print(f"{'Class Name':<22} | {'Clips':<5} | {'Stroke Match':<12} | {'Compound Match':<14}")
    print("-" * 80)
    for c, stat in class_stats.items():
        s_pct = 100.0 * stat["stroke_ok"] / stat["total"]
        cp_pct = 100.0 * stat["compound_ok"] / stat["total"]
        print(f"{c:<22} | {stat['total']:<5} | {stat['stroke_ok']}/5 ({s_pct:5.1f}%) | {stat['compound_ok']}/5 ({cp_pct:5.1f}%)")
    print("=" * 80)


if __name__ == "__main__":
    main()

