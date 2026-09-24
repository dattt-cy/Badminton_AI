"""Evaluate and visualize BFMD 5-minute continuous video clip.

1. Evaluates Hit Detection and Multi-Task Classification against 104 BFMD GT strokes.
2. Renders an annotated visual video clip (with Pose skeleton and Prediction overlay banners).
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from ultralytics import YOLO

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ai_classifier.models import MultiTaskR2Plus1D, SIDE_CLASSES, STROKE_CLASSES
from scripts.inference.classify_shuttleset_rgb_multitask import Record, detect_hitter_crop, decode_clip

VIDEO_5MIN = Path("work_dirs/bfmd_indonesia2025_5min.mp4")
SCAN_JSON = Path("work_dirs/scan_bfmd_5min.json")
BFMD_JSON = Path(r"C:\Users\ADMIN\Downloads\BFMD_data-20260921T175743Z-1-001\BFMD_data\annotations\shot_type\KAPAL-API-Indonesia-Open-2025-Anders-Antonsen-DEN-3-vs.-Chou-Tien-Chen-TPE-6-F.json")
OUTPUT_METRICS = Path("work_dirs/bfmd_5min_eval_results.json")
OUTPUT_DEMO_VIDEO = Path("work_dirs/bfmd_5min_visual_demo.mp4")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_MEAN = torch.tensor([0.43216, 0.394666, 0.37645], device=DEVICE).view(1, 3, 1, 1)
_STD = torch.tensor([0.22803, 0.22145, 0.216989], device=DEVICE).view(1, 3, 1, 1)

LABEL_MAPPING = {
    "flick_serve": "serve",
    "net_kill": "net_attack",
    "push": "drive",  # fast flat push
    "block": "net_shot", # defense block near net
    "press": "drive",
}

def load_gt_events(start_frame_offset=18000, max_frames=9000):
    d = json.load(open(BFMD_JSON, encoding="utf-8"))
    shots = [r for r in d["annotations"][0]["result"] if r.get("from_name") == "shotType"]
    gt_list = []
    for s in shots:
        orig_f = s["value"]["ranges"][0]["start"]
        if start_frame_offset <= orig_f < start_frame_offset + max_frames:
            rel_f = orig_f - start_frame_offset
            raw_lbl = s["value"]["timelinelabels"][0]
            mapped_lbl = LABEL_MAPPING.get(raw_lbl, raw_lbl)
            gt_list.append({
                "orig_frame": orig_f,
                "rel_frame": rel_f,
                "raw_label": raw_lbl,
                "canonical_label": mapped_lbl,
            })
    gt_list.sort(key=lambda x: x["rel_frame"])
    return gt_list


def main():
    print(f"[INIT] Evaluating BFMD 5-minute video on {DEVICE}...")
    
    # 1. Load models
    pose_model = YOLO("models/checkpoints/pose/yolov8n-pose.pt")
    rgb_model = MultiTaskR2Plus1D()
    rgb_ckpt = torch.load("work_dirs/r2plus1d18_mixed_shuttleset_finebadminton/best.pth", map_location="cpu", weights_only=False)
    rgb_model.load_state_dict(rgb_ckpt["model"])
    rgb_model.to(DEVICE).eval()
    print("  [OK] Models loaded successfully.")

    # 2. Load Detections and GT
    scan_data = json.load(open(SCAN_JSON))
    detected_events = scan_data.get("events", [])
    gt_events = load_gt_events()
    print(f"  [INFO] GT strokes: {len(gt_events)} | Detected hits: {len(detected_events)}")

    # 3. Match GT with Detections (tolerance = 12 frames)
    tolerance = 12
    matched_pairs = []
    unmatched_gt = []
    matched_det_indices = set()

    for gt in gt_events:
        gt_f = gt["rel_frame"]
        candidates = []
        for d_idx, det in enumerate(detected_events):
            diff = abs(det["frame"] - gt_f)
            if diff <= tolerance:
                candidates.append((diff, d_idx, det))
        if candidates:
            candidates.sort(key=lambda x: x[0])
            best_diff, best_d_idx, best_det = candidates[0]
            matched_pairs.append({
                "gt": gt,
                "det": best_det,
                "diff_frames": best_diff,
            })
            matched_det_indices.add(best_d_idx)
        else:
            unmatched_gt.append(gt)

    print(f"  [INFO] Matched Hits: {len(matched_pairs)} / {len(gt_events)} (Recall: {100.0 * len(matched_pairs) / len(gt_events):.1f}%)")

    # 4. Classify each matched stroke
    cap = cv2.VideoCapture(str(VIDEO_5MIN))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    eval_results = []
    for pair in matched_pairs:
        gt = pair["gt"]
        det = pair["det"]
        hf = det["frame"]
        hside = det["side"]
        p_side_str = "top" if hside == "upper" else "bottom"

        start_f = max(0, hf - 10)
        end_f = min(total_frames - 1, hf + 10)

        rec = Record(
            sample_id=f"bfmd_{hf}", video_path=VIDEO_5MIN,
            start_frame=start_f, end_frame=end_f,
            coarse_label="lift", stroke_side="forehand",
            player_side=p_side_str, hit_frame=hf,
        )
        crop_box = detect_hitter_crop(rec, pose_model, 0.55)
        tensor = decode_clip(rec, 16, crop_box, 112).to(torch.float32).div_(255.0).to(DEVICE)
        tensor = (tensor - _MEAN) / _STD
        tensor = tensor.permute(1, 0, 2, 3).unsqueeze(0)

        with torch.inference_mode(), torch.autocast(device_type=DEVICE.type, dtype=torch.float16):
            s_logits, side_logits = rgb_model(tensor)
            s_probs = F.softmax(s_logits.float(), dim=-1)[0].cpu().numpy()
            side_probs = F.softmax(side_logits.float(), dim=-1)[0].cpu().numpy()

        pred_stroke = STROKE_CLASSES[int(np.argmax(s_probs))]
        pred_side = SIDE_CLASSES[int(np.argmax(side_probs))]
        s_conf = float(np.max(s_probs))
        side_conf = float(np.max(side_probs))

        is_match = (pred_stroke == gt["canonical_label"])
        # Top-2 match
        top2 = [STROKE_CLASSES[i] for i in np.argsort(s_probs)[::-1][:2]]
        is_top2_match = (gt["canonical_label"] in top2)

        eval_results.append({
            "gt_rel_frame": gt["rel_frame"],
            "detected_hit_frame": hf,
            "diff_frames": pair["diff_frames"],
            "gt_raw_label": gt["raw_label"],
            "gt_canonical_label": gt["canonical_label"],
            "pred_stroke": pred_stroke,
            "pred_side": pred_side,
            "stroke_conf": s_conf,
            "side_conf": side_conf,
            "is_stroke_match": is_match,
            "is_top2_match": is_top2_match,
            "top2_predictions": top2,
            "hitter_side": hside,
        })

    # Metrics
    total_eval = len(eval_results)
    stroke_acc = sum(1 for r in eval_results if r["is_stroke_match"]) / total_eval if total_eval > 0 else 0
    top2_acc = sum(1 for r in eval_results if r["is_top2_match"]) / total_eval if total_eval > 0 else 0

    print("-" * 80)
    print(f"BFMD 5-MINUTE VIDEO EVALUATION SUMMARY:")
    print(f"  Hit Detection Recall:     {len(matched_pairs)} / {len(gt_events)} ({100.0 * len(matched_pairs) / len(gt_events):.1f}%)")
    print(f"  Hit Detection Precision:  {len(matched_det_indices)} / {len(detected_events)} ({100.0 * len(matched_det_indices) / len(detected_events):.1f}%)")
    print(f"  Stroke Accuracy (Top-1):  {sum(1 for r in eval_results if r['is_stroke_match'])} / {total_eval} ({stroke_acc * 100:.1f}%)")
    print(f"  Stroke Accuracy (Top-2):  {sum(1 for r in eval_results if r['is_top2_match'])} / {total_eval} ({top2_acc * 100:.1f}%)")
    print("-" * 80)

    # Per class breakdown
    per_class = {}
    for r in eval_results:
        c = r["gt_canonical_label"]
        if c not in per_class:
            per_class[c] = {"total": 0, "correct": 0, "top2": 0}
        per_class[c]["total"] += 1
        if r["is_stroke_match"]:
            per_class[c]["correct"] += 1
        if r["is_top2_match"]:
            per_class[c]["top2"] += 1

    print(f"{'Class':<15} | {'Count':<6} | {'Top-1 Correct':<15} | {'Top-2 Correct':<15}")
    print("-" * 60)
    for c, stat in sorted(per_class.items(), key=lambda x: -x[1]["total"]):
        acc1 = 100.0 * stat["correct"] / stat["total"]
        acc2 = 100.0 * stat["top2"] / stat["total"]
        print(f"{c:<15} | {stat['total']:<6} | {stat['correct']:2d}/{stat['total']:2d} ({acc1:5.1f}%) | {stat['top2']:2d}/{stat['total']:2d} ({acc2:5.1f}%)")
    print("-" * 80)

    # Save results json
    payload = {
        "video": str(VIDEO_5MIN),
        "total_gt_strokes": len(gt_events),
        "total_detected_hits": len(detected_events),
        "hit_recall": len(matched_pairs) / len(gt_events),
        "hit_precision": len(matched_det_indices) / len(detected_events),
        "stroke_accuracy_top1": stroke_acc,
        "stroke_accuracy_top2": top2_acc,
        "per_class": per_class,
        "strokes": eval_results,
    }
    OUTPUT_METRICS.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Saved metrics to: {OUTPUT_METRICS}")

    # =========================================================================
    # 5. RENDER VISUAL DEMO VIDEO (Trực quan hóa)
    # Render an intense 600-frame rally (Frame 1100 to 1400 = 10 seconds: 6 continuous shots)
    # =========================================================================
    print("\n[RENDER] Creating visual demo video with overlay annotations...")
    demo_start = 1140
    demo_end = 1380  # ~8 seconds of pure rally
    
    cap = cv2.VideoCapture(str(VIDEO_5MIN))
    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out_writer = cv2.VideoWriter(str(OUTPUT_DEMO_VIDEO), fourcc, fps, (w, h))

    # Relevant results in this range
    range_results = [r for r in eval_results if demo_start - 30 <= r["detected_hit_frame"] <= demo_end + 30]

    cap.set(cv2.CAP_PROP_POS_FRAMES, demo_start)
    current_f = demo_start

    # Palette
    COLOR_BG = (20, 20, 20)
    COLOR_GREEN = (40, 200, 40)
    COLOR_RED = (40, 40, 230)
    COLOR_YELLOW = (0, 215, 255)
    COLOR_WHITE = (255, 255, 255)

    while current_f <= demo_end:
        ok, frame = cap.read()
        if not ok:
            break

        # Check if there is an active hit event in window [current_f - 10, current_f + 15]
        active_event = None
        for ev in range_results:
            if abs(ev["detected_hit_frame"] - current_f) <= 15:
                active_event = ev
                break

        # Draw Pose on players
        pose_res = pose_model.predict(frame, imgsz=640, conf=0.25, verbose=False)[0]
        if pose_res.keypoints is not None:
            for kp in pose_res.keypoints.data.cpu().numpy():
                vis = kp[:, 2] > 0.25
                if vis.sum() >= 5:
                    for pt in kp[vis, :2]:
                        cv2.circle(frame, (int(pt[0]), int(pt[1])), 4, (0, 255, 255), -1)

        # Draw Title Banner (Top Left)
        cv2.rectangle(frame, (20, 20), (550, 75), COLOR_BG, -1)
        cv2.putText(frame, "PBL6 AI BADMINTON ANALYZER", (30, 45), cv2.FONT_HERSHEY_DUPLEX, 0.7, (0, 255, 255), 2)
        cv2.putText(frame, f"Indonesia Open 2025 Finals | Frame: {current_f}", (30, 68), cv2.FONT_HERSHEY_PLAIN, 1.1, (200, 200, 200), 1)

        # Draw Active Hit Banner if near hit
        if active_event is not None:
            hit_f = active_event["detected_hit_frame"]
            is_match = active_event["is_stroke_match"]
            border_color = COLOR_GREEN if is_match else COLOR_YELLOW
            status_text = "HIT DETECTED - MATCH [OK]" if is_match else "HIT DETECTED"

            # Overlay box
            cv2.rectangle(frame, (20, h - 140), (620, h - 30), COLOR_BG, -1)
            cv2.rectangle(frame, (20, h - 140), (620, h - 30), border_color, 2)

            pred_text = f"Pred: {active_event['pred_side'].upper()} {active_event['pred_stroke'].upper()} ({active_event['stroke_conf']*100:.0f}%)"
            gt_text = f"GT:   {active_event['gt_raw_label'].upper()} [Diff: {active_event['diff_frames']} frames]"

            cv2.putText(frame, status_text, (35, h - 115), cv2.FONT_HERSHEY_DUPLEX, 0.65, border_color, 2)
            cv2.putText(frame, pred_text, (35, h - 85), cv2.FONT_HERSHEY_SIMPLEX, 0.65, COLOR_WHITE, 2)
            cv2.putText(frame, gt_text, (35, h - 55), cv2.FONT_HERSHEY_SIMPLEX, 0.60, (200, 200, 200), 1)

            # Draw impact circle on center
            cv2.circle(frame, (w - 60, h - 85), 20, border_color, -1)
            cv2.putText(frame, "HIT", (w - 75, h - 80), cv2.FONT_HERSHEY_PLAIN, 0.9, (0, 0, 0), 2)

        out_writer.write(frame)
        current_f += 1

    cap.release()
    out_writer.release()
    print(f"  [OK] Rendered visual demo video to: {OUTPUT_DEMO_VIDEO}")


if __name__ == "__main__":
    main()

