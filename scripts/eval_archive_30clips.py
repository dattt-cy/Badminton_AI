"""Evaluate 30 clips (5 per category) from C:\\Users\\ADMIN\\Downloads\\archive

Pipeline:
1. Stage 1: Hit Detection (R(2+1)D-18 Spatio-Temporal Hit Model) -> detects hit timestamps & hitter side.
2. Stage 2: Stroke & Side Multi-Task Classification (R(2+1)D Multi-Task Backbone) on hitter crops.
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
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ai_classifier.models import MultiTaskR2Plus1D, SIDE_CLASSES, STROKE_CLASSES
from scripts.inference.classify_shuttleset_rgb_multitask import Record, detect_hitter_crop, decode_clip

ARCHIVE_DIR = Path(r"C:\Users\ADMIN\Downloads\archive")
OUTPUT_JSON = Path("work_dirs/archive_30clips_eval_results.json")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

_MEAN = torch.tensor([0.43216, 0.394666, 0.37645], device=DEVICE).view(1, 3, 1, 1)
_STD = torch.tensor([0.22803, 0.22145, 0.216989], device=DEVICE).view(1, 3, 1, 1)


def scan_hits_in_clip(
    video_path: Path,
    hit_model: torch.nn.Module,
    stride: int = 4,
    threshold: float = 0.40,
    nms_radius: int = 10,
) -> list[dict]:
    """Scan clip for hit events using R(2+1)D-18 Hit Detector."""
    cap = cv2.VideoCapture(str(video_path))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames < 16:
        cap.release()
        return [{"frame": total_frames // 2, "side": "upper", "score": 1.0}]

    half = 8
    frame_buffer = deque(maxlen=16)
    pending_clips = []
    pending_centers = []
    first_center = half
    raw_events = []

    def flush():
        if not pending_clips:
            return
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
        if not ok:
            break
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

    # NMS
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


def main():
    print(f"[INIT] Evaluating 30 clips on {DEVICE}...")
    start_time = time.time()

    # 1. Load Hit Model
    from torchvision.models.video import r2plus1d_18
    hit_model = r2plus1d_18(weights=None)
    hit_model.fc = torch.nn.Linear(hit_model.fc.in_features, 3)
    hit_ckpt = torch.load("work_dirs/r2plus1d18_hit_full/best.pth", map_location="cpu", weights_only=False)
    hit_model.load_state_dict(hit_ckpt.get("model", hit_ckpt))
    hit_model.to(DEVICE).eval()
    print("  [OK] Loaded Hit Model (r2plus1d18_hit_full)")

    # 2. Load Pose Model
    pose_model = YOLO("models/checkpoints/pose/yolov8n-pose.pt")
    print("  [OK] Loaded YOLOv8 Pose Model")

    # 3. Load Multitask RGB Model
    rgb_model = MultiTaskR2Plus1D()
    rgb_ckpt = torch.load("work_dirs/r2plus1d18_mixed_shuttleset_finebadminton/best.pth", map_location="cpu", weights_only=False)
    rgb_model.load_state_dict(rgb_ckpt["model"])
    rgb_model.to(DEVICE).eval()
    print("  [OK] Loaded RGB Multitask Classifier")

    folders = [f for f in sorted(os.listdir(ARCHIVE_DIR)) if (ARCHIVE_DIR / f).is_dir()]
    print(f"\n[INFO] Found {len(folders)} classes: {folders}")

    selected_clips = []
    for f in folders:
        clips = sorted([c for c in os.listdir(ARCHIVE_DIR / f) if c.endswith(".mp4") and not c.startswith(".")])
        for c in clips[:5]:
            selected_clips.append((f, c, ARCHIVE_DIR / f / c))

    print(f"[INFO] Total selected clips to test: {len(selected_clips)}")
    print("-" * 80)

    results = []
    for idx, (folder, clip_name, clip_path) in enumerate(selected_clips, 1):
        parts = folder.split("_")
        gt_side = parts[0]
        gt_stroke = "_".join(parts[1:])
        gt_compound = f"{gt_side}_{gt_stroke}"

        cap = cv2.VideoCapture(str(clip_path))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()

        # Step 1: Hit Detection
        events = scan_hits_in_clip(clip_path, hit_model, stride=4, threshold=0.35)

        # Step 2: Classify all detected events
        classified_events = []
        for ev in events:
            hf = ev["frame"]
            hside = ev["side"]
            p_side_str = "top" if hside == "upper" else "bottom"
            start_f = max(0, hf - 10)
            end_f = min(total_frames - 1, hf + 10)

            rec = Record(
                sample_id=f"{clip_path.stem}_{hf}", video_path=clip_path,
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

            classified_events.append({
                "hit_frame": hf,
                "hit_side": hside,
                "hit_score": ev["score"],
                "pred_stroke": pred_stroke,
                "pred_side": pred_side,
                "stroke_conf": s_conf,
                "side_conf": side_conf,
                "compound": f"{pred_side}_{pred_stroke}",
                "is_stroke_match": (pred_stroke == gt_stroke),
                "is_side_match": (pred_side == gt_side),
                "is_compound_match": (pred_stroke == gt_stroke and pred_side == gt_side),
            })

        # Clip-level summary
        stroke_hit_matches = [e for e in classified_events if e["is_stroke_match"]]
        compound_hit_matches = [e for e in classified_events if e["is_compound_match"]]

        # Best confidence event
        best_event = max(classified_events, key=lambda x: x["stroke_conf"] * x["side_conf"]) if classified_events else None

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
                f"      [{mark}] Frame {ev['hit_frame']:2d} ({ev['hit_side']:5s}, hit_sc={ev['hit_score']:.2f}): "
                f"{ev['pred_side']:9s} {ev['pred_stroke']:10s} (conf: {ev['stroke_conf']:.2f}, {ev['side_conf']:.2f})",
                flush=True,
            )

        results.append({
            "index": idx,
            "folder": folder,
            "clip_name": clip_name,
            "total_frames": total_frames,
            "gt_side": gt_side,
            "gt_stroke": gt_stroke,
            "gt_compound": gt_compound,
            "detected_events": classified_events,
            "clip_has_stroke_match": clip_has_stroke,
            "clip_has_compound_match": clip_has_compound,
            "best_event": best_event,
        })

    elapsed = time.time() - start_time
    total_clips = len(results)
    stroke_matches = sum(1 for r in results if r["clip_has_stroke_match"])
    compound_matches = sum(1 for r in results if r["clip_has_compound_match"])

    # Per-class metrics
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
        "seconds_per_clip": round(elapsed / total_clips, 2),
        "per_class_stats": class_stats,
        "clips": results,
    }

    OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_JSON.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print("\n" + "=" * 80)
    print(f"EVALUATION COMPLETE ON 30 CLIPS in {elapsed:.1f}s ({elapsed/total_clips:.2f}s/clip)")
    print(f"Stroke Match Rate:   {stroke_matches}/{total_clips} ({summary['stroke_accuracy']*100:.1f}%)")
    print(f"Compound Match Rate: {compound_matches}/{total_clips} ({summary['compound_accuracy']*100:.1f}%)")
    print("-" * 80)
    print(f"{'Class Name':<22} | {'Clips':<5} | {'Stroke Match':<12} | {'Compound Match':<14}")
    print("-" * 80)
    for c, stat in class_stats.items():
        s_pct = 100.0 * stat["stroke_ok"] / stat["total"]
        cp_pct = 100.0 * stat["compound_ok"] / stat["total"]
        print(f"{c:<22} | {stat['total']:<5} | {stat['stroke_ok']}/5 ({s_pct:5.1f}%) | {stat['compound_ok']}/5 ({cp_pct:5.1f}%)")
    print("=" * 80)
    print(f"Results saved to: {OUTPUT_JSON}")


if __name__ == "__main__":
    main()
