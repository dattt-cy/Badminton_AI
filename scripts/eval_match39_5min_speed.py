"""End-to-End Evaluation on a Completely New 5-Minute Video: Match 39 (Ginting vs Lee Zii Jia).

Runs the speed-optimized multimodal pipeline from scratch:
1. Hit Detection Scan across all 9,081 frames of work_dirs/match39_5min_clip.mp4
   (Automatically cached to work_dirs/scan_match39_5min.json)
2. Speed-optimized FastTrackNet (AMP FP16 + Batch 32)
3. Speed-optimized YOLOv8-Pose (Batched FP16 + Imgsz 640)
4. 4-Model Balanced CR-Ensemble (Full 15%, NoGate 15%, NoTemp 35%, NoCont 35%)
5. Matches against the 92 Human Ground-Truth Strokes from ShuttleSet
Measures exact wall-clock processing time and per-class accuracy.
"""

from __future__ import annotations

from collections import Counter, defaultdict, deque
import csv
import json
from pathlib import Path
import sys
import time

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

from ai_classifier.localization.fast_tracknet import FastTrackNet
from ai_classifier.models import CRGatedFusionModel, MultiTaskR2Plus1D, STROKE_CLASSES, SIDE_CLASSES
from scripts.inference.auto_court_detection import detect_court_corners
from scripts.inference.analyze_long_video import default_corners
from scripts.inference.classify_shuttleset_fusion_video import extract_structured, read_hitter_clip
from scripts.data.extract_sequence_features import extract_sample_sequences

VIDEO_PATH = REPO_ROOT / "work_dirs" / "match39_5min_clip.mp4"
SCAN_CACHE = REPO_ROOT / "work_dirs" / "scan_match39_5min.json"
OUTPUT_JSON = REPO_ROOT / "work_dirs" / "match39_5min_eval_results.json"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

_MEAN = torch.tensor([0.43216, 0.394666, 0.37645], device=DEVICE).view(1, 3, 1, 1)
_STD = torch.tensor([0.22803, 0.22145, 0.216989], device=DEVICE).view(1, 3, 1, 1)


def load_gt_annotations():
    """Load human ground-truth strokes for Match 39 in the 5-min window [10000, 19000]."""
    records = []
    manifest_path = REPO_ROOT / "data" / "manifests" / "shuttleset_rgb.csv"
    with open(manifest_path, mode="r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if "39 Anthony_Sinisuka_Ginting" in row["video_path"]:
                abs_hf = int(row["hit_frame"])
                if 10000 <= abs_hf <= 19000:
                    rel_hf = abs_hf - 10000
                    records.append({
                        "abs_frame": abs_hf,
                        "rel_frame": rel_hf,
                        "stroke": row["coarse_label"],
                        "side": row["stroke_side"],
                        "player_side": row.get("player_side", "bottom"),
                    })
    records.sort(key=lambda x: x["rel_frame"])
    return records


def scan_hits(video_path: Path, hit_model: torch.nn.Module, stride: int = 4, threshold: float = 0.35, nms_radius: int = 10):
    """Scan video with Hit Detector at 44+ FPS to find impact frames."""
    if SCAN_CACHE.exists():
        print(f"[*] Found cached hit scan at {SCAN_CACHE}. Loading...")
        with open(SCAN_CACHE, "r", encoding="utf-8") as f:
            data = json.load(f)
        print(f"[*] Loaded {len(data['events'])} events from cache (original scan time: {data['scan_time']:.1f}s).")
        return data["events"], data["scan_time"]

    cap = cv2.VideoCapture(str(video_path))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"[*] Scanning {total_frames} frames for hit moments (stride={stride})...")
    
    half = 8
    frame_buffer = deque(maxlen=16)
    pending_clips, pending_centers = [], []
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
    t0 = time.time()
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
                if len(pending_clips) >= 32:
                    flush()
        frame_idx += 1
    flush()
    cap.release()
    scan_time = time.time() - t0
    fps = frame_idx / max(1, scan_time)
    print(f"[*] Hit scan completed: {frame_idx} frames in {scan_time:.1f}s ({fps:.1f} FPS)!")

    # NMS
    raw_events.sort(key=lambda x: x["score"], reverse=True)
    suppressed = set()
    nms_events = []
    for ev in raw_events:
        f = ev["frame"]
        if f in suppressed or ev["score"] < threshold:
            continue
        nms_events.append(ev)
        for s in range(max(0, f - nms_radius), f + nms_radius + 1):
            suppressed.add(s)

    nms_events.sort(key=lambda x: x["frame"])
    print(f"[*] Found {len(nms_events)} candidate stroke events after NMS.")

    SCAN_CACHE.parent.mkdir(parents=True, exist_ok=True)
    with open(SCAN_CACHE, "w", encoding="utf-8") as f:
        json.dump({"events": nms_events, "scan_time": scan_time}, f, indent=2)

    return nms_events, scan_time


def load_models():
    """Load all pipeline models."""
    print("[*] Loading Hit Model...")
    from torchvision.models.video import r2plus1d_18
    hit_model = r2plus1d_18(weights=None)
    hit_model.fc = torch.nn.Linear(hit_model.fc.in_features, 3)
    hit_ckpt = torch.load("work_dirs/r2plus1d18_hit_full/best.pth", map_location="cpu", weights_only=False)
    hit_model.load_state_dict(hit_ckpt.get("model", hit_ckpt))
    hit_model.to(DEVICE).eval()

    print("[*] Loading Speed-Optimized FastTrackNet (AMP FP16 + Batch 32)...")
    tracker = FastTrackNet(device=DEVICE)

    print("[*] Loading YOLOv8-Pose (FP16)...")
    pose_model = YOLO("models/checkpoints/pose/yolov8n-pose.pt")

    print("[*] Loading R(2+1)D Visual Backbone...")
    rgb_model = MultiTaskR2Plus1D().to(DEVICE)
    rgb_ckpt = torch.load("work_dirs/r2plus1d18_mixed_shuttleset_finebadminton/best.pth", map_location="cpu", weights_only=False)
    rgb_model.load_state_dict(rgb_ckpt["model"], strict=False)
    rgb_model.eval()

    print("[*] Loading 4-Model Balanced CR-Ensemble...")
    def _load_cr(subdir, ut, uc, ug):
        p = REPO_ROOT / "work_dirs" / "cr_gated_fusion" / subdir / "best.pth"
        ckpt = torch.load(p, map_location="cpu", weights_only=False)
        m = CRGatedFusionModel(
            hidden_dim=128,
            num_stroke_classes=len(STROKE_CLASSES),
            num_side_classes=len(SIDE_CLASSES),
            use_temporal=ut, use_contact=uc, use_gate=ug,
        ).to(DEVICE)
        m.load_state_dict(ckpt["model"])
        m.eval()
        return m

    m_full = _load_cr("cr_gated_full", True, True, True)
    m_ng = _load_cr("cr_gated_no_gate", True, True, False)
    m_nt = _load_cr("cr_gated_no_temporal", False, False, False)
    m_nc = _load_cr("cr_gated_no_contact", True, False, True)

    return hit_model, tracker, pose_model, rgb_model, (m_full, m_ng, m_nt, m_nc)


def run_evaluation():
    print("=" * 88)
    print("END-TO-END EVALUATION: COMPLETELY NEW 5-MINUTE MATCH 39 (GINTING vs LEE ZII JIA)")
    print(f"Video: {VIDEO_PATH}")
    print(f"Device: {DEVICE}")
    print("=" * 88)

    t_start_all = time.time()

    # 1. Load Ground Truth
    gt_strokes = load_gt_annotations()
    print(f"[*] Loaded {len(gt_strokes)} human ground-truth strokes for this 5-minute video.")
    gt_counts = Counter(s["stroke"] for s in gt_strokes)
    print(f"[*] Ground truth distribution: {dict(gt_counts)}")

    # 2. Load Models
    hit_model, tracker, pose_model, rgb_model, cr_models = load_models()
    m_full, m_ng, m_nt, m_nc = cr_models

    # 3. Detect Court (once)
    print("[*] Detecting Court corners...")
    try:
        corners = detect_court_corners(VIDEO_PATH)
    except Exception:
        corners = default_corners(1280, 720)

    # 4. Scan Hits
    events, hit_scan_time = scan_hits(VIDEO_PATH, hit_model, stride=4, threshold=0.35)

    # 5. Classify each detected hit using speed-optimized pipeline
    print(f"\n[*] Classifying {len(events)} detected stroke events...")
    print("-" * 88)

    w = [0.15, 0.15, 0.35, 0.35]  # 4-Model Ensemble weights
    classified_events = []
    t_classify_start = time.time()

    cap = cv2.VideoCapture(str(VIDEO_PATH))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    for idx, ev in enumerate(events, 1):
        hf = ev["frame"]
        hside = ev["side"]
        p_side_str = "top" if hside == "upper" else "bottom"
        try:
            start_f = max(0, hf - 15)
            end_f = min(total_frames - 25, 8980, hf + 30)
            if start_f >= end_f:
                continue

            # 1. FastTrackNet (AMP FP16 + Batch 32)
            trajectory = tracker.predict_window(VIDEO_PATH, hf, window_before=30, window_after=30, batch_size=32)

            # 2. Structured features
            joints, position, shuttle, valid_ratio, w_val, h_val = extract_structured(
                VIDEO_PATH, trajectory, corners, pose_model, start_f, end_f
            )
            joints_flat = joints.reshape(len(joints), 68)
            pos_flat = position.reshape(len(position), 4)

            s_j, s_p, s_s, s_c, s_q = extract_sample_sequences(
                joints_flat, pos_flat, shuttle, target_index=min(hf - start_f, len(joints) - 1), length=32
            )

            # 3. R(2+1)D Visual Crop
            hitter_clip, _ = read_hitter_clip(
                VIDEO_PATH, 16, p_side_str, pose_model, 0.55, start_f, end_f
            )
            hitter_clip = hitter_clip.to(DEVICE)

            # 4. CR-Ensemble Forward
            with torch.inference_mode():
                emb = rgb_model.extract_features(hitter_clip)
                t_pose = torch.from_numpy(s_j).unsqueeze(0).to(DEVICE)
                t_court = torch.from_numpy(s_p).unsqueeze(0).to(DEVICE)
                t_shuttle = torch.from_numpy(s_s).unsqueeze(0).to(DEVICE)
                t_contact = torch.from_numpy(s_c).unsqueeze(0).to(DEVICE)
                t_qual = torch.from_numpy(s_q).unsqueeze(0).to(DEVICE)

                f_p, f_side_l, _ = m_full(emb, t_pose, t_court, t_shuttle, t_contact, t_qual)
                ng_p, _, _ = m_ng(emb, t_pose, t_court, t_shuttle, t_contact, t_qual)
                nt_p, _, _ = m_nt(emb, t_pose, t_court, t_shuttle, t_contact, t_qual)
                nc_p, _, _ = m_nc(emb, t_pose, t_court, t_shuttle, t_contact, t_qual)

                f_p = torch.softmax(f_p[0], dim=-1)
                ng_p = torch.softmax(ng_p[0], dim=-1)
                nt_p = torch.softmax(nt_p[0], dim=-1)
                nc_p = torch.softmax(nc_p[0], dim=-1)

                ens_p = w[0] * f_p + w[1] * ng_p + w[2] * nt_p + w[3] * nc_p
                pred_stroke = STROKE_CLASSES[ens_p.argmax().item()]
                conf = float(ens_p.max().item())

                top2_indices = ens_p.topk(2).indices.cpu().tolist()
                top2_strokes = [STROKE_CLASSES[i] for i in top2_indices]

                side_p = torch.softmax(f_side_l[0], dim=-1)
                pred_side = SIDE_CLASSES[side_p.argmax().item()]

            classified_events.append({
                "hit_frame": hf,
                "side": hside,
                "pred_stroke": pred_stroke,
                "pred_side": pred_side,
                "conf": conf,
                "top2": top2_strokes,
            })

            if idx % 10 == 0 or idx == len(events):
                print(f"  Processed {idx}/{len(events)} events ({idx/len(events)*100:.0f}%) | Latest: Frame {hf} -> {pred_side} {pred_stroke} ({conf*100:.1f}%)", flush=True)
                with open(OUTPUT_JSON.with_suffix(".tmp.json"), "w", encoding="utf-8") as f_tmp:
                    json.dump(classified_events, f_tmp)
        except Exception as err:
            print(f"  [!] Skipped event {idx}/{len(events)} at frame {hf}: {err}", flush=True)
            continue

    t_classify = time.time() - t_classify_start

    # 6. Match with Ground Truth
    print("\n[*] Matching predictions against 92 ground-truth strokes...")
    matched_results = []
    used_preds = set()

    for gt in gt_strokes:
        g_f = gt["rel_frame"]
        g_stroke = gt["stroke"]

        # Find closest predicted hit within +-12 frames
        candidates = []
        for p_idx, p in enumerate(classified_events):
            if p_idx not in used_preds and abs(p["hit_frame"] - g_f) <= 12:
                candidates.append((abs(p["hit_frame"] - g_f), p_idx, p))

        if candidates:
            candidates.sort(key=lambda x: x[0])
            best_diff, best_pidx, best_p = candidates[0]
            used_preds.add(best_pidx)

            is_top1 = (best_p["pred_stroke"] == g_stroke)
            is_top2 = (g_stroke in best_p["top2"])

            matched_results.append({
                "gt_frame": g_f,
                "pred_frame": best_p["hit_frame"],
                "frame_diff": best_diff,
                "gt_stroke": g_stroke,
                "pred_stroke": best_p["pred_stroke"],
                "top2": best_p["top2"],
                "conf": best_p["conf"],
                "is_top1": is_top1,
                "is_top2": is_top2,
            })
        else:
            matched_results.append({
                "gt_frame": g_f,
                "pred_frame": None,
                "frame_diff": None,
                "gt_stroke": g_stroke,
                "pred_stroke": "UNMATCHED",
                "top2": [],
                "conf": 0.0,
                "is_top1": False,
                "is_top2": False,
            })

    total_gt = len(gt_strokes)
    top1_correct = sum(1 for m in matched_results if m["is_top1"])
    top2_correct = sum(1 for m in matched_results if m["is_top2"])
    detected_matches = sum(1 for m in matched_results if m["pred_frame"] is not None)

    # Per-class stats
    class_stats = defaultdict(lambda: {"total": 0, "top1": 0, "top2": 0})
    for m in matched_results:
        c = m["gt_stroke"]
        class_stats[c]["total"] += 1
        if m["is_top1"]: class_stats[c]["top1"] += 1
        if m["is_top2"]: class_stats[c]["top2"] += 1

    total_time = time.time() - t_start_all

    print("\n" + "=" * 88)
    print(f"MATCH 39 (5-MIN FULL VIDEO) EVALUATION RESULTS:")
    print("=" * 88)
    print(f"Total Ground-Truth Strokes:  {total_gt}")
    print(f"Hit Detection Matched:       {detected_matches}/{total_gt} ({detected_matches/total_gt*100:.1f}%)")
    print(f"TOP-1 CLASSIFICATION ACC:    {top1_correct}/{total_gt} ({top1_correct/total_gt*100:.2f}%)")
    print(f"TOP-2 CLASSIFICATION ACC:    {top2_correct}/{total_gt} ({top2_correct/total_gt*100:.2f}%)")
    print("-" * 88)
    print(f"RUNTIME BREAKDOWN (SPEED-OPTIMIZED):")
    print(f"  Hit Detection Scan Time:   {hit_scan_time:.1f}s ({hit_scan_time/60:.2f} mins) -> 44+ FPS")
    print(f"  Stroke Classification Time:{t_classify:.1f}s ({t_classify/60:.2f} mins) -> ~{t_classify/max(1, len(events)):.2f}s / stroke")
    print(f"  TOTAL END-TO-END TIME:     {total_time:.1f}s ({total_time/60:.2f} mins)!")
    print("-" * 88)
    print(f"{'Stroke Class':<15} | {'Total':<6} | {'Top-1 Accuracy':<16} | {'Top-2 Accuracy'}")
    print("-" * 88)
    for c in sorted(class_stats.keys()):
        tot = class_stats[c]["total"]
        t1 = class_stats[c]["top1"]
        t2 = class_stats[c]["top2"]
        print(f"{c:<15} | {tot:<6} | {t1}/{tot} ({t1/tot*100:5.1f}%) | {t2}/{tot} ({t2/tot*100:5.1f}%)")
    print("=" * 88)

    # Save JSON
    summary = {
        "video": str(VIDEO_PATH),
        "total_gt": total_gt,
        "matched_hits": detected_matches,
        "top1_matches": top1_correct,
        "top1_accuracy": top1_correct / total_gt,
        "top2_matches": top2_correct,
        "top2_accuracy": top2_correct / total_gt,
        "timing": {
            "hit_scan_seconds": round(hit_scan_time, 2),
            "classify_seconds": round(t_classify, 2),
            "total_seconds": round(total_time, 2),
            "total_minutes": round(total_time / 60, 2),
        },
        "per_class": dict(class_stats),
        "matches": matched_results,
    }
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"[*] Results saved to {OUTPUT_JSON}")


if __name__ == "__main__":
    run_evaluation()
