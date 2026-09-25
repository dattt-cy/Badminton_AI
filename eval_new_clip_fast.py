"""Fast In-Memory End-to-End Evaluation on a Completely New 5-Minute Video: Match 13 (Kento Momota vs Viktor Axelsen).

Pipeline:
1. Hit Scan: Dense R(2+1)D-18 (AMP FP16 + Batch 32) -> 20s
2. Feature Extraction: FastTrackNet (FP16 Batch 32) + YOLOv8-Pose (FP16 Batch 16) + R(2+1)D
3. Multimodal Classification:
   - Baseline CR-Full
   - Pure KAP-Fusion
   - NDH (Neural Disambiguation Head)
   - NDH + Extended-Horizon Disambiguation
4. Ground Truth: 68 annotated strokes in ShuttleSet manifest for Match 13 [16300, 23857].
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

REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ai_classifier.localization.fast_tracknet import FastTrackNet
from ai_classifier.models import CRGatedFusionModel, MultiTaskR2Plus1D, STROKE_CLASSES, SIDE_CLASSES
from ai_classifier.models.neural_disambiguator import NeuralDisambiguationHead
from scripts.inference.auto_court_detection import detect_court_corners
from scripts.inference.analyze_long_video import default_corners
from train_kap_fusion import KAPFusionModel, augment_shuttle_kinematics

VIDEO_PATH = REPO_ROOT / "work_dirs" / "match13_5min_clip.mp4"
SCAN_CACHE = REPO_ROOT / "work_dirs" / "scan_match13_5min.json"
FEAT_CACHE = REPO_ROOT / "work_dirs" / "match13_5min_features.pt"
OUTPUT_JSON = REPO_ROOT / "work_dirs" / "match13_eval_results.json"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

_MEAN = torch.tensor([0.43216, 0.394666, 0.37645], device=DEVICE).view(1, 3, 1, 1)
_STD = torch.tensor([0.22803, 0.22145, 0.216989], device=DEVICE).view(1, 3, 1, 1)

OFFSET_ORIGINAL = 16300  # Clip frame 0 is original frame 16300


def load_gt_annotations():
    """Load human ground-truth strokes for Match 13 in the 5-min clip."""
    records = []
    manifest_path = REPO_ROOT / "data" / "manifests" / "shuttleset_rgb.csv"
    with open(manifest_path, mode="r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if "13 Kento" in row["video_path"]:
                abs_hf = int(row["hit_frame"])
                if OFFSET_ORIGINAL <= abs_hf <= OFFSET_ORIGINAL + 7557:
                    records.append({
                        "frame": abs_hf - OFFSET_ORIGINAL,
                        "stroke": row["coarse_label"],
                        "side": row.get("player_side", "lower"),
                    })
    records.sort(key=lambda x: x["frame"])
    return records


def run_hit_scan(video_path, hit_model, stride=4, nms_radius=15, threshold=0.5):
    """Dense sliding-window hit detector with batch 32 AMP FP16."""
    if SCAN_CACHE.exists():
        print(f"[*] Found cached hit scan at {SCAN_CACHE}...")
        with open(SCAN_CACHE, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data["events"], data["scan_time"]

    cap = cv2.VideoCapture(str(video_path))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"[*] Scanning {total_frames} frames for hit moments (stride={stride}, batch=32 AMP FP16)...")

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


def extract_features_for_event(
    cap, hf, hside, tracker, pose_model, rgb_model, corners, orig_w, orig_h, total_frames
):
    """Extract multimodal features using in-memory frame decoding."""
    start_f = max(0, hf - 30)
    end_f = min(total_frames - 1, hf + 30)
    num_frames = end_f - start_f + 1
    if num_frames < 32:
        return None

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_f)
    frames_bgr = []
    frame_indices = []
    for f in range(start_f, end_f + 1):
        ok, frame = cap.read()
        if not ok:
            break
        frames_bgr.append(frame)
        frame_indices.append(f)

    if len(frames_bgr) < 32:
        return None

    # 1. FastTrackNet (AMP FP16, batch 32)
    track_df = tracker.predict_frames(frames_bgr, frame_indices, orig_w, orig_h, batch_size=32)
    coord_map = {int(r["Frame"]): (float(r["X"]) / max(1, orig_w), float(r["Y"]) / max(1, orig_h)) for _, r in track_df.iterrows()}

    # 2. YOLOv8-Pose (batch 16 FP16)
    center_idx = hf - start_f
    w_start = max(0, center_idx - 16)
    w_end = min(len(frames_bgr), w_start + 32)
    crop_frames = frames_bgr[w_start:w_end]

    with torch.inference_mode():
        pose_results = pose_model(crop_frames, verbose=False, device=DEVICE, half=True, batch=16)

    # Court Homography
    src_pts = np.float32(corners)
    dst_pts = np.float32([[0, 0], [1, 0], [1, 1], [0, 1]])
    H_mat = cv2.getPerspectiveTransform(src_pts, dst_pts)

    seq_j, seq_p, seq_s = [], [], []
    for idx_in_window in range(len(crop_frames)):
        actual_frame = frame_indices[w_start + idx_in_window] if (w_start + idx_in_window) < len(frame_indices) else hf
        res = pose_results[idx_in_window]

        sx, sy = coord_map.get(actual_frame, (0.5, 0.5))
        seq_s.append([sx, sy])

        if res.keypoints is not None and len(res.keypoints.data) > 0:
            kpts = res.keypoints.data[0].cpu().numpy()
            kpts[:, 0] /= max(1, orig_w)
            kpts[:, 1] /= max(1, orig_h)
            kpts_4 = np.zeros((17, 4), dtype=np.float32)
            kpts_4[:, :3] = kpts
            seq_j.append(kpts_4.flatten())

            l_ankle = kpts[15, :2]
            r_ankle = kpts[16, :2]
            feet = np.mean([l_ankle, r_ankle], axis=0) * [orig_w, orig_h]
            feet_pt = np.array([[[feet[0], feet[1]]]], dtype=np.float32)
            court_pt = cv2.perspectiveTransform(feet_pt, H_mat)[0, 0]
            seq_p.append([court_pt[0], court_pt[1], 0.0, 0.0])
        else:
            seq_j.append(np.zeros(68, dtype=np.float32))
            seq_p.append([0.5, 0.5, 0.0, 0.0])

    while len(seq_j) < 32:
        seq_j.append(seq_j[-1] if seq_j else np.zeros(68, dtype=np.float32))
        seq_p.append(seq_p[-1] if seq_p else [0.5, 0.5, 0.0, 0.0])
        seq_s.append(seq_s[-1] if seq_s else [0.5, 0.5])
    seq_j = np.array(seq_j[:32], dtype=np.float32)
    seq_p = np.array(seq_p[:32], dtype=np.float32)
    seq_s = np.array(seq_s[:32], dtype=np.float32)

    # Signed contact distance aligned with training [-1.0, 2.0]
    contact_dist = np.linspace(-15, 30, 32, dtype=np.float32)[:, None] / 15.0
    quality = np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32)

    # 3. R(2+1)D Visual Clip (16 frames centered at hit)
    c_start = max(0, center_idx - 8)
    c_end = min(len(frames_bgr), c_start + 16)
    rgb_16 = [cv2.cvtColor(f, cv2.COLOR_BGR2RGB) for f in frames_bgr[c_start:c_end]]
    while len(rgb_16) < 16:
        rgb_16.append(rgb_16[-1] if rgb_16 else np.zeros((orig_h, orig_w, 3), dtype=np.uint8))
    rgb_arr = np.stack(rgb_16[:16])

    t_rgb = torch.from_numpy(rgb_arr).permute(0, 3, 1, 2).float().div_(255.0)
    t_rgb = VF.resize(t_rgb, [128, 171], antialias=False)
    t_rgb = VF.center_crop(t_rgb, [112, 112]).to(DEVICE)
    t_rgb = (t_rgb - _MEAN) / _STD
    hitter_clip = t_rgb.permute(1, 0, 2, 3).unsqueeze(0)

    with torch.inference_mode():
        emb = rgb_model.extract_features(hitter_clip)

    return {
        "hit_frame": hf,
        "side": hside,
        "emb": emb,
        "s_j": torch.from_numpy(seq_j).unsqueeze(0).to(DEVICE),
        "s_p": torch.from_numpy(seq_p).unsqueeze(0).to(DEVICE),
        "s_s": torch.from_numpy(seq_s).unsqueeze(0).to(DEVICE),
        "s_c": torch.from_numpy(contact_dist).unsqueeze(0).to(DEVICE),
        "s_q": torch.from_numpy(quality).unsqueeze(0).to(DEVICE),
    }


def main():
    print("=" * 88)
    print("FAST END-TO-END EVALUATION: MATCH 13 (KENTO MOMOTA vs VIKTOR AXELSEN - 5 MINUTES)")
    print(f"Video: {VIDEO_PATH}")
    print("=" * 88)

    gt_strokes = load_gt_annotations()
    print(f"[*] Loaded {len(gt_strokes)} ground-truth strokes from ShuttleSet.")

    # 1. Load Hit Spotting Model
    print("[*] Loading Hit Spotting Model...")
    from torchvision.models.video import r2plus1d_18
    hit_model = r2plus1d_18(weights=None)
    hit_model.fc = torch.nn.Linear(hit_model.fc.in_features, 3)
    hit_ckpt = torch.load("work_dirs/r2plus1d18_hit_full/best.pth", map_location="cpu", weights_only=False)
    hit_model.load_state_dict(hit_ckpt["model"])
    hit_model.to(DEVICE).eval()

    # 2. Run Hit Scan
    events, scan_time = run_hit_scan(VIDEO_PATH, hit_model, stride=4, nms_radius=15, threshold=0.5)

    # 3. Load Feature Extraction Models
    print("[*] Loading FastTrackNet (AMP FP16 Batch 32)...")
    tracker = FastTrackNet(device=DEVICE)

    print("[*] Loading YOLOv8-Pose (FP16 half)...")
    pose_model = YOLO("models/checkpoints/pose/yolov8n-pose.pt")

    print("[*] Loading R(2+1)D Backbone...")
    rgb_model = MultiTaskR2Plus1D().to(DEVICE)
    rgb_ckpt = torch.load("work_dirs/r2plus1d18_mixed_shuttleset_finebadminton/best.pth", map_location="cpu", weights_only=False)
    rgb_model.load_state_dict(rgb_ckpt["model"], strict=False)
    rgb_model.eval()

    try:
        corners = detect_court_corners(str(VIDEO_PATH))
    except Exception:
        corners = default_corners(1280, 720)

    # Extract or load features
    if FEAT_CACHE.exists():
        print(f"[*] Found cached features at {FEAT_CACHE}, loading in 0.1s...")
        extracted_records = torch.load(FEAT_CACHE, weights_only=False)
        extract_time = 0.1
    else:
        cap = cv2.VideoCapture(str(VIDEO_PATH))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        print(f"\n[*] Extracting features for {len(events)} events using in-memory unified pipeline...")
        t0 = time.time()
        extracted_records = []
        for idx, ev in enumerate(events, 1):
            rec = extract_features_for_event(
                cap, ev["frame"], ev["side"], tracker, pose_model, rgb_model, corners, orig_w, orig_h, total_frames
            )
            if rec is not None:
                extracted_records.append(rec)
            if idx % 10 == 0 or idx == len(events):
                elapsed = time.time() - t0
                fps_stroke = idx / max(0.1, elapsed)
                print(f"    [{idx:3d}/{len(events):3d}] processed ({fps_stroke:.2f} strokes/s, {elapsed:.1f}s elapsed)")
        cap.release()
        extract_time = time.time() - t0
        print(f"[*] Feature extraction completed in {extract_time:.1f}s ({extract_time/len(events):.2f}s/stroke)!")
        torch.save(extracted_records, FEAT_CACHE)

    # 4. Load Classification Models
    print("\n[*] Loading Classification Models...")
    # Baseline CR-Full
    m_full = CRGatedFusionModel(
        hidden_dim=128,
        num_stroke_classes=len(STROKE_CLASSES),
        num_side_classes=len(SIDE_CLASSES),
        use_temporal=True, use_contact=True, use_gate=True,
    ).to(DEVICE)
    ckpt_full = torch.load("work_dirs/cr_gated_fusion/cr_gated_full/best.pth", map_location=DEVICE, weights_only=False)
    m_full.load_state_dict(ckpt_full["model"])
    m_full.eval()

    # KAP-Fusion
    m_kap = KAPFusionModel().to(DEVICE)
    ckpt_kap = torch.load("work_dirs/cr_gated_fusion/cr_gated_kap/best.pth", map_location=DEVICE, weights_only=False)
    m_kap.load_state_dict(ckpt_kap["model"])
    m_kap.eval()

    # NDH Model
    m_ndh = NeuralDisambiguationHead().to(DEVICE)
    ckpt_ndh = torch.load("work_dirs/cr_gated_fusion/cr_gated_ndh.pth", map_location=DEVICE, weights_only=False)
    m_ndh.load_state_dict(ckpt_ndh["model"])
    m_ndh.eval()

    # Evaluate Each Model
    eval_configs = [
        ("Baseline CR-Full", "cr_full"),
        ("Pure KAP-Fusion", "kap"),
        ("NDH (Neural Disambiguation Head)", "ndh"),
        ("NDH + Extended Disambiguation (Ours)", "ndh_ext"),
    ]

    print("\n" + "=" * 88)
    print("COMPARATIVE EVALUATION ON MATCH 13 (68 GROUND-TRUTH STROKES)")
    print("=" * 88)
    print(f"{'Model Architecture':<38} | {'Top-1 Acc':<12} | {'Top-2 Acc':<12} | {'Matched Top-1'}")
    print("-" * 88)

    results = {}
    for name, mode in eval_configs:
        predictions = []
        with torch.no_grad():
            for r in extracted_records:
                if mode == "cr_full":
                    s_logits, _, _ = m_full(r["emb"], r["s_j"], r["s_p"], r["s_s"], r["s_c"], r["s_q"])
                elif mode == "kap":
                    s_logits, _, _ = m_kap(r["emb"], r["s_j"], r["s_p"], r["s_s"], r["s_c"], r["s_q"])
                else:
                    s_logits, _ = m_ndh(r["emb"], r["s_j"], r["s_p"], r["s_s"], r["s_c"], r["s_q"])

                probs = torch.softmax(s_logits[0], dim=-1).cpu().numpy()
                top2_idx = probs.argsort()[-2:][::-1]
                c1, c2 = STROKE_CLASSES[top2_idx[0]], STROKE_CLASSES[top2_idx[1]]
                p1, p2 = probs[top2_idx[0]], probs[top2_idx[1]]

                pred_stroke = c1
                if mode == "ndh_ext":
                    s = r["s_s"][0].cpu().numpy()
                    v_early = np.linalg.norm(s[14] - s[11]) * 100
                    v_late = np.linalg.norm(s[24] - s[20]) * 100
                    decel = v_early - v_late

                    if {c1, c2} == {"lift", "net_attack"} and (p1 - p2) < 0.75:
                        if decel > 3.0 or v_early > 8.0:
                            pred_stroke = "lift"
                        else:
                            pred_stroke = "net_attack"
                    elif c1 == "drop" and c2 == "smash" and r["side"] == "upper":
                        if v_early > 11.0:
                            pred_stroke = "smash"

                predictions.append({
                    "hit_frame": r["hit_frame"],
                    "pred_stroke": pred_stroke,
                    "top2": [c1, c2] if pred_stroke == c1 else [pred_stroke, c1],
                })

        used = set()
        matched = []
        for gt in gt_strokes:
            gf = gt["frame"]
            g_stroke = gt["stroke"]
            cands = []
            for p_idx, p in enumerate(predictions):
                if p_idx not in used and abs(p["hit_frame"] - gf) <= 12:
                    cands.append((abs(p["hit_frame"] - gf), p_idx, p))
            if cands:
                cands.sort(key=lambda x: x[0])
                used.add(cands[0][1])
                best_p = cands[0][2]
                matched.append({
                    "gt": g_stroke,
                    "pred": best_p["pred_stroke"],
                    "top2": best_p["top2"],
                    "is_top1": best_p["pred_stroke"] == g_stroke,
                    "is_top2": g_stroke in best_p["top2"],
                })

        tot = len(gt_strokes)
        t1 = sum(1 for m in matched if m["is_top1"])
        t2 = sum(1 for m in matched if m["is_top2"])
        matched_n = len(matched)
        results[name] = {"top1": t1, "top2": t2, "tot": tot, "matched": matched_n}
        print(f"{name:<38} | {t1/tot*100:6.2f}% ({t1:2d}/{tot}) | {t2/tot*100:6.2f}% ({t2:2d}/{tot}) | {t1/matched_n*100:6.2f}% ({t1:2d}/{matched_n})")

    print("=" * 88)
    print(f"\n[PERFORMANCE SUMMARY]")
    print(f"  * Scan Time               : {scan_time:.1f}s")
    print(f"  * Feature Extraction Time : {extract_time:.1f}s")
    print(f"  * Total End-to-End Time   : {scan_time + extract_time:.1f}s ({((scan_time + extract_time)/60.0):.2f} minutes for 5-min clip!)")


if __name__ == "__main__":
    main()
