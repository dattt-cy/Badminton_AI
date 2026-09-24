"""Full-Pipeline CR-Ensemble Evaluation on BFMD 5-minute clip.

Correctly aligns and normalizes all modalities matching ShuttleSet training:
  1. Hitter RGB Crop: (16, 3, 112, 112) centered on player bounding box
  2. Pose Keypoints: normalized relative to bbox center and diagonal [-0.4, 0.4]
  3. Court Position: player feet projected onto court plane [0, 1] via Homography
  4. Shuttle Trajectory: normalized (x, y) coordinates
  5. Multi-Scale CR-Ensemble: Full (40%) + NoGate (30%) + NoTemporal (30%)
"""

from __future__ import annotations

import json
import math
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ai_classifier.models import CRGatedFusionModel, MultiTaskR2Plus1D, STROKE_CLASSES, SIDE_CLASSES
from scripts.data.extract_sequence_features import extract_sample_sequences

# ── Paths ────────────────────────────────────────────────────────────────────
BFMD_ROOT = Path(r"C:\Users\ADMIN\Downloads\BFMD_data-20260921T175743Z-1-001\BFMD_data")
ANNOT_ROOT = BFMD_ROOT / "annotations"
VIDEO_PREFIX = "KAPAL-API-Indonesia-Open-2025-Anders-Antonsen-DEN-3-vs.-Chou-Tien-Chen-TPE-6-F"
VIDEO_5MIN = REPO_ROOT / "work_dirs" / "bfmd_indonesia2025_5min.mp4"
SCAN_JSON = REPO_ROOT / "work_dirs" / "scan_bfmd_5min.json"
RGB_CKPT = REPO_ROOT / "work_dirs" / "r2plus1d18_mixed_shuttleset_finebadminton" / "best.pth"
CR_FULL_CKPT = REPO_ROOT / "work_dirs" / "cr_gated_fusion" / "cr_gated_full" / "best.pth"
CR_NOGATE_CKPT = REPO_ROOT / "work_dirs" / "cr_gated_fusion" / "cr_gated_no_gate" / "best.pth"
CR_NOTEMPORAL_CKPT = REPO_ROOT / "work_dirs" / "cr_gated_fusion" / "cr_gated_no_temporal" / "best.pth"

CLIP_OFFSET = 18_000
CLIP_FRAMES = 9_000

BFMD_LABEL_MAP = {
    "net_shot": "net_shot",
    "lift": "lift",
    "smash": "smash",
    "drop": "drop",
    "clear": "clear",
    "drive": "drive",
    "serve": "serve",
    "push": "net_shot",
    "block": "net_shot",
    "press": "net_attack",
    "net_kill": "net_attack",
    "net_hit": "net_shot",
    "flick_serve": "serve",
}


def load_annotations():
    print("[*] Loading pose annotations...")
    with open(ANNOT_ROOT / "pose" / f"{VIDEO_PREFIX}.json") as f:
        pose_raw = json.load(f)
    pose_by_frame = {}
    for vid_data in pose_raw["videos"].values():
        for fid_str, persons in vid_data["frames"].items():
            pose_by_frame[int(fid_str)] = persons

    print("[*] Loading shuttle annotations...")
    with open(ANNOT_ROOT / "shuttle" / f"{VIDEO_PREFIX}.json") as f:
        shuttle_raw = json.load(f)
    shuttle_by_frame = {}
    for item in shuttle_raw["predictions"][0]["result"][0]["value"]["sequence"]:
        if item.get("enabled", True):
            shuttle_by_frame[int(item["frame"])] = (item["x"] / 100.0, item["y"] / 100.0)

    print("[*] Loading ground truth strokes...")
    with open(ANNOT_ROOT / "shot_type" / f"{VIDEO_PREFIX}.json") as f:
        st = json.load(f)
    gt_events = []
    for r in st["annotations"][0]["result"]:
        v = r["value"]
        label = v.get("timelinelabels", [""])[0]
        if label in BFMD_LABEL_MAP:
            for rng in v.get("ranges", []):
                abs_f = int(rng["start"])
                rel_f = abs_f - CLIP_OFFSET
                if 0 <= rel_f < CLIP_FRAMES:
                    gt_events.append({
                        "abs_frame": abs_f,
                        "rel_frame": rel_f,
                        "label": label,
                        "mapped": BFMD_LABEL_MAP[label],
                    })
    gt_events.sort(key=lambda e: e["rel_frame"])

    # Homography matrix
    tl = [457.3, 286.2]
    tr = [821.9, 282.4]
    br = [946.4, 657.1]
    bl = [334.0, 656.9]
    dest = np.array([[0, 1], [1, 1], [1, 0], [0, 0]], dtype=np.float32)
    H = cv2.getPerspectiveTransform(np.array([bl, br, tr, tl], dtype=np.float32), dest)

    return pose_by_frame, shuttle_by_frame, gt_events, H


def normalize_joints(kps_px: np.ndarray, box: np.ndarray) -> np.ndarray:
    """Normalize keypoints relative to bbox center and diagonal."""
    diag = np.linalg.norm(box[2:] - box[:2])
    center = (box[:2] + box[2:]) / 2.0
    return (kps_px - center) / max(diag, 1e-6)


def extract_multimodal_window(
    abs_frame: int,
    pose_by_frame: dict,
    shuttle_by_frame: dict,
    H: np.ndarray,
    before: int = 15,
    after: int = 30,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict | None]:
    """Extract properly normalized joints (T, 68), position (T, 4), shuttle (T, 2)."""
    window_len = before + 1 + after
    joints_seq = np.zeros((window_len, 2, 17, 2), dtype=np.float32)
    pos_seq = np.zeros((window_len, 2, 2), dtype=np.float32)
    shuttle_seq = np.zeros((window_len, 2), dtype=np.float32)

    hit_hitter_box = None

    for idx, delta in enumerate(range(-before, after + 1)):
        f = abs_frame + delta

        # Shuttle
        if f in shuttle_by_frame:
            shuttle_seq[idx] = shuttle_by_frame[f]

        # Pose & Court
        # Nearest pose within ±3 frames
        persons = None
        for d in range(4):
            for sgn in [0, 1, -1]:
                cand = f + sgn * d
                if cand in pose_by_frame:
                    persons = pose_by_frame[cand]
                    break
            if persons is not None:
                break

        if persons and len(persons) >= 1:
            # Sort persons by y position: upper player first, lower player second
            sorted_persons = sorted(persons, key=lambda p: (p["bbox"][1] + p["bbox"][3]) / 2.0)
            if len(sorted_persons) == 1:
                # Duplicate to lower or upper depending on y
                is_lower = ((sorted_persons[0]["bbox"][1] + sorted_persons[0]["bbox"][3]) / 2.0) > 360
                sorted_persons = [sorted_persons[0], sorted_persons[0]] if is_lower else [sorted_persons[0], sorted_persons[0]]

            for p_idx, person in enumerate(sorted_persons[:2]):
                box = np.array(person["bbox"], dtype=np.float32)
                raw_kps = np.array(person["keypoints"], dtype=np.float32)[:, :2]
                kps_px = raw_kps * np.array([1280.0, 720.0], dtype=np.float32)

                # Normalize joints
                joints_seq[idx, p_idx] = normalize_joints(kps_px, box)

                # Court position from feet (ankles 15, 16)
                feet = kps_px[[15, 16]] if len(kps_px) >= 17 else kps_px[-2:]
                court_pos = cv2.perspectiveTransform(feet[None], H)[0].mean(axis=0)
                pos_seq[idx, p_idx] = court_pos

                if delta == 0 and hit_hitter_box is None:
                    hit_hitter_box = {
                        "upper_box": sorted_persons[0]["bbox"],
                        "lower_box": sorted_persons[1]["bbox"] if len(sorted_persons) > 1 else sorted_persons[0]["bbox"],
                    }

    # Reshape
    joints_flat = joints_seq.reshape(window_len, 68)
    pos_flat = pos_seq.reshape(window_len, 4)

    return joints_flat, pos_flat, shuttle_seq, hit_hitter_box


def crop_hitter_frames(
    cap: cv2.VideoCapture,
    rel_frame: int,
    hitter_side: str,
    boxes_dict: dict | None,
    n_frames: int = 16,
) -> np.ndarray:
    """Crop 16 frames of the active hitter resized to (112, 112)."""
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    start = max(0, rel_frame - n_frames // 2)
    end = min(total - 1, start + n_frames - 1)
    start = max(0, end - n_frames + 1)
    indices = np.linspace(start, end, n_frames, dtype=int)

    # Determine bounding box
    if boxes_dict:
        b = boxes_dict["lower_box" if hitter_side == "lower" else "upper_box"]
        x1, y1, x2, y2 = b
    else:
        # Fallback half court box
        if hitter_side == "lower":
            x1, y1, x2, y2 = 400, 360, 880, 680
        else:
            x1, y1, x2, y2 = 450, 180, 830, 380

    # Expand box slightly (padding 20%)
    w, h = x2 - x1, y2 - y1
    pad_w, pad_h = w * 0.20, h * 0.20
    cx1 = max(0, int(x1 - pad_w))
    cy1 = max(0, int(y1 - pad_h))
    cx2 = min(1280, int(x2 + pad_w))
    cy2 = min(720, int(y2 + pad_h))

    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ok, frame = cap.read()
        if not ok:
            frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        crop = frame[cy1:cy2, cx1:cx2]
        if crop.size == 0:
            crop = np.zeros((112, 112, 3), dtype=np.uint8)
        else:
            crop = cv2.resize(crop, (112, 112))
        crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        frames.append(crop)

    # (16, 112, 112, 3) -> (1, 3, 16, 112, 112)
    clip_np = np.stack(frames).astype(np.float32) / 255.0  # (T, H, W, C)
    mean = np.array([0.43216, 0.394666, 0.37645], dtype=np.float32)
    std = np.array([0.22803, 0.22145, 0.216989], dtype=np.float32)
    clip_norm = (clip_np - mean) / std
    return clip_norm.transpose(3, 0, 1, 2)[None]  # (1, C, T, H, W)


def load_models(device: torch.device):
    print("[*] Loading R(2+1)D RGB backbone...")
    rgb_ckpt = torch.load(RGB_CKPT, map_location="cpu", weights_only=False)
    rgb_model = MultiTaskR2Plus1D().to(device)
    rgb_model.load_state_dict(rgb_ckpt["model"])
    rgb_model.eval()

    def make_cr(ckpt_path, use_temp, use_cont, use_gt):
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        m = CRGatedFusionModel(
            hidden_dim=128,
            num_stroke_classes=len(STROKE_CLASSES),
            num_side_classes=len(SIDE_CLASSES),
            use_temporal=use_temp,
            use_contact=use_cont,
            use_gate=use_gt,
            use_cross_attention=True,
        ).to(device)
        m.load_state_dict(ckpt["model"])
        m.eval()
        return m

    print("[*] Loading CR-Gated variants...")
    cr_full = make_cr(CR_FULL_CKPT, True, True, True)
    cr_nogate = make_cr(CR_NOGATE_CKPT, True, True, False)
    cr_notemp = make_cr(CR_NOTEMPORAL_CKPT, False, False, False)

    return rgb_model, cr_full, cr_nogate, cr_notemp


def main():
    t0 = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 76)
    print("FULL-PIPELINE CR-ENSEMBLE EVALUATION ON BFMD 5-MINUTE CLIP")
    print(f"Device: {device}")
    print("=" * 76)

    pose_dict, shuttle_dict, gt_events, H = load_annotations()
    print(f"  Pose frames:    {len(pose_dict)}")
    print(f"  Shuttle frames: {len(shuttle_dict)}")
    print(f"  GT strokes:     {len(gt_events)}")

    with open(SCAN_JSON) as f:
        scan_data = json.load(f)
    scan_events = scan_data["events"]
    print(f"  Scan events:    {len(scan_events)}")

    rgb_model, cr_full, cr_nogate, cr_notemp = load_models(device)

    cap = cv2.VideoCapture(str(VIDEO_5MIN))
    if not cap.isOpened():
        print(f"Error: Cannot open {VIDEO_5MIN}")
        return

    results = []
    print("\n[*] Processing events through full multimodal pipeline...")

    for idx, ev in enumerate(scan_events):
        rel_f = ev["frame"]
        abs_f = rel_f + CLIP_OFFSET
        side = ev.get("side", "lower")

        if idx % 20 == 0:
            print(f"  Event {idx}/{len(scan_events)} ({time.time()-t0:.1f}s)...")

        # 1. Structured modalities with proper normalization
        joints_flat, pos_flat, shuttle_seq, boxes = extract_multimodal_window(
            abs_f, pose_dict, shuttle_dict, H, before=15, after=30,
        )
        s_joints, s_pos, s_shuttle, s_contact, quality = extract_sample_sequences(
            joints_flat, pos_flat, shuttle_seq, target_index=15, length=32, before=15, after=30,
        )

        # 2. Hitter RGB crop (112x112, 16 frames)
        rgb_tensor = crop_hitter_frames(cap, rel_f, side, boxes, n_frames=16)
        rgb_t = torch.from_numpy(rgb_tensor).to(device)

        with torch.inference_mode():
            emb = rgb_model.backbone(rgb_t)  # (1, 512)

            t_pose = torch.from_numpy(s_joints).unsqueeze(0).to(device)
            t_court = torch.from_numpy(s_pos).unsqueeze(0).to(device)
            t_shuttle = torch.from_numpy(s_shuttle).unsqueeze(0).to(device)
            t_contact = torch.from_numpy(s_contact).unsqueeze(0).to(device)
            t_qual = torch.from_numpy(quality).unsqueeze(0).to(device)

            f_logits, _, _ = cr_full(emb, t_pose, t_court, t_shuttle, t_contact, t_qual)
            ng_logits, _, _ = cr_nogate(emb, t_pose, t_court, t_shuttle, t_contact, t_qual)
            nt_logits, _, _ = cr_notemp(emb, t_pose, t_court, t_shuttle, t_contact, t_qual)

            f_prob = torch.softmax(f_logits[0], dim=-1)
            ng_prob = torch.softmax(ng_logits[0], dim=-1)
            nt_prob = torch.softmax(nt_logits[0], dim=-1)

            # Consensus Ensemble: Full 40% + NoGate 30% + NoTemporal 30%
            ens_prob = f_prob * 0.40 + ng_prob * 0.30 + nt_prob * 0.30

            top1_idx = ens_prob.argmax().item()
            ens_pred = STROKE_CLASSES[top1_idx]
            ens_conf = float(ens_prob[top1_idx].item())

            top2_indices = torch.topk(ens_prob, 2).indices.cpu().numpy().tolist()
            top2_classes = [STROKE_CLASSES[i] for i in top2_indices]

        # Match with GT
        best_gt = None
        min_diff = 999
        for gt in gt_events:
            diff = abs(rel_f - gt["rel_frame"])
            if diff <= 12 and diff < min_diff:
                min_diff = diff
                best_gt = gt

        results.append({
            "rel_frame": rel_f,
            "abs_frame": abs_f,
            "side": side,
            "pred": ens_pred,
            "conf": ens_conf,
            "top2": top2_classes,
            "gt_label": best_gt["mapped"] if best_gt else None,
            "gt_bfmd": best_gt["label"] if best_gt else None,
            "gt_diff": min_diff if best_gt else None,
        })

    cap.release()

    # ── Evaluation Metrics ────────────────────────────────────────────────────
    matched = [r for r in results if r["gt_label"] is not None]
    n_matched = len(matched)
    top1_correct = sum(1 for r in matched if r["pred"] == r["gt_label"])
    top2_correct = sum(1 for r in matched if r["gt_label"] in r["top2"])

    top1_acc = top1_correct / n_matched if n_matched else 0
    top2_acc = top2_correct / n_matched if n_matched else 0

    print("\n" + "=" * 76)
    print("KET QUA FULL-PIPELINE CR-ENSEMBLE TREN BFMD (5 PHUT)")
    print("=" * 76)
    print(f"Tong so cu danh quet duoc:       {len(scan_events)}")
    print(f"So cu danh khop Ground Truth:    {n_matched} / {len(gt_events)} (Recall: {n_matched/len(gt_events):.1%})")
    print(f"\n  DO CHINH XAC:")
    print(f"    Top-1 Accuracy:  {top1_correct}/{n_matched} = {top1_acc:.1%}")
    print(f"    Top-2 Accuracy:  {top2_correct}/{n_matched} = {top2_acc:.1%}")

    print("\n  Phan bo du doan (Prediction Distribution):")
    pred_counts = Counter(r["pred"] for r in matched)
    for cls, cnt in pred_counts.most_common():
        print(f"    {cls:<15}: {cnt:>3} ({cnt/n_matched:.1%})")

    print("\n  Chi tiet theo tung loai cu danh (Per-Class Accuracy):")
    class_stats = defaultdict(lambda: {"correct": 0, "total": 0, "top2": 0})
    for r in matched:
        gt = r["gt_label"]
        class_stats[gt]["total"] += 1
        if r["pred"] == gt:
            class_stats[gt]["correct"] += 1
        if gt in r["top2"]:
            class_stats[gt]["top2"] += 1

    for cls, s in sorted(class_stats.items(), key=lambda x: -x[1]["total"]):
        acc = s["correct"] / s["total"] if s["total"] else 0
        acc2 = s["top2"] / s["total"] if s["total"] else 0
        print(f"    {cls:<15} Top-1: {s['correct']:>2}/{s['total']:>2} ({acc:>5.1%}) | Top-2: {s['top2']:>2}/{s['total']:>2} ({acc2:>5.1%})")

    # Save
    out_file = REPO_ROOT / "work_dirs" / "bfmd_full_pipeline_cr_results.json"
    with open(out_file, "w") as f:
        json.dump({
            "top1_acc": top1_acc,
            "top2_acc": top2_acc,
            "n_matched": n_matched,
            "results": results,
        }, f, indent=2)
    print(f"\nSaved to: {out_file}")
    print(f"Tong thoi gian: {time.time()-t0:.1f}s")
    print("=" * 76)


if __name__ == "__main__":
    main()

