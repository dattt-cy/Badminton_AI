"""Head-to-head comparison: Old Fusion Model vs CR-Ensemble on BFMD 5-minute clip.

Evaluates:
  1. Old Baseline Fusion: FusionHead (512 RGB + 2516 Structured MLP) from Epoch 20
  2. New CR-Ensemble: Multi-scale Consensus (Full + NoGate + NoTemporal)
"""

from __future__ import annotations

import json
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

from ai_classifier.models import CRGatedFusionModel, FusionHead, MultiTaskR2Plus1D, STROKE_CLASSES, SIDE_CLASSES
from scripts.data.extract_sequence_features import extract_sample_sequences
from scripts.training.train_shuttleset_feature_fusion import structured_feature_arrays, resample

# ── Paths ────────────────────────────────────────────────────────────────────
BFMD_ROOT = Path(r"C:\Users\ADMIN\Downloads\BFMD_data-20260921T175743Z-1-001\BFMD_data")
ANNOT_ROOT = BFMD_ROOT / "annotations"
VIDEO_PREFIX = "KAPAL-API-Indonesia-Open-2025-Anders-Antonsen-DEN-3-vs.-Chou-Tien-Chen-TPE-6-F"
VIDEO_5MIN = REPO_ROOT / "work_dirs" / "bfmd_indonesia2025_5min.mp4"
SCAN_JSON = REPO_ROOT / "work_dirs" / "scan_bfmd_5min.json"

RGB_CKPT = REPO_ROOT / "work_dirs" / "r2plus1d18_mixed_shuttleset_finebadminton" / "best.pth"
OLD_FUSION_CKPT = REPO_ROOT / "work_dirs" / "shuttleset_fusion_epoch20_full_b4" / "best.pth"
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


def load_data():
    with open(ANNOT_ROOT / "pose" / f"{VIDEO_PREFIX}.json") as f:
        pose_raw = json.load(f)
    pose_by_frame = {}
    for vid_data in pose_raw["videos"].values():
        for fid_str, persons in vid_data["frames"].items():
            pose_by_frame[int(fid_str)] = persons

    with open(ANNOT_ROOT / "shuttle" / f"{VIDEO_PREFIX}.json") as f:
        shuttle_raw = json.load(f)
    shuttle_by_frame = {}
    for item in shuttle_raw["predictions"][0]["result"][0]["value"]["sequence"]:
        if item.get("enabled", True):
            shuttle_by_frame[int(item["frame"])] = (item["x"] / 100.0, item["y"] / 100.0)

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

    tl, tr, br, bl = [457.3, 286.2], [821.9, 282.4], [946.4, 657.1], [334.0, 656.9]
    dest = np.array([[0, 1], [1, 1], [1, 0], [0, 0]], dtype=np.float32)
    H = cv2.getPerspectiveTransform(np.array([bl, br, tr, tl], dtype=np.float32), dest)

    with open(SCAN_JSON) as f:
        scan_events = json.load(f)["events"]

    return pose_by_frame, shuttle_by_frame, gt_events, H, scan_events


def normalize_joints(kps_px: np.ndarray, box: np.ndarray) -> np.ndarray:
    diag = np.linalg.norm(box[2:] - box[:2])
    center = (box[:2] + box[2:]) / 2.0
    return (kps_px - center) / max(diag, 1e-6)


def extract_features(abs_frame: int, pose_by_frame: dict, shuttle_by_frame: dict, H: np.ndarray, before: int = 15, after: int = 30):
    window_len = before + 1 + after
    joints_seq = np.zeros((window_len, 2, 17, 2), dtype=np.float32)
    pos_seq = np.zeros((window_len, 2, 2), dtype=np.float32)
    shuttle_seq = np.zeros((window_len, 2), dtype=np.float32)
    hit_boxes = None

    for idx, delta in enumerate(range(-before, after + 1)):
        f = abs_frame + delta
        if f in shuttle_by_frame:
            shuttle_seq[idx] = shuttle_by_frame[f]

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
            sorted_p = sorted(persons, key=lambda p: (p["bbox"][1] + p["bbox"][3]) / 2.0)
            if len(sorted_p) == 1:
                sorted_p = [sorted_p[0], sorted_p[0]]

            for p_idx, person in enumerate(sorted_p[:2]):
                box = np.array(person["bbox"], dtype=np.float32)
                raw_kps = np.array(person["keypoints"], dtype=np.float32)[:, :2]
                kps_px = raw_kps * np.array([1280.0, 720.0], dtype=np.float32)

                joints_seq[idx, p_idx] = normalize_joints(kps_px, box)
                feet = kps_px[[15, 16]] if len(kps_px) >= 17 else kps_px[-2:]
                pos_seq[idx, p_idx] = cv2.perspectiveTransform(feet[None], H)[0].mean(axis=0)

                if delta == 0 and hit_boxes is None:
                    hit_boxes = {
                        "upper_box": sorted_p[0]["bbox"],
                        "lower_box": sorted_p[1]["bbox"] if len(sorted_p) > 1 else sorted_p[0]["bbox"],
                    }

    return joints_seq, pos_seq, shuttle_seq, hit_boxes


def crop_hitter(cap: cv2.VideoCapture, rel_frame: int, side: str, hit_boxes: dict | None, n_frames: int = 16):
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    start = max(0, rel_frame - n_frames // 2)
    end = min(total - 1, start + n_frames - 1)
    start = max(0, end - n_frames + 1)
    indices = np.linspace(start, end, n_frames, dtype=int)

    if hit_boxes:
        b = hit_boxes["lower_box" if side == "lower" else "upper_box"]
        x1, y1, x2, y2 = b
    else:
        x1, y1, x2, y2 = (400, 360, 880, 680) if side == "lower" else (450, 180, 830, 380)

    w, h = x2 - x1, y2 - y1
    cx1 = max(0, int(x1 - w * 0.2))
    cy1 = max(0, int(y1 - h * 0.2))
    cx2 = min(1280, int(x2 + w * 0.2))
    cy2 = min(720, int(y2 + h * 0.2))

    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ok, frame = cap.read()
        if not ok:
            frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        crop = frame[cy1:cy2, cx1:cx2]
        crop = cv2.resize(crop, (112, 112)) if crop.size > 0 else np.zeros((112, 112, 3), dtype=np.uint8)
        frames.append(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))

    clip_np = np.stack(frames).astype(np.float32) / 255.0
    mean = np.array([0.43216, 0.394666, 0.37645], dtype=np.float32)
    std = np.array([0.22803, 0.22145, 0.216989], dtype=np.float32)
    return ((clip_np - mean) / std).transpose(3, 0, 1, 2)[None]


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 80)
    print("HEAD-TO-HEAD: OLD FUSION MODEL vs CR-ENSEMBLE ON BFMD (5 PHUT)")
    print(f"Device: {device}")
    print("=" * 80)

    pose_dict, shuttle_dict, gt_events, H, scan_events = load_data()

    # Load RGB
    rgb_ckpt = torch.load(RGB_CKPT, map_location="cpu", weights_only=False)
    rgb_model = MultiTaskR2Plus1D().to(device)
    rgb_model.load_state_dict(rgb_ckpt["model"])
    rgb_model.eval()

    # Load Old Fusion Model
    old_fuse_ckpt = torch.load(OLD_FUSION_CKPT, map_location="cpu", weights_only=False)
    s_dim = old_fuse_ckpt["model"]["structured.1.weight"].shape[1]
    old_fuse_model = FusionHead(512, s_dim).to(device)
    old_fuse_model.load_state_dict(old_fuse_ckpt["model"])
    old_fuse_model.eval()
    old_s_mean = old_fuse_ckpt["structured_mean"]
    old_s_std = old_fuse_ckpt["structured_std"]
    print(f"[*] Loaded Old Fusion Model (Structured dim: {s_dim})")

    # Load CR-Gated models
    def make_cr(ckpt_path, use_temp, use_cont, use_gt):
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        m = CRGatedFusionModel(
            hidden_dim=128, num_stroke_classes=len(STROKE_CLASSES), num_side_classes=len(SIDE_CLASSES),
            use_temporal=use_temp, use_contact=use_cont, use_gate=use_gt, use_cross_attention=True,
        ).to(device)
        m.load_state_dict(ckpt["model"])
        m.eval()
        return m

    cr_full = make_cr(CR_FULL_CKPT, True, True, True)
    cr_nogate = make_cr(CR_NOGATE_CKPT, True, True, False)
    cr_notemp = make_cr(CR_NOTEMPORAL_CKPT, False, False, False)
    print("[*] Loaded CR-Ensemble (Full + NoGate + NoTemporal)")

    cap = cv2.VideoCapture(str(VIDEO_5MIN))
    results = []

    print(f"\n[*] Evaluating on {len(scan_events)} scan events...")
    for idx, ev in enumerate(scan_events):
        rel_f = ev["frame"]
        abs_f = rel_f + CLIP_OFFSET
        side = ev.get("side", "lower")

        joints_seq, pos_seq, shuttle_seq, hit_boxes = extract_features(abs_f, pose_dict, shuttle_dict, H)

        # 1. Feature for Old Fusion Model: structured_feature_arrays (2516-dim)
        raw_struct = structured_feature_arrays(joints_seq, pos_seq, shuttle_seq, length=32)
        norm_struct = (raw_struct - old_s_mean) / np.maximum(old_s_std, 1e-5)
        t_struct = torch.from_numpy(norm_struct).unsqueeze(0).to(device)

        # 2. Features for CR-Ensemble: extract_sample_sequences
        joints_flat = joints_seq.reshape(len(joints_seq), 68)
        pos_flat = pos_seq.reshape(len(pos_seq), 4)
        s_joints, s_pos, s_shuttle, s_contact, quality = extract_sample_sequences(
            joints_flat, pos_flat, shuttle_seq, target_index=15, length=32, before=15, after=30,
        )
        t_pose = torch.from_numpy(s_joints).unsqueeze(0).to(device)
        t_court = torch.from_numpy(s_pos).unsqueeze(0).to(device)
        t_shuttle = torch.from_numpy(s_shuttle).unsqueeze(0).to(device)
        t_contact = torch.from_numpy(s_contact).unsqueeze(0).to(device)
        t_qual = torch.from_numpy(quality).unsqueeze(0).to(device)

        # RGB embedding
        rgb_tensor = crop_hitter(cap, rel_f, side, hit_boxes, n_frames=16)
        with torch.inference_mode():
            emb = rgb_model.backbone(torch.from_numpy(rgb_tensor).to(device))

            # Old Fusion Inference
            old_logits, _ = old_fuse_model(emb, t_struct)
            old_pred = STROKE_CLASSES[old_logits[0].argmax().item()]

            # CR-Ensemble Inference
            f_prob = torch.softmax(cr_full(emb, t_pose, t_court, t_shuttle, t_contact, t_qual)[0][0], dim=-1)
            ng_prob = torch.softmax(cr_nogate(emb, t_pose, t_court, t_shuttle, t_contact, t_qual)[0][0], dim=-1)
            nt_prob = torch.softmax(cr_notemp(emb, t_pose, t_court, t_shuttle, t_contact, t_qual)[0][0], dim=-1)
            ens_prob = f_prob * 0.40 + ng_prob * 0.30 + nt_prob * 0.30
            cr_pred = STROKE_CLASSES[ens_prob.argmax().item()]

        # GT match
        best_gt = None
        min_diff = 999
        for gt in gt_events:
            diff = abs(rel_f - gt["rel_frame"])
            if diff <= 12 and diff < min_diff:
                min_diff = diff
                best_gt = gt

        if best_gt:
            results.append({
                "rel_frame": rel_f,
                "gt": best_gt["mapped"],
                "old_fusion_pred": old_pred,
                "cr_ensemble_pred": cr_pred,
                "old_correct": old_pred == best_gt["mapped"],
                "cr_correct": cr_pred == best_gt["mapped"],
            })

    cap.release()

    # ── Evaluation ────────────────────────────────────────────────────────────
    total = len(results)
    old_correct = sum(1 for r in results if r["old_correct"])
    cr_correct = sum(1 for r in results if r["cr_correct"])

    old_acc = old_correct / total if total else 0
    cr_acc = cr_correct / total if total else 0

    print("\n" + "=" * 80)
    print("KET QUA DOI DAU TRUC TIEP: MODEL FUSION CU vs CR-ENSEMBLE (106 CU DANH)")
    print("=" * 80)
    print(f"Tong so cu danh Ground Truth khop: {total}")
    print(f"\n  [1] MODEL FUSION CU (Concat-MLP): {old_correct}/{total} = {old_acc:.1%} Top-1")
    print(f"  [2] CR-ENSEMBLE MOI (Novel Arch): {cr_correct}/{total} = {cr_acc:.1%} Top-1")
    print(f"\n  -> Chenh lech Top-1: {'CR-Ensemble thang +' if cr_acc >= old_acc else 'Old Fusion thang +'}{abs(cr_acc - old_acc):.1%}")

    print("\n" + "-" * 80)
    print(f"{'Loai cu danh (Class)':<20} {'Tong so':<10} {'Fusion Cu Top-1':<25} {'CR-Ensemble Top-1':<25}")
    print("-" * 80)

    class_data = defaultdict(lambda: {"total": 0, "old": 0, "cr": 0})
    for r in results:
        gt = r["gt"]
        class_data[gt]["total"] += 1
        if r["old_correct"]:
            class_data[gt]["old"] += 1
        if r["cr_correct"]:
            class_data[gt]["cr"] += 1

    for cls, s in sorted(class_data.items(), key=lambda x: -x[1]["total"]):
        old_c_acc = s["old"] / s["total"] if s["total"] else 0
        cr_c_acc = s["cr"] / s["total"] if s["total"] else 0
        print(f"{cls:<20} {s['total']:<10} {s['old']:>2}/{s['total']:<2} ({old_c_acc:>5.1%})             {s['cr']:>2}/{s['total']:<2} ({cr_c_acc:>5.1%})")

    print("=" * 80)


if __name__ == "__main__":
    main()

