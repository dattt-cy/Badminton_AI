"""CR-Ensemble End-to-End Inference on BFMD 5-minute clip.

Pipeline:
  1. Load pre-scanned hit events (work_dirs/scan_bfmd_5min.json)
  2. Extract sequence features from BFMD annotations (pose, court, shuttle)
  3. Extract RGB embedding via R(2+1)D backbone
  4. Run CR-Ensemble (Full 40% + NoGate 30% + NoTemporal 30%)
  5. Compare vs GT shot_type annotations
"""

from __future__ import annotations

import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ai_classifier.models import CRGatedFusionModel, MultiTaskR2Plus1D
from ai_classifier.models import STROKE_CLASSES, SIDE_CLASSES
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

# Video clip offset: our 5-minute clip starts at frame 18000 of original video
CLIP_OFFSET = 18_000
CLIP_FRAMES = 9_000  # 300s × 30fps

# BFMD label → ShuttleSet label mapping
BFMD_LABEL_MAP = {
    "net_shot": "net_shot",
    "lift": "lift",
    "smash": "smash",
    "drop": "drop",
    "clear": "clear",
    "drive": "drive",
    "serve": "serve",
    "push": "net_shot",    # push is similar to net_shot
    "block": "net_shot",   # block similar
    "press": "net_attack",
    "net_kill": "net_attack",
    "net_hit": "net_shot",
    "flick_serve": "serve",
}

CLIP_FRAMES_BEFORE = 16
CLIP_FRAMES_AFTER = 16
SEQUENCE_LEN = 32


# ── Data loading helpers ─────────────────────────────────────────────────────

def load_shuttle_trajectory() -> dict[int, tuple[float, float]]:
    """Return {absolute_frame: (x_norm, y_norm)} from BFMD shuttle annotation."""
    with open(ANNOT_ROOT / "shuttle" / f"{VIDEO_PREFIX}.json") as f:
        data = json.load(f)
    pred = data["predictions"][0]["result"][0]["value"]["sequence"]
    # x,y are percentages (0-100)
    traj = {}
    for item in pred:
        frame = int(item["frame"])
        if item.get("enabled", True):
            traj[frame] = (item["x"] / 100.0, item["y"] / 100.0)
    return traj


def load_court_perframe() -> dict[int, list[list[float]]]:
    """Return {frame_idx: [[x,y]*4]} from court_perframe annotation (sampled at 1fps→30fps interp)."""
    with open(ANNOT_ROOT / "court_perframe" / f"{VIDEO_PREFIX}.json") as f:
        data = json.load(f)
    # frames list: [{sec, frame_idx, corners, ...}]
    frames_list = data["frames"]
    # Build dict frame_idx→corners (absolute frame)
    court_by_frame = {}
    for entry in frames_list:
        fidx = int(entry["frame_idx"])
        corners = entry["corners"]  # [[x,y]*4] in pixel coords
        court_by_frame[fidx] = corners
    return court_by_frame


def load_pose_by_frame() -> dict[int, list[list[float]]]:
    """Return {absolute_frame: [[x,y]*17*2players flat]} from BFMD pose annotation."""
    with open(ANNOT_ROOT / "pose" / f"{VIDEO_PREFIX}.json") as f:
        data = json.load(f)
    videos = data["videos"]  # {rally_id: {frames: {frame_id: [persons]}}}
    pose_by_frame: dict[int, list[list[float]]] = {}
    for vid_data in videos.values():
        w = vid_data.get("width", 1280)
        h = vid_data.get("height", 720)
        frames = vid_data["frames"]
        for fid_str, persons in frames.items():
            fid = int(fid_str)
            kps_all = []
            for person in persons[:2]:  # max 2 players
                kps = person["keypoints"]  # [[x,y,conf], ...] 17 kps, normalized 0-1
                for kp in kps:
                    kps_all.append([kp[0], kp[1]])  # already normalized
            # pad to 2 players if only 1
            while len(kps_all) < 34:  # 2*17
                kps_all.append([0.0, 0.0])
            pose_by_frame[fid] = kps_all[:34]
    return pose_by_frame


def load_gt_strokes() -> list[dict]:
    """Load ground truth shot events from shot_type annotation, within clip window."""
    with open(ANNOT_ROOT / "shot_type" / f"{VIDEO_PREFIX}.json") as f:
        data = json.load(f)
    annots = data.get("annotations", [])
    if not annots:
        return []
    results = annots[0].get("result", [])

    valid_labels = set(BFMD_LABEL_MAP.keys())
    events = []
    for r in results:
        v = r["value"]
        label = v.get("timelinelabels", [""])[0]
        if label not in valid_labels:
            continue
        for rng in v.get("ranges", []):
            abs_frame = int(rng["start"])
            rel_frame = abs_frame - CLIP_OFFSET
            if 0 <= rel_frame < CLIP_FRAMES:
                events.append({
                    "abs_frame": abs_frame,
                    "rel_frame": rel_frame,
                    "label": label,
                    "mapped": BFMD_LABEL_MAP[label],
                })
    events.sort(key=lambda e: e["rel_frame"])
    return events


# ── Feature extraction ───────────────────────────────────────────────────────

def get_court_at_frame(court_by_frame: dict[int, list], abs_frame: int, video_w: int = 1280, video_h: int = 720) -> np.ndarray:
    """Get court corners normalized [0,1] for given absolute frame via nearest lookup."""
    if not court_by_frame:
        return np.array([[0.35, 0.40], [0.64, 0.39], [0.74, 0.91], [0.26, 0.91]], dtype=np.float32).flatten()
    # Find nearest sampled frame
    keys = np.array(sorted(court_by_frame.keys()))
    nearest = keys[np.argmin(np.abs(keys - abs_frame))]
    corners = court_by_frame[nearest]  # [[x,y]*4] pixel coords
    norm = [[c[0] / video_w, c[1] / video_h] for c in corners]
    return np.array(norm, dtype=np.float32).flatten()  # (4,) or (8,)


def build_window_arrays(
    abs_frame: int,
    pose_by_frame: dict[int, list],
    court_by_frame: dict[int, list],
    shuttle_traj: dict[int, tuple],
    before: int = 16,
    after: int = 16,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build joints (T, 2, 17, 2), position (T, 2, 2), shuttle (T, 2) arrays for window."""
    window_size = before + 1 + after
    joints_arr = np.zeros((window_size, 2, 17, 2), dtype=np.float32)
    position_arr = np.zeros((window_size, 2, 2), dtype=np.float32)
    shuttle_arr = np.zeros((window_size, 2), dtype=np.float32)

    for i, offset in enumerate(range(-before, after + 1)):
        frame_abs = abs_frame + offset

        # Pose: try to find in pose dict (approximate lookup within ±5 frames)
        pose_flat = None
        for search_delta in range(6):
            for sign in [0, 1, -1]:
                candidate = frame_abs + sign * search_delta
                if candidate in pose_by_frame:
                    pose_flat = pose_by_frame[candidate]
                    break
            if pose_flat is not None:
                break
        if pose_flat is not None:
            kps = np.array(pose_flat, dtype=np.float32).reshape(2, 17, 2) if len(pose_flat) >= 34 else None
            if kps is not None:
                joints_arr[i] = kps

        # Court (normalized)
        keys = np.array(sorted(court_by_frame.keys())) if court_by_frame else np.array([])
        if len(keys):
            nearest_key = keys[np.argmin(np.abs(keys - frame_abs))]
            corners = court_by_frame[nearest_key]  # [[x,y]*4] pixels
            norm_corners = np.array([[c[0] / 1280, c[1] / 720] for c in corners], dtype=np.float32)  # (4,2)
            # position: use 2 representative court points per player side
            position_arr[i, 0] = norm_corners[0]   # top-left corner
            position_arr[i, 1] = norm_corners[2]   # bottom-right corner

        # Shuttle trajectory
        if frame_abs in shuttle_traj:
            sx, sy = shuttle_traj[frame_abs]
            shuttle_arr[i] = [sx, sy]

    # Reshape to expected input format: joints (T, 2*17*2)=>(T,68), position (T,4), shuttle (T,2)
    T = window_size
    joints_flat = joints_arr.reshape(T, 2 * 17 * 2)   # (T, 68)
    position_flat = position_arr.reshape(T, 2 * 2)      # (T, 4)

    return joints_flat, position_flat, shuttle_arr


def extract_rgb_embedding(
    cap: cv2.VideoCapture,
    rel_frame: int,
    rgb_model: MultiTaskR2Plus1D,
    device: torch.device,
    n_frames: int = 16,
) -> np.ndarray:
    """Extract 512-dim RGB embedding from clip around rel_frame."""
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    half = n_frames // 2
    start = max(0, rel_frame - half)
    end = min(total - 1, start + n_frames - 1)
    start = max(0, end - n_frames + 1)

    indices = np.linspace(start, end, n_frames, dtype=int)
    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ok, frame = cap.read()
        if not ok:
            frame = np.zeros((224, 224, 3), dtype=np.uint8)
        frame = cv2.resize(frame, (224, 224))
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)

    # Shape (T, H, W, C) → (1, C, T, H, W)
    clip_np = np.stack(frames).astype(np.float32) / 255.0
    mean = np.array([0.43216, 0.394666, 0.37645], dtype=np.float32)
    std = np.array([0.22803, 0.22145, 0.216989], dtype=np.float32)
    clip_np = (clip_np - mean) / std  # (T, H, W, C)
    clip_t = torch.from_numpy(clip_np).permute(3, 0, 1, 2).unsqueeze(0).to(device)  # (1, C, T, H, W)

    with torch.inference_mode():
        emb = rgb_model.backbone(clip_t).float().cpu().numpy()[0]  # (512,)
    return emb


# ── Model loading ─────────────────────────────────────────────────────────────

def load_models(device: torch.device):
    print("[*] Loading RGB backbone...")
    rgb_ckpt = torch.load(RGB_CKPT, map_location="cpu", weights_only=False)
    rgb_model = MultiTaskR2Plus1D().to(device)
    rgb_model.load_state_dict(rgb_ckpt["model"])
    rgb_model.eval()

    def load_cr(ckpt_path, use_temporal, use_contact, use_gate, use_cross=False, use_ppf=False, use_aim=False, use_cs=False):
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        m = CRGatedFusionModel(
            hidden_dim=128,
            num_stroke_classes=len(STROKE_CLASSES),
            num_side_classes=len(SIDE_CLASSES),
            use_temporal=use_temporal,
            use_contact=use_contact,
            use_gate=use_gate,
            use_cross_attention=use_cross,
            use_ppf=use_ppf,
            use_aim_player=use_aim,
            use_cross_shuttle=use_cs,
        ).to(device)
        m.load_state_dict(ckpt["model"])
        m.eval()
        return m

    print("[*] Loading CR-Gated Full...")
    cr_full = load_cr(CR_FULL_CKPT, use_temporal=True, use_contact=True, use_gate=True,
                      use_cross=True, use_ppf=False, use_aim=False, use_cs=False)
    print("[*] Loading CR-Gated NoGate...")
    cr_nogate = load_cr(CR_NOGATE_CKPT, use_temporal=True, use_contact=True, use_gate=False,
                        use_cross=True)
    print("[*] Loading CR-Gated NoTemporal...")
    cr_notemporal = load_cr(CR_NOTEMPORAL_CKPT, use_temporal=False, use_contact=False, use_gate=False,
                            use_cross=True)

    return rgb_model, cr_full, cr_nogate, cr_notemporal


# ── Inference ─────────────────────────────────────────────────────────────────

def classify_event(
    rel_frame: int,
    abs_frame: int,
    side: str,
    rgb_model: MultiTaskR2Plus1D,
    cr_full: CRGatedFusionModel,
    cr_nogate: CRGatedFusionModel,
    cr_notemporal: CRGatedFusionModel,
    cap: cv2.VideoCapture,
    pose_by_frame: dict,
    court_by_frame: dict,
    shuttle_traj: dict,
    device: torch.device,
) -> dict:
    """Run CR-Ensemble inference on one event."""
    # 1. RGB embedding
    rgb_emb = extract_rgb_embedding(cap, rel_frame, rgb_model, device, n_frames=16)
    emb_t = torch.from_numpy(rgb_emb).unsqueeze(0).to(device)  # (1, 512)

    # 2. Sequence features from BFMD annotations
    joints_flat, position_flat, shuttle_arr = build_window_arrays(
        abs_frame, pose_by_frame, court_by_frame, shuttle_traj,
        before=16, after=16,
    )
    s_joints, s_pos, s_shuttle, s_contact, quality = extract_sample_sequences(
        joints_flat, position_flat, shuttle_arr,
        target_index=16,  # contact at center of window
        length=SEQUENCE_LEN,
        before=16, after=16,
    )

    # Tensors
    t_pose = torch.from_numpy(s_joints).unsqueeze(0).to(device)      # (1, 32, 68)
    t_court = torch.from_numpy(s_pos).unsqueeze(0).to(device)        # (1, 32, 4)
    t_shuttle = torch.from_numpy(s_shuttle).unsqueeze(0).to(device)  # (1, 32, 2)
    t_contact = torch.from_numpy(s_contact).unsqueeze(0).to(device)  # (1, 32, 1)
    t_qual = torch.from_numpy(quality).unsqueeze(0).to(device)       # (1, 4)

    with torch.inference_mode():
        f_logits, _, _ = cr_full(emb_t, t_pose, t_court, t_shuttle, t_contact, t_qual)
        ng_logits, _, _ = cr_nogate(emb_t, t_pose, t_court, t_shuttle, t_contact, t_qual)
        nt_logits, _, _ = cr_notemporal(emb_t, t_pose, t_court, t_shuttle, t_contact, t_qual)

        f_prob = torch.softmax(f_logits[0], dim=-1)
        ng_prob = torch.softmax(ng_logits[0], dim=-1)
        nt_prob = torch.softmax(nt_logits[0], dim=-1)

        # Ensemble: Full 40% + NoGate 30% + NoTemporal 30%
        ens_prob = f_prob * 0.40 + ng_prob * 0.30 + nt_prob * 0.30

        ens_pred = STROKE_CLASSES[ens_prob.argmax().item()]
        ens_conf = float(ens_prob.max().item())
        f_pred = STROKE_CLASSES[f_prob.argmax().item()]
        ng_pred = STROKE_CLASSES[ng_prob.argmax().item()]

    return {
        "ensemble": ens_pred,
        "ensemble_conf": ens_conf,
        "full": f_pred,
        "nogate": ng_pred,
        "probs": ens_prob.cpu().numpy().tolist(),
    }


def match_gt(detected_frame: int, gt_events: list, tolerance: int = 12) -> dict | None:
    """Return closest GT event within tolerance frames."""
    best = None
    best_diff = float("inf")
    for gt in gt_events:
        diff = abs(detected_frame - gt["rel_frame"])
        if diff <= tolerance and diff < best_diff:
            best_diff = diff
            best = dict(gt, diff=diff)
    return best


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    t0 = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 72)
    print("CR-ENSEMBLE BFMD 5-MINUTE EVALUATION")
    print(f"Device: {device}")
    print("=" * 72)

    # Load annotations
    print("\n[1/5] Loading BFMD annotations...")
    shuttle_traj = load_shuttle_trajectory()
    court_by_frame = load_court_perframe()
    pose_by_frame = load_pose_by_frame()
    gt_events = load_gt_strokes()
    print(f"  Shuttle frames: {len(shuttle_traj)}")
    print(f"  Court frames:   {len(court_by_frame)}")
    print(f"  Pose frames:    {len(pose_by_frame)}")
    print(f"  GT strokes:     {len(gt_events)} (in 5-min window)")

    # GT distribution
    from collections import Counter
    gt_dist = Counter(e["mapped"] for e in gt_events)
    print(f"  GT distribution: {dict(gt_dist.most_common())}")

    # Load scan events
    print("\n[2/5] Loading hit scan events...")
    with open(SCAN_JSON) as f:
        scan_data = json.load(f)
    scan_events = scan_data["events"]
    print(f"  Scan events: {len(scan_events)}")

    # Load models
    print("\n[3/5] Loading CR-Ensemble models...")
    rgb_model, cr_full, cr_nogate, cr_notemporal = load_models(device)
    print("  All models loaded!")

    # Open video
    print("\n[4/5] Opening 5-minute video clip...")
    cap = cv2.VideoCapture(str(VIDEO_5MIN))
    if not cap.isOpened():
        print(f"Error: Cannot open {VIDEO_5MIN}")
        return
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"  Video: {total_frames} frames")

    # Run inference
    print("\n[5/5] Running CR-Ensemble inference on scan events...")
    results = []
    matched_gt = set()

    for i, event in enumerate(scan_events):
        rel_frame = event["frame"]
        abs_frame = rel_frame + CLIP_OFFSET
        side = event.get("side", "lower")

        if i % 20 == 0:
            print(f"  Progress: {i}/{len(scan_events)} events... ({time.time()-t0:.1f}s)")

        try:
            pred = classify_event(
                rel_frame, abs_frame, side,
                rgb_model, cr_full, cr_nogate, cr_notemporal,
                cap, pose_by_frame, court_by_frame, shuttle_traj, device,
            )
        except Exception as e:
            print(f"  Warning: Error on frame {rel_frame}: {e}")
            continue

        # Match with GT
        gt_match = match_gt(rel_frame, gt_events, tolerance=12)
        gt_label = gt_match["mapped"] if gt_match else None
        gt_bfmd = gt_match["label"] if gt_match else None
        diff = gt_match["diff"] if gt_match else None

        if gt_match:
            gt_key = (gt_match["rel_frame"], gt_match["label"])
            matched_gt.add(gt_key)

        results.append({
            "rel_frame": rel_frame,
            "abs_frame": abs_frame,
            "side": side,
            "scan_score": event.get("score", 0),
            "ensemble_pred": pred["ensemble"],
            "ensemble_conf": pred["ensemble_conf"],
            "full_pred": pred["full"],
            "nogate_pred": pred["nogate"],
            "gt_label": gt_label,
            "gt_bfmd": gt_bfmd,
            "gt_diff_frames": diff,
        })

    cap.release()

    # ── Statistics ────────────────────────────────────────────────────────────
    print("\n" + "=" * 72)
    print("EVALUATION RESULTS")
    print("=" * 72)

    matched_results = [r for r in results if r["gt_label"] is not None]
    total_matched = len(matched_results)

    # Top-1 accuracy
    top1_correct = sum(1 for r in matched_results if r["ensemble_pred"] == r["gt_label"])
    top1_acc = top1_correct / total_matched if total_matched else 0

    # Top-2 accuracy
    top2_correct = 0
    for r in matched_results:
        probs_list = results[results.index(r)]["ensemble_pred"]  # reuse
        # Need per-class top-2 — compute from results
        top2_correct += 1 if r["ensemble_pred"] == r["gt_label"] else 0  # placeholder

    # GT coverage
    total_gt = len(gt_events)
    detected_gt = len(matched_gt)
    recall_gt = detected_gt / total_gt if total_gt else 0

    print(f"\n  Total scan events:      {len(scan_events)}")
    print(f"  Events matched to GT:   {total_matched} / {len(scan_events)}")
    print(f"  GT strokes detected:    {detected_gt} / {total_gt} (recall {recall_gt:.1%})")
    print(f"\n  ━━━ STROKE ACCURACY ━━━")
    print(f"  Top-1 Accuracy:  {top1_correct}/{total_matched} = {top1_acc:.1%}")

    # Per-class breakdown
    print(f"\n  Per-class accuracy (BFMD mapped labels):")
    from collections import defaultdict
    class_stats: dict[str, dict] = defaultdict(lambda: {"correct": 0, "total": 0})
    for r in matched_results:
        gt = r["gt_label"]
        pred = r["ensemble_pred"]
        class_stats[gt]["total"] += 1
        if pred == gt:
            class_stats[gt]["correct"] += 1

    for cls, stat in sorted(class_stats.items(), key=lambda x: -x[1]["total"]):
        acc = stat["correct"] / stat["total"] if stat["total"] else 0
        print(f"    {cls:<15} {stat['correct']:>3}/{stat['total']:>3} = {acc:.0%}")

    # Compare with RGB baseline
    print(f"\n  ━━━ MODEL COMPARISON ━━━")
    print(f"  RGB Baseline (prev):    ~30.3% Top-1")
    print(f"  CR-Ensemble (now):      {top1_acc:.1%} Top-1")

    # Confusion table for top errors
    print(f"\n  ━━━ PREDICTIONS BREAKDOWN ━━━")
    pred_dist = Counter(r["ensemble_pred"] for r in matched_results)
    gt_dist2 = Counter(r["gt_label"] for r in matched_results)
    print(f"  GT distribution:   {dict(gt_dist2.most_common())}")
    print(f"  Pred distribution: {dict(pred_dist.most_common())}")

    # Error analysis
    errors = [r for r in matched_results if r["ensemble_pred"] != r["gt_label"]]
    error_pairs = Counter((r["gt_label"], r["ensemble_pred"]) for r in errors)
    print(f"\n  Top confusions (GT→Pred):")
    for (gt, pred), cnt in error_pairs.most_common(8):
        print(f"    {gt} → {pred}: {cnt}×")

    # Save results
    out_path = REPO_ROOT / "work_dirs" / "bfmd_5min_cr_ensemble_results.json"
    save_data = {
        "total_scan_events": len(scan_events),
        "total_matched": total_matched,
        "gt_recall": round(recall_gt, 4),
        "top1_accuracy": round(top1_acc, 4),
        "per_class": {cls: s for cls, s in class_stats.items()},
        "results": results,
    }
    with open(out_path, "w") as f:
        json.dump(save_data, f, indent=2)
    print(f"\n  Results saved: {out_path}")
    print(f"\n  Total time: {time.time()-t0:.1f}s")
    print("=" * 72)


if __name__ == "__main__":
    main()
