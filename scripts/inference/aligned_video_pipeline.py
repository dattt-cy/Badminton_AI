"""Standardized aligned inference with robust KAP and Top-2 reranking.

Canonical Production Pipeline:
- FastTrackNet (AMP FP16)
- YOLOv8-Pose Dual-Player (P0 Upper, P1 Lower) with court boundary filtering
- MultiTask R(2+1)D Visual Embedding (16 frames)
- Frozen RGB backbone + robust scratch-trained KAP + conservative reranker
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import functional as VF
from ultralytics import YOLO

REPO_ROOT = Path(__file__).resolve().parents[2]
import sys
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from ai_classifier.localization.fast_tracknet import FastTrackNet
from ai_classifier.features import modality_quality_vector
from ai_classifier.features.hitter_crop import padded_square
from ai_classifier.models import (
    KAPScratchModel,
    MultiTaskR2Plus1D,
    PairSpecialist,
    STROKE_CLASSES,
    Top2Reranker,
)
from ai_classifier.models.kap_fusion import kap_reranker_features

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_MEAN = torch.tensor([0.43216, 0.394666, 0.37645], device=DEVICE).view(1, 3, 1, 1)
_STD = torch.tensor([0.22803, 0.22145, 0.216989], device=DEVICE).view(1, 3, 1, 1)

COURT_CORNERS_PROFILE = {
    "41": np.float32([[298.0, 656.0], [980.0, 656.0], [858.0, 306.0], [411.0, 306.0]]),
    "42": np.float32([[295.0, 657.0], [986.0, 657.0], [865.0, 304.0], [417.0, 304.0]]),
    "default": np.float32([[298.0, 656.0], [980.0, 656.0], [858.0, 306.0], [411.0, 306.0]]),
}
DST_PTS = np.float32([[0.0, 1.0], [1.0, 1.0], [1.0, 0.0], [0.0, 0.0]])


def resample(array: np.ndarray, target_length: int = 32) -> np.ndarray:
    if len(array) == target_length:
        return array.astype(np.float32)
    indices = np.linspace(0, len(array) - 1, target_length)
    low = np.floor(indices).astype(int)
    high = np.ceil(indices).astype(int)
    weights = (indices - low)[:, None]
    while weights.ndim < array.ndim:
        weights = weights[..., None]
    return ((1 - weights) * array[low] + weights * array[high]).astype(np.float32)


def normalize_person_kpts(kpts_17x2: np.ndarray, box_xyxy: np.ndarray) -> np.ndarray:
    """Match the BST-derived training cache: bbox-center / bbox-diagonal."""
    box = box_xyxy.astype(np.float32)
    center = (box[:2] + box[2:]) / 2.0
    diagonal = max(float(np.linalg.norm(box[2:] - box[:2])), 1e-6)
    normalized = (kpts_17x2.astype(np.float32) - center) / diagonal
    missing = np.all(kpts_17x2 == 0.0, axis=1)
    normalized[missing] = 0.0
    return normalized


def specialist_physics_features(
    raw_shuttle: np.ndarray,
    hitter_wrists: np.ndarray,
    raw_court: np.ndarray,
    hitter_side: str,
    contact_index: int,
) -> np.ndarray:
    """Compact geometry/trajectory cues for pair-specific classifiers."""
    valid_shuttle = np.any(raw_shuttle != 0.0, axis=1)
    valid_wrist = np.any(hitter_wrists != 0.0, axis=1)
    near = range(max(0, contact_index - 2), min(len(raw_shuttle), contact_index + 3))
    distances = [
        float(np.linalg.norm(raw_shuttle[i] - hitter_wrists[i]))
        for i in near if valid_shuttle[i] and valid_wrist[i]
    ]
    wrist_distance = min(distances) if distances else 2.0

    post_ids = [
        i for i in range(contact_index, min(len(raw_shuttle), contact_index + 10))
        if valid_shuttle[i]
    ]
    if len(post_ids) >= 2:
        first, last = raw_shuttle[post_ids[0]], raw_shuttle[post_ids[-1]]
        delta = last - first
        steps = max(post_ids[-1] - post_ids[0], 1)
        velocity = delta / steps
        speed = float(np.linalg.norm(velocity))
        if len(post_ids) >= 3:
            points = raw_shuttle[post_ids]
            curvature = float(np.linalg.norm(np.diff(points, n=2, axis=0), axis=1).mean())
        else:
            curvature = 0.0
    else:
        delta = np.zeros(2, dtype=np.float32)
        velocity = np.zeros(2, dtype=np.float32)
        speed = curvature = 0.0

    court_offset = 0 if hitter_side == "upper" else 2
    court_xy = raw_court[contact_index, court_offset:court_offset + 2]
    court_valid = float(np.any(court_xy != 0.0))
    net_distance = abs(float(court_xy[1]) - 0.5) if court_valid else 1.0
    return np.asarray([
        court_xy[0], court_xy[1], net_distance, court_valid,
        wrist_distance, delta[0], delta[1], velocity[0], velocity[1],
        speed, curvature, float(valid_shuttle[contact_index:contact_index + 10].mean()),
        float(valid_wrist[contact_index]),
    ], dtype=np.float32)


class AlignedVideoPipeline:
    def __init__(
        self,
        ckpt_path: str | Path = "work_dirs/kap_robust_seed20260926/kap_best.pth",
        reranker_path: str | Path = "work_dirs/kap_top2_reranker_seed20260927/best.pth",
        specialist_paths: list[str | Path] | None = None,
        corners: np.ndarray | None = None,
        orig_w: int = 1280,
        orig_h: int = 720,
    ):
        self.orig_w = orig_w
        self.orig_h = orig_h
        corners = corners if corners is not None else COURT_CORNERS_PROFILE["default"]
        self.h_mat = cv2.getPerspectiveTransform(corners, DST_PTS)

        print("[*] Loading FastTrackNet (AMP FP16)...")
        self.tracker = FastTrackNet(device=DEVICE)

        print("[*] Loading YOLOv8-Pose (FP16 half)...")
        self.pose_model = YOLO("models/checkpoints/pose/yolov8n-pose.pt")

        print("[*] Loading R(2+1)D Visual Backbone...")
        self.rgb_model = MultiTaskR2Plus1D().to(DEVICE)
        rgb_ckpt = torch.load("work_dirs/r2plus1d18_mixed_shuttleset_finebadminton/best.pth", map_location="cpu", weights_only=False)
        self.rgb_model.load_state_dict(rgb_ckpt["model"], strict=False)
        self.rgb_model.eval()

        print(f"[*] Loading robust scratch KAP classifier: {ckpt_path}...")
        self.clf = KAPScratchModel().to(DEVICE)
        clf_ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
        self.clf.load_state_dict(clf_ckpt["model"])
        self.clf.eval()

        print(f"[*] Loading conservative Top-2 reranker: {reranker_path}...")
        reranker_ckpt = torch.load(reranker_path, map_location=DEVICE, weights_only=False)
        self.reranker = Top2Reranker(int(reranker_ckpt["input_dim"])).to(DEVICE)
        self.reranker.load_state_dict(reranker_ckpt["model"])
        self.reranker.eval()
        self.reranker_threshold = float(reranker_ckpt["threshold"])
        self.specialists = []
        for path in specialist_paths or []:
            payload = torch.load(path, map_location=DEVICE, weights_only=False)
            specialist = PairSpecialist(int(payload["input_dim"])).to(DEVICE)
            specialist.load_state_dict(payload["model"])
            specialist.eval()
            self.specialists.append((payload, specialist))

    def extract_features(
        self, cap: cv2.VideoCapture, hf_clip: int, total_frames: int,
        hitter_side: str = "lower", before: int = 15, after: int = 30,
    ) -> dict | None:
        start_f = max(0, hf_clip - before)
        end_f = min(total_frames - 1, hf_clip + after)

        cap.set(cv2.CAP_PROP_POS_FRAMES, start_f)
        frames_bgr, frame_indices = [], []
        for f in range(start_f, end_f + 1):
            ok, frame = cap.read()
            if not ok: break
            frames_bgr.append(frame)
            frame_indices.append(f)

        if len(frames_bgr) < 20:
            return None

        # 1. FastTrackNet Shuttlecock Tracking
        track_df = self.tracker.predict_frames(frames_bgr, frame_indices, self.orig_w, self.orig_h, batch_size=32)
        coord_map = {int(r["Frame"]): (float(r["X"]) / self.orig_w, float(r["Y"]) / self.orig_h) for _, r in track_df.iterrows()}

        # 2. Dual-Player Pose & Court Position
        with torch.inference_mode():
            pose_results = self.pose_model(frames_bgr, verbose=False, device=DEVICE, half=True, batch=16)

        raw_court, raw_pose, raw_shuttle = [], [], []
        hitter_wrists = []
        hitter_boxes: list[tuple[int, np.ndarray]] = []

        for idx, f_num in enumerate(frame_indices):
            sx, sy = coord_map.get(f_num, (0.0, 0.0))
            raw_shuttle.append([sx, sy])

            res = pose_results[idx]
            p0_court, p1_court = None, None
            p0_kpts_norm, p1_kpts_norm = None, None
            frame_hitter_wrists = np.zeros((2, 2), dtype=np.float32)

            if res.keypoints is not None and len(res.keypoints.data) > 0:
                boxes_np = res.boxes.xyxy.cpu().numpy() if res.boxes is not None else np.empty((0, 4))
                for b_idx, kpts_t in enumerate(res.keypoints.data):
                    kpts_np = kpts_t[:, :2].cpu().numpy()
                    ankles = (kpts_np[15] + kpts_np[16]) / 2.0
                    feet_pt = np.array([[[ankles[0], ankles[1]]]], dtype=np.float32)
                    court_pt = cv2.perspectiveTransform(feet_pt, self.h_mat)[0, 0]

                    if not (-0.15 <= court_pt[0] <= 1.15 and -0.15 <= court_pt[1] <= 1.15):
                        continue

                    is_upper = court_pt[1] < 0.48 or ankles[1] < 410
                    if b_idx >= len(boxes_np):
                        continue
                    box = boxes_np[b_idx]
                    norm_k = normalize_person_kpts(kpts_np, box)

                    if is_upper and p0_court is None:
                        p0_court = [float(court_pt[0]), float(court_pt[1])]
                        p0_kpts_norm = norm_k
                        if hitter_side == "upper":
                            hitter_boxes.append((abs(f_num - hf_clip), box))
                            frame_hitter_wrists = kpts_np[[9, 10]] / np.asarray(
                                [self.orig_w, self.orig_h], dtype=np.float32
                            )
                    elif not is_upper and p1_court is None:
                        p1_court = [float(court_pt[0]), float(court_pt[1])]
                        p1_kpts_norm = norm_k
                        if hitter_side == "lower":
                            hitter_boxes.append((abs(f_num - hf_clip), box))
                            frame_hitter_wrists = kpts_np[[9, 10]] / np.asarray(
                                [self.orig_w, self.orig_h], dtype=np.float32
                            )

            # Missing detections remain zero, as in the training feature cache.
            c0 = p0_court if p0_court is not None else [0.0, 0.0]
            c1 = p1_court if p1_court is not None else [0.0, 0.0]
            raw_court.append([c0[0], c0[1], c1[0], c1[1]])
            k0_flat = p0_kpts_norm.flatten() if p0_kpts_norm is not None else np.zeros(34, dtype=np.float32)
            k1_flat = p1_kpts_norm.flatten() if p1_kpts_norm is not None else np.zeros(34, dtype=np.float32)
            raw_pose.append(np.concatenate([k0_flat, k1_flat]).astype(np.float32))
            hitter_wrists.append(frame_hitter_wrists)

        raw_court = np.array(raw_court, dtype=np.float32)
        raw_pose = np.array(raw_pose, dtype=np.float32)
        raw_shuttle = np.array(raw_shuttle, dtype=np.float32)
        hitter_wrists = np.array(hitter_wrists, dtype=np.float32)
        # Compare shuttle with the closest of the two wrists.
        wrist_center = np.where(
            np.any(hitter_wrists != 0.0, axis=2, keepdims=True), hitter_wrists, np.nan
        )
        shuttle_expanded = raw_shuttle[:, None, :]
        wrist_distance = np.linalg.norm(wrist_center - shuttle_expanded, axis=2)
        closest = np.nanargmin(
            np.where(np.isnan(wrist_distance), np.inf, wrist_distance), axis=1
        )
        selected_wrists = hitter_wrists[np.arange(len(hitter_wrists)), closest]
        no_wrist = ~np.isfinite(wrist_distance).any(axis=1)
        selected_wrists[no_wrist] = 0.0
        physics_features = specialist_physics_features(
            raw_shuttle, selected_wrists, raw_court, hitter_side,
            min(before, len(raw_shuttle) - 1),
        )

        # 3. Resample to 32 frames (Contact at index 10)
        s_p = resample(raw_court, 32)
        s_j = resample(raw_pose, 32)
        s_s = resample(raw_shuttle, 32)
        s_c = np.linspace(-before, after, 32, dtype=np.float32)[:, None] / float(before)

        # 4. Modality Quality
        s_q = modality_quality_vector(raw_pose, raw_shuttle)

        # 5. RGB: reproduce the training recipe (stable hitter crop, 16 samples
        # spanning the entire event window, padding=0.55).
        if hitter_boxes:
            # Training selected the first valid frame in 0,-2,+2,-4,+4,... order;
            # choosing the closest valid detection reproduces that behavior.
            _, hitter_box = min(hitter_boxes, key=lambda item: item[0])
            crop_box = padded_square(hitter_box, self.orig_w, self.orig_h, padding=0.55)
        else:
            side = int(round(min(self.orig_w, self.orig_h) * 0.62))
            cx = self.orig_w // 2
            cy = int(round(self.orig_h * (0.40 if hitter_side == "upper" else 0.70)))
            crop_box = (
                max(0, cx - side // 2), max(0, cy - side // 2),
                min(self.orig_w, cx + side // 2), min(self.orig_h, cy + side // 2),
            )
        sampled = np.rint(np.linspace(0, len(frames_bgr) - 1, 16)).astype(np.int64)
        x1, y1, x2, y2 = crop_box
        rgb_16 = [
            cv2.cvtColor(frames_bgr[int(index)][y1:y2, x1:x2], cv2.COLOR_BGR2RGB)
            for index in sampled
        ]

        t_rgb = torch.from_numpy(np.stack(rgb_16[:16])).permute(0, 3, 1, 2).float().div_(255.0)
        t_rgb = VF.resize(t_rgb, [112, 112], antialias=False).to(DEVICE)
        t_rgb = (t_rgb - _MEAN) / _STD
        hitter_clip = t_rgb.permute(1, 0, 2, 3).unsqueeze(0)

        with torch.inference_mode():
            emb = self.rgb_model.extract_features(hitter_clip)

        return {
            "hit_frame": hf_clip,
            "emb": emb.cpu(),
            "s_j": torch.from_numpy(s_j).unsqueeze(0),
            "s_p": torch.from_numpy(s_p).unsqueeze(0),
            "s_s": torch.from_numpy(s_s).unsqueeze(0),
            "s_c": torch.from_numpy(s_c).unsqueeze(0),
            "s_q": torch.from_numpy(s_q).unsqueeze(0),
            "crop_box": crop_box,
            "specialist_physics": torch.from_numpy(physics_features),
        }

    def predict_detailed(self, feat: dict) -> dict:
        with torch.no_grad():
            inputs = (
                feat["emb"].to(DEVICE), feat["s_j"].to(DEVICE),
                feat["s_p"].to(DEVICE), feat["s_s"].to(DEVICE),
                feat["s_c"].to(DEVICE), feat["s_q"].to(DEVICE),
            )
            rerank_input, logits, top2 = kap_reranker_features(self.clf, *inputs)
            swap_probability = float(self.reranker(rerank_input).sigmoid().item())
            should_swap = swap_probability >= self.reranker_threshold
            base_logits = logits.clone()
            if should_swap:
                first, second = int(top2[0, 0]), int(top2[0, 1])
                first_value = logits[0, first].clone()
                logits[0, first] = logits[0, second]
                logits[0, second] = first_value
            specialist_applied = None
            specialist_probability = None
            base_pair = {STROKE_CLASSES[int(top2[0, 0])], STROKE_CLASSES[int(top2[0, 1])]}
            specialist_input = torch.cat([
                rerank_input, feat["specialist_physics"].to(DEVICE).unsqueeze(0)
            ], dim=1)
            for payload, specialist in self.specialists:
                class_names = tuple(payload["class_names"])
                if base_pair != set(class_names):
                    continue
                specialist_probability = float(specialist(specialist_input).sigmoid().item())
                chosen = class_names[int(specialist_probability >= float(payload["threshold"]))]
                chosen_index = STROKE_CLASSES.index(chosen)
                other_index = next(int(value) for value in top2[0] if int(value) != chosen_index)
                if logits[0, chosen_index] < logits[0, other_index]:
                    value = logits[0, chosen_index].clone()
                    logits[0, chosen_index] = logits[0, other_index]
                    logits[0, other_index] = value
                specialist_applied = "__".join(class_names)
                break
        probs = F.softmax(logits, dim=-1)[0]
        base_probs = F.softmax(base_logits, dim=-1)[0]
        pred_idx = int(logits.argmax(dim=-1).item())
        base_idx = int(base_logits.argmax(dim=-1).item())
        top2_idx = logits.topk(2, dim=-1).indices[0].tolist()
        return {
            "stroke": STROKE_CLASSES[pred_idx], "confidence": float(probs[pred_idx]),
            "top2": [STROKE_CLASSES[k] for k in top2_idx], "logits": logits[0],
            "base_stroke": STROKE_CLASSES[base_idx],
            "base_confidence": float(base_probs[base_idx]),
            "reranker_swapped": should_swap, "swap_probability": swap_probability,
            "reranker_feature": rerank_input[0].detach().cpu(),
            "base_top2_indices": top2[0].detach().cpu(),
            "specialist_applied": specialist_applied,
            "specialist_probability": specialist_probability,
        }

    def predict(self, feat: dict) -> tuple[str, float, list[str], torch.Tensor]:
        result = self.predict_detailed(feat)
        return result["stroke"], result["confidence"], result["top2"], result["logits"]
