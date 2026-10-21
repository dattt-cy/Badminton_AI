"""Standardized aligned inference with direct Robust KAP.

Canonical Production Pipeline:
- FastTrackNet (AMP FP16)
- YOLOv8-Pose Dual-Player (P0 Upper, P1 Lower) with court boundary filtering
- Conservative shuttle-to-wrist contact refinement (+/- 6 frames)
- MultiTask R(2+1)D Visual Embedding (16 frames)
- Frozen RGB backbone + robust scratch-trained KAP
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch
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
from ai_classifier.localization.contact_refinement import refine_contact_index
from ai_classifier.features import modality_quality_vector
from ai_classifier.features.hitter_crop import padded_square
from ai_classifier.models import (
    KAPScratchModel,
    MultiTaskR2Plus1D,
    STROKE_CLASSES,
)

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


def inpaint_shuttle_trajectory(raw_shuttle: np.ndarray, max_gap: int = 6) -> np.ndarray:
    """Linearly inpaint short dropouts in shuttlecock tracking."""
    arr = raw_shuttle.copy()
    valid = np.any(arr != 0.0, axis=1)
    if not np.any(valid):
        return arr

    n = len(arr)
    i = 0
    while i < n:
        if not valid[i]:
            start = i
            while i < n and not valid[i]:
                i += 1
            gap_len = i - start
            if 0 < gap_len <= max_gap and start > 0 and i < n:
                p_before = arr[start - 1]
                p_after = arr[i]
                for k in range(start, i):
                    alpha = (k - (start - 1)) / float(gap_len + 1)
                    arr[k] = (1.0 - alpha) * p_before + alpha * p_after
        else:
            i += 1
    return arr


class AlignedVideoPipeline:
    def __init__(
        self,
        ckpt_path: str | Path = "work_dirs/kap_online_balanced_selection_20261008/best.pth",
        corners: np.ndarray | None = None,
        orig_w: int = 1280,
        orig_h: int = 720,
        pose_model_path: str | Path = "models/checkpoints/pose/yolov8x-pose.pt",
        tau: float = 0.0,
    ):
        self.orig_w = orig_w
        self.orig_h = orig_h
        self.tau = float(tau)
        corners = corners if corners is not None else COURT_CORNERS_PROFILE["default"]
        self.h_mat = cv2.getPerspectiveTransform(corners, DST_PTS)

        print("[*] Loading FastTrackNet (AMP FP16)...")
        self.tracker = FastTrackNet(device=DEVICE)

        print(f"[*] Loading YOLOv8-Pose: {pose_model_path}...")
        self.pose_model = YOLO(str(pose_model_path))
        # Frame-level caches are populated once for all candidate events in a
        # video.  They intentionally store compact numpy records, not decoded
        # video frames, so a long match does not consume gigabytes of RAM.
        self._pose_cache: dict[int, dict] = {}
        self._shuttle_cache: dict[int, tuple[float, float]] = {}

        print("[*] Loading R(2+1)D Visual Backbone...")
        self.rgb_model = MultiTaskR2Plus1D().to(DEVICE)
        backbone_ckpt = REPO_ROOT / "src" / "ai_classifier" / "models" / "checkpoints" / "visual_backbone.pth"
        if not backbone_ckpt.exists():
            backbone_ckpt = REPO_ROOT / "work_dirs" / "r2plus1d18_mixed_shuttleset_finebadminton" / "best.pth"
        rgb_ckpt = torch.load(str(backbone_ckpt), map_location="cpu", weights_only=False)
        self.rgb_model.load_state_dict(rgb_ckpt["model"], strict=False)
        self.rgb_model.eval()

        print(f"[*] Loading robust scratch KAP classifier: {ckpt_path}...")
        self.clf = KAPScratchModel().to(DEVICE)
        clf_ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
        self.clf.load_state_dict(clf_ckpt["model"])
        self.clf.eval()

        print("[*] Direct Robust KAP Top-1 enabled.")

    def _decode_pose_result(self, res) -> dict:
        """Convert an Ultralytics result into a compact, hitter-neutral record."""
        record = {
            "court": np.zeros(4, dtype=np.float32),
            "pose": np.zeros(68, dtype=np.float32),
            "upper_wrists": np.zeros((2, 2), dtype=np.float32),
            "lower_wrists": np.zeros((2, 2), dtype=np.float32),
            "upper_box": None,
            "lower_box": None,
        }
        if res.keypoints is None or len(res.keypoints.data) == 0:
            return record

        boxes = res.boxes.xyxy.cpu().numpy() if res.boxes is not None else np.empty((0, 4))
        for index, kpts_t in enumerate(res.keypoints.data):
            if index >= len(boxes):
                continue
            kpts = kpts_t[:, :2].cpu().numpy()
            conf = (
                kpts_t[:, 2].cpu().numpy()
                if kpts_t.shape[-1] >= 3 else np.ones(17, dtype=np.float32)
            )
            if conf[15] >= 0.20 and conf[16] >= 0.20:
                ankles = (kpts[15] + kpts[16]) / 2.0
            elif conf[15] >= 0.20:
                ankles = kpts[15]
            elif conf[16] >= 0.20:
                ankles = kpts[16]
            else:
                box = boxes[index]
                ankles = np.asarray([(box[0] + box[2]) / 2.0, box[3]], dtype=np.float32)

            court_pt = cv2.perspectiveTransform(
                np.asarray([[[ankles[0], ankles[1]]]], dtype=np.float32), self.h_mat
            )[0, 0]
            if not (-0.15 <= court_pt[0] <= 1.15 and -0.15 <= court_pt[1] <= 1.15):
                continue

            side = "upper" if court_pt[1] < 0.48 or ankles[1] < 410 else "lower"
            offset = 0 if side == "upper" else 2
            pose_offset = 0 if side == "upper" else 34
            if record[f"{side}_box"] is not None:
                continue
            box = boxes[index].astype(np.float32)
            clean = kpts.copy()
            clean[conf < 0.20] = 0.0
            wrists = kpts[[9, 10]].copy()
            wrists[conf[[9, 10]] < 0.20] = 0.0
            record["court"][offset:offset + 2] = court_pt
            record["pose"][pose_offset:pose_offset + 34] = normalize_person_kpts(clean, box).ravel()
            record[f"{side}_wrists"] = wrists / np.asarray(
                [self.orig_w, self.orig_h], dtype=np.float32
            )
            record[f"{side}_box"] = box
        return record

    @staticmethod
    def _merge_event_ranges(
        event_frames: list[int], total_frames: int, before: int, after: int,
    ) -> list[tuple[int, int]]:
        ranges = sorted(
            (max(0, frame - before), min(total_frames - 1, frame + after))
            for frame in event_frames
        )
        merged: list[list[int]] = []
        for start, end in ranges:
            if merged and start <= merged[-1][1] + 1:
                merged[-1][1] = max(merged[-1][1], end)
            else:
                merged.append([start, end])
        return [(start, end) for start, end in merged]

    def prepare_event_cache(
        self,
        cap: cv2.VideoCapture,
        event_frames: list[int],
        total_frames: int,
        before: int = 15,
        after: int = 30,
        contact_refine_radius: int = 6,
        chunk_frames: int = 192,
        pose_batch: int = 8,
        cache_shuttle: bool = False,
    ) -> dict[str, int]:
        """Precompute pose once for the union of all hit windows.

        This is intended for a brand-new video: the cache is built in memory in
        the same run, then reused by every event. No persistent cache is needed.
        """
        radius = max(0, int(contact_refine_radius))
        ranges = self._merge_event_ranges(
            [int(frame) for frame in event_frames], total_frames,
            before + radius, after + radius,
        )
        requested = sum(end - start + 1 for start, end in ranges)
        print(f"[*] Precomputing shared pose: {len(ranges)} regions, {requested} unique frames")
        for region_start, region_end in ranges:
            chunk_start = region_start
            while chunk_start <= region_end:
                chunk_end = min(region_end, chunk_start + chunk_frames - 1)
                cap.set(cv2.CAP_PROP_POS_FRAMES, chunk_start)
                frames, indices = [], []
                for frame_index in range(chunk_start, chunk_end + 1):
                    ok, frame = cap.read()
                    if not ok:
                        break
                    frames.append(frame)
                    indices.append(frame_index)
                if not frames:
                    break

                # TrackNet depends on the temporal window's median background.
                # Reusing tracks from a longer merged region changes model
                # inputs and was slower in benchmarks, so the canonical path
                # caches only frame-independent pose. This switch remains for
                # controlled experiments.
                if cache_shuttle:
                    track = self.tracker.predict_frames(
                        frames, indices, self.orig_w, self.orig_h, batch_size=32
                    )
                    for _, row in track.iterrows():
                        self._shuttle_cache[int(row["Frame"])] = (
                            float(row["X"]) / self.orig_w,
                            float(row["Y"]) / self.orig_h,
                        )

                # Small outer chunks prevent Ultralytics Results objects from
                # accumulating for an entire rally or match.
                for pose_start in range(0, len(frames), 64):
                    pose_frames = frames[pose_start:pose_start + 64]
                    pose_indices = indices[pose_start:pose_start + 64]
                    with torch.inference_mode():
                        results = self.pose_model(
                            pose_frames, verbose=False, device=DEVICE,
                            half=DEVICE.type == "cuda", batch=pose_batch,
                            imgsz=960, conf=0.20,
                        )
                    for frame_index, result in zip(pose_indices, results):
                        self._pose_cache[frame_index] = self._decode_pose_result(result)
                chunk_start = chunk_end + 1
        return {
            "regions": len(ranges),
            "unique_frames": requested,
            "pose_frames": len(self._pose_cache),
            "shuttle_frames": len(self._shuttle_cache),
        }

    def extract_features(
        self, cap: cv2.VideoCapture, hf_clip: int, total_frames: int,
        hitter_side: str = "lower", before: int = 15, after: int = 30,
        contact_refine_radius: int = 6,
    ) -> dict | None:
        refine_radius = max(0, int(contact_refine_radius))
        start_f = max(0, hf_clip - before - refine_radius)
        end_f = min(total_frames - 1, hf_clip + after + refine_radius)

        cap.set(cv2.CAP_PROP_POS_FRAMES, start_f)
        frames_bgr, frame_indices = [], []
        for f in range(start_f, end_f + 1):
            ok, frame = cap.read()
            if not ok: break
            frames_bgr.append(frame)
            frame_indices.append(f)

        if len(frames_bgr) < 20:
            return None

        if all(frame in self._shuttle_cache for frame in frame_indices):
            coord_map = {frame: self._shuttle_cache[frame] for frame in frame_indices}
        else:
            track_df = self.tracker.predict_frames(
                frames_bgr, frame_indices, self.orig_w, self.orig_h, batch_size=32
            )
            coord_map = {
                int(row["Frame"]): (float(row["X"]) / self.orig_w, float(row["Y"]) / self.orig_h)
                for _, row in track_df.iterrows()
            }

        missing_pose = [frame for frame in frame_indices if frame not in self._pose_cache]
        if missing_pose:
            # Backward-compatible single-event pose path.
            with torch.inference_mode():
                pose_results = self.pose_model(
                    frames_bgr, verbose=False, device=DEVICE,
                    half=DEVICE.type == "cuda", batch=8, imgsz=960, conf=0.20,
                )
            pose_records = [self._decode_pose_result(result) for result in pose_results]
        else:
            pose_records = [self._pose_cache[frame] for frame in frame_indices]

        raw_court, raw_pose, raw_shuttle = [], [], []
        hitter_wrists = []
        hitter_boxes: list[tuple[int, np.ndarray]] = []

        for idx, f_num in enumerate(frame_indices):
            sx, sy = coord_map.get(f_num, (0.0, 0.0))
            raw_shuttle.append([sx, sy])

            record = pose_records[idx]
            raw_court.append(record["court"])
            raw_pose.append(record["pose"])
            hitter_wrists.append(record[f"{hitter_side}_wrists"])
            box = record[f"{hitter_side}_box"]
            if box is not None:
                hitter_boxes.append((f_num, box))

        raw_court = np.array(raw_court, dtype=np.float32)
        raw_pose = np.array(raw_pose, dtype=np.float32)
        raw_shuttle = np.array(raw_shuttle, dtype=np.float32)
        raw_shuttle = inpaint_shuttle_trajectory(raw_shuttle, max_gap=6)
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

        original_contact = min(
            range(len(frame_indices)), key=lambda index: abs(frame_indices[index] - hf_clip)
        )
        refined_contact = refine_contact_index(
            raw_shuttle, selected_wrists, original_contact, refine_radius
        )
        refined_hit_frame = frame_indices[refined_contact]

        # Realign every modality around the refined contact while preserving the
        # same training window. Extra margin was decoded only for refinement.
        window_start = max(0, refined_contact - before)
        window_stop = min(len(frames_bgr), refined_contact + after + 1)
        frames_bgr = frames_bgr[window_start:window_stop]
        raw_court = raw_court[window_start:window_stop]
        raw_pose = raw_pose[window_start:window_stop]
        raw_shuttle = raw_shuttle[window_start:window_stop]
        selected_wrists = selected_wrists[window_start:window_stop]
        analytics_frames = frame_indices[window_start:window_stop]
        contact_index = refined_contact - window_start

        shuttle_pixels = raw_shuttle * np.asarray(
            [self.orig_w, self.orig_h], dtype=np.float32
        )
        shuttle_court = cv2.perspectiveTransform(
            shuttle_pixels[:, None, :].astype(np.float32), self.h_mat
        )[:, 0, :]
        shuttle_court[~np.any(raw_shuttle != 0.0, axis=1)] = np.nan
        player_offset = 0 if hitter_side == "upper" else 2
        opponent_offset = 2 if hitter_side == "upper" else 0

        # 3. Resample to 32 frames (Contact at index 10)
        s_p = resample(raw_court, 32)
        s_j = resample(raw_pose, 32)
        s_s = resample(raw_shuttle, 32)
        s_c = np.linspace(
            -contact_index, len(raw_shuttle) - contact_index - 1, 32, dtype=np.float32
        )[:, None] / float(before)

        # 4. Modality Quality
        s_q = modality_quality_vector(raw_pose, raw_shuttle)
        s_q = np.asarray(s_q, dtype=np.float32)
        s_q[[1, 3]] /= 100.0

        # 5. RGB: reproduce the training recipe (stable hitter crop, 16 samples
        # spanning the entire event window, padding=0.55).
        if hitter_boxes:
            # Training selected the first valid frame in 0,-2,+2,-4,+4,... order;
            # choosing the closest valid detection reproduces that behavior.
            _, hitter_box = min(
                hitter_boxes, key=lambda item: abs(item[0] - refined_hit_frame)
            )
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
            "hit_frame": refined_hit_frame,
            "original_hit_frame": hf_clip,
            "contact_offset": refined_hit_frame - hf_clip,
            "emb": emb.cpu(),
            "s_j": torch.from_numpy(s_j).unsqueeze(0),
            "s_p": torch.from_numpy(s_p).unsqueeze(0),
            "s_s": torch.from_numpy(s_s).unsqueeze(0),
            "s_c": torch.from_numpy(s_c).unsqueeze(0),
            "s_q": torch.from_numpy(s_q).unsqueeze(0),
            "crop_box": crop_box,
            "analytics_context": {
                "frame_indices": analytics_frames,
                "contact_index": contact_index,
                "shuttle_image": raw_shuttle,
                "shuttle_court": shuttle_court,
                "player_court": raw_court[:, player_offset:player_offset + 2],
                "opponent_court": raw_court[:, opponent_offset:opponent_offset + 2],
                "hitter_wrists": selected_wrists,
            },
        }

    def predict_detailed(
        self, feat: dict, classifier: KAPScratchModel | None = None, tau: float | None = None
    ) -> dict:
        model = classifier if classifier is not None else self.clf
        effective_tau = self.tau if tau is None else float(tau)
        with torch.no_grad():
            inputs = (
                feat["emb"].to(DEVICE), feat["s_j"].to(DEVICE),
                feat["s_p"].to(DEVICE), feat["s_s"].to(DEVICE),
                feat["s_c"].to(DEVICE), feat["s_q"].to(DEVICE),
            )
            logits, _side_logits, _gates = model(*inputs)
        if effective_tau > 0.0:
            priors = torch.tensor([0.08, 0.15, 0.12, 0.10, 0.18, 0.07, 0.20, 0.10], device=logits.device)
            logits = logits - effective_tau * torch.log(priors + 1e-8)
        probs = F.softmax(logits, dim=-1)[0]
        pred_idx = int(logits.argmax(dim=-1).item())
        top2_idx = logits.topk(2, dim=-1).indices[0].tolist()
        return {
            "stroke": STROKE_CLASSES[pred_idx], "confidence": float(probs[pred_idx]),
            "top2": [STROKE_CLASSES[k] for k in top2_idx], "logits": logits[0],
        }

    def predict(self, feat: dict) -> tuple[str, float, list[str], torch.Tensor]:
        result = self.predict_detailed(feat)
        return result["stroke"], result["confidence"], result["top2"], result["logits"]

    def predict_detailed_tta(
        self,
        cap: cv2.VideoCapture,
        frame: int,
        total_frames: int,
        hitter_side: str,
        contact_refine_radius: int = 6,
        offsets: tuple[int, ...] = (-2, 0, 2),
        classifier: KAPScratchModel | None = None,
    ) -> tuple[dict | None, dict | None]:
        """Temporal Test-Time Augmentation (TTA) multi-window soft-voting.

        Evaluates multiple temporal shifts around candidate hit frame to smooth out
        contact frame jitter and maximize generalization accuracy.
        """
        base_feature: dict | None = None
        probs_list: list[torch.Tensor] = []

        for offset in offsets:
            shifted_frame = int(frame + offset)
            if shifted_frame < 0 or shifted_frame >= total_frames:
                continue
            feat = self.extract_features(
                cap,
                shifted_frame,
                total_frames,
                hitter_side=hitter_side,
                contact_refine_radius=contact_refine_radius,
            )
            if feat is None:
                continue
            if offset == 0 or base_feature is None:
                base_feature = feat
            detail = self.predict_detailed(feat, classifier=classifier)
            probs = F.softmax(detail["logits"], dim=-1)
            probs_list.append(probs)

        if not probs_list or base_feature is None:
            return None, None

        avg_probs = torch.stack(probs_list).mean(dim=0)
        pred_idx = int(avg_probs.argmax().item())
        top2_idx = avg_probs.topk(2).indices.tolist()

        detail_tta = {
            "stroke": STROKE_CLASSES[pred_idx],
            "confidence": float(avg_probs[pred_idx].item()),
            "top2": [STROKE_CLASSES[k] for k in top2_idx],
            "logits": torch.log(avg_probs.clamp_min(1e-7)),
            "tta_used": len(probs_list) > 1,
            "tta_windows": len(probs_list),
        }
        return base_feature, detail_tta

    def predict_detailed_adaptive_tta(
        self,
        cap: cv2.VideoCapture,
        frame: int,
        total_frames: int,
        hitter_side: str,
        contact_refine_radius: int = 6,
        confidence_threshold: float = 0.75,
        margin_threshold: float = 0.20,
        offsets: tuple[int, ...] = (-2, 2),
        classifier: KAPScratchModel | None = None,
    ) -> tuple[dict | None, dict | None]:
        """Run extra temporal windows only for uncertain base predictions."""
        base_feature = self.extract_features(
            cap,
            frame,
            total_frames,
            hitter_side=hitter_side,
            contact_refine_radius=contact_refine_radius,
        )
        if base_feature is None:
            return None, None
        base_detail = self.predict_detailed(base_feature, classifier=classifier)
        base_probs = F.softmax(base_detail["logits"], dim=-1)
        top2_probs = base_probs.topk(2).values
        margin = float((top2_probs[0] - top2_probs[1]).item())
        if (
            base_detail["confidence"] >= confidence_threshold
            and margin >= margin_threshold
        ):
            base_detail["tta_used"] = False
            base_detail["tta_windows"] = 1
            return base_feature, base_detail

        probs_list = [base_probs]
        for offset in offsets:
            shifted_frame = frame + offset
            if shifted_frame < 0 or shifted_frame >= total_frames:
                continue
            feature = self.extract_features(
                cap,
                shifted_frame,
                total_frames,
                hitter_side=hitter_side,
                contact_refine_radius=contact_refine_radius,
            )
            if feature is None:
                continue
            detail = self.predict_detailed(feature, classifier=classifier)
            probs_list.append(F.softmax(detail["logits"], dim=-1))

        average = torch.stack(probs_list).mean(dim=0)
        pred_idx = int(average.argmax().item())
        top2_idx = average.topk(2).indices.tolist()

        predicted_stroke = STROKE_CLASSES[pred_idx]
        confidence = float(average[pred_idx].item())
        top2_classes = [STROKE_CLASSES[index] for index in top2_idx]

        # Context-aware disambiguation for low-confidence boundary pairs
        if confidence < 0.60:
            if predicted_stroke == "lift" and top2_classes == ["lift", "net_attack"]:
                predicted_stroke = "net_attack"
            elif set(top2_classes) == {"drive", "net_shot"} and base_feature is not None:
                # s_p: normalized court position (x, y) where y is along length
                # Front court near net: y_dist_net < 0.18 -> net_shot, Mid-court: > 0.22 -> drive
                court_pts = base_feature["s_p"][0].detach().cpu().numpy()
                y_pos = float(court_pts[:, 1].mean())
                y_dist_net = y_pos if hitter_side == "upper" else (1.0 - y_pos)
                if y_dist_net > 0.22:
                    predicted_stroke = "drive"
                elif y_dist_net < 0.16:
                    predicted_stroke = "net_shot"

        return base_feature, {
            "stroke": predicted_stroke,
            "confidence": confidence,
            "top2": top2_classes,
            "logits": torch.log(average.clamp_min(1e-7)),
            "tta_used": len(probs_list) > 1,
            "tta_windows": len(probs_list),
        }

