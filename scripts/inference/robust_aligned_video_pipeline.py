"""Standardized aligned inference with direct Robust KAP.

Canonical Production Pipeline:
- FastTrackNet (AMP FP16)
- YOLOv8-Pose Dual-Player (P0 Upper, P1 Lower) with court boundary filtering
- Conservative shuttle-to-wrist contact refinement (+/- 6 frames)
- MultiTask R(2+1)D Visual Embedding (16 frames)
- Frozen RGB backbone + robust scratch-trained KAP
"""

from __future__ import annotations

from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
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
from ai_classifier.model_cache import cached, file_key
from ai_classifier.localization.contact_refinement import refine_contact_index
from ai_classifier.features import modality_quality_vector
from ai_classifier.features.hitter_crop import padded_square
from ai_classifier.models import (
    KAPCrossTransModel,
    KAPCrossTransV2Model,
    KAPScratchModel,
    MultiTaskR2Plus1D,
    OwnRGBEncoder,
    SIDE_CLASSES,
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

# BST "between 2 hits with max limits" clips (30 fps): a stroke's sequence starts at
# the previous hit (first hit of a rally: hit-15) and ends at the next hit+7 (last
# hit: hit+15). Hits further apart than RALLY_GAP_FRAMES belong to different rallies.
RALLY_GAP_FRAMES = 75
RALLY_EDGE_FRAMES = 15
NEXT_HIT_TAIL_FRAMES = 7
SAME_EVENT_FRAMES = 8

# Pose runs on 64-frame groups. Ultralytics treats a list of images as one batch whatever
# ``batch`` says, so a group is one forward pass; the direct path below keeps that grouping.
POSE_GROUP_FRAMES = 64
POSE_ARGS = dict(verbose=False, device=DEVICE, half=DEVICE.type == "cuda", imgsz=960, conf=0.20)
_letterbox_pool: ThreadPoolExecutor | None = None


def _pose_groups(model: YOLO, groups: list[list[np.ndarray]]) -> list[list]:
    """Ultralytics pose Results per group, as ``model(group, **POSE_ARGS)`` returns them.

    Calls the predictor's own inference and post-processing directly and letterboxes the next
    group on CPU threads while the GPU runs the current one. A predictor call also rebuilt its
    data source and timed every stage with GPU synchronisation, and letterboxed on one thread:
    10.2 ms a frame against 5.1 ms here on an RTX 3050, with identical keypoints and boxes.
    """
    global _letterbox_pool
    if not groups:
        return []
    out = []
    if not getattr(model, "_direct_pose_ready", False):
        # A normal call builds the predictor with POSE_ARGS; every pose call in this module uses them.
        out.append(model(groups[0], **POSE_ARGS))
        groups = groups[1:]
        model._direct_pose_ready = True
    if not groups:
        return out
    predictor = model.predictor
    if _letterbox_pool is None:
        _letterbox_pool = ThreadPoolExecutor(4, thread_name_prefix="pose-letterbox")

    def prepare(group: list[np.ndarray]) -> torch.Tensor:
        # pre_transform on one image letterboxes exactly as on a group of same-sized frames.
        boxed = list(_letterbox_pool.map(lambda image: predictor.pre_transform([image])[0], group))
        # The tensor steps of predictor.preprocess for a list of uint8 BGR images.
        tensor = torch.from_numpy(np.stack(boxed)).to(predictor.device).permute(0, 3, 1, 2).flip(1).contiguous()
        return (tensor.half() if predictor.model.fp16 else tensor.float()).div_(255)

    pending = _letterbox_pool.submit(prepare, groups[0])
    for index, group in enumerate(groups):
        image = pending.result()
        if index + 1 < len(groups):
            pending = _letterbox_pool.submit(prepare, groups[index + 1])
        with torch.inference_mode():
            preds = predictor.inference(image)
            predictor.batch = ([""] * len(group), group, [""] * len(group))
            out.append(predictor.postprocess(preds, image, group))
    return out


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


class SequentialFrameReader:
    """Serve frame ranges from one VideoCapture with as few seeks as possible.

    Seeking an H.264 stream decodes again from the previous keyframe, which made
    seeking the largest single cost of a match analysis. Consecutive hits and the
    TTA windows of one hit overlap, so recently decoded frames are kept and a
    short forward gap is decoded through instead of seeking.
    """

    def __init__(self, cap: cv2.VideoCapture, capacity: int = 192, max_forward_gap: int = 90):
        self.cap = cap
        self.capacity = capacity
        self.max_forward_gap = max_forward_gap
        self.position: int | None = None  # index of the frame the next cap.read() returns
        self.frames: OrderedDict[int, np.ndarray] = OrderedDict()

    def _move_to(self, frame_index: int) -> None:
        if self.position == frame_index:
            return
        if self.position is not None and self.position < frame_index <= self.position + self.max_forward_gap:
            while self.position < frame_index:
                if not self.cap.grab():
                    break
                self.position += 1
            if self.position == frame_index:
                return
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        self.position = frame_index

    def read(self, start: int, end: int) -> tuple[list[np.ndarray], list[int]]:
        """Frames start..end inclusive, stopping at the first frame that cannot be read."""
        frames, indices = [], []
        for frame_index in range(start, end + 1):
            frame = self.frames.get(frame_index)
            if frame is None:
                self._move_to(frame_index)
                ok, frame = self.cap.read()
                if not ok:
                    self.position = None
                    break
                self.position = frame_index + 1
                self.frames[frame_index] = frame
                if len(self.frames) > self.capacity:
                    self.frames.popitem(last=False)
            else:
                self.frames.move_to_end(frame_index)
            frames.append(frame)
            indices.append(frame_index)
        return frames, indices


class AlignedVideoPipeline:
    def __init__(
        self,
        ckpt_path: str | Path = "work_dirs/kap_online_balanced_selection_20261008/best.pth",
        corners: np.ndarray | None = None,
        orig_w: int = 1280,
        orig_h: int = 720,
        pose_model_path: str | Path = "models/checkpoints/pose/yolov8x-pose.pt",
        tau: float = 0.0,
        rgb_before: int = 30,
    ):
        self.orig_w = orig_w
        self.orig_h = orig_h
        self.tau = float(tau)
        # The visual backbone was trained on 16 frames spanning hit-30..hit+30;
        # pose and shuttle keep the shorter before/after window of extract_features.
        self.rgb_before = int(rgb_before)
        corners = corners if corners is not None else COURT_CORNERS_PROFILE["default"]
        self.h_mat = cv2.getPerspectiveTransform(corners, DST_PTS)

        # Models are loaded once per process and shared by every video; only the
        # per-video state below is rebuilt.
        self.tracker = cached("tracknet", (str(DEVICE),), self._load_tracker)
        self.pose_model = cached("pose", file_key(pose_model_path), lambda: self._load_pose(pose_model_path))
        # Frame-level caches are populated once for all candidate events in a
        # video.  They intentionally store compact numpy records, not decoded
        # video frames, so a long match does not consume gigabytes of RAM.
        self._pose_cache: dict[int, dict] = {}
        self._shuttle_cache: dict[int, tuple[float, float]] = {}
        self._tracknet_resize_cache: dict[int, np.ndarray] = {}
        self._reader: SequentialFrameReader | None = None

        self.clf, self.rgb_model, self.context_window = cached(
            "kap", file_key(ckpt_path), lambda: self._load_classifier(ckpt_path)
        )
        self._event_frames: np.ndarray | None = None
        if self.context_window:
            print(f"[*] KAP context window: hit-{self.context_window[0]}..hit+{self.context_window[1]}")

        print("[*] Direct Robust KAP Top-1 enabled.")

    def _read_range(self, cap: cv2.VideoCapture, start: int, end: int) -> tuple[list[np.ndarray], list[int]]:
        if self._reader is None or self._reader.cap is not cap:
            self._reader = SequentialFrameReader(cap)
        return self._reader.read(start, end)

    @staticmethod
    def _load_tracker() -> FastTrackNet:
        print("[*] Loading FastTrackNet (AMP FP16)...")
        return FastTrackNet(device=DEVICE)

    @staticmethod
    def _load_pose(pose_model_path: str | Path) -> YOLO:
        print(f"[*] Loading YOLOv8-Pose: {pose_model_path}...")
        return YOLO(str(pose_model_path))

    @staticmethod
    def _load_classifier(ckpt_path: str | Path):
        """Return (KAP classifier, RGB encoder, context window) from a KAP checkpoint."""
        print(f"[*] Loading robust scratch KAP classifier: {ckpt_path}...")
        clf_ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
        model_type = clf_ckpt.get("model_type", "baseline")
        if model_type in ("cross_v2", "cross_v3"):
            model_class = KAPCrossTransV2Model
        elif model_type == "cross":
            model_class = KAPCrossTransModel
        else:
            model_class = KAPScratchModel
        # Checkpoints built on the team's RGB encoder carry its weights and input size.
        if "rgb_encoder" in clf_ckpt:
            rgb_phases = clf_ckpt.get("rgb_phases", 0)
            clf = KAPCrossTransV2Model(rgb_dim=clf_ckpt["rgb_dim"], gate_free=clf_ckpt["gate_free"], rgb_phases=rgb_phases).to(DEVICE)
            rgb_model = OwnRGBEncoder(phases=bool(rgb_phases)).to(DEVICE)
            rgb_model.load_state_dict(clf_ckpt["rgb_encoder"])
            rgb_model.eval()
            print("[*] RGB branch: team-built encoder from the KAP checkpoint.")
        else:
            clf = model_class().to(DEVICE)
            print("[*] Loading R(2+1)D Visual Backbone...")
            rgb_model = MultiTaskR2Plus1D().to(DEVICE)
            backbone_ckpt = REPO_ROOT / "src" / "ai_classifier" / "models" / "checkpoints" / "visual_backbone.pth"
            if not backbone_ckpt.exists():
                backbone_ckpt = REPO_ROOT / "work_dirs" / "r2plus1d18_mixed_shuttleset_finebadminton" / "best.pth"
            rgb_ckpt = torch.load(str(backbone_ckpt), map_location="cpu", weights_only=False)
            rgb_model.load_state_dict(rgb_ckpt["model"], strict=False)
            rgb_model.eval()
        clf.load_state_dict(clf_ckpt["model"])
        clf.eval()
        # Checkpoints trained on BST's longer between-hit context store their
        # (before, after) window; older checkpoints keep the 15/30 extract_features window.
        window = clf_ckpt.get("context_window")
        context_window = tuple(int(v) for v in window) if window else None
        return clf, rgb_model, context_window

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
        cache_shuttle: bool = False,
    ) -> dict[str, int]:
        """Precompute pose once for the union of all hit windows.

        This is intended for a brand-new video: the cache is built in memory in
        the same run, then reused by every event. No persistent cache is needed.
        """
        radius = max(0, int(contact_refine_radius))
        if self.context_window:
            before, after = max(before, self.context_window[0]), max(after, self.context_window[1])
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
                frames, indices = self._read_range(cap, chunk_start, chunk_end)
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

                # Small groups prevent Ultralytics Results objects from
                # accumulating for an entire rally or match.
                starts = range(0, len(frames), POSE_GROUP_FRAMES)
                groups = [frames[s:s + POSE_GROUP_FRAMES] for s in starts]
                for start, results in zip(starts, _pose_groups(self.pose_model, groups)):
                    for frame_index, result in zip(indices[start:start + POSE_GROUP_FRAMES], results):
                        self._pose_cache[frame_index] = self._decode_pose_result(result)
                chunk_start = chunk_end + 1
        return {
            "regions": len(ranges),
            "unique_frames": requested,
            "pose_frames": len(self._pose_cache),
            "shuttle_frames": len(self._shuttle_cache),
        }

    def set_event_frames(self, frames: list[int]) -> None:
        """Register every candidate hit of the video so a long context window can stop at its neighbours."""
        self._event_frames = np.sort(np.asarray(frames, dtype=np.int64))

    def _context_bounds(self, frame: int, hit: int) -> tuple[int, int]:
        """First and last frame of the BST between-hit sequence of the hit at ``hit``."""
        before, after = self.context_window
        if self._event_frames is None:
            return hit - before, hit + after
        events = self._event_frames
        previous = events[(events < frame - SAME_EVENT_FRAMES) & (events >= hit - RALLY_GAP_FRAMES)]
        following = events[(events > frame + SAME_EVENT_FRAMES) & (events <= hit + RALLY_GAP_FRAMES)]
        first = max(int(previous[-1]), hit - before) if len(previous) else hit - RALLY_EDGE_FRAMES
        last = (min(int(following[0]) + NEXT_HIT_TAIL_FRAMES, hit + after)
                if len(following) else hit + RALLY_EDGE_FRAMES)
        return first, last

    def _context_window_inputs(
        self, frame: int, frame_indices: list[int], contact: int,
        raw_pose: np.ndarray, raw_court: np.ndarray, raw_shuttle: np.ndarray,
    ) -> tuple[np.ndarray, ...]:
        """KAP streams over hit-before..hit+after, zero outside the between-hit sequence as in training."""
        before, after = self.context_window
        hit = frame_indices[contact]
        first, last = self._context_bounds(frame, hit)
        offsets = np.arange(-before, after + 1)
        source = contact + offsets
        valid = (source >= 0) & (source < len(frame_indices)) & (hit + offsets >= first) & (hit + offsets <= last)
        windows = []
        for array in (raw_pose, raw_court, raw_shuttle):
            padded = np.zeros((len(offsets), array.shape[1]), dtype=np.float32)
            padded[valid] = np.nan_to_num(array[source[valid]])
            windows.append(padded)
        pose, court, shuttle = windows
        s_q = np.asarray(modality_quality_vector(pose, shuttle), dtype=np.float32)
        s_q[[1, 3]] /= 100.0
        s_c = np.linspace(-before, after, 32, dtype=np.float32)[:, None] / float(before)
        return resample(pose, 32), resample(court, 32), resample(shuttle, 32), s_c, s_q

    def extract_features(
        self, cap: cv2.VideoCapture, hf_clip: int, total_frames: int,
        hitter_side: str = "lower", before: int = 15, after: int = 30,
        contact_refine_radius: int = 6,
    ) -> dict | None:
        refine_radius = max(0, int(contact_refine_radius))
        decode_before, decode_after = (
            (max(before, self.context_window[0]), max(after, self.context_window[1]))
            if self.context_window else (before, after)
        )
        start_f = max(0, hf_clip - decode_before - refine_radius)
        end_f = min(total_frames - 1, hf_clip + decode_after + refine_radius)

        frames_bgr, frame_indices = self._read_range(cap, start_f, end_f)

        if len(frames_bgr) < 20:
            return None
        decoded = dict(zip(frame_indices, frames_bgr))

        if all(frame in self._shuttle_cache for frame in frame_indices):
            coord_map = {frame: self._shuttle_cache[frame] for frame in frame_indices}
        else:
            track_df = self.tracker.predict_frames(
                frames_bgr, frame_indices, self.orig_w, self.orig_h, batch_size=32,
                resize_cache=self._tracknet_resize_cache,
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
        context_inputs = (
            self._context_window_inputs(hf_clip, frame_indices, refined_contact, raw_pose, raw_court, raw_shuttle)
            if self.context_window else None
        )

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
        if context_inputs is not None:
            # The classifier sees the long window; analytics keep the before/after window above.
            s_j, s_p, s_s, s_c, s_q = context_inputs

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
        emb = self.rgb_embedding(cap, refined_hit_frame, crop_box, total_frames, after, decoded)

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

    def rgb_embedding(
        self, cap: cv2.VideoCapture, hit_frame: int, crop_box, total_frames: int,
        after: int = 30, decoded: dict[int, np.ndarray] | None = None,
    ) -> torch.Tensor:
        """R(2+1)D embedding of 16 hitter crops spanning hit-rgb_before..hit+after.

        ``decoded`` holds frames already read for this event; only the missing
        (earlier) frames are decoded again.
        """
        decoded = dict(decoded or {})
        first = max(0, hit_frame - self.rgb_before)
        last = min(total_frames - 1, hit_frame + after)
        wanted = [int(f) for f in np.rint(np.linspace(first, last, 16)).astype(np.int64)]
        missing = [f for f in wanted if f not in decoded]
        if missing:
            frames, indices = self._read_range(cap, missing[0], missing[-1])
            decoded.update(zip(indices, frames))
        available = sorted(decoded)
        x1, y1, x2, y2 = crop_box
        rgb_16 = [
            cv2.cvtColor(
                decoded[f if f in decoded else min(available, key=lambda a: abs(a - f))][y1:y2, x1:x2],
                cv2.COLOR_BGR2RGB,
            )
            for f in wanted
        ]
        t_rgb = torch.from_numpy(np.stack(rgb_16)).permute(0, 3, 1, 2).float().div_(255.0)
        t_rgb = VF.resize(t_rgb, [112, 112], antialias=False).to(DEVICE)
        t_rgb = (t_rgb - _MEAN) / _STD
        hitter_clip = t_rgb.permute(1, 0, 2, 3).unsqueeze(0)
        with torch.inference_mode():
            return self.rgb_model.extract_features(hitter_clip)

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
            logits, side_logits, _gates = model(*inputs)
        if effective_tau > 0.0:
            priors = torch.tensor([0.08, 0.15, 0.12, 0.10, 0.18, 0.07, 0.20, 0.10], device=logits.device)
            logits = logits - effective_tau * torch.log(priors + 1e-8)
        probs = F.softmax(logits, dim=-1)[0]
        pred_idx = int(logits.argmax(dim=-1).item())
        top2_idx = logits.topk(2, dim=-1).indices[0].tolist()

        side_probs = F.softmax(side_logits, dim=-1)[0]
        side_idx = int(side_logits.argmax(dim=-1).item())
        side_label = SIDE_CLASSES[side_idx]
        side_conf = float(side_probs[side_idx].item())

        return {
            "stroke": STROKE_CLASSES[pred_idx], "confidence": float(probs[pred_idx]),
            "top2": [STROKE_CLASSES[k] for k in top2_idx], "logits": logits[0],
            "stroke_side": side_label, "stroke_side_confidence": side_conf,
            "side_logits": side_logits[0],
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
            "stroke_side": base_detail.get("stroke_side", "forehand"),
            "stroke_side_confidence": base_detail.get("stroke_side_confidence", 0.85),
            "tta_used": len(probs_list) > 1,
            "tta_windows": len(probs_list),
        }

