"""Classify stroke type and stroke side for one pre-segmented RGB clip."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from ultralytics import YOLO

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.inference.classify_fine_badminton_rgb import read_clip
from scripts.training.train_shuttleset_rgb_multitask import (
    MultiTaskR2Plus1D,
    Record,
    decode_clip,
    detect_hitter_crop,
)
from ai_classifier.localization.player_tracking import (
    PoseDetection,
    build_tracks,
    select_active_track,
    track_motion,
    union_box,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video", type=Path)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--crop-hitter", action="store_true")
    parser.add_argument("--player-side", choices=("top", "bottom", "auto"), default="auto")
    parser.add_argument(
        "--pose-model", type=Path, default=Path("models/checkpoints/pose/yolov8n-pose.pt")
    )
    parser.add_argument(
        "--sliding-window-seconds", type=float,
        help="Run overlapping temporal windows and average their probabilities.",
    )
    parser.add_argument("--sliding-stride-seconds", type=float, default=0.5)
    parser.add_argument(
        "--rally-only", action="store_true",
        help="Suppress 'serve' classification during in-rally stroke analysis.",
    )
    return parser.parse_args()


CLASS_PRIOR_WEIGHTS = {
    "drive": 4.160,
    "clear": 2.858,
    "lift": 1.487,
    "smash": 0.748,
    "drop": 0.654,
    "net_attack": 0.600,
    "net_shot": 0.394,
    "serve": 1.000,
}


def auto_detect_player_side(path: Path, pose_model: YOLO) -> str:
    capture = cv2.VideoCapture(str(path))
    try:
        total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        h = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        w = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    finally:
        capture.release()
    if total_frames <= 0:
        return "bottom"

    top_overhead_frames = 0
    bot_overhead_frames = 0
    top_motions = []
    bot_motions = []
    prev_top = None
    prev_bot = None

    step = max(1, total_frames // 16)
    capture = cv2.VideoCapture(str(path))
    try:
        for fi in range(0, total_frames, step):
            capture.set(cv2.CAP_PROP_POS_FRAMES, fi)
            ok, frame = capture.read()
            if not ok:
                break
            t_crop = cv2.resize(frame[int(h * 0.18):int(h * 0.52), int(w * 0.20):int(w * 0.80)], (96, 64))
            b_crop = cv2.resize(frame[int(h * 0.52):int(h * 0.95), int(w * 0.15):int(w * 0.85)], (96, 64))
            t_gray = cv2.cvtColor(t_crop, cv2.COLOR_BGR2GRAY).astype(np.float32)
            b_gray = cv2.cvtColor(b_crop, cv2.COLOR_BGR2GRAY).astype(np.float32)
            if prev_top is not None:
                top_motions.append(float(np.mean(np.abs(t_gray - prev_top))))
                bot_motions.append(float(np.mean(np.abs(b_gray - prev_bot))))
            prev_top = t_gray
            prev_bot = b_gray

            res = pose_model.predict(frame, imgsz=640, conf=0.20, verbose=False)[0]
            if res.keypoints is not None:
                for kp in res.keypoints.data.cpu().numpy():
                    vis = kp[:, 2] > 0.20
                    if not vis.any():
                        continue
                    cy = kp[vis, 1].mean()
                    is_top = cy < h * 0.55
                    for s_i, w_i in [(6, 10), (5, 9)]:
                        if kp[s_i, 2] > 0.20 and kp[w_i, 2] > 0.20 and kp[w_i, 1] < kp[s_i, 1]:
                            if is_top:
                                top_overhead_frames += 1
                            else:
                                bot_overhead_frames += 1
    finally:
        capture.release()

    if top_overhead_frames >= 2 and bot_overhead_frames == 0:
        return "top"
    if bot_overhead_frames >= 2 and top_overhead_frames == 0:
        return "bottom"
    if top_overhead_frames > bot_overhead_frames * 2:
        return "top"
    if bot_overhead_frames > top_overhead_frames * 2:
        return "bottom"

    top_peak = max(top_motions) if top_motions else 0.0
    bot_peak = max(bot_motions) if bot_motions else 0.0
    return "top" if top_peak > bot_peak * 1.35 else "bottom"


def read_hitter_clip(
    path: Path, frame_count: int, player_side: str, pose_model: YOLO,
    crop_padding: float, start_frame: int, end_frame: int, crop_size: int = 112,
):
    capture = cv2.VideoCapture(str(path))
    try:
        total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    finally:
        capture.release()
    if total_frames <= 0:
        raise ValueError(f"Cannot inspect video: {path}")
    record = Record(
        sample_id=path.stem, video_path=path, start_frame=start_frame,
        end_frame=end_frame, coarse_label="lift", stroke_side="forehand",
        player_side=player_side, hit_frame=(start_frame + end_frame) // 2,
    )
    crop_box = detect_hitter_crop(record, pose_model, crop_padding)
    tensor = decode_clip(record, frame_count, crop_box, crop_size).to(torch.float32).div_(255.0)
    mean = tensor.new_tensor([0.43216, 0.394666, 0.37645]).view(1, 3, 1, 1)
    std = tensor.new_tensor([0.22803, 0.22145, 0.216989]).view(1, 3, 1, 1)
    clip = ((tensor - mean) / std).permute(1, 0, 2, 3).unsqueeze(0)
    metadata = {
        "frames": total_frames, "fps": fps, "width": width, "height": height,
        "crop_box": list(crop_box), "player_side": player_side,
        "crop_padding": crop_padding,
        "crop_size": crop_size,
        "start_frame": start_frame, "end_frame": end_frame,
    }
    return clip, metadata


def read_tracked_hitter_clip(
    path: Path, frame_count: int, player_side: str, pose_model: YOLO,
    crop_padding: float, start_frame: int, end_frame: int, hit_frame: int,
    tracking_radius: int = 6, crop_size: int = 112,
):
    capture = cv2.VideoCapture(str(path))
    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    detections_by_frame = []
    try:
        for frame_index in range(
            max(0, hit_frame - tracking_radius),
            min(total_frames - 1, hit_frame + tracking_radius) + 1,
            2,
        ):
            capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            ok, frame = capture.read()
            if not ok:
                continue
            result = pose_model.predict(frame, imgsz=960, conf=0.20, verbose=False)[0]
            frame_detections = []
            if result.boxes is not None and result.keypoints is not None:
                for box, class_id, pose in zip(
                    result.boxes.xyxy.cpu().numpy(),
                    result.boxes.cls.cpu().numpy(),
                    result.keypoints.data.cpu().numpy(),
                ):
                    if int(class_id) != 0:
                        continue
                    visible = pose[:, 2] >= 0.25
                    if visible.sum() < 5:
                        continue
                    center = pose[visible, :2].mean(axis=0)
                    wrists = np.full((2, 2), np.nan, dtype=np.float32)
                    for output_index, keypoint_index in enumerate((9, 10)):
                        if pose[keypoint_index, 2] >= 0.20:
                            wrists[output_index] = pose[keypoint_index, :2]
                    frame_detections.append(PoseDetection(
                        frame_index, box.astype(np.float32), center.astype(np.float32), wrists
                    ))
            detections_by_frame.append(frame_detections)
    finally:
        capture.release()
    tracks = build_tracks(
        detections_by_frame, max_center_distance=max(width, height) * 0.10
    )
    track = select_active_track(
        tracks, player_side=player_side, width=width, height=height
    )
    if track is None:
        return read_hitter_clip(
            path, frame_count, player_side, pose_model, crop_padding, start_frame, end_frame,
            crop_size,
        )
    crop_box = union_box(track, width=width, height=height, padding=crop_padding)
    crop_width = crop_box[2] - crop_box[0]
    crop_height = crop_box[3] - crop_box[1]
    if crop_width > 0.48 * width or crop_height > 0.72 * height:
        return read_hitter_clip(
            path, frame_count, player_side, pose_model, crop_padding, start_frame, end_frame,
            crop_size,
        )
    record = Record(
        sample_id=path.stem, video_path=path, start_frame=start_frame,
        end_frame=end_frame, coarse_label="lift", stroke_side="forehand",
        player_side=player_side, hit_frame=hit_frame,
    )
    tensor = decode_clip(record, frame_count, crop_box, crop_size).to(torch.float32).div_(255.0)
    mean = tensor.new_tensor([0.43216, 0.394666, 0.37645]).view(1, 3, 1, 1)
    std = tensor.new_tensor([0.22803, 0.22145, 0.216989]).view(1, 3, 1, 1)
    clip = ((tensor - mean) / std).permute(1, 0, 2, 3).unsqueeze(0)
    return clip, {
        "frames": total_frames, "fps": fps, "width": width, "height": height,
        "crop_box": list(crop_box), "player_side": player_side,
        "crop_padding": crop_padding, "start_frame": start_frame, "end_frame": end_frame,
        "crop_size": crop_size,
        "tracking": True, "track_frames": len(track.detections),
        "track_motion": track_motion(track),
    }


def video_metadata(path: Path) -> tuple[int, float]:
    capture = cv2.VideoCapture(str(path))
    try:
        total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = float(capture.get(cv2.CAP_PROP_FPS))
    finally:
        capture.release()
    if total_frames <= 0 or fps <= 0:
        raise ValueError(f"Cannot inspect video: {path}")
    return total_frames, fps


def sliding_ranges(total_frames: int, fps: float, seconds: float, stride_seconds: float):
    if seconds <= 0 or stride_seconds <= 0:
        raise ValueError("Sliding window and stride must be positive")
    window = min(total_frames, max(16, int(round(seconds * fps))))
    stride = max(1, int(round(stride_seconds * fps)))
    starts = list(range(0, max(total_frames - window + 1, 1), stride))
    final_start = max(0, total_frames - window)
    if not starts or starts[-1] != final_start:
        starts.append(final_start)
    return [(start, start + window - 1) for start in starts]


def action_motion_score(path: Path, metadata: dict[str, object]) -> float:
    x1, y1, x2, y2 = (int(value) for value in metadata["crop_box"])
    start_frame = int(metadata["start_frame"])
    end_frame = int(metadata["end_frame"])
    capture = cv2.VideoCapture(str(path))
    frames = []
    try:
        capture.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        for _ in range(start_frame, end_frame + 1):
            ok, frame = capture.read()
            if not ok:
                break
            crop = frame[y1:y2, x1:x2]
            if crop.size == 0:
                continue
            h = crop.shape[0]
            # Focus on upper 65% of the hitter crop (arms, shoulders, racket swing)
            upper_crop = crop[:max(1, int(h * 0.65)), :]
            gray = cv2.cvtColor(upper_crop, cv2.COLOR_BGR2GRAY)
            frames.append(cv2.resize(gray, (96, 64)).astype(np.float32))
    finally:
        capture.release()
    if len(frames) < 2:
        return 0.0
    differences = [
        float(np.mean(np.abs(current - previous)))
        for previous, current in zip(frames, frames[1:])
    ]
    # Peak motion is more useful than the mean because racket contact is brief.
    # Peak upper-body motion captures racket swing and hit contact best
    return max(differences)


def ranked(probabilities: torch.Tensor, classes: list[str]) -> list[dict[str, object]]:
    values, indices = torch.sort(probabilities, descending=True)
    return [
        {"label": classes[int(index)], "probability": float(value)}
        for value, index in zip(values, indices)
    ]


def constrain_side_probabilities(
    probabilities: torch.Tensor, classes: list[str], stroke_label: str
) -> tuple[torch.Tensor, list[str]]:
    allowed = set(classes)
    if stroke_label not in {"clear", "smash", "drop"}:
        allowed.discard("aroundhead")
    constrained = probabilities.clone()
    for index, name in enumerate(classes):
        if name not in allowed:
            constrained[index] = 0.0
    total = constrained.sum()
    if total > 0:
        constrained = constrained / total
    return constrained, sorted(allowed)


def main() -> None:
    args = parse_args()
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    stroke_classes = [str(value) for value in checkpoint["stroke_classes"]]
    side_classes = [str(value) for value in checkpoint["side_classes"]]
    frame_count = int(checkpoint.get("frames", 16))
    window_metadata = []
    selected_window_index = 0
    full_clip_index = None
    if args.crop_hitter:
        total_frames, fps = video_metadata(args.video)
        ranges = (
            sliding_ranges(
                total_frames, fps, args.sliding_window_seconds,
                args.sliding_stride_seconds,
            )
            if args.sliding_window_seconds else [(0, total_frames - 1)]
        )
        pose_model = YOLO(str(args.pose_model))
        active_player_side = (
            auto_detect_player_side(args.video, pose_model)
            if args.player_side == "auto" else args.player_side
        )
        clips = []
        for start_frame, end_frame in ranges:
            window_clip, item_metadata = read_hitter_clip(
                args.video, frame_count, active_player_side, pose_model,
                float(checkpoint.get("crop_padding", 0.30)), start_frame, end_frame,
                int(checkpoint.get("crop_size", 112)),
            )
            clips.append(window_clip)
            item_metadata["action_motion_score"] = action_motion_score(
                args.video, item_metadata
            )
            window_metadata.append(item_metadata)
        if args.sliding_window_seconds:
            selected_window_index = max(
                range(len(window_metadata)),
                key=lambda index: float(window_metadata[index]["action_motion_score"]),
            )
            full_clip, _full_metadata = read_hitter_clip(
                args.video, frame_count, active_player_side, pose_model,
                float(checkpoint.get("crop_padding", 0.30)), 0, total_frames - 1,
                int(checkpoint.get("crop_size", 112)),
            )
            full_clip_index = len(clips)
            clips.append(full_clip)
        clip = torch.cat(clips, dim=0)
        metadata = {
            "frames": total_frames, "fps": fps, "player_side": active_player_side,
            "requested_player_side": args.player_side,
            "window_seconds": args.sliding_window_seconds,
            "stride_seconds": args.sliding_stride_seconds if args.sliding_window_seconds else None,
            "windows": window_metadata,
            "selected_window_index": selected_window_index,
        }
    else:
        active_player_side = args.player_side
        if args.sliding_window_seconds:
            raise ValueError("Sliding inference currently requires --crop-hitter")
        clip, metadata = read_clip(args.video, frame_count)
    model = MultiTaskR2Plus1D()
    model.load_state_dict(checkpoint["model"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()
    with torch.inference_mode(), torch.autocast(
        device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"
    ):
        stroke_logits, side_logits = model(clip.to(device))
        all_stroke_probabilities = torch.softmax(stroke_logits, dim=1).cpu()
        all_side_probabilities = torch.softmax(side_logits, dim=1).cpu()
        window_count = len(window_metadata) if window_metadata else 1
        window_stroke_probabilities = all_stroke_probabilities[:window_count]
        window_side_probabilities = all_side_probabilities[:window_count]
        mean_stroke_probabilities = window_stroke_probabilities.mean(dim=0)
        stroke_probabilities = window_stroke_probabilities[selected_window_index]

        # Top-2 motion windows aggregation + full clip embedding for stroke
        sorted_window_indices = sorted(
            range(window_count),
            key=lambda index: float(window_metadata[index]["action_motion_score"]) if window_metadata else 0.0,
            reverse=True,
        )
        # Strike-Phase Window Selector:
        # Avoid early preparation / footwork false triggers.
        # Genuine stroke contact exhibits both active upper-body motion and sharp stroke confidence.
        scored_windows = []
        for index in range(window_count):
            meta = window_metadata[index] if window_metadata else {}
            motion = float(meta.get("action_motion_score", 1.0))
            stroke_probs = window_stroke_probabilities[index]
            max_prob = float(stroke_probs.max())
            top_label = stroke_classes[int(stroke_probs.argmax())]

            pos = index / max(1, window_count - 1)
            # Prioritize action phase (middle-to-late part of the clip)
            temporal_weight = 0.40 if pos < 0.25 else 1.0
            # Running/waiting posture in early clip falsely triggers 'serve'
            if top_label == "serve" and pos < 0.50:
                temporal_weight *= 0.15
            if top_label == "serve":
                temporal_weight *= 0.05 if args.rally_only else (0.15 if pos < 0.50 else 0.50)

            composite_score = motion * (max_prob ** 1.5) * temporal_weight
            scored_windows.append((composite_score, index))

        scored_windows.sort(key=lambda item: item[0], reverse=True)
        top_k = min(2, window_count)
        top_indices = sorted_window_indices[:top_k]
        top_indices = [item[1] for item in scored_windows[:top_k]]
        selected_window_index = top_indices[0]
        top_stroke_probs = window_stroke_probabilities[top_indices].mean(dim=0)
        
        # Combine Top-motion windows (75%) with full-clip embedding (25%) if available
        # Combine Top strike-phase windows (80%) with full-clip embedding (20%) if available
        if full_clip_index is not None:
            full_stroke_probs = all_stroke_probabilities[full_clip_index]
            stroke_probabilities = 0.75 * top_stroke_probs + 0.25 * full_stroke_probs
            stroke_probabilities = 0.80 * top_stroke_probs + 0.20 * full_stroke_probs
        else:
            stroke_probabilities = top_stroke_probs

        side_probabilities = (
            all_side_probabilities[full_clip_index]
            if full_clip_index is not None else window_side_probabilities[0]
        )
        if args.rally_only:
            serve_idx = stroke_classes.index("serve") if "serve" in stroke_classes else -1
            if serve_idx >= 0:
                stroke_probabilities = stroke_probabilities.clone()
                stroke_probabilities[serve_idx] *= 0.05
                stroke_probabilities = stroke_probabilities / stroke_probabilities.sum()

        # Method 1 Prior: Neutralize training class weight bias
        calibrated_stroke = stroke_probabilities.clone()
        for c_idx, c_name in enumerate(stroke_classes):
            w = CLASS_PRIOR_WEIGHTS.get(c_name, 1.0)
            calibrated_stroke[c_idx] = calibrated_stroke[c_idx] / (w ** 0.30)
        calibrated_stroke = calibrated_stroke / calibrated_stroke.sum()
        stroke_probabilities = calibrated_stroke

        # Method 1 Prior: Court Depth Biomechanics Prior
        if window_metadata and selected_window_index < len(window_metadata):
            sel_meta = window_metadata[selected_window_index]
            box = sel_meta.get("crop_box", [0, 0, 0, 0])
            h_vid = sel_meta.get("height", 1080)
            yc = (box[1] + box[3]) / 2.0 / max(1, h_vid)
            is_backcourt = (active_player_side == "bottom" and yc >= 0.68) or (active_player_side == "top" and yc <= 0.32)
            is_net = (active_player_side == "bottom" and yc <= 0.52) or (active_player_side == "top" and yc >= 0.46)
            if is_backcourt:
                stroke_probabilities = stroke_probabilities.clone()
                for c_idx, c_name in enumerate(stroke_classes):
                    if c_name in {"clear", "smash", "drop"}:
                        stroke_probabilities[c_idx] *= 2.2
                    elif c_name in {"net_shot", "lift", "net_attack"}:
                        stroke_probabilities[c_idx] *= 0.25
                stroke_probabilities = stroke_probabilities / stroke_probabilities.sum()
            elif is_net:
                stroke_probabilities = stroke_probabilities.clone()
                for c_idx, c_name in enumerate(stroke_classes):
                    if c_name in {"net_shot", "lift", "net_attack"}:
                        stroke_probabilities[c_idx] *= 2.0
                    elif c_name in {"clear", "smash"}:
                        stroke_probabilities[c_idx] *= 0.20
                stroke_probabilities = stroke_probabilities / stroke_probabilities.sum()

        # For side (forehand/backhand): blend best strike window with full clip
        if full_clip_index is not None:
            best_win_side = window_side_probabilities[selected_window_index]
            full_side = all_side_probabilities[full_clip_index]
            side_probabilities = 0.35 * best_win_side + 0.65 * full_side
        else:
            side_probabilities = window_side_probabilities[selected_window_index]

        # Perspective bias compensation for upper/top player vs bottom player
        if active_player_side == "top":
            forehand_idx = side_classes.index("forehand") if "forehand" in side_classes else -1
            if forehand_idx >= 0:
                side_probabilities = side_probabilities.clone()
                # Compensate for camera angle bias where right-handed forehands resemble backhands
                side_probabilities[forehand_idx] = side_probabilities[forehand_idx] * 1.65
                side_probabilities = side_probabilities / side_probabilities.sum()
        elif active_player_side == "bottom":
            forehand_idx = side_classes.index("forehand") if "forehand" in side_classes else -1
            if forehand_idx >= 0:
                side_probabilities = side_probabilities.clone()
                side_probabilities[forehand_idx] = side_probabilities[forehand_idx] * 1.25
                side_probabilities = side_probabilities / side_probabilities.sum()
    stroke_ranking = ranked(stroke_probabilities, stroke_classes)
    # In badminton technique, around-the-head is a forehand overhead shot.
    # Combine aroundhead into forehand to prevent probability splitting.
    merged_side_probs = side_probabilities.clone()
    fh_idx = side_classes.index("forehand") if "forehand" in side_classes else -1
    ah_idx = side_classes.index("aroundhead") if "aroundhead" in side_classes else -1
    if fh_idx >= 0 and ah_idx >= 0:
        merged_side_probs[fh_idx] += merged_side_probs[ah_idx]
        merged_side_probs[ah_idx] = 0.0
        merged_side_probs = merged_side_probs / merged_side_probs.sum()

    raw_side_ranking = ranked(merged_side_probs, side_classes)
    constrained_side_probabilities, allowed_sides = constrain_side_probabilities(
        merged_side_probs, side_classes, str(stroke_ranking[0]["label"])
    )
    side_ranking = ranked(constrained_side_probabilities, side_classes)
    primary_side = side_ranking[0]["label"]
    result = {
        "video": str(args.video.resolve()),
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_epoch": int(checkpoint["epoch"]),
        "device": str(device),
        "prediction": {
            "stroke": stroke_ranking[0],
            "stroke_side": side_ranking[0],
            "combined_label": f"{primary_side} {stroke_ranking[0]['label']}",
        },
        "stroke_ranking": stroke_ranking,
        "stroke_side_ranking": side_ranking,
        "raw_stroke_side_ranking": raw_side_ranking,
        "allowed_stroke_sides": allowed_sides,
        "mean_window_stroke_ranking": ranked(mean_stroke_probabilities, stroke_classes),
        "selection_method": (
            "peak_player_motion_for_stroke_full_clip_for_side"
            if args.sliding_window_seconds else "single_full_clip"
        ),
        "window_predictions": [
            {
                "start_frame": item["start_frame"],
                "end_frame": item["end_frame"],
                "stroke_ranking": ranked(stroke_probs, stroke_classes),
                "stroke_side_ranking": ranked(side_probs, side_classes),
            }
            for item, stroke_probs, side_probs in zip(
                window_metadata, window_stroke_probabilities, window_side_probabilities
            )
        ] if window_metadata else [],
        "video_metadata": metadata,
        "scope_warning": (
            "Pilot model trained on broadcast singles footage and pre-segmented "
            "single-stroke clips; confidence is not calibrated for practice videos."
        ),
    }
    rendered = json.dumps(result, ensure_ascii=False, indent=2)
    print(rendered)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
