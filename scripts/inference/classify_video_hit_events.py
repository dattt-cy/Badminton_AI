"""Detect and classify every badminton hit in a video as a timeline."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import torch
from ultralytics import YOLO

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ai_classifier.localization import temporal_nms
from ai_classifier.localization.live_gate import frame_is_live
from scripts.inference.classify_shuttleset_rgb_multitask import (
    constrain_side_probabilities,
    read_hitter_clip,
    read_tracked_hitter_clip,
)
from scripts.inference.scan_video_hit_rgb import load_model as load_hit_model, scan_video
from scripts.training.train_shuttleset_rgb_multitask import MultiTaskR2Plus1D


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--hit-checkpoint", type=Path, required=True)
    parser.add_argument("--stroke-checkpoint", type=Path, required=True)
    parser.add_argument(
        "--pose-model", type=Path,
        default=Path("models/checkpoints/pose/yolov8n-pose.pt"),
    )
    parser.add_argument("--hit-stride", type=int, default=2)
    parser.add_argument("--hit-threshold", type=float, default=0.5)
    parser.add_argument("--nms-radius-seconds", type=float, default=0.5)
    parser.add_argument(
        "--aggregate-radius-seconds", type=float, default=0.0,
        help="Optional radius for same-player probability averaging; zero preserves individual events.",
    )
    parser.add_argument(
        "--event-radius-seconds", type=float, default=1.0,
        help="Temporal context before/after each detected hit.",
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--gate-json", type=Path, help="LIVE segments from detect_live_play_gate.py")
    parser.add_argument("--motion-track-crop", action="store_true")
    return parser.parse_args()


def metadata(path: Path) -> tuple[int, float]:
    capture = cv2.VideoCapture(str(path))
    try:
        frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = float(capture.get(cv2.CAP_PROP_FPS))
    finally:
        capture.release()
    if frames <= 0 or fps <= 0:
        raise ValueError(f"Cannot inspect video: {path}")
    return frames, fps


def ranking(probabilities: torch.Tensor, classes: list[str]) -> list[dict[str, object]]:
    values, indices = torch.sort(probabilities, descending=True)
    return [
        {"label": classes[int(index)], "probability": round(float(value), 6)}
        for value, index in zip(values, indices)
    ]


def group_same_player_events(events, radius_frames: int):
    if radius_frames <= 0:
        return [[event] for event in sorted(events, key=lambda item: item.frame)]
    groups = []
    for side in sorted({event.side for event in events}):
        side_groups = []
        for event in sorted(
            (item for item in events if item.side == side), key=lambda item: item.frame
        ):
            if side_groups and event.frame - side_groups[-1][-1].frame <= radius_frames:
                side_groups[-1].append(event)
            else:
                side_groups.append([event])
        groups.extend(side_groups)
    return sorted(groups, key=lambda group: group[0].frame)


def main() -> None:
    args = parse_args()
    total_frames, fps = metadata(args.video)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    hit_model, hit_checkpoint = load_hit_model(args.hit_checkpoint)
    hit_model.to(device).eval()
    raw_events = scan_video(
        args.video,
        hit_model,
        int(hit_checkpoint.get("frames", 16)),
        stride=args.hit_stride,
        batch_size=args.batch_size,
        device=device,
        use_amp=device.type == "cuda",
    )
    events = temporal_nms(
        raw_events,
        threshold=args.hit_threshold,
        radius=max(1, round(fps * args.nms_radius_seconds)),
        side_aware=True,
    )
    gate_segments = None
    if args.gate_json is not None:
        gate_payload = json.loads(args.gate_json.read_text(encoding="utf-8"))
        gate_segments = gate_payload.get("live_segments", [])
        events = [event for event in events if frame_is_live(event.frame, gate_segments)]

    stroke_checkpoint = torch.load(
        args.stroke_checkpoint, map_location="cpu", weights_only=False
    )
    stroke_model = MultiTaskR2Plus1D()
    stroke_model.load_state_dict(stroke_checkpoint["model"])
    stroke_model.to(device).eval()
    pose_model = YOLO(str(args.pose_model))
    stroke_classes = list(stroke_checkpoint["stroke_classes"])
    side_classes = list(stroke_checkpoint["side_classes"])
    radius = max(1, round(fps * args.event_radius_seconds))

    timeline = []
    event_groups = group_same_player_events(
        events, round(fps * args.aggregate_radius_seconds)
    )
    for group in event_groups:
        player_side = "top" if group[0].side == "upper" else "bottom"
        centers = []
        stroke_vectors = []
        side_vectors = []
        for event in group:
            start_frame = max(0, event.frame - radius)
            end_frame = min(total_frames - 1, event.frame + radius)
            crop_size = int(stroke_checkpoint.get("crop_size", 112))
            if args.motion_track_crop:
                clip, crop = read_tracked_hitter_clip(
                    args.video, int(stroke_checkpoint.get("frames", 16)), player_side,
                    pose_model, float(stroke_checkpoint.get("crop_padding", 0.55)),
                    start_frame, end_frame, event.frame, crop_size=crop_size,
                )
            else:
                clip, crop = read_hitter_clip(
                    args.video, int(stroke_checkpoint.get("frames", 16)), player_side,
                    pose_model, float(stroke_checkpoint.get("crop_padding", 0.55)),
                    start_frame, end_frame, crop_size,
                )
            with torch.inference_mode(), torch.autocast(
                device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"
            ):
                stroke_logits, side_logits = stroke_model(clip.to(device))
            stroke_vectors.append(torch.softmax(stroke_logits[0], dim=0).cpu())
            side_vectors.append(torch.softmax(side_logits[0], dim=0).cpu())
            centers.append({
                "frame": event.frame,
                "hit_score": round(event.score, 6),
                "clip_range": [start_frame, end_frame],
                "crop_box": crop["crop_box"],
                "stroke_ranking": ranking(stroke_vectors[-1], stroke_classes),
            })
        stroke_probs = torch.stack(stroke_vectors).mean(dim=0)
        side_probs = torch.stack(side_vectors).mean(dim=0)
        if "forehand" in side_classes and "aroundhead" in side_classes:
            forehand = side_classes.index("forehand")
            aroundhead = side_classes.index("aroundhead")
            side_probs[forehand] += side_probs[aroundhead]
            side_probs[aroundhead] = 0.0
            side_probs /= side_probs.sum()
        stroke = stroke_classes[int(stroke_probs.argmax())]
        side_probs, allowed_sides = constrain_side_probabilities(
            side_probs, side_classes, stroke
        )
        side = side_classes[int(side_probs.argmax())]
        representative = max(group, key=lambda item: item.score)
        timeline.append({
            "frame": representative.frame,
            "time_seconds": round(representative.frame / fps, 3),
            "hit_score": round(representative.score, 6),
            "court_player": player_side,
            "stroke": stroke,
            "stroke_side": side,
            "combined_label": f"{side} {stroke}",
            "stroke_ranking": ranking(stroke_probs, stroke_classes),
            "stroke_side_ranking": ranking(side_probs, side_classes),
            "allowed_stroke_sides": allowed_sides,
            "aggregation": "mean_probabilities_same_player",
            "candidate_centers": centers,
        })

    result = {
        "video": str(args.video.resolve()),
        "frames": total_frames,
        "fps": fps,
        "duration_seconds": total_frames / fps,
        "hit_checkpoint": str(args.hit_checkpoint.resolve()),
        "hit_checkpoint_epoch": hit_checkpoint.get("epoch"),
        "stroke_checkpoint": str(args.stroke_checkpoint.resolve()),
        "stroke_checkpoint_epoch": stroke_checkpoint.get("epoch"),
        "hit_threshold": args.hit_threshold,
        "hit_stride": args.hit_stride,
        "nms_radius_seconds": args.nms_radius_seconds,
        "aggregate_radius_seconds": args.aggregate_radius_seconds,
        "motion_track_crop": args.motion_track_crop,
        "event_radius_seconds": args.event_radius_seconds,
        "event_count": len(timeline),
        "gate_json": str(args.gate_json.resolve()) if args.gate_json else None,
        "gate_applied": gate_segments is not None,
        "events": timeline,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"[DONE] events={len(timeline)} output={args.output}")


if __name__ == "__main__":
    main()
