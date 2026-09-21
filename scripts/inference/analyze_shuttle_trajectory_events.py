"""Detect shuttle direction-change events and assign the nearest court player."""

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

from ai_classifier.localization.shuttle_events import direction_change_events, load_tracknet_csv
from scripts.inference.classify_shuttleset_rgb_multitask import (
    constrain_side_probabilities,
    ranked,
    read_hitter_clip,
)
from scripts.training.train_shuttleset_rgb_multitask import MultiTaskR2Plus1D


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video", type=Path)
    parser.add_argument("--trajectory", type=Path, required=True)
    parser.add_argument("--pose-model", type=Path, default=Path("models/checkpoints/pose/yolov8n-pose.pt"))
    parser.add_argument("--stroke-checkpoint", type=Path)
    parser.add_argument("--window", type=int, default=5)
    parser.add_argument("--min-score", type=float, default=0.0015)
    parser.add_argument("--nms-radius", type=int, default=8)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def assign_player(result, shuttle: np.ndarray, width: int, height: int) -> tuple[str, float] | None:
    if result.boxes is None or result.keypoints is None:
        return None
    boxes = result.boxes.xyxy.detach().cpu().numpy()
    classes = result.boxes.cls.detach().cpu().numpy()
    keypoints = result.keypoints.data.detach().cpu().numpy()
    players: list[tuple[float, float, float]] = []
    for box, class_id, pose in zip(boxes, classes, keypoints):
        if int(class_id) != 0:
            continue
        center = np.asarray([(box[0] + box[2]) / 2, (box[1] + box[3]) / 2])
        if not (0.15 * width <= center[0] <= 0.85 * width and 0.18 * height <= center[1] <= 0.95 * height):
            continue
        wrists = pose[[9, 10]]
        visible_wrists = wrists[wrists[:, 2] >= 0.15, :2]
        anchor = visible_wrists[np.argmin(np.linalg.norm(visible_wrists - shuttle, axis=1))] if len(visible_wrists) else center
        players.append((float(center[0]), float(center[1]), float(np.linalg.norm(anchor - shuttle))))
    
    if len(players) < 2:
        return None
    # Find the two main players: the ones closest to the horizontal center line
    players.sort(key=lambda item: abs(item[0] - width / 2))
    main_players = players[:2]
    
    # Sort the two main players by Y coordinate (0=upper, 1=lower)
    main_players.sort(key=lambda item: item[1])
    
    selected = min(range(2), key=lambda index: main_players[index][2])
    return (
        "upper" if selected == 0 else "lower",
        main_players[selected][2],
    )


def main() -> None:
    args = parse_args()
    capture = cv2.VideoCapture(str(args.video))
    width, height = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    capture.release()
    frames, points = load_tracknet_csv(args.trajectory)
    events = direction_change_events(
        frames, points, width=width, height=height, window=args.window,
        min_score=args.min_score, nms_radius=args.nms_radius,
    )
    pose_model = YOLO(str(args.pose_model))
    stroke_checkpoint = None
    stroke_model = None
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.stroke_checkpoint:
        stroke_checkpoint = torch.load(args.stroke_checkpoint, map_location="cpu", weights_only=False)
        stroke_model = MultiTaskR2Plus1D()
        stroke_model.load_state_dict(stroke_checkpoint["model"])
        stroke_model.to(device).eval()
    output_events = []
    for event in events:
        capture = cv2.VideoCapture(str(args.video))
        capture.set(cv2.CAP_PROP_POS_FRAMES, event.frame)
        ok, frame = capture.read()
        capture.release()
        assignment = None
        if ok:
            result = pose_model.predict(frame, imgsz=960, conf=0.20, verbose=False)[0]
            assignment = assign_player(result, np.asarray([event.x, event.y]), width, height)
        output_event = {
            "frame": event.frame, "x": event.x, "y": event.y, "trajectory_score": event.score,
            "player_side": assignment[0] if assignment else "unknown",
            "shuttle_to_wrist_distance": assignment[1] if assignment else None,
        }
        if assignment and stroke_model is not None and stroke_checkpoint is not None:
            total_frames = int(frames[-1]) + 1
            radius = 30
            start, end = max(0, event.frame - radius), min(total_frames - 1, event.frame + radius)
            clip, crop_metadata = read_hitter_clip(
                args.video, int(stroke_checkpoint.get("frames", 16)),
                "top" if assignment[0] == "upper" else "bottom", pose_model,
                float(stroke_checkpoint.get("crop_padding", 0.55)), start, end,
            )
            with torch.inference_mode(), torch.autocast(
                device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"
            ):
                stroke_logits, side_logits = stroke_model(clip.to(device))
            stroke_probabilities = torch.softmax(stroke_logits, 1).cpu()[0]
            side_probabilities = torch.softmax(side_logits, 1).cpu()[0]
            stroke_ranking = ranked(stroke_probabilities, list(stroke_checkpoint["stroke_classes"]))
            side_classes = list(stroke_checkpoint["side_classes"])
            # Around-the-head is a forehand overhead technique for the UI's
            # binary forehand/backhand posture label.
            if "forehand" in side_classes and "aroundhead" in side_classes:
                forehand_index = side_classes.index("forehand")
                aroundhead_index = side_classes.index("aroundhead")
                side_probabilities[forehand_index] += side_probabilities[aroundhead_index]
                side_probabilities[aroundhead_index] = 0.0
                side_probabilities /= side_probabilities.sum()
            constrained_side, _ = constrain_side_probabilities(
                side_probabilities, side_classes,
                str(stroke_ranking[0]["label"]),
            )
            side_ranking = ranked(constrained_side, side_classes)
            output_event.update({
                "clip_start_frame": start, "clip_end_frame": end,
                "crop_box": crop_metadata["crop_box"],
                "stroke": stroke_ranking[0], "stroke_side": side_ranking[0],
                "stroke_ranking": stroke_ranking, "stroke_side_ranking": side_ranking,
            })
        output_events.append(output_event)
    output = {"video": str(args.video.resolve()), "trajectory": str(args.trajectory.resolve()), "events": output_events}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
