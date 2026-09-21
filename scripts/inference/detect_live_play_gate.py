"""Detect full-court two-player LIVE segments before hit inference."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ai_classifier.localization.live_gate import GateSample, samples_to_segments


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sample-fps", type=float, default=2.0)
    parser.add_argument("--min-court-ratio", type=float, default=0.12)
    parser.add_argument("--pose-model", type=Path, default=Path("models/checkpoints/pose/yolov8n-pose.pt"))
    parser.add_argument("--pose-device", default="cpu")
    return parser.parse_args()


def court_ratio(frame: np.ndarray) -> float:
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array([30, 30, 30]), np.array([100, 255, 255]))
    return float(np.count_nonzero(mask) / mask.size)


def player_layout(result, width: int, height: int) -> tuple[bool, bool, int]:
    if result.keypoints is None:
        return False, False, 0
    top = bottom = False
    count = 0
    for pose in result.keypoints.data.cpu().numpy():
        visible = pose[:, 2] > 0.20
        if visible.sum() < 5:
            continue
        cx = float(pose[visible, 0].mean()) / width
        cy = float(pose[visible, 1].mean()) / height
        if not 0.12 <= cx <= 0.88 or not 0.12 <= cy <= 0.95:
            continue
        count += 1
        top |= cy < 0.55
        bottom |= cy >= 0.45
    return top, bottom, count


def main() -> None:
    args = parse_args()
    if args.sample_fps <= 0:
        raise ValueError("sample-fps must be positive")
    capture = cv2.VideoCapture(str(args.video))
    total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    if total <= 0 or fps <= 0:
        capture.release()
        raise ValueError(f"Cannot inspect video: {args.video}")
    step = max(1, round(fps / args.sample_fps))
    pose_model = YOLO(str(args.pose_model))
    samples = []
    for frame_index in range(0, total, step):
        capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ok, frame = capture.read()
        if not ok:
            continue
        ratio = court_ratio(frame)
        result = pose_model.predict(
            frame, imgsz=640, conf=0.20, verbose=False, device=args.pose_device
        )[0]
        top, bottom, player_count = player_layout(result, frame.shape[1], frame.shape[0])
        layout_score = min(1.0, ratio / max(args.min_court_ratio, 1e-6))
        layout_score *= 1.0 if top and bottom else (0.35 if player_count >= 1 else 0.0)
        live = ratio >= args.min_court_ratio and top and bottom
        samples.append(GateSample(frame_index, live, layout_score))
        if len(samples) % 100 == 0:
            print(f"gate_progress={frame_index}/{total}", flush=True)
    capture.release()
    segments = samples_to_segments(samples, total_frames=total, max_gap_frames=step * 2)
    result = {
        "video": str(args.video.resolve()),
        "frames": total,
        "fps": fps,
        "sample_fps": args.sample_fps,
        "gate_version": "v1_court_two_player",
        "scope_warning": "V1 filters close-ups/non-play; full-court replay deduplication is not implemented.",
        "live_segments": segments,
        "live_frame_fraction": sum(
            int(item["end_frame"]) - int(item["start_frame"]) + 1 for item in segments
        ) / total,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"[DONE] live_segments={len(segments)} output={args.output}")


if __name__ == "__main__":
    main()
