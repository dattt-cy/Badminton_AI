"""Evaluate fusion on archive events in-memory using FastTrackNet and shared models."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from ultralytics import YOLO

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ai_classifier.localization.fast_tracknet import FastTrackNet
from scripts.inference.auto_court_detection import detect_court_corners
from scripts.inference.classify_shuttleset_fusion_video import classify_fusion_event
from scripts.training.train_shuttleset_feature_fusion import FusionHead
from scripts.training.train_shuttleset_rgb_multitask import MultiTaskR2Plus1D


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hit-report", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--trajectory-cache-dir", type=Path)
    parser.add_argument("--fusion-checkpoint", type=Path, default=Path("work_dirs/shuttleset_fusion_full/best.pth"))
    parser.add_argument(
        "--rgb-checkpoint", type=Path,
        default=Path("work_dirs/r2plus1d18_shuttleset_mixed_crop_full_e8_e10_b4/best.pth"),
        help="Must match the RGB backbone used to build the fusion feature cache.",
    )
    parser.add_argument("--pose-model", type=Path, default=Path("models/checkpoints/pose/yolov8n-pose.pt"))
    parser.add_argument("--tracknet-model", type=Path, default=Path("external/TrackNetV3/ckpts/TrackNet_best.pt"))
    return parser.parse_args()


def trajectory_cache_path(cache_dir: Path, video: Path, event_frame: int) -> Path:
    """Return a cache path that cannot be reused for a different hit window."""
    video_key = str(video.resolve()).lower().encode("utf-8")
    digest = hashlib.sha1(video_key).hexdigest()[:10]
    return cache_dir / f"{video.parent.name}_{video.stem}_{digest}_f{event_frame}_ball.csv"


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    trajectory_dir = args.trajectory_cache_dir or (args.output_dir / "trajectory")
    trajectory_dir.mkdir(exist_ok=True)
    report_path = args.output_dir / "report.json"
    source = json.loads(args.hit_report.read_text(encoding="utf-8"))
    previous = json.loads(report_path.read_text(encoding="utf-8"))["results"] if report_path.is_file() else []
    completed = {row["video"]: row for row in previous if "predicted_stroke" in row}

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading models in-memory on {device}...")
    t_load0 = time.perf_counter()

    tracker = FastTrackNet(checkpoint_path=args.tracknet_model, device=device)
    pose_model = YOLO(str(args.pose_model))

    rgb_checkpoint = torch.load(args.rgb_checkpoint, map_location="cpu", weights_only=False)
    rgb_model = MultiTaskR2Plus1D()
    rgb_model.load_state_dict(rgb_checkpoint["model"])
    rgb_model.to(device).eval()

    fusion_checkpoint = torch.load(args.fusion_checkpoint, map_location="cpu", weights_only=False)
    structured_dim = fusion_checkpoint["model"]["structured.1.weight"].shape[1]
    fusion_model = FusionHead(512, structured_dim)
    fusion_model.load_state_dict(fusion_checkpoint["model"])
    fusion_model.to(device).eval()

    print(f"All models loaded in {time.perf_counter() - t_load0:.2f}s.")

    rows = []
    started = time.perf_counter()
    total_items = len(source["results"])

    for index, item in enumerate(source["results"], 1):
        video = Path(item["video"])
        if str(video) in completed:
            rows.append(completed[str(video)])
            continue

        clip_start = time.perf_counter()
        row = {key: item[key] for key in ("video", "label", "expected_side", "expected_stroke")}

        try:
            if "error" in item.get("with_hit", {}):
                raise RuntimeError(item["with_hit"]["error"])
            event = item["with_hit"]

            # 1. Fast TrackNet trajectory
            trajectory_csv = trajectory_cache_path(trajectory_dir, video, int(event["event_frame"]))
            if trajectory_csv.is_file():
                trajectory = pd.read_csv(trajectory_csv)
            else:
                trajectory = tracker.predict_window(
                    video, center_frame=event["event_frame"], window_before=32, window_after=32
                )
                trajectory.to_csv(trajectory_csv, index=False)

            # 2. Court corners
            try:
                corners = detect_court_corners(str(video)).reshape(4, 2)
                court_fallback = False
            except ValueError:
                capture = cv2.VideoCapture(str(video))
                width = int(capture.get(3))
                height = int(capture.get(4))
                capture.release()
                corners = np.asarray([
                    [0.27 * width, 0.83 * height],
                    [0.73 * width, 0.83 * height],
                    [0.84 * width, 0.46 * height],
                    [0.16 * width, 0.46 * height],
                ], dtype=np.float32)
                court_fallback = True

            # 3. In-memory Fusion Classification
            player_side = "upper" if event["player_side"] == "top" else "lower"
            result = classify_fusion_event(
                video=video,
                trajectory=trajectory,
                event_frame=event["event_frame"],
                player_side=player_side,
                corners=corners,
                pose_model=pose_model,
                rgb_model=rgb_model,
                fusion_model=fusion_model,
                rgb_checkpoint=rgb_checkpoint,
                fusion_checkpoint=fusion_checkpoint,
                device=device,
            )

            stroke = result["stroke"]["label"]
            side = result["stroke_side"]["label"]
            clip_time = time.perf_counter() - clip_start

            row.update({
                "event_frame": event["event_frame"],
                "court_fallback": court_fallback,
                "predicted_stroke": stroke,
                "predicted_side": side,
                "stroke_confidence": result["stroke"]["probability"],
                "side_confidence": result["stroke_side"]["probability"],
                "stroke_correct": stroke == item["expected_stroke"],
                "side_correct": side == item["expected_side"],
                "joint_correct": stroke == item["expected_stroke"] and side == item["expected_side"],
                "pose_two_player_valid_ratio": result["pose_two_player_valid_ratio"],
                "structured_quality": result["structured_quality"],
                "raw_stroke": result["raw_stroke"],
                "raw_stroke_side": result["raw_stroke_side"],
                "physics_adjustment": result["physics_adjustment"],
                "clip_elapsed_seconds": round(clip_time, 3),
            })
        except Exception as error:
            row["error"] = f"{type(error).__name__}: {error}"

        rows.append(row)
        valid = [v for v in rows if "predicted_stroke" in v]
        report = {
            "pipeline": [
                "hit_model", "learned_selector", "motion_tracking_crop",
                "fast_tracknet", "fusion", "physics",
            ],
            "hit_report": str(args.hit_report.resolve()),
            "fusion_checkpoint": str(args.fusion_checkpoint.resolve()),
            "rgb_checkpoint": str(args.rgb_checkpoint.resolve()),
            "tracknet_model": str(args.tracknet_model.resolve()),
            "evaluated": len(valid),
            "failures": sum("error" in v for v in rows),
            "stroke_correct": sum(v["stroke_correct"] for v in valid),
            "side_correct": sum(v["side_correct"] for v in valid),
            "joint_correct": sum(v["joint_correct"] for v in valid),
            "elapsed_seconds": round(time.perf_counter() - started, 2),
            "results": rows,
        }
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(
            f"[{index:02d}/{total_items:02d}] {video.parent.name}/{video.name} -> "
            f"{row.get('predicted_side', 'ERR')} {row.get('predicted_stroke', '')} "
            f"joint={row.get('joint_correct', False)} ({row.get('clip_elapsed_seconds', 0.0):.2f}s)",
            flush=True,
        )


if __name__ == "__main__":
    main()
