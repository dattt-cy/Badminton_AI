"""Run full Epoch 20 Fusion pipeline on a single clip (FastTrackNet + Pose + Fusion Head + Physics)."""

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
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ai_classifier.localization.fast_tracknet import FastTrackNet
from scripts.inference.analyze_long_video import default_corners
from scripts.inference.auto_court_detection import detect_court_corners
from scripts.inference.classify_shuttleset_fusion_video import classify_fusion_event
from scripts.inference.classify_shuttleset_rgb_multitask import auto_detect_player_side
from scripts.training.train_shuttleset_feature_fusion import FusionHead
from scripts.training.train_shuttleset_rgb_multitask import MultiTaskR2Plus1D

VIETNAMESE_STROKES = {
    "clear": "Phông cầu (Clear)",
    "smash": "Đập cầu (Smash)",
    "drop": "Bỏ nhỏ (Drop)",
    "net_shot": "Gài lưới (Net Shot)",
    "lift": "Vút cầu (Lift)",
    "drive": "Tạt cầu (Drive)",
    "net_attack": "Vồ lưới (Net Attack)",
    "serve": "Giao cầu (Serve)",
}

VIETNAMESE_SIDES = {
    "forehand": "Thuận tay (Forehand)",
    "backhand": "Trái tay (Backhand)",
    "aroundhead": "Vòng đầu (Aroundhead)",
}


def run_fusion_on_clip(
    video_path: Path,
    player_side: str = "auto",
    device_str: str = "cuda" if torch.cuda.is_available() else "cpu",
    output_path: Path | None = None,
) -> dict:
    device = torch.device(device_str)
    cap = cv2.VideoCapture(str(video_path))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    if total_frames <= 0:
        raise ValueError(f"Cannot inspect video: {video_path}")

    event_frame = total_frames // 2

    # Court corners
    try:
        corners = detect_court_corners(video_path)
    except Exception:
        corners = default_corners(width, height)

    # Models
    pose_model = YOLO("models/checkpoints/pose/yolov8n-pose.pt")

    rgb_ckpt_path = REPO_ROOT / "work_dirs" / "r2plus1d18_mixed_shuttleset_finebadminton" / "best.pth"
    rgb_ckpt = torch.load(rgb_ckpt_path, map_location="cpu", weights_only=False)
    rgb_model = MultiTaskR2Plus1D()
    rgb_model.load_state_dict(rgb_ckpt["model"])
    rgb_model.to(device).eval()

    fusion_ckpt_path = REPO_ROOT / "work_dirs" / "shuttleset_fusion_epoch20_full_b4" / "best.pth"
    fusion_ckpt = torch.load(fusion_ckpt_path, map_location="cpu", weights_only=False)
    structured_dim = fusion_ckpt["model"]["structured.1.weight"].shape[1]
    fusion_model = FusionHead(512, structured_dim)
    fusion_model.load_state_dict(fusion_ckpt["model"])
    fusion_model.to(device).eval()

    # TrackNet
    tracker = FastTrackNet(device=device)
    trajectory = tracker.predict_window(video_path, event_frame, window_before=30, window_after=30)

    # Map player_side
    if player_side in ("top", "upper"):
        p_side = "upper"
    elif player_side in ("bottom", "lower"):
        p_side = "lower"
    else:
        # Check filename first (e.g. _xa_ or _top_ vs _gan_ or _bottom_)
        fname = video_path.name.lower()
        if "_xa_" in fname or "_top_" in fname:
            p_side = "upper"
        elif "_gan_" in fname or "_bottom_" in fname:
            p_side = "lower"
        else:
            try:
                det = auto_detect_player_side(video_path, pose_model)
                p_side = "upper" if det == "top" else "lower"
            except Exception:
                p_side = "upper"

    raw_res = classify_fusion_event(
        video_path, trajectory, event_frame, p_side, corners,
        pose_model, rgb_model, fusion_model, rgb_ckpt, fusion_ckpt, device,
    )

    pred = {
        "stroke": {
            "label": raw_res["stroke"]["label"],
            "probability": float(raw_res["stroke"]["probability"]),
            "vn_label": VIETNAMESE_STROKES.get(raw_res["stroke"]["label"], raw_res["stroke"]["label"]),
        },
        "stroke_side": {
            "label": raw_res["stroke_side"]["label"],
            "probability": float(raw_res["stroke_side"]["probability"]),
            "vn_label": VIETNAMESE_SIDES.get(raw_res["stroke_side"]["label"], raw_res["stroke_side"]["label"]),
        },
        "combined_label": f"{raw_res['stroke_side']['label']} {raw_res['stroke']['label']}",
    }

    stroke_ranking = [
        {
            "label": item["label"],
            "probability": float(item["probability"]),
            "vn_label": VIETNAMESE_STROKES.get(item["label"], item["label"]),
        }
        for item in raw_res["stroke_ranking"]
    ]

    stroke_side_ranking = [
        {
            "label": item["label"],
            "probability": float(item["probability"]),
            "vn_label": VIETNAMESE_SIDES.get(item["label"], item["label"]),
        }
        for item in raw_res["stroke_side_ranking"]
    ]

    result = {
        "video": str(video_path),
        "pipeline": "epoch20_fusion_full",
        "prediction": pred,
        "stroke_ranking": stroke_ranking,
        "stroke_side_ranking": stroke_side_ranking,
        "video_metadata": {
            "frames": total_frames,
            "fps": fps,
            "width": width,
            "height": height,
            "duration_seconds": round(total_frames / max(1.0, fps), 2),
            "player_side": p_side,
        },
        "physics_adjustment": raw_res.get("physics_adjustment", {}),
    }

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video", type=Path)
    parser.add_argument("--player-side", choices=("top", "bottom", "upper", "lower", "auto"), default="auto")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    res = run_fusion_on_clip(args.video, args.player_side, output_path=args.output)
    print(json.dumps(res, indent=2, ensure_ascii=False))

