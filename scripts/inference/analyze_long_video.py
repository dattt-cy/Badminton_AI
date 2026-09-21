"""Build a stroke timeline from a long badminton singles video.

Pipeline:
    HIT scan -> learned candidate selector -> FastTrackNet -> fusion -> physics

The runner persists the expensive HIT scan and every completed event. Re-running
the same command resumes incomplete work. Cache identities include the video,
selected event frame, and relevant model files.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from ultralytics import YOLO

REPO_ROOT = Path(__file__).resolve().parents[2]
for root in (REPO_ROOT, REPO_ROOT / "src"):
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

from ai_classifier.localization import HitEvent, temporal_nms
from ai_classifier.localization.fast_tracknet import FastTrackNet
from ai_classifier.localization.live_gate import frame_is_live
from scripts.inference.auto_court_detection import detect_court_corners
from scripts.inference.classify_shuttleset_fusion_video import classify_fusion_event
from scripts.inference.scan_video_hit_rgb import load_model as load_hit_model, scan_video
from scripts.training.train_hit_candidate_selector import candidate_features, selector_probability
from scripts.training.train_shuttleset_feature_fusion import FusionHead
from scripts.training.train_shuttleset_rgb_multitask import MultiTaskR2Plus1D


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--hit-checkpoint", type=Path, required=True)
    parser.add_argument("--hit-selector", type=Path, required=True)
    parser.add_argument(
        "--rgb-checkpoint", type=Path,
        default=Path("work_dirs/r2plus1d18_shuttleset_mixed_crop_full_e8_e10_b4/best.pth"),
    )
    parser.add_argument(
        "--fusion-checkpoint", type=Path,
        default=Path("work_dirs/shuttleset_fusion_full/best.pth"),
    )
    parser.add_argument(
        "--pose-model", type=Path,
        default=Path("models/checkpoints/pose/yolov8n-pose.pt"),
    )
    parser.add_argument(
        "--tracknet-model", type=Path,
        default=Path("external/TrackNetV3/ckpts/TrackNet_best.pt"),
    )
    parser.add_argument("--gate-json", type=Path)
    parser.add_argument("--hit-stride", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--selector-threshold", type=float)
    parser.add_argument(
        "--candidate-hit-threshold", type=float,
        help="Override the selector's candidate HIT threshold for controlled sweeps.",
    )
    parser.add_argument(
        "--candidate-nms-radius", type=int,
        help="Override the selector's side-aware candidate NMS radius.",
    )
    parser.add_argument(
        "--cross-side-nms-radius", type=int, default=0,
        help="Optional final NMS radius across both players; zero disables it.",
    )
    parser.add_argument("--max-events", type=int, help="Debug/smoke-test limit after selection.")
    parser.add_argument("--force-scan", action="store_true")
    parser.add_argument("--force-events", action="store_true")
    return parser.parse_args()


def video_metadata(path: Path) -> dict[str, float | int]:
    capture = cv2.VideoCapture(str(path))
    try:
        frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    finally:
        capture.release()
    if frames <= 0 or fps <= 0:
        raise ValueError(f"Cannot inspect video: {path}")
    return {
        "frames": frames, "fps": fps, "width": width, "height": height,
        "duration_seconds": frames / fps,
    }


def file_identity(path: Path) -> dict[str, object]:
    resolved = path.resolve()
    stat = resolved.stat()
    return {"path": str(resolved), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def stable_digest(payload: object, length: int = 16) -> str:
    encoded = json.dumps(payload, sort_keys=True, ensure_ascii=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:length]


def select_hit_events(
    raw_events: list[HitEvent], selector: dict[str, object], threshold: float,
    *, candidate_hit_threshold: float | None = None,
    candidate_nms_radius: int | None = None,
    cross_side_nms_radius: int = 0,
) -> tuple[list[HitEvent], list[dict[str, object]]]:
    generation = selector["candidate_generation"]
    candidates = temporal_nms(
        raw_events,
        threshold=float(
            generation["hit_threshold"]
            if candidate_hit_threshold is None else candidate_hit_threshold
        ),
        radius=int(
            generation["nms_radius"]
            if candidate_nms_radius is None else candidate_nms_radius
        ),
        side_aware=bool(generation.get("side_aware", True)),
    )
    scored = []
    selected = []
    for index, event in enumerate(candidates):
        probability = selector_probability(candidate_features(candidates, index), selector)
        row = {
            "frame": event.frame, "side": event.side, "hit_score": event.score,
            "selector_probability": probability,
        }
        scored.append(row)
        if probability >= threshold:
            selected.append(event)
    if cross_side_nms_radius > 0:
        probabilities = {
            (int(row["frame"]), str(row["side"])): float(row["selector_probability"])
            for row in scored
        }
        kept: list[HitEvent] = []
        for event in sorted(
            selected,
            key=lambda item: probabilities[(item.frame, item.side)],
            reverse=True,
        ):
            if all(abs(event.frame - previous.frame) > cross_side_nms_radius for previous in kept):
                kept.append(event)
        selected = sorted(kept, key=lambda item: item.frame)
    return selected, scored


def default_corners(width: int, height: int) -> np.ndarray:
    return np.asarray([
        [0.27 * width, 0.83 * height], [0.73 * width, 0.83 * height],
        [0.84 * width, 0.46 * height], [0.16 * width, 0.46 * height],
    ], dtype=np.float32)


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def main() -> None:
    args = parse_args()
    started = time.perf_counter()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    trajectory_dir = args.output_dir / "trajectory"
    event_dir = args.output_dir / "events"
    trajectory_dir.mkdir(exist_ok=True)
    event_dir.mkdir(exist_ok=True)
    report_path = args.output_dir / "timeline.json"
    scan_path = args.output_dir / "hit_scan.json"

    meta = video_metadata(args.video)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    selector = json.loads(args.hit_selector.read_text(encoding="utf-8"))
    selector_threshold = float(
        args.selector_threshold if args.selector_threshold is not None
        else selector.get("threshold", 0.5)
    )
    scan_identity = {
        "video": file_identity(args.video),
        "hit_checkpoint": file_identity(args.hit_checkpoint),
        "hit_stride": args.hit_stride,
    }

    scan_payload = None
    if scan_path.is_file() and not args.force_scan:
        cached = json.loads(scan_path.read_text(encoding="utf-8"))
        if cached.get("identity") == scan_identity:
            scan_payload = cached
            print(f"[resume] HIT scan: {scan_path}", flush=True)
    if scan_payload is None:
        hit_model, hit_checkpoint = load_hit_model(args.hit_checkpoint)
        hit_model.to(device).eval()
        raw = scan_video(
            args.video, hit_model, int(hit_checkpoint.get("frames", 16)),
            stride=args.hit_stride, batch_size=args.batch_size, device=device,
            use_amp=device.type == "cuda",
        )
        scan_payload = {
            "identity": scan_identity,
            "checkpoint_epoch": hit_checkpoint.get("epoch"),
            "raw_events": [event.__dict__ for event in raw],
        }
        write_json(scan_path, scan_payload)
    raw_events = [HitEvent(**item) for item in scan_payload["raw_events"]]
    selected, scored = select_hit_events(
        raw_events, selector, selector_threshold,
        candidate_hit_threshold=args.candidate_hit_threshold,
        candidate_nms_radius=args.candidate_nms_radius,
        cross_side_nms_radius=args.cross_side_nms_radius,
    )

    if args.gate_json:
        gate = json.loads(args.gate_json.read_text(encoding="utf-8"))
        selected = [event for event in selected if frame_is_live(event.frame, gate["live_segments"])]
    if args.max_events is not None:
        selected = selected[:args.max_events]
    print(f"[selector] candidates={len(scored)} selected={len(selected)}", flush=True)

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

    try:
        corners = detect_court_corners(str(args.video)).reshape(4, 2)
        court_fallback = False
    except ValueError:
        corners = default_corners(int(meta["width"]), int(meta["height"]))
        court_fallback = True

    model_identity = stable_digest({
        "tracknet": file_identity(args.tracknet_model),
        "rgb": file_identity(args.rgb_checkpoint),
        "fusion": file_identity(args.fusion_checkpoint),
    })
    completed = []
    for index, event in enumerate(selected, 1):
        event_key = f"f{event.frame}_{event.side}_{model_identity}"
        event_path = event_dir / f"{event_key}.json"
        if event_path.is_file() and not args.force_events:
            row = json.loads(event_path.read_text(encoding="utf-8"))
            print(f"[{index}/{len(selected)}] resume frame={event.frame}", flush=True)
        else:
            event_started = time.perf_counter()
            trajectory_path = trajectory_dir / f"{event_key}.csv"
            if trajectory_path.is_file() and not args.force_events:
                trajectory = trajectory_path
            else:
                trajectory = tracker.predict_window(
                    args.video, center_frame=event.frame, window_before=32, window_after=32
                )
                trajectory.to_csv(trajectory_path, index=False)
            result = classify_fusion_event(
                video=args.video, trajectory=trajectory, event_frame=event.frame,
                player_side=event.side, corners=corners, pose_model=pose_model,
                rgb_model=rgb_model, fusion_model=fusion_model,
                rgb_checkpoint=rgb_checkpoint, fusion_checkpoint=fusion_checkpoint,
                device=device,
            )
            probability = next(
                row["selector_probability"] for row in scored
                if row["frame"] == event.frame and row["side"] == event.side
            )
            row = {
                "event_id": event_key,
                "frame": event.frame,
                "time_seconds": round(event.frame / float(meta["fps"]), 3),
                "court_player": event.side,
                "hit_score": event.score,
                "selector_probability": probability,
                "stroke": result["stroke"],
                "stroke_side": result["stroke_side"],
                "raw_stroke": result["raw_stroke"],
                "raw_stroke_side": result["raw_stroke_side"],
                "physics_adjustment": result["physics_adjustment"],
                "pose_two_player_valid_ratio": result["pose_two_player_valid_ratio"],
                "structured_quality": result["structured_quality"],
                "clip_range": result["clip_range"],
                "crop_box": result["crop_box"],
                "elapsed_seconds": round(time.perf_counter() - event_started, 3),
            }
            write_json(event_path, row)
            print(
                f"[{index}/{len(selected)}] frame={event.frame} "
                f"{row['stroke_side']['label']} {row['stroke']['label']} "
                f"({row['elapsed_seconds']:.1f}s)", flush=True,
            )
        completed.append(row)
        report = {
            "status": "running" if index < len(selected) else "complete",
            "pipeline": ["hit_model", "learned_selector", "fast_tracknet", "fusion", "physics"],
            "video": file_identity(args.video),
            "metadata": meta,
            "device": str(device),
            "models": {
                "hit": file_identity(args.hit_checkpoint),
                "selector": file_identity(args.hit_selector),
                "tracknet": file_identity(args.tracknet_model),
                "rgb": file_identity(args.rgb_checkpoint),
                "fusion": file_identity(args.fusion_checkpoint),
                "pose": file_identity(args.pose_model),
            },
            "selector_threshold": selector_threshold,
            "candidate_hit_threshold": args.candidate_hit_threshold,
            "candidate_nms_radius": args.candidate_nms_radius,
            "cross_side_nms_radius": args.cross_side_nms_radius,
            "candidate_count": len(scored),
            "selected_event_count": len(selected),
            "completed_event_count": len(completed),
            "court_corners": corners.tolist(),
            "court_fallback": court_fallback,
            "elapsed_seconds": round(time.perf_counter() - started, 2),
            "events": completed,
        }
        write_json(report_path, report)
    if not selected:
        write_json(report_path, {
            "status": "complete", "video": file_identity(args.video), "metadata": meta,
            "candidate_count": len(scored), "selected_event_count": 0, "events": [],
        })
    print(f"[DONE] events={len(completed)} output={report_path}", flush=True)


if __name__ == "__main__":
    main()
