"""Canonical end-to-end inference for a new badminton video.

Production defaults use the best validated configuration in this repository:
R(2+1)D hit detection at 0.50, two-stage NMS, YOLOv8x pose, contact
refinement, and direct Robust KAP classification. Pose features are precomputed
once for the union of candidate windows and kept only in memory. TrackNet keeps
its original event windows so its background-dependent inputs do not change.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ai_classifier.localization import two_stage_temporal_nms  # noqa: E402
from scripts.inference.auto_court_detection import detect_court_corners  # noqa: E402
from scripts.inference.robust_aligned_video_pipeline import AlignedVideoPipeline  # noqa: E402
from scripts.inference.scan_video_hit_rgb import load_model, scan_video  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--hit-checkpoint", type=Path,
                        default=Path("work_dirs/r2plus1d18_hit_full/best.pth"))
    parser.add_argument("--kap-checkpoint", type=Path,
                        default=Path("work_dirs/kap_robust_seed20260926/kap_best.pth"))
    parser.add_argument("--pose-model", type=Path,
                        default=Path("models/checkpoints/pose/yolov8x-pose.pt"))
    parser.add_argument("--threshold", type=float, default=0.50)
    parser.add_argument("--stride", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--nms-radius", type=int, default=15)
    parser.add_argument("--same-side-nms-radius", type=int, default=18)
    parser.add_argument("--contact-refine-radius", type=int, default=6)
    parser.add_argument("--gate", type=Path, default=None,
                        help="Optional rally-gate JSON; off by default to preserve recall.")
    parser.add_argument("--gate-padding", type=int, default=30)
    parser.add_argument("--no-shared-cache", action="store_true",
                        help="Use the legacy slow per-event feature path.")
    parser.add_argument("--no-amp", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    started = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    hit_model, hit_meta = load_model(args.hit_checkpoint)
    hit_model.to(device).eval()
    scan_started = time.time()
    raw_events = scan_video(
        args.video, hit_model, int(hit_meta.get("frames", 16)),
        stride=args.stride, batch_size=args.batch_size, device=device,
        use_amp=not args.no_amp and device.type == "cuda",
        gate_path=args.gate, gate_padding=args.gate_padding,
    )
    events = two_stage_temporal_nms(
        raw_events, threshold=args.threshold, radius=args.nms_radius,
        same_side_radius=args.same_side_nms_radius,
    )
    scan_seconds = time.time() - scan_started
    del hit_model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    cap = cv2.VideoCapture(str(args.video))
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {args.video}")
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    corners = detect_court_corners(str(args.video))
    pipeline = AlignedVideoPipeline(
        ckpt_path=args.kap_checkpoint, corners=corners,
        orig_w=width, orig_h=height, pose_model_path=args.pose_model,
    )
    cache_stats = None
    cache_started = time.time()
    if not args.no_shared_cache:
        cache_stats = pipeline.prepare_event_cache(
            cap, [event.frame for event in events], total_frames,
            contact_refine_radius=args.contact_refine_radius,
        )
    cache_seconds = time.time() - cache_started

    classify_started = time.time()
    predictions = []
    for index, event in enumerate(events, 1):
        feature = pipeline.extract_features(
            cap, event.frame, total_frames, hitter_side=event.side,
            contact_refine_radius=args.contact_refine_radius,
        )
        if feature is None:
            continue
        detail = pipeline.predict_detailed(feature)
        predictions.append({
            "frame": event.frame,
            "time_seconds": event.frame / fps if fps > 0 else None,
            "refined_frame": int(feature["hit_frame"]),
            "contact_offset": int(feature["contact_offset"]),
            "hitter": event.side,
            "hit_score": float(event.score),
            "stroke": detail["stroke"],
            "confidence": detail["confidence"],
            "top2": detail["top2"],
            "crop_box": list(feature["crop_box"]),
        })
        print(f"[{index}/{len(events)}] frame={event.frame} "
              f"{detail['stroke']} {detail['confidence']:.3f}", flush=True)
    cap.release()
    result = {
        "pipeline": "canonical_robust_kap_v2_pose_cache",
        "video": str(args.video.resolve()), "video_frames": total_frames, "fps": fps,
        "configuration": {
            "hit_checkpoint": str(args.hit_checkpoint.resolve()),
            "kap_checkpoint": str(args.kap_checkpoint.resolve()),
            "pose_model": str(args.pose_model.resolve()),
            "threshold": args.threshold, "stride": args.stride,
            "nms_radius": args.nms_radius,
            "same_side_nms_radius": args.same_side_nms_radius,
            "contact_refine_radius": args.contact_refine_radius,
            "gate": str(args.gate.resolve()) if args.gate else None,
            "shared_cache": not args.no_shared_cache,
        },
        "raw_scores": len(raw_events), "candidate_count": len(events),
        "classified_count": len(predictions), "cache": cache_stats,
        "timing_seconds": {
            "hit_scan": scan_seconds,
            "pose_cache": cache_seconds,
            "classification_after_cache": time.time() - classify_started,
            "total": time.time() - started,
        },
        "predictions": predictions,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n",
                           encoding="utf-8")
    print(f"[DONE] candidates={len(events)} classified={len(predictions)} "
          f"seconds={result['timing_seconds']['total']:.1f} output={args.output}")


if __name__ == "__main__":
    main()
