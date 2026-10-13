"""Unified End-to-End Badminton Video Analytics Pipeline (PBL6 Production).

Single CLI command to process a raw badminton match video from A to Z:
1. Court Corner Detection (Homography).
2. Dense Hit Spotting (R(2+1)D-18) + Two-Stage Temporal NMS.
3. Keypoint & Shuttle Tracking (YOLOv8n-pose + FastTrackNet with shared cache).
4. Multi-Modal KAP Feature Fusion (8-class balanced stroke classification).
5. ML Rally Segmentation (same_rally vs new_rally boundary detector).
6. CoachAI-aligned Tactical Intelligence (Scoreboard, N-grams, 3x3 Spatial Grid).

Usage:
    python run_pipeline.py --video "path/to/match.mp4" --output "work_dirs/analysis.json"
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

import cv2
import joblib
import torch

REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from ai_classifier.analytics import (  # noqa: E402
    assign_rallies,
    build_stroke_analytics,
    build_match_analysis,
    extract_score_events,
    suppress_weak_duplicate_serves,
)
from ai_classifier.localization import (  # noqa: E402
    rescue_alternating_bridges,
    two_stage_temporal_nms,
    rescue_pre_rally_serves,
)
from scripts.inference.auto_court_detection import detect_court_corners  # noqa: E402
from scripts.inference.robust_aligned_video_pipeline import AlignedVideoPipeline  # noqa: E402
from scripts.inference.scan_video_hit_rgb import load_model, scan_video  # noqa: E402



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Unified End-to-End Badminton Video Analytics Pipeline"
    )
    parser.add_argument("--video", type=Path, required=True, help="Path to input match video (.mp4)")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Path to output JSON file (defaults to work_dirs/<video_stem>_analysis.json)",
    )
    parser.add_argument("--player-upper", type=str, default="Player Upper", help="Name of upper player")
    parser.add_argument("--player-lower", type=str, default="Player Lower", help="Name of lower player")

    # Canonical production model checkpoints in src
    ckpts = REPO_ROOT / "src" / "ai_classifier" / "models" / "checkpoints"
    hit_default = (ckpts / "hit_detector.pth") if (ckpts / "hit_detector.pth").exists() else (REPO_ROOT / "work_dirs" / "r2plus1d18_hit_full" / "best.pth")
    kap_default = (ckpts / "kap_classifier.pth") if (ckpts / "kap_classifier.pth").exists() else (REPO_ROOT / "work_dirs" / "kap_online_balanced_selection_20261008" / "best.pth")
    pose_default = (ckpts / "yolov8n-pose.pt") if (ckpts / "yolov8n-pose.pt").exists() else (REPO_ROOT / "models" / "checkpoints" / "pose" / "yolov8n-pose.pt")
    rally_default = (ckpts / "rally_boundary_detector.joblib") if (ckpts / "rally_boundary_detector.joblib").exists() else (REPO_ROOT / "work_dirs" / "rally_boundary_detector.joblib")

    parser.add_argument("--hit-checkpoint", type=Path, default=hit_default)
    parser.add_argument("--kap-checkpoint", type=Path, default=kap_default)
    parser.add_argument("--pose-model", type=Path, default=pose_default)
    parser.add_argument("--rally-model", type=Path, default=rally_default)
    parser.add_argument("--threshold", type=float, default=0.75, help="Hit detection score threshold")
    parser.add_argument("--stride", type=int, default=4, help="Hit scan frame stride")
    parser.add_argument("--batch-size", type=int, default=8, help="GPU batch size for scan")
    parser.add_argument("--nms-radius", type=int, default=8, help="First-stage temporal NMS radius")
    parser.add_argument(
        "--same-side-nms-radius",
        type=int,
        default=18,
        help="Second-stage same-side duplicate suppression radius",
    )
    parser.add_argument(
        "--contact-refine-radius",
        type=int,
        default=6,
        help="Contact refinement window radius (+/- frames)",
    )
    parser.add_argument("--no-shared-cache", action="store_true", help="Disable shared-pose memory cache")
    parser.add_argument("--no-amp", action="store_true", help="Disable mixed precision")
    parser.add_argument(
        "--score-roi",
        type=int,
        nargs=4,
        metavar=("Y1", "Y2", "X1", "X2"),
        help="Optional calibrated scoreboard digit ROI for EasyOCR",
    )
    parser.add_argument(
        "--auto-ocr",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Automatically scan standard BWF broadcast scoreboard overlay with EasyOCR",
    )
    parser.add_argument(
        "--score-ocr-interval",
        type=int,
        default=30,
        help="Frame interval between optional scoreboard OCR reads",
    )
    parser.add_argument(
        "--use-tta",
        action="store_true",
        default=True,
        help="Enable adaptive temporal soft-voting (default)",
    )
    parser.add_argument(
        "--no-tta",
        dest="use_tta",
        action="store_false",
        help="Disable temporal multi-window soft-voting",
    )
    parser.add_argument(
        "--full-tta",
        action="store_true",
        help="Run all three temporal windows for every hit (slower)",
    )
    parser.add_argument("--tta-confidence-threshold", type=float, default=0.75)
    parser.add_argument("--tta-margin-threshold", type=float, default=0.20)
    parser.add_argument("--serve-rescue-threshold", type=float, default=0.40)
    parser.add_argument("--bridge-rescue-threshold", type=float, default=0.55)
    parser.add_argument("--no-hit-rescue", action="store_true")
    return parser.parse_args()


def analyze_badminton_video(
    video_path: Path | str,
    output_path: Path | str | None = None,
    player_upper: str = "Player Upper",
    player_lower: str = "Player Lower",
    hit_checkpoint: Path | str = REPO_ROOT / "work_dirs" / "r2plus1d18_hit_full" / "best.pth",
    kap_checkpoint: Path | str = REPO_ROOT / "work_dirs" / "kap_online_balanced_selection_20261008" / "best.pth",
    pose_model: Path | str = REPO_ROOT / "models" / "checkpoints" / "pose" / "yolov8n-pose.pt",
    rally_model: Path | str = REPO_ROOT / "work_dirs" / "rally_boundary_detector.joblib",
    threshold: float = 0.75,
    stride: int = 4,
    batch_size: int = 8,
    nms_radius: int = 8,
    same_side_nms_radius: int = 15,
    contact_refine_radius: int = 6,
    shared_cache: bool = True,
    use_amp: bool = True,
    use_tta: bool = True,
    full_tta: bool = False,
    tta_confidence_threshold: float = 0.75,
    tta_margin_threshold: float = 0.20,
    hit_rescue: bool = True,
    serve_rescue_threshold: float = 0.40,
    bridge_rescue_threshold: float = 0.55,
    score_roi: tuple[int, int, int, int] | None = None,
    score_ocr_interval: int = 30,
    auto_ocr: bool = True,
) -> dict[str, Any]:
    """Execute full end-to-end video analysis pipeline and return structured dict."""
    video_path = Path(video_path)
    if not video_path.exists():
        raise FileNotFoundError(f"Video file not found: {video_path}")

    started_total = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n=======================================================")
    print(f"[*] STARTING END-TO-END PIPELINE: {video_path.name}")
    print(f"[*] Hardware: {device} ({torch.cuda.get_device_name(0) if device.type == 'cuda' else 'CPU'})")
    print(f"[*] Upper: {player_upper} | Lower: {player_lower}")
    print(f"=======================================================")

    # 1. Hit Spotting (R(2+1)D-18)
    print("\n--- STEP 1/5: DENSE HIT SCANNING ---")
    t0_scan = time.time()
    hit_model, hit_meta = load_model(Path(hit_checkpoint))
    hit_model.to(device).eval()
    n_frames = int(hit_meta.get("frames", 16))

    raw_events = scan_video(
        video_path,
        hit_model,
        n_frames,
        stride=stride,
        batch_size=batch_size,
        device=device,
        use_amp=use_amp and device.type == "cuda",
    )
    events = two_stage_temporal_nms(
        raw_events,
        threshold=threshold,
        radius=nms_radius,
        same_side_radius=same_side_nms_radius,
    )
    if hit_rescue:
        events = rescue_pre_rally_serves(
            events,
            raw_events,
            min_serve_score=serve_rescue_threshold,
            nms_radius=nms_radius,
            same_side_radius=same_side_nms_radius,
        )
        events = rescue_alternating_bridges(
            events,
            raw_events,
            min_score=bridge_rescue_threshold,
            nms_radius=nms_radius,
            same_side_radius=same_side_nms_radius,
        )
    scan_time = time.time() - t0_scan
    del hit_model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    print(f"[+] Scored {len(raw_events)} windows -> Found {len(events)} candidate hits ({scan_time:.1f}s)")

    # 2. Setup Video & Auto Court Corner Detection
    print("\n--- STEP 2/5: COURT DETECTION & POSE/KAP PIPELINE INIT ---")
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video stream: {video_path}")
    try:
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = float(cap.get(cv2.CAP_PROP_FPS)) or 30.0

        corners = detect_court_corners(str(video_path))
        pipeline = AlignedVideoPipeline(
            ckpt_path=Path(kap_checkpoint),
            corners=corners,
            orig_w=width,
            orig_h=height,
            pose_model_path=Path(pose_model),
        )

        # 3. Shared Pose Cache & KAP Feature Classification
        print(f"\n--- STEP 3/5: MULTI-MODAL STROKE CLASSIFICATION ({len(events)} candidates) ---")
        t0_classify = time.time()
        cache_stats = None
        if shared_cache:
            cache_stats = pipeline.prepare_event_cache(
                cap,
                [event.frame for event in events],
                total_frames,
                contact_refine_radius=contact_refine_radius,
            )

        predictions = []
        for idx, event in enumerate(events, 1):
            if use_tta and full_tta:
                feature, detail = pipeline.predict_detailed_tta(
                    cap,
                    event.frame,
                    total_frames,
                    hitter_side=event.side,
                    contact_refine_radius=contact_refine_radius,
                )
            elif use_tta:
                feature, detail = pipeline.predict_detailed_adaptive_tta(
                    cap,
                    event.frame,
                    total_frames,
                    hitter_side=event.side,
                    contact_refine_radius=contact_refine_radius,
                    confidence_threshold=tta_confidence_threshold,
                    margin_threshold=tta_margin_threshold,
                )
            else:
                feature = pipeline.extract_features(
                    cap,
                    event.frame,
                    total_frames,
                    hitter_side=event.side,
                    contact_refine_radius=contact_refine_radius,
                )
                detail = pipeline.predict_detailed(feature) if feature is not None else None

            if feature is None or detail is None:
                continue
            predictions.append({
                "hit_index": idx,
                "frame": event.frame,
                "time_seconds": round(event.frame / fps, 3),
                "refined_frame": int(feature["hit_frame"]),
                "contact_offset": int(feature["contact_offset"]),
                "hitter": event.side,
                "hitter_name": player_upper if event.side == "upper" else player_lower,
                "hit_score": round(float(event.score), 4),
                "stroke": detail["stroke"],
                "confidence": round(float(detail["confidence"]), 4),
                "top2": detail["top2"],
                "tta_used": bool(detail.get("tta_used", False)),
                "tta_windows": int(detail.get("tta_windows", 1)),
                "crop_box": list(feature["crop_box"]),
                "_analytics_context": feature["analytics_context"],
            })
            if idx % 20 == 0 or idx == len(events):
                print(f"  [{idx}/{len(events)}] frame={event.frame} ({detail['stroke']}, conf={detail['confidence']:.2f})")
    finally:
        cap.release()

    # Stroke analytics
    for i, pred in enumerate(predictions):
        next_f = int(predictions[i + 1]["refined_frame"]) if i + 1 < len(predictions) else None
        pred["analytics"] = build_stroke_analytics(
            pred.pop("_analytics_context"),
            stroke=pred["stroke"],
            fps=fps,
            next_hit_frame=next_f,
        )
    predictions, suppressed_duplicate_serves = suppress_weak_duplicate_serves(
        predictions
    )
    if suppressed_duplicate_serves:
        print(
            f"[+] Suppressed {len(suppressed_duplicate_serves)} weak duplicate serve(s)"
        )
    classify_time = time.time() - t0_classify
    print(f"[+] Classified {len(predictions)} hits ({classify_time:.1f}s)")

    # 4. ML Rally Boundary Segmentation
    print("\n--- STEP 4/5: RALLY SEGMENTATION ---")
    t0_rally = time.time()
    rally_clf = None
    rally_model_path = Path(rally_model)
    if rally_model_path.exists():
        try:
            payload = joblib.load(rally_model_path)
            rally_clf = payload.get("model", payload)
            print(f"[*] Loaded Rally Boundary ML Model: {rally_model_path.name}")
        except Exception as e:
            print(f"[!] Warning: Could not load rally model ({e}), falling back to adaptive rule")

    assign_rallies(predictions, fps=fps, model=rally_clf)
    rally_count = max((p.get("rally", 1) for p in predictions), default=0)
    rally_time = time.time() - t0_rally
    print(f"[+] Segmented into {rally_count} rallies ({rally_time:.2f}s)")

    # 5. CoachAI Tactical Analytics
    print("\n--- STEP 5/5: COACH-AI TACTICAL ANALYSIS ---")
    t0_tactics = time.time()
    player_names = {"upper": player_upper, "lower": player_lower}
    tactics = build_match_analysis(
        predictions,
        player_names=player_names,
        fps=fps,
        total_frames=total_frames,
    )
    tactics_time = time.time() - t0_tactics

    scoreboard_ocr = {"enabled": False, "reason": "score_roi_not_configured"}
    effective_score_roi = score_roi
    if effective_score_roi is None and auto_ocr:
        # Standard BWF broadcast scoreboard overlay ROI scaled to frame resolution
        effective_score_roi = (
            int(40 * height / 720),
            int(110 * height / 720),
            int(410 * width / 1280),
            int(445 * width / 1280),
        )
    if effective_score_roi is not None:
        try:
            print(f"[*] Scanning BWF broadcast scoreboard ROI {effective_score_roi} with EasyOCR...")
            scoreboard_ocr = extract_score_events(
                video_path,
                effective_score_roi,
                sample_interval_frames=score_ocr_interval,
            )
        except Exception as ocr_err:
            print(f"[!] Warning: Scoreboard OCR scan skipped or failed ({ocr_err})")
            scoreboard_ocr = {"enabled": False, "reason": str(ocr_err)}

    total_time = time.time() - started_total
    duration_seconds = round(total_frames / fps, 2)
    speed_factor = round(duration_seconds / total_time, 2) if total_time > 0 else 0.0

    counts = Counter(p["stroke"] for p in predictions)
    total_hits = len(predictions)
    stroke_percentages = {
        s: round((c / total_hits) * 100, 2) for s, c in sorted(counts.items())
    }
    rally_rows = tactics["scoreboard"]["rallies"]
    reliability = {
        "score_is_estimated": True,
        "rally_winners_are_estimated": True,
        "partial_first_rally": bool(rally_rows and rally_rows[0]["partial_start"]),
        "partial_last_rally": bool(rally_rows and rally_rows[-1]["partial_end"]),
        "low_confidence_threshold": 0.70,
        "low_confidence_hits": sum(p["confidence"] < 0.70 for p in predictions),
        "landing_is_2d_proxy": True,
        "suppressed_weak_duplicate_serves": len(suppressed_duplicate_serves),
    }

    result = {
        "schema_version": "2.0_pbl6_coachai",
        "video": {
            "path": str(video_path.resolve()),
            "filename": video_path.name,
            "total_frames": total_frames,
            "fps": fps,
            "duration_seconds": duration_seconds,
            "court_corners": corners.tolist(),
        },
        "players": {
            "upper": player_upper,
            "lower": player_lower,
        },
        "summary": {
            "hit_count": total_hits,
            "estimated_rally_count": rally_count,
            "hits_per_minute": round((total_hits / duration_seconds) * 60, 2) if duration_seconds > 0 else 0.0,
            "mean_confidence": round(float(sum(p["confidence"] for p in predictions) / max(total_hits, 1)), 4),
            "stroke_counts": dict(sorted(counts.items())),
            "stroke_percentages": stroke_percentages,
            "estimated_score": tactics["scoreboard"]["estimated_score"],
        },
        "reliability": reliability,
        "scoreboard": tactics["scoreboard"],
        "tactical_patterns": tactics["tactical_patterns"],
        "structured_tactics": tactics["structured_tactics"],
        "movement": tactics["movement"],
        "scoreboard_ocr": scoreboard_ocr,
        "suppression_audit": {
            "weak_duplicate_serves": suppressed_duplicate_serves,
        },
        "spatial_distribution_3x3": tactics["spatial_distribution_3x3"],
        "player_profiles": tactics.get("player_profiles", {}),
        "radar_chart": tactics.get("radar_chart", {}),
        "response_transition_matrix": tactics.get("response_transition_matrix", {}),
        "coach_insights": tactics.get("coach_insights", []),
        "predictions": predictions,
        "timing_seconds": {
            "hit_scan": round(scan_time, 2),
            "pose_and_classification": round(classify_time, 2),
            "rally_segmentation": round(rally_time, 2),
            "tactical_analysis": round(tactics_time, 2),
            "total": round(total_time, 2),
            "speed_factor_realtime": f"{speed_factor}x RT",
        },
    }

    if output_path is not None:
        out_p = Path(output_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        out_p.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"\n=======================================================")
        print(f"[DONE] Saved full analysis to: {out_p}")
        print(f"[*] Processed {duration_seconds}s video in {total_time:.1f}s ({speed_factor}x Realtime)")
        print(f"[*] Estimated score: {tactics['scoreboard']['estimated_score']['text']} | Rallies: {rally_count} | Hits: {total_hits}")
        print(f"=======================================================\n")

    return result


def main() -> None:
    args = parse_args()
    out_path = args.output
    if out_path is None:
        out_path = REPO_ROOT / "work_dirs" / f"{args.video.stem}_analysis.json"

    analyze_badminton_video(
        video_path=args.video,
        output_path=out_path,
        player_upper=args.player_upper,
        player_lower=args.player_lower,
        hit_checkpoint=args.hit_checkpoint,
        kap_checkpoint=args.kap_checkpoint,
        pose_model=args.pose_model,
        rally_model=args.rally_model,
        threshold=args.threshold,
        stride=args.stride,
        batch_size=args.batch_size,
        nms_radius=args.nms_radius,
        same_side_nms_radius=args.same_side_nms_radius,
        contact_refine_radius=args.contact_refine_radius,
        shared_cache=not args.no_shared_cache,
        use_amp=not args.no_amp,
        use_tta=args.use_tta,
        full_tta=args.full_tta,
        tta_confidence_threshold=args.tta_confidence_threshold,
        tta_margin_threshold=args.tta_margin_threshold,
        hit_rescue=not args.no_hit_rescue,
        serve_rescue_threshold=args.serve_rescue_threshold,
        bridge_rescue_threshold=args.bridge_rescue_threshold,
        score_roi=tuple(args.score_roi) if args.score_roi else None,
        score_ocr_interval=args.score_ocr_interval,
        auto_ocr=args.auto_ocr,
    )


if __name__ == "__main__":
    main()

