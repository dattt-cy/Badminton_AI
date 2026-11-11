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
import bisect
import copy
import json
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Callable

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
    calibrate_bwf_scoreboard_roi,
    crop_score_roi,
    extract_score_events,
    refine_strokes_by_physics,
    suppress_weak_duplicate_serves,
    TacticalSequenceDecoder,
    refine_predictions_by_sequence,
    fit_flight_time_model,
    rescore_strokes_by_flight_time,
)
from ai_classifier.localization import (  # noqa: E402
    HitEvent,
    enforce_alternation,
    rescue_alternating_bridges,
    two_stage_temporal_nms,
    rescue_pre_rally_serves,
)
from scripts.inference.auto_court_detection import detect_court_corners, detect_court_line_corners  # noqa: E402
from scripts.inference.robust_aligned_video_pipeline import AlignedVideoPipeline  # noqa: E402
from scripts.inference.scan_video_hit_rgb import load_model, scan_video  # noqa: E402
from ai_classifier.model_cache import cached, file_key  # noqa: E402



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

    # Canonical production model checkpoints in src (Hit Detector: BadmintonHitNetV2)
    ckpts = REPO_ROOT / "src" / "ai_classifier" / "models" / "checkpoints"
    hit_default = (ckpts / "badminton_hitnet_v2.pth") if (ckpts / "badminton_hitnet_v2.pth").exists() else (ckpts / "hit_detector.pth")
    kap_default = (ckpts / "kap_classifier.pth") if (ckpts / "kap_classifier.pth").exists() else (REPO_ROOT / "work_dirs" / "kap_online_balanced_selection_20261008" / "best.pth")
    pose_default = (ckpts / "yolov8n-pose.pt") if (ckpts / "yolov8n-pose.pt").exists() else (REPO_ROOT / "models" / "checkpoints" / "pose" / "yolov8n-pose.pt")
    rally_default = (ckpts / "rally_boundary_detector.joblib") if (ckpts / "rally_boundary_detector.joblib").exists() else (REPO_ROOT / "work_dirs" / "rally_boundary_detector.joblib")

    parser.add_argument("--hit-checkpoint", type=Path, default=hit_default, help="Hit spotter checkpoint (default: BadmintonHitNetV2)")
    parser.add_argument("--kap-checkpoint", type=Path, default=kap_default)
    parser.add_argument("--pose-model", type=Path, default=pose_default)
    parser.add_argument("--rally-model", type=Path, default=rally_default)
    parser.add_argument("--threshold", type=float, default=0.72, help="Hit detection score threshold")
    parser.add_argument("--stride", type=int, default=4, help="Hit scan frame stride")
    parser.add_argument("--batch-size", type=int, default=8, help="GPU batch size for scan")
    parser.add_argument("--nms-radius", type=int, default=6, help="First-stage temporal NMS radius")
    parser.add_argument(
        "--same-side-nms-radius",
        type=int,
        default=20,
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
    parser.add_argument("--serve-rescue-threshold", type=float, default=0.55)
    parser.add_argument("--bridge-rescue-threshold", type=float, default=0.55)
    parser.add_argument("--no-hit-rescue", action="store_true")
    parser.add_argument(
        "--tau",
        type=float,
        default=0.20,
        help="Logit adjustment temperature tau (default 0.20, 0.0 to disable)",
    )
    parser.add_argument(
        "--context-rules",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Apply badminton sport context rules (soft serve rescue and no-mid-rally serve)",
    )
    parser.add_argument(
        "--use-physics-refinement",
        action="store_true",
        default=False,
        help="Enable geometric court and opponent position disambiguation (Drive vs Net Shot, Net Attack vs Lift)",
    )
    parser.add_argument(
        "--use-tactical-decoder",
        action="store_true",
        default=False,
        help="Enable statistical sequence decoding (Turn-taking Viterbi with Laplace smoothing). "
             "Off by default: it gained only ~0.3 top-1 and undoes flight-time re-scoring.",
    )
    parser.add_argument(
        "--no-tactical-decoder",
        dest="use_tactical_decoder",
        action="store_false",
        help="Disable statistical sequence decoding",
    )
    parser.add_argument(
        "--no-flight-time-rescoring",
        dest="use_flight_time_rescoring",
        action="store_false",
        default=True,
        help="Disable re-scoring strokes with the flight time to the next hit",
    )
    parser.add_argument(
        "--flight-time-weight",
        type=float,
        default=0.75,
        help="Weight of log p(flight time | stroke) in flight-time re-scoring (default 0.75)",
    )
    parser.add_argument(
        "--tactical-lambda",
        type=float,
        default=0.20,
        help="Transition weight for tactical sequence decoder (default 0.20)",
    )
    parser.add_argument(
        "--min-shuttle-visibility",
        type=float,
        default=0.25,
        help="Minimum TrackNet shuttlecock visibility ratio to keep candidate hit (default 0.25, 0.0 to disable)",
    )
    return parser.parse_args()


def _load_hit_model(kind: str, checkpoint: Path, device: torch.device):
    """HitNet on ``device`` in eval mode, loaded once per process."""
    def load():
        model, meta = load_model(checkpoint)
        return model.to(device).eval(), meta
    return cached(kind, (*file_key(checkpoint), str(device)), load)


def _load_rally_model(path: Path):
    payload = joblib.load(path)
    print(f"[*] Loaded Rally Boundary ML Model: {path.name}")
    return payload.get("model", payload)


# A segment of the match closes after this many frames without a raw hit score any
# candidate filter could use. The widest filter (alternating-bridge rescue) spans two
# 85-frame legs, so 180 frames keep neighbouring segments independent.
SEGMENT_DEAD_FRAMES = 180
SEGMENT_DEAD_SECONDS = 6.0
# Progress-only reports (no new hits) are sent at most this often.
PROGRESS_REPORT_SECONDS = 2.0
# Closed clusters are looked for every this many scored batches (4 x 32 frames).
CLUSTER_CHECK_BATCHES = 4
# A viewer's seek into unscanned video starts a pass this far before the seek point, so the
# rally around it closes; seeks the running pass reaches within FOCUS_SOON_SECONDS of video
# do not interrupt it.
FOCUS_LEAD_SECONDS = 10.0
FOCUS_SOON_SECONDS = 30.0


def _postprocess_predictions(
    predictions: list[dict[str, Any]],
    *,
    fps: float,
    rally_clf,
    min_shuttle_visibility: float,
    use_flight_time_rescoring: bool,
    flight_time_weight: float,
    context_rules: bool,
    use_physics_refinement: bool,
    use_tactical_decoder: bool,
    tactical_lambda: float,
    verbose: bool,
) -> dict[str, Any]:
    """Match-level refinement of classified hits: analytics, rallies and sport rules.

    It mutates and returns ``predictions``; provisional calls pass a deep copy.
    """
    log = print if verbose else (lambda *args, **kwargs: None)

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
        log(
            f"[+] Suppressed {len(suppressed_duplicate_serves)} weak duplicate serve(s)"
        )
    if min_shuttle_visibility > 0.0:
        kept_preds = []
        pruned_ghost = 0
        for p in predictions:
            vis = p.get("analytics", {}).get("shuttle_visibility_ratio", 1.0)
            if vis is not None and vis < min_shuttle_visibility:
                pruned_ghost += 1
                continue
            kept_preds.append(p)
        if pruned_ghost > 0:
            log(f"[+] Pruned {pruned_ghost} phantom candidate(s) with low TrackNet shuttle visibility (< {min_shuttle_visibility:.2f})")
        predictions = kept_preds

    # ML Rally Boundary Segmentation
    assign_rallies(predictions, fps=fps, model=rally_clf)
    rally_count = max((p.get("rally", 1) for p in predictions), default=0)

    # Flight-time re-scoring: the classifier window ends before the opponent's
    # return, whose timing separates push/lift and drive/push.
    flight_changes: list[dict[str, Any]] = []
    if use_flight_time_rescoring:
        flight_model = fit_flight_time_model(REPO_ROOT / "data" / "manifests" / "shuttleset_rgb.csv")
        predictions, flight_changes = rescore_strokes_by_flight_time(
            predictions, flight_model, fps=fps, weight=flight_time_weight
        )
        log(f"[+] Flight-time re-scoring: adjusted {len(flight_changes)} stroke(s)")
    for p in predictions:
        p.pop("_stroke_log_probs", None)

    # Badminton Soft Context Rules (Sport Domain Logic)
    if context_rules:
        rule_changes = 0
        for p in predictions:
            r = p.get("ball_round", 1)
            top2 = p.get("top2", [])
            # Rule 1: Soft serve rescue at rally opening: only if serve in top2
            if r == 1 and "serve" in top2 and p["stroke"] != "serve":
                p["stroke"] = "serve"
                rule_changes += 1
            # Rule 2: Middle of rally cannot be serve
            elif r > 1 and p["stroke"] == "serve":
                p["stroke"] = top2[1] if (len(top2) > 1 and top2[0] == "serve") else (top2[0] if top2 else "net_shot")
                rule_changes += 1
        if rule_changes:
            log(f"[+] Badminton Context Rules: adjusted {rule_changes} stroke(s) to match sport logic")

    # Geometric & Physics Disambiguation (Drive vs Net Shot, Net Attack vs Lift)
    physics_changes: list[dict[str, Any]] = []
    if use_physics_refinement:
        predictions, physics_changes = refine_strokes_by_physics(predictions)
        if physics_changes:
            log(f"[+] Physics Refinement: disambiguated {len(physics_changes)} ambiguous shot(s):")
            for ch in physics_changes:
                log(f"    - Frame {ch['frame']}: {ch['original_stroke']} -> {ch['refined_stroke']} ({ch['reason']})")
        else:
            log("[+] Physics Refinement: all predictions physically consistent (0 changes)")

    # Tactical Sequence Decoding (Turn-taking Viterbi with Laplace smoothing)
    sequence_changes: list[dict[str, Any]] = []
    if use_tactical_decoder:
        manifest_p = REPO_ROOT / "data" / "manifests" / "shuttleset_rgb.csv"
        decoder = TacticalSequenceDecoder(manifest_p, alpha=0.5, default_lambda=tactical_lambda)
        predictions, sequence_changes = refine_predictions_by_sequence(predictions, decoder, lam=tactical_lambda)
        if sequence_changes:
            log(f"[+] Tactical Sequence Decoder: refined {len(sequence_changes)} ambiguous shot(s):")
            for ch in sequence_changes:
                log(f"    - Frame {ch['frame']}: {ch['original_stroke']} -> {ch['refined_stroke']}")
        else:
            log("[+] Tactical Sequence Decoder: all predictions sequence-consistent (0 changes)")

    return {
        "predictions": predictions,
        "rally_count": rally_count,
        "suppressed_duplicate_serves": suppressed_duplicate_serves,
        "flight_changes": flight_changes,
        "physics_changes": physics_changes,
        "sequence_changes": sequence_changes,
    }


def _summarize_rallies(predictions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Rally spans for provisional results; the final rallies come from build_match_analysis."""
    spans: dict[int, dict[str, Any]] = {}
    for p in predictions:
        rally_id = int(p.get("rally", 1))
        time_s = float(p["time_seconds"])
        span = spans.setdefault(rally_id, {
            "rally_id": rally_id, "start_time_seconds": time_s, "end_time_seconds": time_s, "hits_count": 0,
        })
        span["end_time_seconds"] = time_s
        span["hits_count"] += 1
    for span in spans.values():
        span["duration_seconds"] = round(span["end_time_seconds"] - span["start_time_seconds"], 3)
    return [spans[key] for key in sorted(spans)]


def analyze_badminton_video(
    video_path: Path | str,
    output_path: Path | str | None = None,
    player_upper: str = "Player Upper",
    player_lower: str = "Player Lower",
    hit_checkpoint: Path | str = REPO_ROOT / "work_dirs" / "r2plus1d18_hit_full" / "best.pth",
    extra_hit_checkpoint: Path | str | None = None,
    alternation_gap_frames: int | None = None,
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
    use_physics_refinement: bool = False,
    use_tactical_decoder: bool = False,
    tactical_lambda: float = 0.20,
    use_flight_time_rescoring: bool = True,
    flight_time_weight: float = 0.75,
    tau: float = 0.20,
    context_rules: bool = True,
    min_shuttle_visibility: float = 0.25,
    on_progress: Callable[[dict[str, Any]], None] | None = None,
    focus: Callable[[], tuple[Any, float] | None] | None = None,
) -> dict[str, Any]:
    """Execute full end-to-end video analysis pipeline and return structured dict.

    ``on_progress`` receives provisional results while the video is analysed:
    ``analyzed_until_seconds`` (results are complete up to there), ``duration_seconds``
    ``analyzed_ranges`` (spans whose hits are final) and, when new hits were classified,
    ``predictions`` and ``rallies``. ``focus`` returns ``(version, seconds)`` of the
    viewer's latest seek (or None); unscanned video there is analysed first. The returned
    dict is identical to a run without either callback.
    """
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

    # 1. Set-up: the hit detector, court geometry and the KAP pipeline are ready before
    # the scan starts, so rallies can be classified while the scan is still running.
    print("\n--- STEP 1/5: HIT DETECTOR, COURT DETECTION & POSE/KAP PIPELINE INIT ---")
    hit_model, hit_meta = _load_hit_model("hit", Path(hit_checkpoint), device)
    n_frames = int(hit_meta.get("frames", 16))

    # The scoreboard ROI depends only on the frame size, so its OCR samples are cut
    # during this sequential scan instead of seeking the video once per sample later.
    effective_score_roi = score_roi
    if effective_score_roi is None and auto_ocr:
        probe = cv2.VideoCapture(str(video_path))
        try:
            probe_h = int(probe.get(cv2.CAP_PROP_FRAME_HEIGHT))
            probe_w = int(probe.get(cv2.CAP_PROP_FRAME_WIDTH))
        finally:
            probe.release()
        effective_score_roi = calibrate_bwf_scoreboard_roi(probe_h, probe_w)
    ocr_interval = max(1, score_ocr_interval)
    score_crops: dict[int, Any] = {}

    def keep_score_crop(frame_index: int, frame) -> None:
        if frame_index % ocr_interval == 0:
            score_crops[frame_index] = crop_score_roi(frame, effective_score_roi)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video stream: {video_path}")
    try:
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = float(cap.get(cv2.CAP_PROP_FPS)) or 30.0

        # Checkpoints fine-tuned on court-line coordinates say so; older ones were adapted
        # to the green-mat quad and keep it.
        court_reference = cached(
            "kap_court_reference", file_key(kap_checkpoint),
            lambda: torch.load(kap_checkpoint, map_location="cpu", weights_only=False).get("court_reference", "mat"),
        )
        detect = detect_court_line_corners if court_reference == "lines" else detect_court_corners
        corners = detect(str(video_path))
        pipeline = AlignedVideoPipeline(
            ckpt_path=Path(kap_checkpoint),
            corners=corners,
            orig_w=width,
            orig_h=height,
            pose_model_path=Path(pose_model),
            tau=tau,
        )
        rally_clf = None
        rally_model_path = Path(rally_model)
        if rally_model_path.exists():
            try:
                rally_clf = cached("rally", file_key(rally_model_path), lambda: _load_rally_model(rally_model_path))
            except Exception as e:
                print(f"[!] Warning: Could not load rally model ({e}), falling back to adaptive rule")
        postprocess_options = dict(
            fps=fps,
            rally_clf=rally_clf,
            min_shuttle_visibility=min_shuttle_visibility,
            use_flight_time_rescoring=use_flight_time_rescoring,
            flight_time_weight=flight_time_weight,
            context_rules=context_rules,
            use_physics_refinement=use_physics_refinement,
            use_tactical_decoder=use_tactical_decoder,
            tactical_lambda=tactical_lambda,
        )

        # 2-3. Scan the video in passes and classify each closed cluster of hits meanwhile.
        # Raw scores >= min_candidate_score ("strong") are the only ones candidate filtering
        # uses, and every filter acts within at most 170 frames (two 85-frame bridge legs).
        # Strong scores further apart than dead_frames therefore belong to independent
        # clusters, and classifying cluster by cluster gives the hits of one whole-video pass.
        # A cluster is classified once the scanned frames on both sides of it confirm such
        # a gap. Passes start on the stride * batch_size grid, so every scored batch holds
        # the same clips as in one pass; a pass may start where the viewer seeks (``focus``).
        print("\n--- STEP 2/5: DENSE HIT SCANNING + STREAMED STROKE CLASSIFICATION ---")
        t0_scan = time.time()
        min_candidate_score = (
            min(threshold, serve_rescue_threshold, bridge_rescue_threshold) if hit_rescue else threshold
        )
        dead_frames = max(SEGMENT_DEAD_FRAMES, int(round(SEGMENT_DEAD_SECONDS * fps)))
        half = n_frames // 2
        grid = stride * batch_size
        first_centre = half
        last_centre = first_centre + stride * ((total_frames - 2 * half - 1) // stride)
        scores: dict[int, HitEvent] = {}   # window centre -> raw event, over all passes
        strong: list[int] = []              # sorted centres with a strong score
        runs: list[list[int]] = []          # sorted, merged [first, last] scanned centres
        cluster_candidates: dict[int, list[HitEvent]] = {}  # first strong centre -> candidates
        classified: list[dict[str, Any]] = []
        stream = {
            "reported_hits": 0,
            "last_report": 0.0,
            "classify_time": 0.0,
            "focus_version": None,
            "focus_frame": None,  # unscanned frame the viewer asked for, or None
            "batches_since_check": 0,
        }

        def filter_candidates(raw: list[HitEvent]) -> list[HitEvent]:
            found = two_stage_temporal_nms(
                raw,
                threshold=threshold,
                radius=nms_radius,
                same_side_radius=same_side_nms_radius,
            )
            if hit_rescue:
                found = rescue_pre_rally_serves(
                    found,
                    raw,
                    min_serve_score=serve_rescue_threshold,
                    nms_radius=nms_radius,
                    same_side_radius=same_side_nms_radius,
                )
                found = rescue_alternating_bridges(
                    found,
                    raw,
                    min_score=bridge_rescue_threshold,
                    nms_radius=nms_radius,
                    same_side_radius=same_side_nms_radius,
                )
            if alternation_gap_frames:
                found = enforce_alternation(found, alternation_gap_frames)
            return found

        def all_candidates() -> list[HitEvent]:
            return [event for key in sorted(cluster_candidates) for event in cluster_candidates[key]]

        def classify(cluster_events: list[HitEvent]) -> None:
            t0 = time.time()
            pipeline.set_event_frames([event.frame for event in all_candidates()])
            if shared_cache:
                pipeline.prepare_event_cache(
                    cap,
                    [event.frame for event in cluster_events],
                    total_frames,
                    contact_refine_radius=contact_refine_radius,
                )
            for event in cluster_events:
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
                classified.append({
                    "hit_index": 0,  # position among all candidates, set once all are known
                    "frame": event.frame,
                    "time_seconds": round(event.frame / fps, 3),
                    "refined_frame": int(feature["hit_frame"]),
                    "contact_offset": int(feature["contact_offset"]),
                    "hitter": event.side,
                    "hitter_name": player_upper if event.side == "upper" else player_lower,
                    "hit_score": round(float(event.score), 4),
                    "stroke": detail["stroke"],
                    "stroke_side": detail.get("stroke_side", "unknown"),
                    "stroke_side_confidence": round(float(detail.get("stroke_side_confidence", 0.0)), 4),
                    "confidence": round(float(detail["confidence"]), 4),
                    "top2": detail["top2"],
                    "tta_used": bool(detail.get("tta_used", False)),
                    "tta_windows": int(detail.get("tta_windows", 1)),
                    "crop_box": list(feature["crop_box"]),
                    "_analytics_context": feature["analytics_context"],
                    "_stroke_log_probs": torch.log_softmax(
                        detail["logits"].reshape(-1).float(), dim=-1
                    ).tolist(),
                })
            print(f"  [cluster] classified {len(cluster_events)} hit(s); {len(classified)} so far")
            stream["classify_time"] += time.time() - t0

        def ordered_predictions() -> list[dict[str, Any]]:
            """Classified hits in frame order with their position among all candidates."""
            index = {(event.frame, event.side): i for i, event in enumerate(all_candidates(), 1)}
            for pred in classified:
                pred["hit_index"] = index[(pred["frame"], pred["hitter"])]
            classified.sort(key=lambda pred: pred["hit_index"])
            return classified

        def run_of(centre: int) -> list[int] | None:
            position = bisect.bisect_right(runs, [centre, float("inf")]) - 1
            if position >= 0 and runs[position][0] <= centre <= runs[position][1]:
                return runs[position]
            return None

        def mark_scanned(first: int, last: int) -> list[int]:
            runs.append([first, last])
            runs.sort()
            merged: list[list[int]] = []
            for run in runs:
                if merged and run[0] <= merged[-1][1] + stride:
                    merged[-1][1] = max(merged[-1][1], run[1])
                else:
                    merged.append(run)
            runs[:] = merged
            return run_of(last)

        def clusters_in(run: list[int]) -> list[tuple[int, int, bool]]:
            """(first strong, last strong, closable) for every cluster inside a scanned run."""
            lo, hi = run
            inside = strong[bisect.bisect_left(strong, lo):bisect.bisect_right(strong, hi)]
            if not inside:
                return []
            groups: list[list[int]] = [[inside[0], inside[0]]]
            for centre in inside[1:]:
                if centre - groups[-1][1] >= dead_frames:
                    groups.append([centre, centre])
                else:
                    groups[-1][1] = centre
            result = []
            for i, (first, last) in enumerate(groups):
                left_ok = i > 0 or lo == first_centre or first - lo >= dead_frames
                right_ok = i < len(groups) - 1 or hi == last_centre or hi - last >= dead_frames
                result.append((first, last, left_ok and right_ok))
            return result

        def close_clusters(run: list[int]) -> None:
            for first, last, closable in clusters_in(run):
                if closable and first not in cluster_candidates:
                    raw = [scores[c] for c in range(first, last + 1, stride) if c in scores]
                    found = filter_candidates(raw)
                    cluster_candidates[first] = found
                    if found:
                        classify(found)

        def analyzed_ranges() -> list[list[float]]:
            """Video spans (seconds) whose hits are final: scanned, outside pending clusters."""
            spans: list[list[float]] = []
            for run in runs:
                start = 0 if run[0] == first_centre else run[0]
                end = total_frames if run[1] == last_centre else run[1]
                cursor = start
                for first, last, _ in clusters_in(run):
                    if first in cluster_candidates:
                        continue
                    if first > cursor:
                        spans.append([cursor, first])
                    cursor = max(cursor, last)
                if end > cursor:
                    spans.append([cursor, end])
            return [[round(a / fps, 2), round(b / fps, 2)] for a, b in spans]

        def report(force: bool = False) -> None:
            """Hand provisional results to ``on_progress``; failures there never stop the analysis."""
            if on_progress is None:
                return
            new_hits = len(classified) != stream["reported_hits"]
            now = time.time()
            if not force and not new_hits and now - stream["last_report"] < PROGRESS_REPORT_SECONDS:
                return
            stream["last_report"] = now
            ranges = analyzed_ranges()
            payload: dict[str, Any] = {
                "duration_seconds": round(total_frames / fps, 2),
                "analyzed_ranges": ranges,
                "analyzed_until_seconds": ranges[0][1] if ranges and ranges[0][0] == 0 else 0.0,
            }
            try:
                if new_hits:
                    stream["reported_hits"] = len(classified)
                    provisional = copy.deepcopy(sorted(classified, key=lambda pred: pred["frame"]))
                    provisional = _postprocess_predictions(provisional, verbose=False, **postprocess_options)
                    payload["predictions"] = provisional["predictions"]
                    payload["rallies"] = _summarize_rallies(provisional["predictions"])
                on_progress(payload)
            except Exception as progress_err:  # noqa: BLE001 - progress must not fail the job
                print(f"[!] Warning: progress callback failed ({progress_err})")

        def poll_focus() -> None:
            """Remember a new seek of the viewer into video that is not scanned yet."""
            if focus is None:
                return
            try:
                request = focus()
            except Exception as focus_err:  # noqa: BLE001 - a focus hint is optional
                print(f"[!] Warning: focus provider failed ({focus_err})")
                return
            if request is None or request[0] == stream["focus_version"]:
                return
            stream["focus_version"] = request[0]
            frame = int(round(float(request[1]) * fps))
            if 0 <= frame < total_frames and run_of(min(max(frame, first_centre), last_centre)) is None:
                stream["focus_frame"] = frame
                print(f"[*] Viewer focus at {request[1]:.1f}s: scanning there first")

        def next_pass_start() -> int | None:
            def first_unscanned(centre: int) -> int | None:
                while centre <= last_centre:
                    run = run_of(centre)
                    if run is None:
                        return centre
                    centre = run[1] + stride
                return None

            if stream["focus_frame"] is not None:
                target = max(first_centre, stream["focus_frame"] - int(FOCUS_LEAD_SECONDS * fps))
                aligned = first_centre + grid * ((target - first_centre) // grid)
                start = first_unscanned(aligned)
                if start is not None:
                    return start
                stream["focus_frame"] = None  # the rest after the focus is done
            return first_unscanned(first_centre)

        def run_pass(start: int) -> None:
            following = [run[0] for run in runs if run[0] > start]
            stop_at = min(following) if following else None
            pass_state = {"seen": 0}

            def on_scored(raw: list[HitEvent], frontier: int) -> bool:
                for event in raw[pass_state["seen"]:]:
                    scores[event.frame] = event
                    if event.score >= min_candidate_score:
                        bisect.insort(strong, event.frame)
                pass_state["seen"] = len(raw)
                run = mark_scanned(start, frontier)
                stream["batches_since_check"] += 1
                if stream["batches_since_check"] >= CLUSTER_CHECK_BATCHES:
                    stream["batches_since_check"] = 0
                    close_clusters(run)
                    report()
                    poll_focus()
                    focus_frame = stream["focus_frame"]
                    if focus_frame is not None and not (frontier <= focus_frame <= frontier + int(FOCUS_SOON_SECONDS * fps)):
                        if run_of(min(max(focus_frame, first_centre), last_centre)) is None:
                            return True  # jump to where the viewer is watching
                return False

            print(f"  [pass] scanning from {start / fps:.1f}s" + (f" to {stop_at / fps:.1f}s" if stop_at else ""))
            scan_video(
                video_path,
                hit_model,
                n_frames,
                stride=stride,
                batch_size=batch_size,
                device=device,
                use_amp=use_amp and device.type == "cuda",
                start_frame=start - half,
                max_frames=(stop_at + half) if stop_at is not None else None,
                frame_callback=keep_score_crop if effective_score_roi is not None else None,
                on_scored=on_scored,
            )
            for run in list(runs):
                close_clusters(run)
            report(force=True)

        if extra_hit_checkpoint is None:
            start = next_pass_start()
            while start is not None:
                run_pass(start)
                poll_focus()
                start = next_pass_start()
        else:
            # Two hit detectors scanned at the same window centres; each window keeps the higher
            # score. The production detector is strongest on 2018-2021 broadcasts, the
            # ShuttleSet22-tuned one on newer venues, and the max keeps both recalls.
            raw_events = scan_video(
                video_path, hit_model, n_frames, stride=stride, batch_size=batch_size,
                device=device, use_amp=use_amp and device.type == "cuda",
                frame_callback=keep_score_crop if effective_score_roi is not None else None,
            )
            hit_model, _ = _load_hit_model("hit_extra", Path(extra_hit_checkpoint), device)
            extra = {(e.frame, e.side): e.score for e in scan_video(
                video_path, hit_model, n_frames, stride=stride, batch_size=batch_size,
                device=device, use_amp=use_amp and device.type == "cuda",
            )}
            for e in raw_events:
                scores[e.frame] = HitEvent(e.frame, e.side, max(e.score, extra.get((e.frame, e.side), 0.0)))
            strong[:] = sorted(c for c, e in scores.items() if e.score >= min_candidate_score)
            if scores:
                close_clusters(mark_scanned(first_centre, last_centre))
        if any(not closable or first not in cluster_candidates
               for run in runs for first, _, closable in clusters_in(run)):
            raise RuntimeError("streamed scan left hit clusters unclassified")
        events = all_candidates()
        classified = ordered_predictions()
        classify_time = stream["classify_time"]
        scan_time = time.time() - t0_scan - classify_time
        print(
            f"[+] Scored {len(scores)} windows -> Found {len(events)} candidate hits "
            f"(scan {scan_time:.1f}s, classification {classify_time:.1f}s)"
        )
    finally:
        cap.release()

    print("\n--- STEP 4/5: RALLY SEGMENTATION & REFINEMENT ---")
    t0_rally = time.time()
    post = _postprocess_predictions(classified, verbose=True, **postprocess_options)
    predictions = post["predictions"]
    rally_count = post["rally_count"]
    suppressed_duplicate_serves = post["suppressed_duplicate_serves"]
    flight_changes = post["flight_changes"]
    physics_changes = post["physics_changes"]
    sequence_changes = post["sequence_changes"]
    rally_time = time.time() - t0_rally
    print(f"[+] Classified {len(predictions)} hits; segmented into {rally_count} rallies ({rally_time:.2f}s)")

    # 4.5 Scoreboard OCR Evidence Detection
    scoreboard_ocr = {"enabled": False, "reason": "score_roi_not_configured"}
    if effective_score_roi is not None:
        try:
            print(f"[*] Scanning BWF broadcast scoreboard ROI {effective_score_roi} with EasyOCR...")
            scoreboard_ocr = extract_score_events(
                video_path,
                effective_score_roi,
                sample_interval_frames=score_ocr_interval,
                roi_crops=score_crops,
            )
            print(
                f"[+] Scoreboard OCR: {scoreboard_ocr.get('successful_reads', 0)} reads, "
                f"initial={scoreboard_ocr.get('initial_score')}, "
                f"events={len(scoreboard_ocr.get('events', []))}, "
                f"observed={scoreboard_ocr.get('observed_score')}"
            )
        except Exception as ocr_err:
            print(f"[!] Warning: Scoreboard OCR scan skipped or failed ({ocr_err})")
            scoreboard_ocr = {"enabled": False, "reason": str(ocr_err)}

    # 5. CoachAI Tactical Analytics
    print("\n--- STEP 5/5: COACH-AI TACTICAL ANALYSIS ---")
    t0_tactics = time.time()
    player_names = {"upper": player_upper, "lower": player_lower}
    tactics = build_match_analysis(
        predictions,
        player_names=player_names,
        fps=fps,
        total_frames=total_frames,
        scoreboard_ocr=scoreboard_ocr,
    )
    tactics_time = time.time() - t0_tactics

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
        "physics_refinement_enabled": bool(use_physics_refinement),
        "physics_refinement_count": len(physics_changes),
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
            "physics_disambiguations": physics_changes,
            "tactical_sequence_refinements": sequence_changes,
            "flight_time_refinements": flight_changes,
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
        use_physics_refinement=args.use_physics_refinement,
        use_tactical_decoder=args.use_tactical_decoder,
        tactical_lambda=args.tactical_lambda,
        use_flight_time_rescoring=args.use_flight_time_rescoring,
        flight_time_weight=args.flight_time_weight,
        tau=args.tau,
        context_rules=args.context_rules,
        min_shuttle_visibility=args.min_shuttle_visibility,
    )


if __name__ == "__main__":
    main()

