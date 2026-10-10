"""Evaluate the canonical direct Robust KAP pipeline on an annotated video."""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import cv2
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.inference.robust_aligned_video_pipeline import AlignedVideoPipeline
from scripts.inference.auto_court_detection import detect_court_corners
from ai_classifier.localization import (
    HitEvent,
    two_stage_temporal_nms,
    rescue_pre_rally_serves,
)
from ai_classifier.models import KAPScratchModel


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--scan", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=Path("data/manifests/shuttleset_rgb.csv"))
    parser.add_argument("--match-id", required=True)
    parser.add_argument("--offset", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--kap-checkpoint", type=Path,
        default=Path("work_dirs/kap_online_balanced_selection_20261008/best.pth"),
    )
    parser.add_argument(
        "--comparison-kap-checkpoint", type=Path, default=None,
        help="Optional second KAP checkpoint evaluated on the exact same features.",
    )
    parser.add_argument(
        "--pose-model", type=Path,
        default=Path("models/checkpoints/pose/yolov8x-pose.pt"),
    )
    parser.add_argument("--tolerance", type=int, default=12)
    parser.add_argument(
        "--min-hit-score", type=float, default=0.0,
        help="Optionally filter events from an existing scan JSON by score.",
    )
    parser.add_argument(
        "--raw-threshold", type=float,
        help="Rebuild events from raw_events at this threshold instead of using stored events.",
    )
    parser.add_argument("--raw-nms-radius", type=int, default=8)
    parser.add_argument("--raw-same-side-nms-radius", type=int, default=15)
    parser.add_argument(
        "--contact-refine-radius", type=int, default=6,
        help="Search +/- this many frames for a closer shuttle/wrist contact; use 0 to disable.",
    )
    parser.add_argument(
        "--gate", type=Path, default=None,
        help="Optional path to live segments gate JSON.",
    )
    parser.add_argument(
        "--no-shared-cache", action="store_true",
        help="Disable in-memory shared-pose precomputation (legacy slow path).",
    )
    parser.add_argument(
        "--use-tta", action="store_true", default=False,
        help="Enable temporal multi-window soft-voting (TTA)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    started = time.time()
    scan = json.loads(args.scan.read_text(encoding="utf-8"))
    if args.raw_threshold is not None:
        if not scan.get("raw_events"):
            raise ValueError("--raw-threshold requires raw_events in the scan JSON")
        raw = [HitEvent(int(row["frame"]), row["side"], float(row["score"]))
               for row in scan["raw_events"]]
        base_events = two_stage_temporal_nms(
            raw, threshold=args.raw_threshold, radius=args.raw_nms_radius,
            same_side_radius=args.raw_same_side_nms_radius,
        )
        rescued = rescue_pre_rally_serves(
            base_events, raw,
            nms_radius=args.raw_nms_radius,
            same_side_radius=args.raw_same_side_nms_radius,
        )
        events = [event.__dict__ for event in rescued]
    else:
        events = [
            event for event in scan["events"]
            if float(event["score"]) >= args.min_hit_score
        ]
    if args.gate is not None:
        from ai_classifier.localization.live_gate import frame_is_live
        gate_payload = json.loads(args.gate.read_text(encoding="utf-8"))
        live_segments = gate_payload.get("live_segments", [])
        events = [e for e in events if frame_is_live(int(e["frame"]), live_segments)]
    gt = []
    with args.manifest.open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            absolute = int(row["hit_frame"])
            row_match_id = row.get("match_id", "")
            name = Path(row["video_path"]).name
            same_match = row_match_id == args.match_id or name.startswith(args.match_id + " ")
            if same_match and args.offset <= absolute < args.offset + 9000:
                gt.append({"frame": absolute - args.offset, "stroke": row["coarse_label"]})
    gt.sort(key=lambda row: row["frame"])

    cap = cv2.VideoCapture(str(args.video))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    corners = detect_court_corners(str(args.video))
    pipeline = AlignedVideoPipeline(
        ckpt_path=args.kap_checkpoint,
        corners=corners, orig_w=width, orig_h=height,
        pose_model_path=args.pose_model,
    )
    comparison_model = None
    if args.comparison_kap_checkpoint is not None:
        comparison_payload = torch.load(
            args.comparison_kap_checkpoint, map_location=pipeline.clf.stroke_head.weight.device,
            weights_only=False,
        )
        comparison_model = KAPScratchModel().to(pipeline.clf.stroke_head.weight.device)
        comparison_model.load_state_dict(comparison_payload["model"])
        comparison_model.eval()
    cache_stats = None
    if not args.no_shared_cache:
        cache_stats = pipeline.prepare_event_cache(
            cap, [int(event["frame"]) for event in events], total_frames,
            contact_refine_radius=args.contact_refine_radius,
        )
    predictions = []
    for index, event in enumerate(events, 1):
        if args.use_tta:
            feature, detail = pipeline.predict_detailed_tta(
                cap, int(event["frame"]), total_frames, hitter_side=event["side"],
                contact_refine_radius=args.contact_refine_radius,
            )
        else:
            feature = pipeline.extract_features(
                cap, int(event["frame"]), total_frames, hitter_side=event["side"],
                contact_refine_radius=args.contact_refine_radius,
            )
            detail = pipeline.predict_detailed(feature) if feature is not None else None

        if feature is None or detail is None:
            continue
        comparison = (
            pipeline.predict_detailed(feature, comparison_model)
            if comparison_model is not None else None
        )
        predictions.append({
            "frame": int(event["frame"]), "hitter": event["side"],
            "refined_frame": int(feature["hit_frame"]),
            "contact_offset": int(feature["contact_offset"]),
            "hit_score": float(event["score"]), "stroke": detail["stroke"],
            "confidence": detail["confidence"], "top2": detail["top2"],
            "crop_box": list(feature["crop_box"]),
            "comparison_stroke": comparison["stroke"] if comparison else None,
            "comparison_confidence": comparison["confidence"] if comparison else None,
            "comparison_top2": comparison["top2"] if comparison else None,
        })
        print(f"[{index}/{len(events)}] frame={event['frame']} {detail['stroke']} {detail['confidence']:.3f}", flush=True)
    cap.release()

    used = set()
    matches = []
    for truth in gt:
        choices = [
            (abs(pred["frame"] - truth["frame"]), idx, pred)
            for idx, pred in enumerate(predictions)
            if idx not in used and abs(pred["frame"] - truth["frame"]) <= args.tolerance
        ]
        if not choices:
            matches.append({"gt": truth, "prediction": None, "top1": False, "top2": False})
            continue
        distance, idx, pred = min(choices)
        used.add(idx)
        matches.append({
            "gt": truth, "prediction": pred, "frame_error": distance,
            "top1": pred["stroke"] == truth["stroke"],
            "top2": truth["stroke"] in pred["top2"],
            "comparison_top1": (
                pred.get("comparison_stroke") == truth["stroke"]
                if comparison_model is not None else False
            ),
            "comparison_top2": (
                truth["stroke"] in (pred.get("comparison_top2") or [])
                if comparison_model is not None else False
            ),
        })

    matched = sum(item["prediction"] is not None for item in matches)
    top1 = sum(item["top1"] for item in matches)
    top2 = sum(item["top2"] for item in matches)
    comparison_top1 = sum(item.get("comparison_top1", False) for item in matches)
    comparison_top2 = sum(item.get("comparison_top2", False) for item in matches)
    by_class = defaultdict(lambda: {"total": 0, "matched": 0, "top1": 0, "top2": 0})
    for item in matches:
        stats = by_class[item["gt"]["stroke"]]
        stats["total"] += 1
        stats["matched"] += item["prediction"] is not None
        stats["top1"] += item["top1"]
        stats["top2"] += item["top2"]
    result = {
        "video": str(args.video.resolve()), "offset": args.offset,
        "minimum_hit_score": args.min_hit_score,
        "raw_threshold": args.raw_threshold,
        "raw_nms_radius": args.raw_nms_radius if args.raw_threshold is not None else None,
        "raw_same_side_nms_radius": (
            args.raw_same_side_nms_radius if args.raw_threshold is not None else None
        ),
        "contact_refine_radius": args.contact_refine_radius,
        "contact_refined_count": sum(row["contact_offset"] != 0 for row in predictions),
        "gate": str(args.gate.resolve()) if args.gate is not None else None,
        "shared_cache": cache_stats,
        "court_corners": corners.tolist(),
        "gt_count": len(gt), "candidate_count": len(events),
        "classified_count": len(predictions), "matched_count": matched,
        "hit_recall_at_tolerance": matched / len(gt),
        "end_to_end_top1": top1 / len(gt), "end_to_end_top2": top2 / len(gt),
        "conditional_top1": top1 / matched if matched else 0.0,
        "conditional_top2": top2 / matched if matched else 0.0,
        "comparison_checkpoint": (
            str(args.comparison_kap_checkpoint.resolve())
            if args.comparison_kap_checkpoint is not None else None
        ),
        "comparison_end_to_end_top1": (
            comparison_top1 / len(gt) if comparison_model is not None else None
        ),
        "comparison_end_to_end_top2": (
            comparison_top2 / len(gt) if comparison_model is not None else None
        ),
        "comparison_conditional_top1": (
            comparison_top1 / matched if comparison_model is not None and matched else None
        ),
        "comparison_conditional_top2": (
            comparison_top2 / matched if comparison_model is not None and matched else None
        ),
        "runtime_seconds": time.time() - started,
        "per_class": dict(by_class), "predictions": predictions, "matches": matches,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in result.items() if key not in {"predictions", "matches"}}, indent=2))


if __name__ == "__main__":
    main()
