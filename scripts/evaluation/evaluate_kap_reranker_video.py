"""Run Robust KAP + Top-2 reranker on hit detections from a video clip."""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import cv2

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.inference.aligned_video_pipeline import AlignedVideoPipeline
from scripts.inference.auto_court_detection import detect_court_corners


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--scan", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=Path("data/manifests/shuttleset_rgb.csv"))
    parser.add_argument("--match-id", required=True)
    parser.add_argument("--offset", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--reranker-checkpoint", type=Path,
        default=Path("work_dirs/kap_top2_reranker_seed20260927/best.pth"),
    )
    parser.add_argument("--specialist", type=Path, action="append", default=[])
    parser.add_argument("--tolerance", type=int, default=12)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    started = time.time()
    scan = json.loads(args.scan.read_text(encoding="utf-8"))
    events = scan["events"]
    gt = []
    with args.manifest.open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            name = Path(row["video_path"]).name
            absolute = int(row["hit_frame"])
            if name.startswith(args.match_id + " ") and args.offset <= absolute < args.offset + 9000:
                gt.append({"frame": absolute - args.offset, "stroke": row["coarse_label"]})
    gt.sort(key=lambda row: row["frame"])

    cap = cv2.VideoCapture(str(args.video))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    corners = detect_court_corners(str(args.video))
    pipeline = AlignedVideoPipeline(
        reranker_path=args.reranker_checkpoint,
        specialist_paths=args.specialist,
        corners=corners, orig_w=width, orig_h=height,
    )
    predictions = []
    for index, event in enumerate(events, 1):
        feature = pipeline.extract_features(
            cap, int(event["frame"]), total_frames, hitter_side=event["side"]
        )
        if feature is None:
            continue
        detail = pipeline.predict_detailed(feature)
        predictions.append({
            "frame": int(event["frame"]), "hitter": event["side"],
            "hit_score": float(event["score"]), "stroke": detail["stroke"],
            "confidence": detail["confidence"], "top2": detail["top2"],
            "base_stroke": detail["base_stroke"],
            "base_confidence": detail["base_confidence"],
            "reranker_swapped": detail["reranker_swapped"],
            "swap_probability": detail["swap_probability"],
            "specialist_applied": detail["specialist_applied"],
            "specialist_probability": detail["specialist_probability"],
            "crop_box": list(feature["crop_box"]),
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
            "base_top1": pred["base_stroke"] == truth["stroke"],
            "top1": pred["stroke"] == truth["stroke"],
            "top2": truth["stroke"] in pred["top2"],
        })

    matched = sum(item["prediction"] is not None for item in matches)
    top1 = sum(item["top1"] for item in matches)
    top2 = sum(item["top2"] for item in matches)
    base_top1 = sum(item.get("base_top1", False) for item in matches)
    by_class = defaultdict(lambda: {"total": 0, "matched": 0, "top1": 0, "top2": 0})
    for item in matches:
        stats = by_class[item["gt"]["stroke"]]
        stats["total"] += 1
        stats["matched"] += item["prediction"] is not None
        stats["top1"] += item["top1"]
        stats["top2"] += item["top2"]
    result = {
        "video": str(args.video.resolve()), "offset": args.offset,
        "court_corners": corners.tolist(),
        "gt_count": len(gt), "candidate_count": len(events),
        "classified_count": len(predictions), "matched_count": matched,
        "hit_recall_at_tolerance": matched / len(gt),
        "end_to_end_top1": top1 / len(gt), "end_to_end_top2": top2 / len(gt),
        "base_end_to_end_top1": base_top1 / len(gt),
        "conditional_top1": top1 / matched if matched else 0.0,
        "conditional_top2": top2 / matched if matched else 0.0,
        "base_conditional_top1": base_top1 / matched if matched else 0.0,
        "reranker_swap_count": sum(row["reranker_swapped"] for row in predictions),
        "specialist_apply_count": sum(row["specialist_applied"] is not None for row in predictions),
        "runtime_seconds": time.time() - started,
        "per_class": dict(by_class), "predictions": predictions, "matches": matches,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in result.items() if key not in {"predictions", "matches"}}, indent=2))


if __name__ == "__main__":
    main()
