"""Scan ShuttleSet videos and evaluate full-video hit spotting."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict, deque
from pathlib import Path

import cv2
import joblib
import numpy as np

from ai_classifier.localization import HitEvent, match_hit_events, temporal_nms


SIDE_LABELS = {1: "upper", 2: "lower"}
SOURCE_SIDES = {"top": "upper", "upper": "upper", "bottom": "lower", "lower": "lower"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path("data/manifests/shuttleset_rgb.csv"))
    parser.add_argument("--model", type=Path, default=Path("work_dirs/shuttleset_hit_detector_motion_pilot/hit_detector.joblib"))
    parser.add_argument("--model-metrics", type=Path, default=Path("work_dirs/shuttleset_hit_detector_motion_pilot/metrics.json"))
    parser.add_argument("--split", choices=("train", "val", "test"), default="val")
    parser.add_argument("--match-ids", nargs="*")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--threshold", type=float, help="Fixed hit threshold; otherwise tune on this split.")
    parser.add_argument(
        "--thresholds", type=float, nargs="+",
        default=[0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95],
    )
    parser.add_argument("--nms-radius", type=int, default=8)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--max-frames", type=int, help="Diagnostic limit per video.")
    return parser.parse_args()


def load_ground_truth(path: Path, split: str, match_ids: set[str] | None) -> dict[str, dict]:
    matches: dict[str, dict] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            match_id = row["match_id"]
            if row["split"] != split or (match_ids is not None and match_id not in match_ids):
                continue
            side = SOURCE_SIDES.get(row["player_side"].strip().lower())
            if side is None:
                continue
            item = matches.setdefault(match_id, {"video_path": Path(row["video_path"]), "truth": []})
            item["truth"].append(HitEvent(int(float(row["hit_frame"])), side, 1.0))
    for item in matches.values():
        item["truth"] = sorted(set(item["truth"]), key=lambda event: event.frame)
    return matches


def grid_stats(image: np.ndarray, rows: int, cols: int) -> list[float]:
    values: list[float] = []
    for row in np.array_split(image, rows, axis=0):
        for cell in np.array_split(row, cols, axis=1):
            values.extend((float(cell.mean()), float(cell.std()), float(np.quantile(cell, 0.9))))
    return values


def feature_from_frames(before: np.ndarray, middle: np.ndarray, after: np.ndarray, rows: int, cols: int) -> np.ndarray:
    pre_motion = np.abs(middle - before)
    post_motion = np.abs(after - middle)
    span_motion = np.abs(after - before)
    values: list[float] = []
    for motion in (pre_motion, post_motion, span_motion):
        values.extend(grid_stats(motion, rows, cols))
    height = middle.shape[0]
    top, bottom = span_motion[: height // 2], span_motion[height // 2 :]
    values.extend([
        float(pre_motion.mean()), float(post_motion.mean()), float(span_motion.mean()),
        float(top.mean()), float(bottom.mean()), float(top.mean() - bottom.mean()),
        float(middle.mean()), float(middle.std()),
    ])
    return np.asarray(values, dtype=np.float32)


def scan_video(video_path: Path, model, config: dict, *, stride: int, batch_size: int, max_frames: int | None) -> list[HitEvent]:
    radius = int(config["frame_radius"])
    width, height = int(config["width"]), int(config["height"])
    rows, cols = int(config["grid_rows"]), int(config["grid_cols"])
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise FileNotFoundError(f"Cannot open {video_path}")
    window: deque[tuple[int, np.ndarray]] = deque(maxlen=2 * radius + 1)
    batch_features: list[np.ndarray] = []
    batch_frames: list[int] = []
    events: list[HitEvent] = []

    def flush() -> None:
        if not batch_features:
            return
        probabilities = model.predict_proba(np.stack(batch_features))
        class_indices = {int(label): index for index, label in enumerate(model.classes_)}
        upper_index, lower_index = class_indices[1], class_indices[2]
        for frame, probability in zip(batch_frames, probabilities):
            upper, lower = float(probability[upper_index]), float(probability[lower_index])
            events.append(HitEvent(frame, "upper" if upper >= lower else "lower", upper + lower))
        batch_features.clear()
        batch_frames.clear()

    frame_index = 0
    try:
        while max_frames is None or frame_index < max_frames:
            ok, frame = capture.read()
            if not ok:
                break
            gray = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (width, height), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
            window.append((frame_index, gray))
            if len(window) == window.maxlen:
                center_frame = window[radius][0]
                if center_frame % stride == 0:
                    batch_features.append(feature_from_frames(window[0][1], window[radius][1], window[-1][1], rows, cols))
                    batch_frames.append(center_frame)
                    if len(batch_features) >= batch_size:
                        flush()
            frame_index += 1
    finally:
        capture.release()
    flush()
    return events


def aggregate(matches: dict[str, dict], scored: dict[str, list[HitEvent]], threshold: float, radius: int, tolerance: int) -> dict:
    totals = defaultdict(float)
    all_errors: list[int] = []
    per_match = {}
    for match_id, item in matches.items():
        predicted = temporal_nms(scored[match_id], threshold=threshold, radius=radius)
        result = match_hit_events(predicted, item["truth"], tolerance=tolerance)
        per_match[match_id] = {key: value for key, value in result.items() if key != "matches"}
        for key in ("truth", "predicted", "true_positives", "false_positives", "false_negatives", "joint_true_positives"):
            totals[key] += result[key]
        all_errors.extend(abs(match["frame_error"]) for match in result["matches"])
    tp, predicted_count, truth_count = totals["true_positives"], totals["predicted"], totals["truth"]
    precision = tp / predicted_count if predicted_count else 0.0
    recall = tp / truth_count if truth_count else 0.0
    side_correct = totals["joint_true_positives"]
    return {
        "truth": int(truth_count), "predicted": int(predicted_count), "true_positives": int(tp),
        "false_positives": int(totals["false_positives"]), "false_negatives": int(totals["false_negatives"]),
        "precision": precision, "recall": recall,
        "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
        "mean_absolute_frame_error": float(np.mean(all_errors)) if all_errors else None,
        "median_absolute_frame_error": float(np.median(all_errors)) if all_errors else None,
        "side_accuracy": side_correct / tp if tp else None,
        "joint_precision": side_correct / predicted_count if predicted_count else 0.0,
        "joint_recall": side_correct / truth_count if truth_count else 0.0,
        "per_match": per_match,
    }


def main() -> None:
    args = parse_args()
    if args.stride <= 0 or args.batch_size <= 0:
        raise ValueError("stride and batch-size must be positive")
    matches = load_ground_truth(args.manifest, args.split, set(args.match_ids) if args.match_ids else None)
    if not matches:
        raise ValueError("No matching ShuttleSet rows found")
    if args.max_frames is not None:
        for item in matches.values():
            item["truth"] = [event for event in item["truth"] if event.frame < args.max_frames]
    model = joblib.load(args.model)
    artifact = json.loads(args.model_metrics.read_text(encoding="utf-8"))
    config = artifact["feature_config"]
    scored: dict[str, list[HitEvent]] = {}
    for match_id, item in sorted(matches.items(), key=lambda pair: int(pair[0])):
        scored[match_id] = scan_video(
            item["video_path"], model, config, stride=args.stride,
            batch_size=args.batch_size, max_frames=args.max_frames,
        )
        print(f"match={match_id} scored_frames={len(scored[match_id])} truth_hits={len(item['truth'])}", flush=True)

    thresholds = [args.threshold] if args.threshold is not None else args.thresholds
    selection = []
    for threshold in thresholds:
        metric = aggregate(matches, scored, threshold, args.nms_radius, tolerance=5)
        selection.append({"threshold": threshold, "precision": metric["precision"], "recall": metric["recall"], "f1": metric["f1"]})
    best_threshold = max(selection, key=lambda row: row["f1"])["threshold"]
    metrics = {
        str(tolerance): aggregate(matches, scored, best_threshold, args.nms_radius, tolerance)
        for tolerance in (2, 5, 15)
    }
    predictions = {
        match_id: [event.__dict__ for event in temporal_nms(events, threshold=best_threshold, radius=args.nms_radius)]
        for match_id, events in scored.items()
    }
    result = {
        "split": args.split, "match_ids": sorted(matches, key=int),
        "model": str(args.model.resolve()), "feature_config": config,
        "stride": args.stride, "nms_radius": args.nms_radius,
        "threshold_selection_tolerance": 5, "threshold_selection": selection,
        "selected_threshold": best_threshold, "metrics_by_tolerance": metrics,
        "predictions": predictions,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    selected = metrics["5"]
    print(f"threshold={best_threshold:.2f} precision@5={selected['precision']:.4f} recall@5={selected['recall']:.4f} f1@5={selected['f1']:.4f} output={args.output}")


if __name__ == "__main__":
    main()
