"""Evaluate predicted hit boundaries with the ShuttleSet stroke classifier."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from ultralytics import YOLO

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ai_classifier.localization import HitEvent, match_hit_events
from scripts.inference.classify_shuttleset_rgb_multitask import constrain_side_probabilities
from scripts.training.train_fine_badminton_rgb import confusion_metrics
from scripts.training.train_shuttleset_rgb_multitask import (
    SIDE_CLASSES,
    STROKE_CLASSES,
    MultiTaskR2Plus1D,
    Record,
    ShuttleSetRGBDataset,
)


PLAYER_SIDE = {"top": "upper", "upper": "upper", "bottom": "lower", "lower": "lower"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path("data/manifests/shuttleset_rgb.csv"))
    parser.add_argument("--cascade-report", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--tolerance", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--pose-model", type=Path, default=Path("models/checkpoints/pose/yolov8n-pose.pt"))
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def load_rows(path: Path, split: str, match_ids: set[str]) -> dict[str, list[dict]]:
    rows: dict[str, list[dict]] = defaultdict(list)
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            if row["split"] == split and row["match_id"] in match_ids:
                rows[row["match_id"]].append(row)
    return dict(rows)


def main() -> None:
    args = parse_args()
    cascade = json.loads(args.cascade_report.read_text(encoding="utf-8"))
    split = cascade["split"]
    predictions = cascade["predictions"]
    rows = load_rows(args.manifest, split, set(predictions))
    records, metadata = [], []
    localization = {}
    for match_id, predicted_rows in predictions.items():
        truth_rows = sorted(rows[match_id], key=lambda row: int(float(row["hit_frame"])))
        predicted_events = [HitEvent(int(item["frame"]), item["side"], float(item["score"])) for item in predicted_rows]
        truth_events = [HitEvent(int(float(row["hit_frame"])), PLAYER_SIDE[row["player_side"].lower()], 1.0) for row in truth_rows]
        matched = match_hit_events(predicted_events, truth_events, tolerance=args.tolerance)
        localization[match_id] = {key: value for key, value in matched.items() if key != "matches"}
        for index, pair in enumerate(matched["matches"]):
            candidates = [
                row for row in truth_rows
                if int(float(row["hit_frame"])) == int(pair["truth_frame"])
                and PLAYER_SIDE[row["player_side"].lower()] == pair["truth_side"]
            ]
            if not candidates:
                raise ValueError(f"Missing matched truth row: {match_id=} {pair=}")
            truth = candidates[0]
            predicted_side = "top" if pair["predicted_side"] == "upper" else "bottom"
            center = int(pair["predicted_frame"])
            records.append(Record(
                sample_id=f"pred_{match_id}_{center}_{index}", video_path=Path(truth["video_path"]),
                start_frame=max(0, center - 30), end_frame=center + 30,
                coarse_label=truth["coarse_label"], stroke_side=truth["stroke_side"],
                player_side=predicted_side, hit_frame=center,
            ))
            metadata.append({**pair, "expected_stroke": truth["coarse_label"], "expected_stroke_side": truth["stroke_side"]})
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    pose_model = YOLO(str(args.pose_model))
    dataset = ShuttleSetRGBDataset(
        records, int(checkpoint.get("frames", 16)), False, crop_hitter=True,
        pose_model=pose_model, cache_dir=args.cache_dir,
        crop_padding=float(checkpoint.get("crop_padding", 0.55)),
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, num_workers=0)
    model = MultiTaskR2Plus1D()
    model.load_state_dict(checkpoint["model"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()
    stroke_confusion = torch.zeros(8, 8, dtype=torch.int64)
    side_confusion = torch.zeros(3, 3, dtype=torch.int64)
    results, offset = [], 0
    with torch.inference_mode():
        for clips, stroke_truth, side_truth, sample_ids in loader:
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                stroke_logits, side_logits = model(clips.to(device))
            stroke_probs = torch.softmax(stroke_logits, 1).cpu()
            side_probs = torch.softmax(side_logits, 1).cpu()
            for sample_id, expected_stroke, expected_side, stroke_p, side_p in zip(sample_ids, stroke_truth, side_truth, stroke_probs, side_probs):
                stroke_id = int(stroke_p.argmax())
                constrained, _ = constrain_side_probabilities(side_p, SIDE_CLASSES, STROKE_CLASSES[stroke_id])
                side_id = int(constrained.argmax())
                stroke_confusion[int(expected_stroke), stroke_id] += 1
                side_confusion[int(expected_side), side_id] += 1
                row = dict(metadata[offset])
                row.update({
                    "sample_id": sample_id, "predicted_stroke": STROKE_CLASSES[stroke_id],
                    "predicted_stroke_side": SIDE_CLASSES[side_id],
                    "stroke_correct": STROKE_CLASSES[stroke_id] == row["expected_stroke"],
                    "stroke_side_correct": SIDE_CLASSES[side_id] == row["expected_stroke_side"],
                })
                results.append(row)
                offset += 1
    stroke_metrics = confusion_metrics(stroke_confusion)
    stroke_side_metrics = confusion_metrics(side_confusion)
    count = len(results)
    correct_player = sum(row["side_correct"] for row in results)
    correct_stroke = sum(row["stroke_correct"] for row in results)
    correct_stroke_side = sum(row["stroke_side_correct"] for row in results)
    correct_all = sum(row["side_correct"] and row["stroke_correct"] and row["stroke_side_correct"] for row in results)
    total_truth = sum(item["truth"] for item in localization.values())
    total_predictions = sum(item["predicted"] for item in localization.values())
    output = {
        "split": split, "tolerance": args.tolerance, "matched_events": count,
        "total_truth_events": total_truth, "total_predicted_events": total_predictions,
        "localization": localization, "stroke_metrics_on_matched": stroke_metrics,
        "stroke_side_metrics_on_matched": stroke_side_metrics,
        "player_accuracy_on_matched": correct_player / count if count else 0.0,
        "stroke_accuracy_on_matched": correct_stroke / count if count else 0.0,
        "stroke_side_accuracy_on_matched": correct_stroke_side / count if count else 0.0,
        "joint_accuracy_on_matched": correct_all / count if count else 0.0,
        "end_to_end_joint_recall": correct_all / total_truth if total_truth else 0.0,
        "end_to_end_joint_precision": correct_all / total_predictions if total_predictions else 0.0,
        "events": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(f"matched={count} stroke_accuracy={output['stroke_accuracy_on_matched']:.4f} joint_matched={output['joint_accuracy_on_matched']:.4f} end_to_end_joint_recall={output['end_to_end_joint_recall']:.4f} output={args.output}")


if __name__ == "__main__":
    main()
