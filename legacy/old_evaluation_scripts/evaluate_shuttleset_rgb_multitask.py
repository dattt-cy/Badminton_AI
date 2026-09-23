"""Evaluate a multitask RGB checkpoint on a balanced ShuttleSet split."""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from ultralytics import YOLO

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.inference.classify_shuttleset_rgb_multitask import constrain_side_probabilities
from scripts.training.train_fine_badminton_rgb import confusion_metrics
from scripts.training.train_shuttleset_rgb_multitask import (
    SIDE_CLASSES,
    STROKE_CLASSES,
    MultiTaskR2Plus1D,
    ShuttleSetRGBDataset,
    load_records,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path("data/manifests/shuttleset_rgb.csv"))
    parser.add_argument("--split", default="test")
    parser.add_argument(
        "--match-id",
        help="Evaluate only one match_id from the selected split.",
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--samples-per-stroke", type=int, default=25)
    parser.add_argument(
        "--all-samples",
        action="store_true",
        help="Evaluate every selected record instead of a balanced subset.",
    )
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument(
        "--pose-model", type=Path, default=Path("models/checkpoints/pose/yolov8n-pose.pt")
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260917)
    return parser.parse_args()


def balanced_records(records, count: int, seed: int):
    rng = random.Random(seed)
    by_stroke_match = defaultdict(lambda: defaultdict(list))
    for record in records:
        match_id = record.sample_id.split("_", 1)[0]
        by_stroke_match[record.coarse_label][match_id].append(record)
    selected = []
    for stroke in STROKE_CLASSES:
        buckets = list(by_stroke_match[stroke].values())
        for bucket in buckets:
            rng.shuffle(bucket)
        while buckets and sum(item.coarse_label == stroke for item in selected) < count:
            remaining = []
            for bucket in buckets:
                if sum(item.coarse_label == stroke for item in selected) >= count:
                    break
                selected.append(bucket.pop())
                if bucket:
                    remaining.append(bucket)
            buckets = remaining
    return selected


def main() -> None:
    args = parse_args()
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    records = load_records(args.manifest, args.split)
    if args.match_id is not None:
        records = [
            record for record in records
            if record.sample_id.split("_", 1)[0] == str(args.match_id)
        ]
    if not records:
        raise ValueError(
            f"No records found for split={args.split!r}, match_id={args.match_id!r}"
        )
    if not args.all_samples:
        records = balanced_records(records, args.samples_per_stroke, args.seed)
    pose_model = YOLO(str(args.pose_model))
    dataset = ShuttleSetRGBDataset(
        records, int(checkpoint.get("frames", 16)), training=False,
        crop_hitter=True, pose_model=pose_model, cache_dir=args.cache_dir,
        crop_padding=float(checkpoint.get("crop_padding", 0.55)),
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    model = MultiTaskR2Plus1D()
    model.load_state_dict(checkpoint["model"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()

    stroke_confusion = torch.zeros(8, 8, dtype=torch.int64)
    side_confusion = torch.zeros(3, 3, dtype=torch.int64)
    constrained_side_confusion = torch.zeros(3, 3, dtype=torch.int64)
    rows = []
    with torch.inference_mode():
        for clips, stroke_truth, side_truth, sample_ids in loader:
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                stroke_logits, side_logits = model(clips.to(device))
            stroke_probs = torch.softmax(stroke_logits, dim=1).cpu()
            side_probs = torch.softmax(side_logits, dim=1).cpu()
            for sample_id, expected_stroke, expected_side, sp, hp in zip(
                sample_ids, stroke_truth, side_truth, stroke_probs, side_probs
            ):
                stroke_prediction = int(sp.argmax())
                side_prediction = int(hp.argmax())
                constrained, allowed = constrain_side_probabilities(
                    hp, SIDE_CLASSES, STROKE_CLASSES[stroke_prediction]
                )
                constrained_prediction = int(constrained.argmax())
                stroke_confusion[int(expected_stroke), stroke_prediction] += 1
                side_confusion[int(expected_side), side_prediction] += 1
                constrained_side_confusion[int(expected_side), constrained_prediction] += 1
                rows.append({
                    "sample_id": sample_id,
                    "expected_stroke": STROKE_CLASSES[int(expected_stroke)],
                    "predicted_stroke": STROKE_CLASSES[stroke_prediction],
                    "stroke_probability": float(sp[stroke_prediction]),
                    "expected_side": SIDE_CLASSES[int(expected_side)],
                    "predicted_side_raw": SIDE_CLASSES[side_prediction],
                    "predicted_side": SIDE_CLASSES[constrained_prediction],
                    "side_probability": float(constrained[constrained_prediction]),
                    "allowed_sides": allowed,
                })

    stroke_metrics = confusion_metrics(stroke_confusion)
    side_metrics = confusion_metrics(side_confusion)
    constrained_side_metrics = confusion_metrics(constrained_side_confusion)
    combined_accuracy = sum(
        row["expected_stroke"] == row["predicted_stroke"]
        and row["expected_side"] == row["predicted_side"]
        for row in rows
    ) / len(rows)
    result = {
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_epoch": int(checkpoint["epoch"]),
        "device": str(device),
        "split": args.split,
        "match_id": args.match_id,
        "samples": len(rows),
        "samples_per_stroke": None if args.all_samples else args.samples_per_stroke,
        "stroke_metrics": stroke_metrics,
        "side_metrics_raw": side_metrics,
        "side_metrics_constrained": constrained_side_metrics,
        "combined_accuracy_constrained": combined_accuracy,
        "classes": {"stroke": STROKE_CLASSES, "side": SIDE_CLASSES},
        "predictions": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(
        f"samples={len(rows)} stroke_accuracy={stroke_metrics['accuracy']:.4f} "
        f"stroke_macro_f1={stroke_metrics['macro_f1']:.4f} "
        f"side_accuracy_raw={side_metrics['accuracy']:.4f} "
        f"side_accuracy_constrained={constrained_side_metrics['accuracy']:.4f} "
        f"combined_accuracy={combined_accuracy:.4f} output={args.output}"
    )


if __name__ == "__main__":
    main()
