"""Diagnose epoch-20 drive recognition using ShuttleSet ground-truth hits."""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from collections import Counter, defaultdict
from dataclasses import replace
from pathlib import Path

import torch
from ultralytics import YOLO

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.inference.classify_shuttleset_rgb_multitask import constrain_side_probabilities
from scripts.training.train_shuttleset_rgb_multitask import (
    SIDE_CLASSES,
    STROKE_CLASSES,
    MultiTaskR2Plus1D,
    decode_clip,
    detect_hitter_crop,
    load_records,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path("data/manifests/shuttleset_rgb.csv"))
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--radii", type=int, nargs="+", default=[12, 24, 30])
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument(
        "--samples-per-raw-label", type=int,
        help="Deterministically sample this many records from each original drive subtype.",
    )
    parser.add_argument("--seed", type=int, default=20260920)
    parser.add_argument(
        "--pose-model", type=Path, default=Path("models/checkpoints/pose/yolov8n-pose.pt")
    )
    return parser.parse_args()


def manifest_metadata(path: Path, split: str) -> dict[str, dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return {
            row["sample_id"]: row
            for row in csv.DictReader(handle)
            if row["split"] == split and row["coarse_label"] == "drive"
        }


def normalize(tensor: torch.Tensor) -> torch.Tensor:
    tensor = tensor.to(torch.float32).div_(255.0)
    mean = tensor.new_tensor([0.43216, 0.394666, 0.37645]).view(1, 3, 1, 1)
    std = tensor.new_tensor([0.22803, 0.22145, 0.216989]).view(1, 3, 1, 1)
    return ((tensor - mean) / std).permute(1, 0, 2, 3)


def summarize(rows: list[dict[str, object]], key: str | None = None) -> dict[str, object]:
    groups: dict[str, list[dict[str, object]]] = defaultdict(list)
    if key is None:
        groups["all"] = rows
    else:
        for row in rows:
            groups[str(row[key])].append(row)
    output = {}
    for name, items in sorted(groups.items()):
        total = len(items)
        stroke_correct = sum(row["predicted_stroke"] == "drive" for row in items)
        side_correct = sum(row["predicted_side"] == row["expected_side"] for row in items)
        joint_correct = sum(
            row["predicted_stroke"] == "drive"
            and row["predicted_side"] == row["expected_side"]
            for row in items
        )
        output[name] = {
            "samples": total,
            "drive_recall": stroke_correct / total if total else 0.0,
            "side_accuracy": side_correct / total if total else 0.0,
            "joint_accuracy": joint_correct / total if total else 0.0,
            "predicted_strokes": dict(Counter(row["predicted_stroke"] for row in items)),
        }
    return output


def main() -> None:
    args = parse_args()
    if any(radius <= 0 for radius in args.radii):
        raise ValueError("All radii must be positive")
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    records = [
        record for record in load_records(args.manifest, args.split)
        if record.coarse_label == "drive"
    ]
    metadata = manifest_metadata(args.manifest, args.split)
    if args.samples_per_raw_label is not None:
        if args.samples_per_raw_label <= 0:
            raise ValueError("samples-per-raw-label must be positive")
        grouped = defaultdict(list)
        for record in records:
            grouped[metadata[record.sample_id]["raw_label"]].append(record)
        rng = random.Random(args.seed)
        records = [
            record
            for raw_label in sorted(grouped)
            for record in rng.sample(
                sorted(grouped[raw_label], key=lambda item: item.sample_id),
                min(args.samples_per_raw_label, len(grouped[raw_label])),
            )
        ]
    if not records:
        raise ValueError(f"No drive records in split={args.split}")

    pose_model = YOLO(str(args.pose_model))
    crop_padding = float(checkpoint.get("crop_padding", 0.55))
    crop_boxes = {}
    for index, record in enumerate(records, 1):
        crop_boxes[record.sample_id] = detect_hitter_crop(record, pose_model, crop_padding)
        if index % 25 == 0 or index == len(records):
            print(f"crop_progress={index}/{len(records)}", flush=True)

    model = MultiTaskR2Plus1D()
    model.load_state_dict(checkpoint["model"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()
    frame_count = int(checkpoint.get("frames", 16))
    results = {}

    for radius in args.radii:
        rows = []
        pending_tensors = []
        pending_records = []

        def flush() -> None:
            if not pending_tensors:
                return
            clips = torch.stack(pending_tensors).to(device)
            with torch.inference_mode(), torch.autocast(
                device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"
            ):
                stroke_logits, side_logits = model(clips)
            stroke_probs = torch.softmax(stroke_logits, dim=1).cpu()
            side_probs = torch.softmax(side_logits, dim=1).cpu()
            for record, sp, hp in zip(pending_records, stroke_probs, side_probs):
                stroke_id = int(sp.argmax())
                constrained, _ = constrain_side_probabilities(
                    hp, SIDE_CLASSES, STROKE_CLASSES[stroke_id]
                )
                side_id = int(constrained.argmax())
                item = metadata[record.sample_id]
                rows.append({
                    "sample_id": record.sample_id,
                    "raw_label": item["raw_label"],
                    "player_side": record.player_side,
                    "expected_side": record.stroke_side,
                    "predicted_stroke": STROKE_CLASSES[stroke_id],
                    "stroke_probability": float(sp[stroke_id]),
                    "drive_probability": float(sp[STROKE_CLASSES.index("drive")]),
                    "predicted_side": SIDE_CLASSES[side_id],
                    "side_probability": float(constrained[side_id]),
                })
            pending_tensors.clear()
            pending_records.clear()

        for index, record in enumerate(records, 1):
            window = replace(
                record,
                start_frame=max(0, record.hit_frame - radius),
                end_frame=record.hit_frame + radius,
            )
            tensor = decode_clip(window, frame_count, crop_boxes[record.sample_id])
            pending_tensors.append(normalize(tensor))
            pending_records.append(record)
            if len(pending_tensors) >= args.batch_size:
                flush()
            if index % 50 == 0:
                print(f"radius={radius} inference_progress={index}/{len(records)}", flush=True)
        flush()
        results[str(radius)] = {
            "overall": summarize(rows)["all"],
            "by_stroke_side": summarize(rows, "expected_side"),
            "by_player_side": summarize(rows, "player_side"),
            "by_raw_label": summarize(rows, "raw_label"),
            "predictions": rows,
        }
        overall = results[str(radius)]["overall"]
        print(
            f"radius={radius} drive_recall={overall['drive_recall']:.4f} "
            f"side_accuracy={overall['side_accuracy']:.4f} "
            f"joint_accuracy={overall['joint_accuracy']:.4f}",
            flush=True,
        )

    report = {
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_epoch": checkpoint.get("epoch"),
        "split": args.split,
        "ground_truth_hit_and_player": True,
        "drive_samples": len(records),
        "samples_per_raw_label": args.samples_per_raw_label,
        "radii_frames": args.radii,
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"[DONE] output={args.output}")


if __name__ == "__main__":
    main()
