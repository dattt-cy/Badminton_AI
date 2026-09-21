"""Evaluate motion proposals filtered by an RGB hit-detector checkpoint."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision.models.video import r2plus1d_18

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ai_classifier.localization import HitEvent, match_hit_events, temporal_nms
from scripts.training.train_shuttleset_rgb_hit_detector import HitDataset, Sample


SIDE_MAP = {"top": "upper", "upper": "upper", "bottom": "lower", "lower": "lower"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path("data/manifests/shuttleset_rgb.csv"))
    parser.add_argument("--proposal-report", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--split", choices=("val", "test"), default="val")
    parser.add_argument(
        "--thresholds", type=float, nargs="+",
        default=[0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.97, 0.99],
    )
    parser.add_argument("--nms-radius", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def load_truth(path: Path, split: str, match_ids: set[str]) -> tuple[dict[str, Path], dict[str, list[HitEvent]]]:
    videos: dict[str, Path] = {}
    truth: dict[str, list[HitEvent]] = defaultdict(list)
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            match_id = row["match_id"]
            if row["split"] != split or match_id not in match_ids:
                continue
            side = SIDE_MAP.get(row["player_side"].strip().lower())
            if side is None:
                continue
            videos[match_id] = Path(row["video_path"])
            truth[match_id].append(HitEvent(int(float(row["hit_frame"])), side, 1.0))
    return videos, {key: sorted(set(value), key=lambda event: event.frame) for key, value in truth.items()}


def aggregate(scored: dict[str, list[HitEvent]], truth: dict[str, list[HitEvent]], threshold: float, radius: int, tolerance: int) -> dict:
    totals = defaultdict(int)
    errors = []
    per_match = {}
    for match_id in sorted(truth, key=int):
        predicted = temporal_nms(scored[match_id], threshold=threshold, radius=radius)
        metric = match_hit_events(predicted, truth[match_id], tolerance=tolerance)
        per_match[match_id] = {key: value for key, value in metric.items() if key != "matches"}
        for key in ("truth", "predicted", "true_positives", "false_positives", "false_negatives", "joint_true_positives"):
            totals[key] += metric[key]
        errors.extend(abs(pair["frame_error"]) for pair in metric["matches"])
    tp, count, expected = totals["true_positives"], totals["predicted"], totals["truth"]
    precision, recall = (tp / count if count else 0.0), (tp / expected if expected else 0.0)
    side_correct = totals["joint_true_positives"]
    return {
        **{key: int(value) for key, value in totals.items()},
        "precision": precision, "recall": recall,
        "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
        "mean_absolute_frame_error": sum(errors) / len(errors) if errors else None,
        "side_accuracy": side_correct / tp if tp else None,
        "joint_precision": side_correct / count if count else 0.0,
        "joint_recall": side_correct / expected if expected else 0.0,
        "per_match": per_match,
    }


def main() -> None:
    args = parse_args()
    proposal_report = json.loads(args.proposal_report.read_text(encoding="utf-8"))
    raw_predictions = proposal_report["predictions"]
    match_ids = set(raw_predictions)
    videos, truth = load_truth(args.manifest, args.split, match_ids)
    if set(truth) != match_ids:
        raise ValueError(f"Proposal matches do not match {args.split} manifest")
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    samples, identities = [], []
    for match_id, events in raw_predictions.items():
        for event in events:
            samples.append(Sample(match_id, videos[match_id], int(event["frame"]), 0))
            identities.append((match_id, int(event["frame"])))
    loader = DataLoader(HitDataset(samples, int(checkpoint["frames"]), False, args.cache_dir), batch_size=args.batch_size, num_workers=0, pin_memory=True)
    model = r2plus1d_18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 3)
    model.load_state_dict(checkpoint["model"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()
    scored: dict[str, list[HitEvent]] = defaultdict(list)
    offset = 0
    with torch.inference_mode():
        for clips, _labels in loader:
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                probabilities = torch.softmax(model(clips.to(device)), dim=1).cpu()
            for probability in probabilities:
                match_id, frame = identities[offset]
                upper, lower = float(probability[1]), float(probability[2])
                scored[match_id].append(HitEvent(frame, "upper" if upper >= lower else "lower", upper + lower))
                offset += 1
    selection = []
    for threshold in args.thresholds:
        metric = aggregate(scored, truth, threshold, args.nms_radius, 5)
        selection.append({"threshold": threshold, "precision": metric["precision"], "recall": metric["recall"], "f1": metric["f1"]})
    selected = max(selection, key=lambda row: row["f1"])["threshold"]
    result = {
        "split": args.split, "match_ids": sorted(match_ids, key=int),
        "proposal_report": str(args.proposal_report.resolve()), "checkpoint": str(args.checkpoint.resolve()),
        "candidates": len(samples), "threshold_selection": selection, "selected_threshold": selected,
        "metrics_by_tolerance": {str(t): aggregate(scored, truth, selected, args.nms_radius, t) for t in (2, 5, 15)},
        "predictions": {
            match_id: [event.__dict__ for event in temporal_nms(events, threshold=selected, radius=args.nms_radius)]
            for match_id, events in scored.items()
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    metric = result["metrics_by_tolerance"]["5"]
    print(f"candidates={len(samples)} threshold={selected:.2f} precision@5={metric['precision']:.4f} recall@5={metric['recall']:.4f} f1@5={metric['f1']:.4f} output={args.output}")


if __name__ == "__main__":
    main()
