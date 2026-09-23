"""Evaluate an RGB ShuttleSet hit-detector checkpoint on held-out matches."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.training.train_fine_badminton_rgb import confusion_metrics
from scripts.training.train_shuttleset_rgb_hit_detector import (
    CLASSES,
    HitDataset,
    build_samples,
    load_matches,
)
from torchvision.models.video import r2plus1d_18


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path("data/manifests/shuttleset_rgb.csv"))
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--split", choices=("val", "test"), default="test")
    parser.add_argument("--positives-per-match", type=int, default=50)
    parser.add_argument("--negative-ratio", type=float, default=2.0)
    parser.add_argument("--negative-exclusion", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260919)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    samples = build_samples(
        load_matches(args.manifest, args.split), args.positives_per_match,
        args.negative_ratio, args.negative_exclusion, 0, args.seed,
    )
    dataset = HitDataset(samples, int(checkpoint["frames"]), False, args.cache_dir)
    loader = DataLoader(dataset, batch_size=args.batch_size, num_workers=args.workers, pin_memory=True)
    model = r2plus1d_18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, len(CLASSES))
    model.load_state_dict(checkpoint["model"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()
    confusion = torch.zeros(3, 3, dtype=torch.int64)
    with torch.inference_mode():
        for clips, labels in loader:
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                predictions = model(clips.to(device)).argmax(1).cpu()
            for truth, prediction in zip(labels, predictions):
                confusion[int(truth), int(prediction)] += 1
    metrics = confusion_metrics(confusion)
    result = {
        "checkpoint": str(args.checkpoint.resolve()), "checkpoint_epoch": int(checkpoint["epoch"]),
        "split": args.split, "samples": len(samples), "positives_per_match": args.positives_per_match,
        "negative_ratio": args.negative_ratio, "classes": CLASSES, "metrics": metrics,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"samples={len(samples)} accuracy={metrics['accuracy']:.4f} macro_f1={metrics['macro_f1']:.4f} output={args.output}")


if __name__ == "__main__":
    main()
