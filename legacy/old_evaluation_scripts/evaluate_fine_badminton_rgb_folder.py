"""Evaluate an RGB Fine-Badminton checkpoint on a label-named video folder."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import torch
from torch import nn
from torchvision.models.video import r2plus1d_18

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.inference.classify_fine_badminton_rgb import read_clip


VIDEO_SUFFIXES = {".avi", ".mkv", ".mov", ".mp4", ".webm"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video_dir", type=Path)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--expected-label", required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.batch_size <= 0:
        raise ValueError("batch-size must be positive")
    videos = sorted(
        path
        for path in args.video_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in VIDEO_SUFFIXES
    )
    if not videos:
        raise ValueError(f"No videos found in {args.video_dir}")

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    classes = [str(item) for item in checkpoint["classes"]]
    if args.expected_label not in classes:
        raise ValueError(f"Unknown expected label: {args.expected_label}")
    frame_count = int(checkpoint.get("frames", 16))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = r2plus1d_18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, len(classes))
    model.load_state_dict(checkpoint["model"])
    model.to(device).eval()

    results: list[dict[str, object]] = []
    for offset in range(0, len(videos), args.batch_size):
        paths = videos[offset : offset + args.batch_size]
        loaded = [(path, *read_clip(path, frame_count)) for path in paths]
        clips = torch.cat([item[1] for item in loaded]).to(device)
        with torch.inference_mode(), torch.autocast(
            device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"
        ):
            probabilities = torch.softmax(model(clips), dim=1).cpu()
        for (path, _clip, metadata), scores in zip(loaded, probabilities):
            confidence, prediction_id = scores.max(dim=0)
            expected_score = scores[classes.index(args.expected_label)]
            results.append(
                {
                    "video": str(path.resolve()),
                    "prediction": classes[int(prediction_id)],
                    "confidence": float(confidence),
                    "expected_label_probability": float(expected_score),
                    "correct": classes[int(prediction_id)] == args.expected_label,
                    "scores": {
                        class_name: float(scores[index])
                        for index, class_name in enumerate(classes)
                    },
                    "video_metadata": metadata,
                }
            )
        print(f"Processed {min(offset + args.batch_size, len(videos))}/{len(videos)}", flush=True)

    prediction_counts = Counter(str(item["prediction"]) for item in results)
    correct = sum(bool(item["correct"]) for item in results)
    summary = {
        "video_dir": str(args.video_dir.resolve()),
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_epoch": int(checkpoint["epoch"]),
        "device": str(device),
        "expected_label": args.expected_label,
        "total": len(results),
        "correct": correct,
        "accuracy": correct / len(results),
        "mean_expected_label_probability": sum(
            float(item["expected_label_probability"]) for item in results
        )
        / len(results),
        "prediction_counts": dict(prediction_counts.most_common()),
    }
    report = {"summary": summary, "results": results}
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
