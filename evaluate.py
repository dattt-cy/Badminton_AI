"""Comprehensive Evaluation Script for Badminton Multimodal Fusion Model.

Evaluates the optimal checkpoint on the ShuttleSet test set (871 samples)
and calculates:
- Top-1 Accuracy, Top-2 Accuracy (Acc-2), Macro-F1
- Per-class Precision, Recall, F1 breakdown
- Side (Forehand / Backhand / Aroundhead) metrics

Usage:
    python evaluate.py
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from ai_classifier.models import FusionHead, SIDE_CLASSES, STROKE_CLASSES
from scripts.training.train_fine_badminton_rgb import confusion_metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate Fusion Model")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=REPO_ROOT / "work_dirs" / "shuttleset_fusion_epoch20_full_b4" / "best.pth",
        help="Path to best.pth checkpoint",
    )
    parser.add_argument(
        "--feature-cache",
        type=Path,
        default=REPO_ROOT / "work_dirs" / "shuttleset_fusion_epoch20_full_b4" / "features.npz",
        help="Path to features.npz",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Evaluating Fusion Model on device: {device}")

    if not args.checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")
    if not args.feature_cache.is_file():
        raise FileNotFoundError(f"Feature cache not found: {args.feature_cache}")

    data = np.load(args.feature_cache)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)

    rgb = torch.from_numpy(data["rgb"]).float()
    structured_np = data["structured"].astype(np.float32)

    mean = checkpoint.get("structured_mean", structured_np.mean(axis=0))
    std = checkpoint.get("structured_std", structured_np.std(axis=0))
    structured = torch.from_numpy((structured_np - mean) / np.maximum(std, 1e-5)).float()

    stroke = torch.from_numpy(data["stroke"])
    side = torch.from_numpy(data["side"])
    test_indices = np.flatnonzero(data["split"] == "test")

    rgb_dim = rgb.shape[1]
    structured_dim = structured.shape[1]
    model = FusionHead(rgb_dim=rgb_dim, structured_dim=structured_dim).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    test_rgb = rgb[test_indices].to(device)
    test_struct = structured[test_indices].to(device)
    test_stroke = stroke[test_indices].to(device)
    test_side = side[test_indices].to(device)

    with torch.inference_mode():
        stroke_logits, side_logits = model(test_rgb, test_struct)

    # Top-1 & Top-2 Accuracy
    pred_top1 = stroke_logits.argmax(dim=1)
    top2_preds = stroke_logits.topk(2, dim=1).indices
    acc_top1 = (pred_top1 == test_stroke).float().mean().item() * 100
    acc_top2 = ((test_stroke.unsqueeze(1) == top2_preds).any(dim=1)).float().mean().item() * 100

    pred_side = side_logits.argmax(dim=1)
    acc_side = (pred_side == test_side).float().mean().item() * 100

    # Confusion metrics
    stroke_cm = torch.zeros(len(STROKE_CLASSES), len(STROKE_CLASSES), dtype=torch.int64)
    for t, p in zip(test_stroke.cpu(), pred_top1.cpu()):
        stroke_cm[int(t), int(p)] += 1
    metrics_stroke = confusion_metrics(stroke_cm)

    side_cm = torch.zeros(len(SIDE_CLASSES), len(SIDE_CLASSES), dtype=torch.int64)
    for t, p in zip(test_side.cpu(), pred_side.cpu()):
        side_cm[int(t), int(p)] += 1
    metrics_side = confusion_metrics(side_cm)

    print("\n" + "=" * 65)
    print(f"SHUTTLESET TEST SET BENCHMARK RESULTS ({len(test_indices)} SAMPLES)")
    print("=" * 65)
    print(f"  * Stroke Top-1 Accuracy : {acc_top1:.2f}%")
    print(f"  * Stroke Top-2 Accuracy : {acc_top2:.2f}%  (Acc-2)")
    print(f"  * Stroke Macro-F1       : {metrics_stroke['macro_f1']*100:.2f}%")
    print("-" * 65)
    print(f"  * Side Top-1 Accuracy   : {acc_side:.2f}%")
    print(f"  * Side Macro-F1         : {metrics_side['macro_f1']*100:.2f}%")
    print("=" * 65)

    print("\nPER-CLASS STROKE BREAKDOWN:")
    print(f"{'Class':<14} | {'Precision':<10} | {'Recall':<10} | {'F1-Score':<10} | {'Samples':<8}")
    print("-" * 60)
    for i, name in enumerate(STROKE_CLASSES):
        p = metrics_stroke["precision"][i] * 100
        r = metrics_stroke["recall"][i] * 100
        f1 = metrics_stroke["f1"][i] * 100
        cnt = int(stroke_cm[i].sum())
        print(f"{name:<14} | {p:7.1f}%   | {r:7.1f}%   | {f1:7.1f}%   | {cnt:<8}")
    print("-" * 60)


if __name__ == "__main__":
    main()

