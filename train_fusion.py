"""Main Training Script for Multimodal Fusion Network on Badminton ShuttleSet.

Usage:
    python train_fusion.py
    python train_fusion.py --epochs 20 --batch-size 4 --learning-rate 1e-3
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch
from torch import nn

REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from ai_classifier.models import FusionHead, SIDE_CLASSES, STROKE_CLASSES
from scripts.training.train_fine_badminton_rgb import confusion_metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Multimodal Fusion Model")
    parser.add_argument(
        "--feature-cache",
        type=Path,
        default=REPO_ROOT / "work_dirs" / "shuttleset_fusion_epoch20_full_b4" / "features.npz",
        help="Path to precomputed features.npz cache file.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "work_dirs" / "shuttleset_fusion_epoch20_full_b4",
        help="Directory to save model checkpoints and metrics.",
    )
    parser.add_argument("--epochs", type=int, default=20, help="Number of training epochs.")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size for training.")
    parser.add_argument("--learning-rate", type=float, default=1e-3, help="Initial learning rate.")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="Weight decay for AdamW.")
    parser.add_argument("--seed", type=int, default=20260917, help="Random seed for reproducibility.")
    parser.add_argument("--drive-boost", type=float, default=1.0, help="Multiplier for drive stroke loss weight.")
    return parser.parse_args()


def calculate_class_weights(labels: torch.Tensor, ids: np.ndarray, num_classes: int, device: torch.device) -> torch.Tensor:
def calculate_class_weights(
    labels: torch.Tensor,
    ids: np.ndarray,
    num_classes: int,
    device: torch.device,
    boost_dict: dict[int, float] | None = None,
) -> torch.Tensor:
    """Compute balanced inverse class frequencies to handle imbalanced strokes."""
    counts = torch.bincount(labels[ids], minlength=num_classes).float()
    weights = len(ids) / (num_classes * counts.clamp_min(1.0))
    if boost_dict:
        for class_idx, multiplier in boost_dict.items():
            if class_idx < len(weights):
                weights[class_idx] *= multiplier
    return weights.to(device)


def evaluate(
    model: nn.Module,
    rgb: torch.Tensor,
    structured: torch.Tensor,
    stroke: torch.Tensor,
    side: torch.Tensor,
    indices: np.ndarray,
    device: torch.device,
) -> tuple[dict, dict]:
    """Evaluate model performance across stroke and side heads."""
    model.eval()
    stroke_confusion = torch.zeros(len(STROKE_CLASSES), len(STROKE_CLASSES), dtype=torch.int64)
    side_confusion = torch.zeros(len(SIDE_CLASSES), len(SIDE_CLASSES), dtype=torch.int64)

    with torch.inference_mode():
        for start in range(0, len(indices), 128):
            ids = indices[start:start + 128]
            stroke_logits, side_logits = model(rgb[ids].to(device), structured[ids].to(device))
            for truth, pred in zip(stroke[ids], stroke_logits.argmax(dim=1).cpu()):
                stroke_confusion[int(truth), int(pred)] += 1
            for truth, pred in zip(side[ids], side_logits.argmax(dim=1).cpu()):
                side_confusion[int(truth), int(pred)] += 1

    return confusion_metrics(stroke_confusion), confusion_metrics(side_confusion)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Training Multimodal Fusion Network on device: {device}")

    if not args.feature_cache.is_file():
        raise FileNotFoundError(f"Feature cache not found at: {args.feature_cache}")

    print(f"[*] Loading precomputed features from: {args.feature_cache}")
    data = np.load(args.feature_cache)

    rgb = torch.from_numpy(data["rgb"]).float()
    structured_np = data["structured"].astype(np.float32)

    # Standard feature scaling based on training split
    train_mask = data["split"] == "train"
    mean = structured_np[train_mask].mean(axis=0)
    std = structured_np[train_mask].std(axis=0)
    structured = torch.from_numpy((structured_np - mean) / np.maximum(std, 1e-5)).float()

    stroke = torch.from_numpy(data["stroke"])
    side = torch.from_numpy(data["side"])
    indices = {split: np.flatnonzero(data["split"] == split) for split in ("train", "val", "test")}

    print(f"[*] Dataset splits: Train={len(indices['train'])}, Val={len(indices['val'])}, Test={len(indices['test'])}")

    # Initialize model
    rgb_dim = rgb.shape[1]
    structured_dim = structured.shape[1]
    model = FusionHead(rgb_dim=rgb_dim, structured_dim=structured_dim).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    stroke_loss_fn = nn.CrossEntropyLoss(weight=calculate_class_weights(stroke, indices["train"], len(STROKE_CLASSES), device))
    drive_idx = STROKE_CLASSES.index("drive") if "drive" in STROKE_CLASSES else 6
    boost_dict = {drive_idx: args.drive_boost} if args.drive_boost != 1.0 else None
    if boost_dict:
        print(f"[*] Applying Drive class weight boost: x{args.drive_boost:.2f}")
    stroke_loss_fn = nn.CrossEntropyLoss(weight=calculate_class_weights(stroke, indices["train"], len(STROKE_CLASSES), device, boost_dict))
    side_loss_fn = nn.CrossEntropyLoss(weight=calculate_class_weights(side, indices["train"], len(SIDE_CLASSES), device))

    rng = np.random.default_rng(args.seed)
    best_score = -1.0
    history = []

    print("\n" + "=" * 70)
    print(f"{'Epoch':<8} | {'Stroke Val F1':<15} | {'Side Val F1':<15} | {'Total Score':<12}")
    print("=" * 70)

    for epoch in range(1, args.epochs + 1):
        model.train()
        order = rng.permutation(indices["train"])

        for start in range(0, len(order), args.batch_size):
            batch_ids = order[start:start + args.batch_size]
            optimizer.zero_grad(set_to_none=True)

            stroke_logits, side_logits = model(rgb[batch_ids].to(device), structured[batch_ids].to(device))
            loss = stroke_loss_fn(stroke_logits, stroke[batch_ids].to(device)) + side_loss_fn(side_logits, side[batch_ids].to(device))

            loss.backward()
            optimizer.step()

        val_stroke, val_side = evaluate(model, rgb, structured, stroke, side, indices["val"], device)
        total_score = val_stroke["macro_f1"] + val_side["macro_f1"]
        history.append({"epoch": epoch, "val_stroke": val_stroke, "val_side": val_side})

        print(f"{epoch:<8} | {val_stroke['macro_f1']*100:6.2f}%         | {val_side['macro_f1']*100:6.2f}%         | {total_score*100:6.2f}%")

        if total_score > best_score:
            best_score = total_score
            torch.save({
                "model": model.state_dict(),
                "structured_mean": mean,
                "structured_std": std,
                "epoch": epoch,
                "rgb_dim": rgb_dim,
                "structured_dim": structured_dim,
            }, args.output_dir / "best.pth")

    print("=" * 70)
    print(f"[*] Training finished! Best checkpoint saved to {args.output_dir / 'best.pth'}")

    # Evaluate best model on test set
    best_ckpt = torch.load(args.output_dir / "best.pth", map_location=device, weights_only=False)
    model.load_state_dict(best_ckpt["model"])
    test_stroke, test_side = evaluate(model, rgb, structured, stroke, side, indices["test"], device)

    print(f"[*] Test Set Performance:")
    print(f"    - Stroke Accuracy : {test_stroke['accuracy']*100:.2f}% | Macro-F1: {test_stroke['macro_f1']*100:.2f}%")
    print(f"    - Side Accuracy   : {test_side['accuracy']*100:.2f}% | Macro-F1: {test_side['macro_f1']*100:.2f}%")


if __name__ == "__main__":
    main()

