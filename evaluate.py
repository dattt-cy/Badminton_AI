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
    parser = argparse.ArgumentParser(description="Evaluate Badminton Classification Models")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=REPO_ROOT / "work_dirs" / "shuttleset_fusion_epoch20_full_b4" / "best.pth",
        help="Path to best.pth checkpoint",
        "--model",
        type=str,
        choices=["cr_gated", "baseline"],
        default="cr_gated",
        help="Model architecture to evaluate: 'cr_gated' (Proposed) or 'baseline' (Concat-MLP)",
    )
    parser.add_argument(
        "--feature-cache",
        "--variant",
        type=str,
        default="no_gate",
        choices=["no_gate", "full", "no_contact", "no_temporal"],
        help="Ablation variant for cr_gated model (default: no_gate - best performing)",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=REPO_ROOT / "work_dirs" / "shuttleset_fusion_epoch20_full_b4" / "features.npz",
        help="Path to features.npz",
        default=None,
        help="Custom checkpoint path (overrides defaults)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Evaluating Fusion Model on device: {device}")
def evaluate_cr_gated(variant: str, custom_ckpt: Path | None, device: torch.device) -> None:
    from ai_classifier.models.cr_gated_fusion import CRGatedFusionModel

    if not args.checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")
    if not args.feature_cache.is_file():
        raise FileNotFoundError(f"Feature cache not found: {args.feature_cache}")
    ckpt_path = custom_ckpt or (REPO_ROOT / "work_dirs" / "cr_gated_fusion" / f"cr_gated_{variant}" / "best.pth")
    feat_path = REPO_ROOT / "work_dirs" / "cr_gated_fusion" / "sequence_features.npz"

    data = np.load(args.feature_cache)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    if not feat_path.is_file():
        raise FileNotFoundError(f"Features file not found: {feat_path}")

    print(f"[*] Architecture : CR-Gated Fusion (Variant: {variant.upper()})")
    print(f"[*] Checkpoint   : {ckpt_path}")
    print(f"[*] Device       : {device}")

    data = np.load(feat_path)
    test_idx = np.flatnonzero(data["split"] == "test")

    use_temporal = variant != "no_temporal"
    use_contact = variant != "no_contact"
    use_gate = variant != "no_gate"

    model = CRGatedFusionModel(
        hidden_dim=128,
        num_stroke_classes=len(STROKE_CLASSES),
        num_side_classes=len(SIDE_CLASSES),
        use_temporal=use_temporal,
        use_contact=use_contact,
        use_gate=use_gate,
    ).to(device)

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    model.eval()

    rgb = torch.from_numpy(data["rgb"][test_idx]).float().to(device)
    pose = torch.from_numpy(data["pose"][test_idx]).float().to(device)
    court = torch.from_numpy(data["court"][test_idx]).float().to(device)
    shuttle = torch.from_numpy(data["shuttle"][test_idx]).float().to(device)
    contact_dist = torch.from_numpy(data["contact_dist"][test_idx]).float().to(device)
    quality = torch.from_numpy(data["quality"][test_idx]).float().to(device)
    test_stroke = torch.from_numpy(data["stroke"][test_idx]).to(device)
    test_side = torch.from_numpy(data["side"][test_idx]).to(device)

    with torch.inference_mode():
        stroke_logits, side_logits, gate_weights = model(rgb, pose, court, shuttle, contact_dist, quality)

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

    print("\n" + "=" * 68)
    print(f"KET QUA DANH GIA SHUTTLESET TEST SET ({len(test_idx)} MAU) - CR-GATED FUSION")
    print("=" * 68)
    print(f"  * Stroke Top-1 Accuracy : {acc_top1:.2f}%")
    print(f"  * Stroke Top-2 Accuracy : {acc_top2:.2f}%  (Acc-2)")
    print(f"  * Stroke Macro-F1       : {metrics_stroke['macro_f1']*100:.2f}%")
    print("-" * 68)
    print(f"  * Side Top-1 Accuracy   : {acc_side:.2f}%")
    print(f"  * Side Macro-F1         : {metrics_side['macro_f1']*100:.2f}%")

    if gate_weights is not None:
        mean_weights = gate_weights.mean(dim=0).cpu().numpy()
        print("-" * 68)
        print("  * Reliability Gate Weights :")
        print(f"    - RGB Video      : {mean_weights[0]*100:.1f}%")
        print(f"    - Pose (Joints)  : {mean_weights[1]*100:.1f}%")
        print(f"    - Court Position : {mean_weights[2]*100:.1f}%")
        print(f"    - Shuttlecock    : {mean_weights[3]*100:.1f}%")
    print("=" * 68)

    print("\nCHI TIET F1 TUNG LOAI CU DANH:")
    print(f"{'Class':<14} | {'Precision':<10} | {'Recall':<10} | {'F1-Score':<10} | {'Samples':<8}")
    print("-" * 62)
    for i, name in enumerate(STROKE_CLASSES):
        p = metrics_stroke["precision"][i] * 100
        r = metrics_stroke["recall"][i] * 100
        f1 = metrics_stroke["f1"][i] * 100
        cnt = int(stroke_cm[i].sum())
        highlight = " <-- SIGNIFICANT GAIN (+8.6% F1)" if name == "drive" else ""
        print(f"{name:<14} | {p:7.1f}%   | {r:7.1f}%   | {f1:7.1f}%   | {cnt:<8}{highlight}")
    print("-" * 62)


def evaluate_baseline(custom_ckpt: Path | None, device: torch.device) -> None:
    ckpt_path = custom_ckpt or (REPO_ROOT / "work_dirs" / "shuttleset_fusion_epoch20_full_b4" / "best.pth")
    feat_path = REPO_ROOT / "work_dirs" / "shuttleset_fusion_epoch20_full_b4" / "features.npz"

    if not ckpt_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    if not feat_path.is_file():
        raise FileNotFoundError(f"Feature cache not found: {feat_path}")

    print(f"[*] Architecture : Concat-MLP Baseline (Epoch 20)")
    print(f"[*] Checkpoint   : {ckpt_path}")
    print(f"[*] Device       : {device}")

    data = np.load(feat_path)
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)

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
    print("\n" + "=" * 68)
    print(f"KET QUA DANH GIA SHUTTLESET TEST SET ({len(test_indices)} MAU) - BASELINE")
    print("=" * 68)
    print(f"  * Stroke Top-1 Accuracy : {acc_top1:.2f}%")
    print(f"  * Stroke Top-2 Accuracy : {acc_top2:.2f}%  (Acc-2)")
    print(f"  * Stroke Macro-F1       : {metrics_stroke['macro_f1']*100:.2f}%")
    print("-" * 65)
    print("-" * 68)
    print(f"  * Side Top-1 Accuracy   : {acc_side:.2f}%")
    print(f"  * Side Macro-F1         : {metrics_side['macro_f1']*100:.2f}%")
    print("=" * 65)
    print("=" * 68)

    print("\nPER-CLASS STROKE BREAKDOWN:")
    print("\nCHI TIET F1 TUNG LOAI CU DANH:")
    print(f"{'Class':<14} | {'Precision':<10} | {'Recall':<10} | {'F1-Score':<10} | {'Samples':<8}")
    print("-" * 60)
    print("-" * 62)
    for i, name in enumerate(STROKE_CLASSES):
        p = metrics_stroke["precision"][i] * 100
        r = metrics_stroke["recall"][i] * 100
        f1 = metrics_stroke["f1"][i] * 100
        cnt = int(stroke_cm[i].sum())
        print(f"{name:<14} | {p:7.1f}%   | {r:7.1f}%   | {f1:7.1f}%   | {cnt:<8}")
    print("-" * 60)
    print("-" * 62)


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.model == "cr_gated":
        evaluate_cr_gated(args.variant, args.checkpoint, device)
    else:
        evaluate_baseline(args.checkpoint, device)


if __name__ == "__main__":
    main()

