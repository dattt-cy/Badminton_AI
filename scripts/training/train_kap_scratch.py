"""Train domain-robust KAP from random initialization on frozen RGB features.

This is the canonical no-fine-tuning training path:
1. External feature extractors are absent from the optimizer and computation graph.
2. KAP is initialized randomly and trained on the train split.
Example:
    python scripts/training/train_kap_scratch.py --evaluate-test
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ai_classifier.models import (  # noqa: E402
    KAPScratchModel,
    SIDE_CLASSES,
    STROKE_CLASSES,
)


def confusion_metrics(confusion: torch.Tensor) -> dict:
    """Return class-wise and macro metrics from an integer confusion matrix."""
    matrix = confusion.to(torch.float64)
    true_positive = matrix.diag()
    support = matrix.sum(dim=1)
    predicted = matrix.sum(dim=0)
    precision = true_positive / predicted.clamp_min(1)
    recall = true_positive / support.clamp_min(1)
    f1 = 2 * precision * recall / (precision + recall).clamp_min(1e-12)
    return {
        "accuracy": float(true_positive.sum() / matrix.sum().clamp_min(1)),
        "balanced_accuracy": float(recall.mean()),
        "macro_f1": float(f1.mean()),
        "precision": precision.tolist(),
        "recall": recall.tolist(),
        "f1": f1.tolist(),
        "support": support.to(torch.int64).tolist(),
        "confusion_matrix": confusion.tolist(),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--features", type=Path,
        default=Path("work_dirs/cr_gated_fusion/sequence_features.npz"),
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("work_dirs/kap_robust_scratch")
    )
    parser.add_argument("--kap-epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--kap-learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--side-loss-weight", type=float, default=0.5)
    parser.add_argument("--consistency-weight", type=float, default=0.35)
    parser.add_argument("--noisy-loss-weight", type=float, default=0.75)
    parser.add_argument("--max-temporal-shift", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20260926)
    parser.add_argument(
        "--feature-provenance", default="frozen_cached_feature_extractors",
        help="Human-readable provenance recorded in checkpoints and metrics.",
    )
    parser.add_argument(
        "--evaluate-test", action="store_true",
        help="Evaluate the held-out test set after model selection. Omit while tuning.",
    )
    return parser.parse_args()


class VideoDomainAugmentor:
    """Simulate pose, court, TrackNet and contact errors seen at inference."""

    def __init__(self, max_shift: int = 2) -> None:
        self.max_shift = max_shift

    @staticmethod
    def _shift(value: torch.Tensor, shifts: torch.Tensor) -> torch.Tensor:
        output = torch.empty_like(value)
        for index, shift_tensor in enumerate(shifts):
            shift = int(shift_tensor)
            if shift == 0:
                output[index] = value[index]
            elif shift > 0:
                output[index, :shift] = value[index, :1]
                output[index, shift:] = value[index, :-shift]
            else:
                amount = -shift
                output[index, :-amount] = value[index, amount:]
                output[index, -amount:] = value[index, -1:]
        return output

    @staticmethod
    def _quality(pose: torch.Tensor, shuttle: torch.Tensor) -> torch.Tensor:
        shuttle_missing = (shuttle.abs().sum(dim=-1) < 1e-8).float().mean(dim=1)
        shuttle_speed = torch.diff(shuttle, dim=1).norm(dim=-1).std(dim=1)
        pose_missing = (pose.abs() < 1e-8).float().mean(dim=(1, 2))
        pose_motion = pose.std(dim=1).mean(dim=1)
        return torch.stack(
            [shuttle_missing, shuttle_speed, pose_missing, pose_motion], dim=1
        )

    def __call__(self, inputs: list[torch.Tensor]) -> list[torch.Tensor]:
        rgb, pose, court, shuttle, contact, _quality = inputs
        pose, court, shuttle, contact = (
            pose.clone(), court.clone(), shuttle.clone(), contact.clone()
        )
        batch, frames = pose.shape[:2]

        # Per-sample temporal misalignment of the structured modalities. Keep
        # contact coordinates fixed: this represents an imperfect hit estimate.
        shifts = torch.randint(
            -self.max_shift, self.max_shift + 1, (batch,), device=pose.device
        )
        apply_shift = torch.rand(batch, device=pose.device) < 0.65
        shifts = torch.where(apply_shift, shifts, torch.zeros_like(shifts))
        pose = self._shift(pose, shifts)
        court = self._shift(court, shifts)
        shuttle = self._shift(shuttle, shifts)

        # Detector jitter. Noise is sample-gated so clean examples remain in
        # every batch as anchors for consistency learning.
        pose_gate = (torch.rand(batch, 1, 1, device=pose.device) < 0.65).float()
        pose = pose + pose_gate * torch.randn_like(pose) * 0.012
        court_gate = (torch.rand(batch, 1, 1, device=court.device) < 0.45).float()
        court = court + court_gate * torch.randn_like(court) * 0.008
        shuttle_gate = (torch.rand(batch, 1, 1, device=shuttle.device) < 0.75).float()
        shuttle = shuttle + shuttle_gate * torch.randn_like(shuttle) * 0.010

        # Missing keypoints and TrackNet gaps. Drop whole (x,y) pairs rather
        # than scalar coordinates so zero retains its missing-data meaning.
        joints = pose.reshape(batch, frames, -1, 2)
        joint_drop = torch.rand(batch, frames, joints.shape[2], 1, device=pose.device) < 0.035
        joints = joints.masked_fill(joint_drop, 0.0)
        pose = joints.reshape_as(pose)
        for index in range(batch):
            if torch.rand((), device=pose.device) < 0.60:
                length = int(torch.randint(2, 7, (), device=pose.device))
                start = int(torch.randint(4, max(5, frames - length), (), device=pose.device))
                shuttle[index, start:start + length] = 0.0

        pose = pose.clamp(-1.0, 1.0)
        court = court.clamp(-0.1, 1.1)
        shuttle = shuttle.clamp(0.0, 1.0)
        quality = self._quality(pose, shuttle)
        return [rgb, pose, court, shuttle, contact, quality]


def symmetric_consistency(clean_logits: torch.Tensor, noisy_logits: torch.Tensor) -> torch.Tensor:
    """Symmetric KL with stopped teacher probabilities in both directions."""
    clean_prob = clean_logits.detach().softmax(dim=1)
    noisy_prob = noisy_logits.detach().softmax(dim=1)
    clean_log = clean_logits.log_softmax(dim=1)
    noisy_log = noisy_logits.log_softmax(dim=1)
    return 0.5 * (
        F.kl_div(noisy_log, clean_prob, reduction="batchmean")
        + F.kl_div(clean_log, noisy_prob, reduction="batchmean")
    )


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_data(path: Path) -> tuple[dict[str, torch.Tensor], dict[str, np.ndarray]]:
    raw = np.load(path)
    required = {
        "rgb", "pose", "court", "shuttle", "contact_dist", "quality",
        "stroke", "side", "split",
    }
    missing = required.difference(raw.files)
    if missing:
        raise ValueError(f"Feature cache is missing arrays: {sorted(missing)}")

    arrays = {name: raw[name] for name in required}
    finite = np.ones(len(arrays["stroke"]), dtype=bool)
    for name in ("rgb", "pose", "court", "shuttle", "contact_dist", "quality"):
        finite &= np.isfinite(arrays[name]).reshape(len(finite), -1).all(axis=1)
    structured_present = ~(
        np.all(arrays["pose"] == 0, axis=(1, 2))
        & np.all(arrays["court"] == 0, axis=(1, 2))
        & np.all(arrays["shuttle"] == 0, axis=(1, 2))
    )
    valid = finite & structured_present
    if "valid" in raw.files:
        valid &= raw["valid"].astype(bool)
    rejected = int((~valid).sum())
    if rejected:
        print(f"[data] Excluding {rejected} invalid/all-zero samples.", flush=True)

    tensors = {
        name: torch.from_numpy(arrays[name]).float()
        for name in ("rgb", "pose", "court", "shuttle", "contact_dist", "quality")
    }
    tensors["stroke"] = torch.from_numpy(arrays["stroke"]).long()
    tensors["side"] = torch.from_numpy(arrays["side"]).long()
    indices = {
        split: np.flatnonzero((arrays["split"] == split) & valid)
        for split in ("train", "val", "test")
    }
    return tensors, indices


def make_loader(
    tensors: dict[str, torch.Tensor], indices: np.ndarray, batch_size: int,
    *, shuffle: bool,
) -> DataLoader:
    ids = torch.from_numpy(indices).long()
    dataset = TensorDataset(*[
        tensors[name][ids]
        for name in (
            "rgb", "pose", "court", "shuttle", "contact_dist", "quality",
            "stroke", "side",
        )
    ])
    return DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle,
        pin_memory=torch.cuda.is_available(),
    )


def feature_statistics(
    tensors: dict[str, torch.Tensor], train_indices: np.ndarray
) -> dict[str, torch.Tensor]:
    ids = torch.from_numpy(train_indices).long()
    stats = {}
    for name in ("pose", "court", "shuttle"):
        values = tensors[name][ids]
        stats[f"{name}_mean"] = values.mean(dim=(0, 1))
        stats[f"{name}_std"] = values.std(dim=(0, 1)).clamp_min(1e-5)
    quality = tensors["quality"][ids]
    stats["quality_mean"] = quality.mean(dim=0)
    stats["quality_std"] = quality.std(dim=0).clamp_min(1e-5)
    return stats


def balanced_weights(
    labels: torch.Tensor, indices: np.ndarray, classes: int, device: torch.device
) -> torch.Tensor:
    ids = torch.from_numpy(indices).long()
    counts = torch.bincount(labels[ids], minlength=classes).float()
    return (len(ids) / (classes * counts.clamp_min(1))).to(device)


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[dict, dict]:
    model.eval()
    stroke_cm = torch.zeros(len(STROKE_CLASSES), len(STROKE_CLASSES), dtype=torch.int64)
    side_cm = torch.zeros(len(SIDE_CLASSES), len(SIDE_CLASSES), dtype=torch.int64)
    with torch.inference_mode():
        for batch in loader:
            inputs = [value.to(device, non_blocking=True) for value in batch[:6]]
            stroke, side, _ = model(*inputs)
            for truth, prediction in zip(batch[6], stroke.argmax(dim=1).cpu()):
                stroke_cm[int(truth), int(prediction)] += 1
            for truth, prediction in zip(batch[7], side.argmax(dim=1).cpu()):
                side_cm[int(truth), int(prediction)] += 1
    return confusion_metrics(stroke_cm), confusion_metrics(side_cm)


def train_kap(
    model: KAPScratchModel,
    train_loader: DataLoader,
    val_loader: DataLoader,
    tensors: dict[str, torch.Tensor],
    train_indices: np.ndarray,
    args: argparse.Namespace,
    device: torch.device,
    checkpoint: Path,
) -> list[dict]:
    stroke_loss = nn.CrossEntropyLoss(weight=balanced_weights(
        tensors["stroke"], train_indices, len(STROKE_CLASSES), device
    ))
    side_loss = nn.CrossEntropyLoss(weight=balanced_weights(
        tensors["side"], train_indices, len(SIDE_CLASSES), device
    ))
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.kap_learning_rate, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.kap_epochs)
    augmentor = VideoDomainAugmentor(args.max_temporal_shift)
    best, history = -1.0, []
    for epoch in range(1, args.kap_epochs + 1):
        model.train()
        loss_sum, samples = 0.0, 0
        for batch in train_loader:
            inputs = [value.to(device, non_blocking=True) for value in batch[:6]]
            stroke_gt, side_gt = batch[6].to(device), batch[7].to(device)
            optimizer.zero_grad(set_to_none=True)
            stroke, side, _ = model(*inputs)
            noisy_inputs = augmentor(inputs)
            noisy_stroke, noisy_side, _ = model(*noisy_inputs)
            clean_loss = stroke_loss(stroke, stroke_gt) + args.side_loss_weight * side_loss(side, side_gt)
            noisy_loss = stroke_loss(noisy_stroke, stroke_gt) + args.side_loss_weight * side_loss(noisy_side, side_gt)
            consistency = symmetric_consistency(stroke, noisy_stroke)
            consistency = consistency + args.side_loss_weight * symmetric_consistency(side, noisy_side)
            loss = clean_loss + args.noisy_loss_weight * noisy_loss + args.consistency_weight * consistency
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            loss_sum += float(loss.detach()) * len(stroke_gt)
            samples += len(stroke_gt)
        scheduler.step()
        val_stroke, val_side = evaluate(model, val_loader, device)
        score = val_stroke["macro_f1"] + args.side_loss_weight * val_side["macro_f1"]
        row = {
            "epoch": epoch, "train_loss": loss_sum / max(samples, 1),
            "val_stroke": val_stroke, "val_side": val_side, "selection_score": score,
        }
        history.append(row)
        print(
            f"[KAP] {epoch:02d}/{args.kap_epochs} loss={row['train_loss']:.4f} "
            f"stroke_f1={val_stroke['macro_f1']:.4f} side_f1={val_side['macro_f1']:.4f}",
            flush=True,
        )
        if score > best:
            best = score
            torch.save({
                "model": model.state_dict(), "epoch": epoch,
                "training_mode": "random_initialization_no_external_checkpoint",
                "external_feature_extractors_trainable": False,
                "feature_provenance": args.feature_provenance,
                "selection_metric": "val_stroke_macro_f1_plus_weighted_side_macro_f1",
                "selection_score": score,
                "robust_training": {
                    "noisy_loss_weight": args.noisy_loss_weight,
                    "consistency_weight": args.consistency_weight,
                    "max_temporal_shift": args.max_temporal_shift,
                },
            }, checkpoint)
    return history


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tensors, indices = load_data(args.features)
    print(
        f"[data] train={len(indices['train'])} val={len(indices['val'])} "
        f"test={len(indices['test'])} device={device}", flush=True,
    )
    train_loader = make_loader(tensors, indices["train"], args.batch_size, shuffle=True)
    val_loader = make_loader(tensors, indices["val"], args.batch_size, shuffle=False)
    test_loader = make_loader(tensors, indices["test"], args.batch_size, shuffle=False)

    kap_checkpoint = args.output_dir / "kap_best.pth"
    kap = KAPScratchModel()
    kap.set_normalization(**feature_statistics(tensors, indices["train"]))
    kap.to(device)
    kap_history = train_kap(
        kap, train_loader, val_loader, tensors, indices["train"],
        args, device, kap_checkpoint,
    )
    if not kap_checkpoint.is_file():
        raise FileNotFoundError(f"KAP checkpoint not found: {kap_checkpoint}")
    kap_payload = torch.load(kap_checkpoint, map_location=device, weights_only=False)
    kap.load_state_dict(kap_payload["model"])

    result = {
        "protocol": "frozen_rgb_features_domain_robust_kap_scratch",
        "seed": args.seed,
        "features": str(args.features),
        "feature_provenance": args.feature_provenance,
        "external_feature_extractors_trainable": False,
        "splits": {name: len(value) for name, value in indices.items()},
        "kap_best_epoch": int(kap_payload["epoch"]),
        "kap_history": kap_history,
    }
    if args.evaluate_test:
        result["kap_test_stroke"], result["kap_test_side"] = evaluate(kap, test_loader, device)
        print(
            f"[test] KAP stroke F1={result['kap_test_stroke']['macro_f1']:.4f}",
            flush=True,
        )
    (args.output_dir / "metrics.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
