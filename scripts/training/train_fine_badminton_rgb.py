"""Train an R(2+1)D-18 RGB baseline on Fine-Badminton virtual clips."""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
import yaml
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision.models.video import R2Plus1D_18_Weights, r2plus1d_18
from torchvision.transforms import functional as vision_functional

from ai_classifier.datasets import (
    FineBadmintonRecord,
    decode_virtual_clip,
    load_fine_badminton_manifest,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset_root", type=Path)
    parser.add_argument(
        "--train-manifest",
        type=Path,
        default=Path("data/manifests/fine_badminton_splits/train.csv"),
    )
    parser.add_argument(
        "--val-manifest",
        type=Path,
        default=Path("data/manifests/fine_badminton_splits/val.csv"),
    )
    parser.add_argument(
        "--taxonomy",
        type=Path,
        default=Path("configs/taxonomy/fine_badminton_8class.yaml"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("work_dirs/r2plus1d18_fine_badminton_8class"),
    )
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--frames", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument(
        "--cache-dir",
        type=Path,
        help="Optional cache of sampled uint8 clips; missing entries are created lazily.",
    )
    parser.add_argument("--seed", type=int, default=20260916)
    parser.add_argument(
        "--smoke-samples-per-class",
        type=int,
        help="Use a small balanced subset from each split.",
    )
    parser.add_argument("--freeze-backbone", action="store_true")
    parser.add_argument("--no-pretrained", action="store_true")
    parser.add_argument(
        "--amp",
        action="store_true",
        help="Use CUDA automatic mixed precision when a CUDA device is available.",
    )
    parser.add_argument(
        "--overfit",
        action="store_true",
        help="Evaluate on the exact selected training samples as a pipeline sanity check.",
    )
    return parser.parse_args()


def load_classes(path: Path) -> list[str]:
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    return [str(item) for item in config["coarse_classes"]]


def balanced_subset(
    records: list[FineBadmintonRecord], count: int | None, seed: int
) -> list[FineBadmintonRecord]:
    if count is None:
        return records
    if count <= 0:
        raise ValueError("smoke-samples-per-class must be positive")
    grouped: dict[str, list[FineBadmintonRecord]] = defaultdict(list)
    for record in records:
        grouped[record.coarse_label].append(record)
    rng = random.Random(seed)
    selected: list[FineBadmintonRecord] = []
    for class_name in sorted(grouped):
        candidates = sorted(grouped[class_name], key=lambda item: item.sample_id)
        selected.extend(rng.sample(candidates, min(count, len(candidates))))
    rng.shuffle(selected)
    return selected


class FineBadmintonRGBDataset(Dataset):
    def __init__(
        self,
        records: list[FineBadmintonRecord],
        dataset_root: Path,
        classes: list[str],
        *,
        frame_count: int,
        transform,
        training: bool,
        cache_dir: Path | None = None,
    ) -> None:
        self.records = records
        self.dataset_root = dataset_root
        self.class_to_id = {name: index for index, name in enumerate(classes)}
        self.frame_count = frame_count
        self.transform = transform
        self.training = training
        self.cache_dir = cache_dir

    def _load_frames(self, record: FineBadmintonRecord) -> torch.Tensor:
        cache_path = (
            self.cache_dir / f"frames_{self.frame_count}" / f"{record.sample_id}.npy"
            if self.cache_dir is not None
            else None
        )
        if cache_path is not None and cache_path.is_file():
            return torch.from_numpy(np.load(cache_path))

        frames = decode_virtual_clip(
            record, self.dataset_root, frame_count=self.frame_count
        )
        tensor = torch.from_numpy(frames.copy()).permute(0, 3, 1, 2)
        # Cache the deterministic spatial transform as uint8. Normalization
        # and stochastic augmentation remain in the training process.
        tensor = vision_functional.resize(
            tensor, [128, 171], antialias=False
        )
        tensor = vision_functional.center_crop(tensor, [112, 112]).contiguous()
        if cache_path is not None:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            temporary = cache_path.with_suffix(".tmp.npy")
            np.save(temporary, tensor.numpy())
            temporary.replace(cache_path)
        return tensor

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int):
        record = self.records[index]
        tensor = self._load_frames(record)
        # The official weights transform would resize/crop again. Cache has
        # already completed that deterministic part, so only normalize here.
        tensor = tensor.to(torch.float32).div_(255.0)
        mean = tensor.new_tensor([0.43216, 0.394666, 0.37645]).view(1, 3, 1, 1)
        std = tensor.new_tensor([0.22803, 0.22145, 0.216989]).view(1, 3, 1, 1)
        tensor = (tensor - mean) / std
        tensor = tensor.permute(1, 0, 2, 3)
        if self.training and torch.rand(()) < 0.5:
            tensor = torch.flip(tensor, dims=(-1,))
        return tensor, self.class_to_id[record.coarse_label], record.sample_id


def confusion_metrics(confusion: torch.Tensor) -> dict[str, object]:
    confusion = confusion.to(torch.float64)
    tp = confusion.diag()
    support = confusion.sum(dim=1)
    predicted = confusion.sum(dim=0)
    precision = tp / predicted.clamp_min(1)
    recall = tp / support.clamp_min(1)
    f1 = 2 * precision * recall / (precision + recall).clamp_min(1e-12)
    present = support > 0
    return {
        "accuracy": float(tp.sum() / support.sum().clamp_min(1)),
        "balanced_accuracy": float(recall[present].mean()),
        "macro_f1": float(f1[present].mean()),
        "precision": precision.tolist(),
        "recall": recall.tolist(),
        "f1": f1.tolist(),
        "support": support.to(torch.int64).tolist(),
        "confusion_matrix": confusion.to(torch.int64).tolist(),
    }


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    *,
    optimizer: torch.optim.Optimizer | None,
    freeze_backbone: bool,
    use_amp: bool,
    scaler: torch.amp.GradScaler | None,
) -> tuple[float, torch.Tensor]:
    training = optimizer is not None
    if training and not freeze_backbone:
        model.train()
    elif training:
        model.eval()
        model.fc.train()
    else:
        model.eval()
    confusion = torch.zeros(model.fc.out_features, model.fc.out_features, dtype=torch.int64)
    loss_sum = 0.0
    sample_count = 0
    context = torch.enable_grad() if training else torch.inference_mode()
    with context:
        for clips, labels, _sample_ids in loader:
            clips = clips.to(device)
            labels = labels.to(device)
            if training:
                optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=use_amp,
            ):
                logits = model(clips)
                loss = criterion(logits, labels)
            if training:
                assert scaler is not None
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            predictions = logits.argmax(dim=1).cpu()
            for truth, prediction in zip(labels.cpu(), predictions):
                confusion[int(truth), int(prediction)] += 1
            batch_size = labels.shape[0]
            loss_sum += float(loss.detach()) * batch_size
            sample_count += batch_size
    return loss_sum / max(sample_count, 1), confusion


def main() -> None:
    args = parse_args()
    if args.epochs <= 0 or args.batch_size <= 0 or args.frames <= 0:
        raise ValueError("epochs, batch-size and frames must be positive")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    classes = load_classes(args.taxonomy)
    train_records = balanced_subset(
        load_fine_badminton_manifest(args.train_manifest),
        args.smoke_samples_per_class,
        args.seed,
    )
    val_records = balanced_subset(
        load_fine_badminton_manifest(args.val_manifest),
        args.smoke_samples_per_class,
        args.seed + 1,
    )
    if args.overfit:
        if args.smoke_samples_per_class is None:
            raise ValueError("--overfit requires --smoke-samples-per-class")
        val_records = list(train_records)
    unknown = sorted(
        {record.coarse_label for record in train_records + val_records}.difference(classes)
    )
    if unknown:
        raise ValueError(f"Manifest contains unknown classes: {unknown}")

    weights = None if args.no_pretrained else R2Plus1D_18_Weights.DEFAULT
    transform = R2Plus1D_18_Weights.DEFAULT.transforms()
    train_dataset = FineBadmintonRGBDataset(
        train_records,
        args.dataset_root,
        classes,
        frame_count=args.frames,
        transform=transform,
        training=True,
        cache_dir=args.cache_dir,
    )
    val_dataset = FineBadmintonRGBDataset(
        val_records,
        args.dataset_root,
        classes,
        frame_count=args.frames,
        transform=transform,
        training=False,
        cache_dir=args.cache_dir,
    )
    generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        generator=generator,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = r2plus1d_18(weights=weights)
    model.fc = nn.Linear(model.fc.in_features, len(classes))
    if args.freeze_backbone:
        for parameter in model.parameters():
            parameter.requires_grad = False
        for parameter in model.fc.parameters():
            parameter.requires_grad = True
    model.to(device)

    train_counts = Counter(record.coarse_label for record in train_records)
    class_weights = torch.tensor(
        [len(train_records) / (len(classes) * train_counts[name]) for name in classes],
        dtype=torch.float32,
        device=device,
    )
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    use_amp = bool(args.amp and device.type == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    history: list[dict[str, object]] = []
    best_macro_f1 = -1.0
    for epoch in range(1, args.epochs + 1):
        train_loss, train_confusion = run_epoch(
            model,
            train_loader,
            criterion,
            device,
            optimizer=optimizer,
            freeze_backbone=args.freeze_backbone,
            use_amp=use_amp,
            scaler=scaler,
        )
        val_loss, val_confusion = run_epoch(
            model,
            val_loader,
            criterion,
            device,
            optimizer=None,
            freeze_backbone=args.freeze_backbone,
            use_amp=use_amp,
            scaler=scaler,
        )
        train_metrics = confusion_metrics(train_confusion)
        val_metrics = confusion_metrics(val_confusion)
        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "train": train_metrics,
            "val": val_metrics,
        }
        history.append(row)
        print(
            f"epoch={epoch} train_loss={train_loss:.4f} "
            f"train_macro_f1={train_metrics['macro_f1']:.4f} "
            f"val_loss={val_loss:.4f} val_macro_f1={val_metrics['macro_f1']:.4f}"
        )
        if float(val_metrics["macro_f1"]) > best_macro_f1:
            best_macro_f1 = float(val_metrics["macro_f1"])
            torch.save(
                {
                    "model": model.state_dict(),
                    "classes": classes,
                    "epoch": epoch,
                    "val_metrics": val_metrics,
                    "frames": args.frames,
                },
                args.output_dir / "best.pth",
            )

    result = {
        "model": "r2plus1d_18",
        "pretrained": not args.no_pretrained,
        "freeze_backbone": args.freeze_backbone,
        "overfit": args.overfit,
        "device": str(device),
        "amp": use_amp,
        "classes": classes,
        "train_samples": len(train_dataset),
        "val_samples": len(val_dataset),
        "frames": args.frames,
        "cache_dir": str(args.cache_dir) if args.cache_dir else None,
        "history": history,
        "best_val_macro_f1": best_macro_f1,
    }
    (args.output_dir / "metrics.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Saved results to {args.output_dir}")


if __name__ == "__main__":
    main()
