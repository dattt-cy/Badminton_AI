"""Train an R(2+1)D-18 hit verifier from ShuttleSet match videos."""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision.models.video import r2plus1d_18
from torchvision.transforms import functional as VF

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.training.train_fine_badminton_rgb import confusion_metrics


CLASSES = ["no_hit", "upper_hit", "lower_hit"]
SIDE_LABEL = {"top": 1, "upper": 1, "bottom": 2, "lower": 2}


@dataclass(frozen=True)
class Sample:
    match_id: str
    video_path: Path
    center: int
    label: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path("data/manifests/shuttleset_rgb.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("work_dirs/r2plus1d18_shuttleset_hit"))
    parser.add_argument("--source-checkpoint", type=Path, default=Path("work_dirs/r2plus1d18_shuttleset_mixed_crop_full_e8_e10_b4/best.pth"))
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--frames", type=int, default=16)
    parser.add_argument("--positive-radius", type=int, default=2)
    parser.add_argument("--negative-exclusion", type=int, default=12)
    parser.add_argument("--positives-per-match", type=int, default=200)
    parser.add_argument("--val-positives-per-match", type=int, default=100)
    parser.add_argument("--negative-ratio", type=float, default=2.0)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--backbone-learning-rate", type=float, default=3e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--unfreeze-layer4", action="store_true")
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--seed", type=int, default=20260917)
    return parser.parse_args()


def load_matches(path: Path, split: str) -> dict[str, dict]:
    matches: dict[str, dict] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            if row["split"] != split:
                continue
            side = SIDE_LABEL.get(row["player_side"].strip().lower())
            if side is None:
                continue
            item = matches.setdefault(row["match_id"], {"video_path": Path(row["video_path"]), "hits": []})
            item["hits"].append((int(float(row["hit_frame"])), side))
    for item in matches.values():
        item["hits"] = sorted(set(item["hits"]))
    return matches


def build_samples(matches: dict[str, dict], positives_per_match: int, negative_ratio: float, exclusion: int, radius: int, seed: int) -> list[Sample]:
    rng = random.Random(seed)
    samples: list[Sample] = []
    for match_id, item in sorted(matches.items(), key=lambda pair: int(pair[0])):
        hits = list(item["hits"])
        rng.shuffle(hits)
        positives = hits[: min(positives_per_match, len(hits))]
        for frame, label in positives:
            samples.append(Sample(match_id, item["video_path"], max(0, frame + rng.randint(-radius, radius)), label))
        all_frames = np.asarray([frame for frame, _ in item["hits"]], dtype=np.int64)
        low, high = int(all_frames.min()), int(all_frames.max())
        wanted = int(round(len(positives) * negative_ratio))
        negatives: set[int] = set()
        attempts = 0
        while len(negatives) < wanted and attempts < wanted * 200:
            attempts += 1
            frame = rng.randint(low, high)
            if int(np.min(np.abs(all_frames - frame))) > exclusion:
                negatives.add(frame)
        samples.extend(Sample(match_id, item["video_path"], frame, 0) for frame in negatives)
    rng.shuffle(samples)
    return samples


class HitDataset(Dataset):
    def __init__(self, samples: list[Sample], frames: int, training: bool, cache_dir: Path | None) -> None:
        self.samples, self.frames, self.training, self.cache_dir = samples, frames, training, cache_dir

    def __len__(self) -> int:
        return len(self.samples)

    def _decode(self, sample: Sample) -> torch.Tensor:
        key = f"{sample.match_id}_{sample.center}_{self.frames}.npy"
        cache_path = self.cache_dir / key if self.cache_dir else None
        if cache_path is not None and cache_path.is_file():
            return torch.from_numpy(np.load(cache_path))
        half = self.frames // 2
        start = max(0, sample.center - half)
        cap = cv2.VideoCapture(str(sample.video_path))
        cap.set(cv2.CAP_PROP_POS_FRAMES, start)
        decoded = []
        try:
            for _ in range(self.frames):
                ok, frame = cap.read()
                if not ok:
                    break
                decoded.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        finally:
            cap.release()
        if not decoded:
            raise ValueError(f"Cannot decode {sample.video_path} at frame {start}")
        while len(decoded) < self.frames:
            decoded.append(decoded[-1])
        tensor = torch.from_numpy(np.stack(decoded)).permute(0, 3, 1, 2)
        tensor = VF.resize(tensor, [128, 171], antialias=False)
        tensor = VF.center_crop(tensor, [112, 112]).contiguous()
        if cache_path is not None:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            temporary = cache_path.with_suffix(".tmp.npy")
            np.save(temporary, tensor.numpy())
            temporary.replace(cache_path)
        return tensor

    def __getitem__(self, index: int):
        sample = self.samples[index]
        tensor = self._decode(sample).float().div_(255)
        mean = tensor.new_tensor([0.43216, 0.394666, 0.37645]).view(1, 3, 1, 1)
        std = tensor.new_tensor([0.22803, 0.22145, 0.216989]).view(1, 3, 1, 1)
        tensor = ((tensor - mean) / std).permute(1, 0, 2, 3)
        if self.training and torch.rand(()) < 0.5:
            tensor = torch.flip(tensor, dims=(-1,))
        return tensor, sample.label


def initialize_model(checkpoint_path: Path, unfreeze_layer4: bool) -> nn.Module:
    model = r2plus1d_18(weights=None)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    source = {
        key.removeprefix("backbone."): value
        for key, value in checkpoint["model"].items()
        if key.startswith("backbone.") and not key.startswith("backbone.fc.")
    }
    missing, unexpected = model.load_state_dict(source, strict=False)
    if unexpected or any(key != "fc.weight" and key != "fc.bias" for key in missing):
        raise ValueError(f"Backbone mismatch: missing={missing}, unexpected={unexpected}")
    model.fc = nn.Linear(model.fc.in_features, len(CLASSES))
    for parameter in model.parameters():
        parameter.requires_grad = False
    for parameter in model.fc.parameters():
        parameter.requires_grad = True
    if unfreeze_layer4:
        for parameter in model.layer4.parameters():
            parameter.requires_grad = True
    return model


def run_epoch(model, loader, device, criterion, optimizer, use_amp, scaler=None) -> tuple[float, torch.Tensor]:
    training = optimizer is not None
    model.train(training)
    if training:
        # Frozen batch-normalization statistics must remain fixed.
        for module in model.modules():
            if isinstance(module, nn.BatchNorm3d) and not any(p.requires_grad for p in module.parameters()):
                module.eval()
    confusion = torch.zeros(3, 3, dtype=torch.int64)
    loss_sum = count = 0
    context = torch.enable_grad() if training else torch.inference_mode()
    with context:
        for clips, labels in loader:
            clips, labels = clips.to(device), labels.to(device)
            if training:
                optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                logits = model(clips)
                loss = criterion(logits, labels)
            if training:
                if scaler is not None:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()
            for truth, prediction in zip(labels.cpu(), logits.argmax(1).cpu()):
                confusion[int(truth), int(prediction)] += 1
            loss_sum += float(loss.detach()) * len(labels)
            count += len(labels)
    return loss_sum / max(count, 1), confusion


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    train_samples = build_samples(load_matches(args.manifest, "train"), args.positives_per_match, args.negative_ratio, args.negative_exclusion, args.positive_radius, args.seed)
    val_samples = build_samples(load_matches(args.manifest, "val"), args.val_positives_per_match, args.negative_ratio, args.negative_exclusion, 0, args.seed + 1)
    train_loader = DataLoader(HitDataset(train_samples, args.frames, True, args.cache_dir / "train" if args.cache_dir else None), batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=True)
    val_loader = DataLoader(HitDataset(val_samples, args.frames, False, args.cache_dir / "val" if args.cache_dir else None), batch_size=args.batch_size, num_workers=args.workers, pin_memory=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = initialize_model(args.source_checkpoint, args.unfreeze_layer4).to(device)
    parameter_groups = [{"params": model.fc.parameters(), "lr": args.learning_rate}]
    if args.unfreeze_layer4:
        parameter_groups.append({"params": model.layer4.parameters(), "lr": args.backbone_learning_rate})
    optimizer = torch.optim.AdamW(parameter_groups, weight_decay=args.weight_decay)
    counts = np.bincount([sample.label for sample in train_samples], minlength=3)
    weights = torch.tensor(len(train_samples) / (3 * counts), dtype=torch.float32, device=device)
    criterion = nn.CrossEntropyLoss(weight=weights)
    history, best = [], -1.0
    args.output_dir.mkdir(parents=True, exist_ok=True)
    use_amp = args.amp and device.type == "cuda"
    scaler = torch.amp.GradScaler(device.type, enabled=use_amp)
    for epoch in range(1, args.epochs + 1):
        train_loss, train_confusion = run_epoch(model, train_loader, device, criterion, optimizer, use_amp, scaler)
        val_loss, val_confusion = run_epoch(model, val_loader, device, criterion, None, use_amp)
        train_metrics, val_metrics = confusion_metrics(train_confusion), confusion_metrics(val_confusion)
        row = {"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss, "train": train_metrics, "val": val_metrics}
        history.append(row)
        print(f"epoch={epoch} train_loss={train_loss:.4f} val_loss={val_loss:.4f} val_macro_f1={val_metrics['macro_f1']:.4f}", flush=True)
        if val_metrics["macro_f1"] > best:
            best = val_metrics["macro_f1"]
            torch.save({"model": model.state_dict(), "epoch": epoch, "classes": CLASSES, "frames": args.frames, "source_checkpoint": str(args.source_checkpoint), "val_metrics": val_metrics}, args.output_dir / "best.pth")
        (args.output_dir / "metrics.json").write_text(json.dumps({"device": str(device), "train_samples": len(train_samples), "val_samples": len(val_samples), "class_counts": counts.tolist(), "history": history}, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
