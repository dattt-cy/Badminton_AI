"""Pilot a frozen-backbone three-view temporal consensus stroke head."""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch import nn

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.training.train_fine_badminton_rgb import confusion_metrics
from scripts.training.train_shuttleset_rgb_multitask import MultiTaskR2Plus1D, STROKE_CLASSES


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path("data/manifests/shuttleset_rgb.csv"))
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--train-per-class", type=int, default=300)
    parser.add_argument("--val-per-class", type=int, default=100)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--feature-cache", type=Path)
    parser.add_argument("--seed", type=int, default=20260921)
    return parser.parse_args()


def select_rows(path: Path, cache: Path, split: str, per_class: int, seed: int):
    grouped = defaultdict(list)
    with path.open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            if row["split"] == split and (cache / f"{row['sample_id']}.npy").is_file():
                grouped[row["coarse_label"]].append(row)
    rng = random.Random(seed)
    rows = []
    for label in STROKE_CLASSES:
        candidates = sorted(grouped[label], key=lambda row: row["sample_id"])
        rows.extend(rng.sample(candidates, min(per_class, len(candidates))))
    rng.shuffle(rows)
    return rows


def normalize(array: np.ndarray) -> torch.Tensor:
    tensor = torch.from_numpy(array).float().div_(255.0)
    mean = tensor.new_tensor([0.43216, 0.394666, 0.37645]).view(1, 3, 1, 1)
    std = tensor.new_tensor([0.22803, 0.22145, 0.216989]).view(1, 3, 1, 1)
    return ((tensor - mean) / std).permute(1, 0, 2, 3)


def shifted(array: np.ndarray, offset: int) -> np.ndarray:
    indices = np.clip(np.arange(len(array)) + offset, 0, len(array) - 1)
    return array[indices]


def extract(rows, cache, backbone, device, batch_size):
    features, labels, pending, pending_labels = [], [], [], []
    with torch.inference_mode():
        for index, row in enumerate(rows):
            array = np.load(cache / f"{row['sample_id']}.npy")
            pending.extend(normalize(shifted(array, offset)) for offset in (-1, 0, 1))
            pending_labels.append(STROKE_CLASSES.index(row["coarse_label"]))
            if len(pending_labels) == batch_size or index == len(rows) - 1:
                batch = torch.stack(pending).to(device)
                with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                    embedding = backbone(batch).float().cpu().reshape(len(pending_labels), 3, -1)
                features.append(embedding)
                labels.extend(pending_labels)
                pending, pending_labels = [], []
    return torch.cat(features), torch.tensor(labels, dtype=torch.long)


class ConsensusHead(nn.Module):
    def __init__(self, dimension: int):
        super().__init__()
        self.network = nn.Sequential(
            nn.LayerNorm(dimension * 3), nn.Linear(dimension * 3, 512),
            nn.GELU(), nn.Dropout(0.3), nn.Linear(512, len(STROKE_CLASSES)),
        )

    def forward(self, values):
        return self.network(values.flatten(1))


def metrics(head, features, labels, device):
    head.eval()
    confusion = torch.zeros(8, 8, dtype=torch.int64)
    with torch.inference_mode():
        prediction = head(features.to(device)).argmax(1).cpu()
    for truth, predicted in zip(labels, prediction):
        confusion[int(truth), int(predicted)] += 1
    return confusion_metrics(confusion)


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    cache = args.cache_dir / "frames_16"
    train_rows = select_rows(args.manifest, cache, "train", args.train_per_class, args.seed)
    val_rows = select_rows(args.manifest, cache, "val", args.val_per_class, args.seed + 1)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model = MultiTaskR2Plus1D()
    model.load_state_dict(checkpoint["model"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    backbone = model.backbone.to(device).eval()
    for parameter in backbone.parameters():
        parameter.requires_grad = False
    feature_cache = args.feature_cache or (args.output_dir / "features.pt")
    if feature_cache.is_file():
        cached = torch.load(feature_cache, map_location="cpu", weights_only=False)
        train_x, train_y = cached["train_x"], cached["train_y"]
        val_x, val_y = cached["val_x"], cached["val_y"]
        print(f"loaded features: {feature_cache}", flush=True)
    else:
        print(f"extract train={len(train_rows)} val={len(val_rows)}", flush=True)
        train_x, train_y = extract(train_rows, cache, backbone, device, args.batch_size)
        val_x, val_y = extract(val_rows, cache, backbone, device, args.batch_size)
        feature_cache.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "train_x": train_x, "train_y": train_y,
            "val_x": val_x, "val_y": val_y,
            "checkpoint": str(args.checkpoint.resolve()),
        }, feature_cache)
        print(f"saved features: {feature_cache}", flush=True)
    head = ConsensusHead(train_x.shape[-1]).to(device)
    optimizer = torch.optim.AdamW(head.parameters(), lr=3e-4, weight_decay=1e-4)
    counts = torch.bincount(train_y, minlength=8).float()
    criterion = nn.CrossEntropyLoss(weight=(len(train_y) / (8 * counts)).to(device))
    rng = np.random.default_rng(args.seed)
    history, best = [], -1.0
    for epoch in range(1, args.epochs + 1):
        head.train()
        order = rng.permutation(len(train_y))
        for start in range(0, len(train_y), args.batch_size):
            ids = torch.from_numpy(order[start:start + args.batch_size])
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(head(train_x[ids].to(device)), train_y[ids].to(device))
            loss.backward(); optimizer.step()
        result = metrics(head, val_x, val_y, device)
        history.append({"epoch": epoch, "val": result})
        print(f"epoch={epoch} val_macro_f1={result['macro_f1']:.4f}", flush=True)
        if result["macro_f1"] > best:
            best = result["macro_f1"]
            torch.save({"model": head.state_dict(), "epoch": epoch, "classes": STROKE_CLASSES}, args.output_dir / "best.pth")
    (args.output_dir / "metrics.json").write_text(json.dumps({"history": history, "best_macro_f1": best}, indent=2) + "\n")


if __name__ == "__main__":
    main()
