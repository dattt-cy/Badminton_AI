"""Train leakage-safe pair specialists from online validation-pipeline features."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from ai_classifier.models import PairSpecialist, STROKE_CLASSES

PAIRS = (("net_attack", "lift"), ("net_attack", "net_shot"), ("smash", "drop"))


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, nargs="+", required=True)
    parser.add_argument("--validation", type=Path, nargs="+", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--seed", type=int, default=20260928)
    return parser.parse_args()


def load(paths):
    return [row for path in paths for row in torch.load(path, weights_only=False)]


def select(rows, pair):
    ids = tuple(STROKE_CLASSES.index(name) for name in pair)
    selected = [row for row in rows if set(map(int, row["top2"])) == set(ids) and int(row["target"]) in ids]
    if not selected:
        raise ValueError(f"No samples for {pair}")
    features = torch.stack([torch.cat([row["feature"], row["physics"]]) for row in selected])
    labels = torch.tensor([int(row["target"]) == ids[1] for row in selected], dtype=torch.float32)
    return features, labels


@torch.inference_mode()
def best_threshold(model, features, labels, device):
    probabilities = model(features.to(device)).sigmoid().cpu()
    choices = []
    for threshold in np.linspace(0.25, 0.75, 101):
        predictions = probabilities >= threshold
        accuracy = float((predictions == labels.bool()).float().mean())
        balanced = 0.5 * sum(
            float((predictions[labels.bool() == value] == value).float().mean())
            for value in (False, True) if (labels.bool() == value).any()
        )
        choices.append((accuracy, balanced, float(threshold)))
    return max(choices, key=lambda item: (item[0], item[1]))


def main():
    args = parse_args()
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_rows, val_rows = load(args.train), load(args.validation)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary = {}
    for pair in PAIRS:
        train_x, train_y = select(train_rows, pair)
        val_x, val_y = select(val_rows, pair)
        positives = int(train_y.sum()); negatives = len(train_y) - positives
        loader = DataLoader(TensorDataset(train_x, train_y), batch_size=64, shuffle=True)
        model = PairSpecialist(train_x.shape[1]).to(device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([negatives / max(positives, 1)], device=device))
        optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=2e-4)
        best = (-1.0, -1.0)
        filename = f"{pair[0]}__{pair[1]}.pth"
        for epoch in range(1, args.epochs + 1):
            model.train()
            for features, labels in loader:
                optimizer.zero_grad(set_to_none=True)
                loss = criterion(model(features.to(device)), labels.to(device))
                loss.backward(); optimizer.step()
            model.eval()
            accuracy, balanced, threshold = best_threshold(model, val_x, val_y, device)
            if (accuracy, balanced) > best:
                best = (accuracy, balanced)
                torch.save({
                    "model": model.state_dict(), "input_dim": train_x.shape[1],
                    "class_names": pair, "threshold": threshold, "epoch": epoch,
                    "validation_accuracy": accuracy, "validation_balanced_accuracy": balanced,
                    "train_samples": len(train_y), "validation_samples": len(val_y),
                }, args.output_dir / filename)
        summary["__".join(pair)] = torch.load(args.output_dir / filename, map_location="cpu", weights_only=False)
        summary["__".join(pair)].pop("model")
        print(pair, summary["__".join(pair)], flush=True)
    (args.output_dir / "metrics.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
