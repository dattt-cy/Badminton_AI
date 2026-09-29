"""Train a Top-2 reranker on online validation-pipeline features only."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from ai_classifier.models import Top2Reranker


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, nargs="+", required=True)
    parser.add_argument("--validation", type=Path, nargs="+", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--seed", type=int, default=20260928)
    return parser.parse_args()


def load(paths: list[Path]) -> dict[str, torch.Tensor]:
    rows = [row for path in paths for row in torch.load(path, weights_only=False)]
    return {
        "features": torch.stack([row["feature"] for row in rows]),
        "top2": torch.stack([row["top2"] for row in rows]).long(),
        "targets": torch.tensor([row["target"] for row in rows]).long(),
    }


def decisive(data: dict[str, torch.Tensor]) -> TensorDataset:
    keep = data["targets"] == data["top2"][:, 0]
    swap = data["targets"] == data["top2"][:, 1]
    eligible = keep | swap
    return TensorDataset(data["features"][eligible], swap[eligible].float())


@torch.inference_mode()
def score(model, data, threshold, device):
    probabilities = model(data["features"].to(device)).sigmoid().cpu()
    predictions = torch.where(probabilities >= threshold, data["top2"][:, 1], data["top2"][:, 0])
    return {
        "accuracy": float((predictions == data["targets"]).float().mean()),
        "swap_rate": float((probabilities >= threshold).float().mean()),
        "top2_oracle": float((data["top2"] == data["targets"][:, None]).any(1).float().mean()),
    }


def main() -> None:
    args = parse_args()
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train, validation = load(args.train), load(args.validation)
    dataset = decisive(train)
    positives = int(dataset.tensors[1].sum())
    negatives = len(dataset) - positives
    loader = DataLoader(dataset, batch_size=128, shuffle=True)
    model = Top2Reranker(train["features"].shape[1]).to(device)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([negatives / max(positives, 1)], device=device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    best = (-1.0, None)
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        for features, labels in loader:
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(features.to(device)), labels.to(device))
            loss.backward(); optimizer.step()
        model.eval()
        choices = [(float(t), score(model, validation, float(t), device)) for t in np.linspace(0.35, 0.95, 61)]
        threshold, metrics = max(choices, key=lambda item: item[1]["accuracy"])
        history.append({"epoch": epoch, "threshold": threshold, **metrics})
        if metrics["accuracy"] > best[0]:
            best = (metrics["accuracy"], epoch)
            torch.save({"model": model.state_dict(), "input_dim": train["features"].shape[1],
                        "threshold": threshold, "epoch": epoch, "val_metrics": metrics},
                       args.output_dir / "best.pth")
        print(f"epoch={epoch:02d} val_acc={metrics['accuracy']:.4f} threshold={threshold:.2f}", flush=True)
    payload = torch.load(args.output_dir / "best.pth", map_location=device, weights_only=False)
    (args.output_dir / "metrics.json").write_text(json.dumps({"best": payload["val_metrics"], "history": history}, indent=2)+"\n")


if __name__ == "__main__":
    main()
