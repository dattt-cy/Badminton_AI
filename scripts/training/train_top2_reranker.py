"""Train a leakage-safe neural keep/swap reranker for robust KAP Top-2.

The base model is frozen. Training examples are restricted to cases where the
ground-truth label is either rank 1 (KEEP) or rank 2 (SWAP). Model selection
and the swap threshold use only the validation split.
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
from torch.utils.data import DataLoader, TensorDataset

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ai_classifier.models import (  # noqa: E402
    KAPScratchModel, Top2Reranker,
)
from ai_classifier.models.kap_fusion import kap_reranker_features  # noqa: E402
from scripts.training.train_kap_scratch import (  # noqa: E402
    confusion_metrics,
    load_data,
    make_loader,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--features", type=Path,
        default=Path("work_dirs/cr_gated_fusion/sequence_features.npz"),
    )
    parser.add_argument(
        "--base-checkpoint", type=Path,
        default=Path("work_dirs/kap_robust_seed20260926/kap_best.pth"),
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path("work_dirs/kap_top2_reranker_seed20260927"),
    )
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=20260927)
    parser.add_argument("--evaluate-test", action="store_true")
    return parser.parse_args()


def load_base(path: Path, device: torch.device) -> KAPScratchModel:
    model = KAPScratchModel().to(device)
    payload = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(payload["model"])
    model.eval()
    return model


def meta_features(
    model: KAPScratchModel, inputs: list[torch.Tensor]
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return kap_reranker_features(model, *inputs)


@torch.inference_mode()
def collect(
    base: KAPScratchModel, loader: DataLoader, device: torch.device,
) -> dict[str, torch.Tensor]:
    features, logits, top2, targets = [], [], [], []
    for batch in loader:
        inputs = [value.to(device, non_blocking=True) for value in batch[:6]]
        feat, batch_logits, batch_top2 = meta_features(base, inputs)
        features.append(feat.cpu())
        logits.append(batch_logits.cpu())
        top2.append(batch_top2.cpu())
        targets.append(batch[6].cpu())
    return {
        "features": torch.cat(features), "logits": torch.cat(logits),
        "top2": torch.cat(top2), "targets": torch.cat(targets),
    }


def decisive_dataset(data: dict[str, torch.Tensor]) -> TensorDataset:
    top2, targets = data["top2"], data["targets"]
    keep = targets == top2[:, 0]
    swap = targets == top2[:, 1]
    eligible = keep | swap
    labels = swap[eligible].float()
    return TensorDataset(data["features"][eligible], labels)


def rerank_metrics(
    reranker: Top2Reranker,
    data: dict[str, torch.Tensor],
    threshold: float,
    device: torch.device,
) -> dict:
    reranker.eval()
    with torch.inference_mode():
        swap_probability = reranker(data["features"].to(device)).sigmoid().cpu()
    top2, targets = data["top2"], data["targets"]
    predictions = torch.where(swap_probability >= threshold, top2[:, 1], top2[:, 0])
    confusion = torch.zeros(8, 8, dtype=torch.int64)
    for truth, prediction in zip(targets, predictions):
        confusion[int(truth), int(prediction)] += 1
    metrics = confusion_metrics(confusion)
    metrics.update({
        "swap_threshold": threshold,
        "swap_rate": float((swap_probability >= threshold).float().mean()),
        "top2_oracle_accuracy": float((top2 == targets[:, None]).any(dim=1).float().mean()),
    })
    return metrics


def best_threshold(
    reranker: Top2Reranker, data: dict[str, torch.Tensor], device: torch.device
) -> tuple[float, dict]:
    candidates = np.linspace(0.50, 0.95, 46)
    scored = [(float(value), rerank_metrics(reranker, data, float(value), device)) for value in candidates]
    # Accuracy is the explicit optimization target; macro-F1 breaks ties.
    return max(scored, key=lambda item: (item[1]["accuracy"], item[1]["macro_f1"]))


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tensors, indices = load_data(args.features)
    loaders = {
        split: make_loader(tensors, indices[split], args.batch_size, shuffle=False)
        for split in ("train", "val", "test")
    }
    base = load_base(args.base_checkpoint, device)
    collected = {split: collect(base, loader, device) for split, loader in loaders.items()}
    train_set = decisive_dataset(collected["train"])
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    positives = int(train_set.tensors[1].sum())
    negatives = len(train_set) - positives
    print(f"[reranker] decisive train={len(train_set)} keep={negatives} swap={positives}")

    reranker = Top2Reranker(collected["train"]["features"].shape[1]).to(device)
    positive_weight = torch.tensor([negatives / max(positives, 1)], device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=positive_weight)
    optimizer = torch.optim.AdamW(
        reranker.parameters(), lr=args.learning_rate, weight_decay=1e-4
    )
    best_accuracy, history = -1.0, []
    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = args.output_dir / "best.pth"
    for epoch in range(1, args.epochs + 1):
        reranker.train()
        total = 0.0
        for features, labels in train_loader:
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(reranker(features.to(device)), labels.to(device))
            loss.backward()
            optimizer.step()
            total += float(loss.detach()) * len(labels)
        threshold, val_metrics = best_threshold(reranker, collected["val"], device)
        row = {
            "epoch": epoch, "train_loss": total / len(train_set),
            "threshold": threshold, "val": val_metrics,
        }
        history.append(row)
        print(
            f"epoch={epoch:02d} loss={row['train_loss']:.4f} "
            f"val_acc={val_metrics['accuracy']:.4f} val_f1={val_metrics['macro_f1']:.4f} "
            f"threshold={threshold:.2f} swap_rate={val_metrics['swap_rate']:.3f}",
            flush=True,
        )
        if val_metrics["accuracy"] > best_accuracy:
            best_accuracy = val_metrics["accuracy"]
            torch.save({
                "model": reranker.state_dict(), "input_dim": collected["train"]["features"].shape[1],
                "epoch": epoch, "threshold": threshold, "val_metrics": val_metrics,
                "base_checkpoint": str(args.base_checkpoint),
            }, checkpoint)

    payload = torch.load(checkpoint, map_location=device, weights_only=False)
    reranker.load_state_dict(payload["model"])
    result = {
        "best_epoch": payload["epoch"], "threshold": payload["threshold"],
        "base_checkpoint": str(args.base_checkpoint), "history": history,
        "base_val": rerank_metrics(reranker, collected["val"], 1.01, device),
        "reranked_val": rerank_metrics(reranker, collected["val"], payload["threshold"], device),
    }
    if args.evaluate_test:
        result["base_test"] = rerank_metrics(reranker, collected["test"], 1.01, device)
        result["reranked_test"] = rerank_metrics(
            reranker, collected["test"], payload["threshold"], device
        )
    (args.output_dir / "metrics.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({key: value for key, value in result.items() if key != "history"}, indent=2))


if __name__ == "__main__":
    main()
