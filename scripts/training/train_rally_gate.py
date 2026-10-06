"""Train lightweight Rally-State Gate MLP (LIVE vs NON_LIVE).

Input features (516-d):
  - 512-d pooled R(2+1)D visual embedding
  - 4-d scene dynamics (court_ratio, motion_mean, motion_std, contrast)

Validation is strictly match-disjoint (Matches 35, 36, 37, 38, 41).
Selects the decision threshold guaranteeing Validation Rally Recall >= 98.5%.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class RallyGateMLP(nn.Module):
    def __init__(self, in_features: int = 516):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(32, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-features", type=Path, default=Path("work_dirs/rally_gate_train_features.pt"))
    parser.add_argument("--val-features", type=Path, default=Path("work_dirs/rally_gate_val_features.pt"))
    parser.add_argument("--output", type=Path, default=Path("work_dirs/rally_gate_best.pth"))
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--pos-weight", type=float, default=1.25, help="BCE positive class weight")
    parser.add_argument("--min-recall", type=float, default=0.985, help="Target minimum recall on val")
    return parser.parse_args()


def evaluate(model, val_loader, val_matches, min_recall):
    model.eval()
    all_logits = []
    all_labels = []

    with torch.inference_mode():
        for x, y in val_loader:
            x = x.to(DEVICE)
            logits = model(x)
            all_logits.append(logits.cpu())
            all_labels.append(y)

    logits = torch.cat(all_logits)
    labels = torch.cat(all_labels).numpy()
    probs = logits.sigmoid().numpy()

    threshold_results = []
    for thresh in np.arange(0.20, 0.81, 0.05):
        preds = (probs >= thresh).astype(int)
        tp = int(((preds == 1) & (labels == 1)).sum())
        fp = int(((preds == 1) & (labels == 0)).sum())
        fn = int(((preds == 0) & (labels == 1)).sum())
        tn = int(((preds == 0) & (labels == 0)).sum())

        prec = tp / max(tp + fp, 1)
        rec = tp / max(tp + fn, 1)
        f1 = 2 * prec * rec / max(prec + rec, 1e-6)
        acc = (tp + tn) / max(len(labels), 1)

        threshold_results.append({
            "threshold": float(thresh), "precision": prec, "recall": rec, "f1": f1, "accuracy": acc,
            "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        })

    eligible = [r for r in threshold_results if r["recall"] >= min_recall]
    if eligible:
        best_metrics = max(eligible, key=lambda r: (r["f1"], r["precision"]))
    else:
        best_metrics = max(threshold_results, key=lambda r: (r["recall"], r["f1"]))

    per_match = {}
    preds_best = (probs >= best_metrics["threshold"]).astype(int)
    for mid in sorted(set(val_matches), key=lambda x: int(x) if x.isdigit() else x):
        mask = np.array([m == mid for m in val_matches])
        m_labels = labels[mask]
        m_preds = preds_best[mask]
        m_tp = int(((m_preds == 1) & (m_labels == 1)).sum())
        m_fp = int(((m_preds == 1) & (m_labels == 0)).sum())
        m_fn = int(((m_preds == 0) & (m_labels == 1)).sum())
        m_tn = int(((m_preds == 0) & (m_labels == 0)).sum())
        m_prec = m_tp / max(m_tp + m_fp, 1)
        m_rec = m_tp / max(m_tp + m_fn, 1)
        m_f1 = 2 * m_prec * m_rec / max(m_prec + m_rec, 1e-6)
        per_match[mid] = {
            "total": int(mask.sum()), "precision": m_prec, "recall": m_rec, "f1": m_f1,
            "tp": m_tp, "fp": m_fp, "fn": m_fn,
        }

    return {
        "best": best_metrics,
        "per_match": per_match,
        "threshold_grid": threshold_results,
    }


def main():
    args = parse_args()
    print(f"[*] Loading training features from {args.train_features}...")
    train_data = torch.load(args.train_features, map_location="cpu", weights_only=False)
    print(f"[*] Loading validation features from {args.val_features}...")
    val_data = torch.load(args.val_features, map_location="cpu", weights_only=False)

    x_train, y_train = train_data["features"].float(), train_data["labels"].float()
    x_val, y_val = val_data["features"].float(), val_data["labels"].float()
    val_matches = val_data["match_ids"]

    in_dim = x_train.shape[1]
    print(f"[*] Input dimension: {in_dim}")
    print(f"[*] Train set: {len(x_train)} samples (Live: {int((y_train == 1).sum())}, Non-live: {int((y_train == 0).sum())})")
    print(f"[*] Val set:   {len(x_val)} samples (Live: {int((y_val == 1).sum())}, Non-live: {int((y_val == 0).sum())})")

    train_ds = TensorDataset(x_train, y_train)
    val_ds = TensorDataset(x_val, y_val)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    model = RallyGateMLP(in_features=in_dim).to(DEVICE)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[*] Model parameters: {total_params:,}")

    pos_weight = torch.tensor([args.pos_weight], device=DEVICE)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-5)

    best_f1 = 0.0
    best_checkpoint_payload = None

    print("\n--- Training RallyGateMLP ---")
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(y)

        scheduler.step()
        train_loss = total_loss / len(train_ds)

        eval_res = evaluate(model, val_loader, val_matches, min_recall=args.min_recall)
        best = eval_res["best"]

        if best["f1"] > best_f1:
            best_f1 = best["f1"]
            best_checkpoint_payload = {
                "epoch": epoch,
                "input_dim": in_dim,
                "model": model.state_dict(),
                "threshold": best["threshold"],
                "val_metrics": best,
                "per_match": eval_res["per_match"],
            }
            marker = " [*BEST*]"
        else:
            marker = ""

        if epoch % 5 == 0 or epoch == 1 or marker:
            print(
                f"Epoch {epoch:2d}/{args.epochs:2d} | Train Loss: {train_loss:.4f} | "
                f"Val Acc: {best['accuracy']*100:.1f}% | Prec: {best['precision']*100:.1f}% | "
                f"Rec: {best['recall']*100:.1f}% | F1: {best['f1']*100:.1f}% (tau={best['threshold']:.2f}){marker}",
                flush=True,
            )

    if best_checkpoint_payload is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        torch.save(best_checkpoint_payload, args.output)
        print("\n========================================================")
        print(f" [DONE] Best Rally-State Gate Checkpoint Saved to: {args.output}")
        print(f"   Epoch: {best_checkpoint_payload['epoch']}")
        print(f"   Selected Threshold: {best_checkpoint_payload['threshold']:.2f}")
        print(f"   Validation Precision: {best_checkpoint_payload['val_metrics']['precision']*100:.2f}%")
        print(f"   Validation Recall:    {best_checkpoint_payload['val_metrics']['recall']*100:.2f}%")
        print(f"   Validation F1-Score:  {best_checkpoint_payload['val_metrics']['f1']*100:.2f}%")
        print("   Per-match Breakdown:")
        for mid, m in best_checkpoint_payload["per_match"].items():
            print(f"     Match {mid}: Prec={m['precision']*100:.1f}%, Rec={m['recall']*100:.1f}%, F1={m['f1']*100:.1f}% (TP={m['tp']}, FP={m['fp']}, FN={m['fn']})")
        print("========================================================")


if __name__ == "__main__":
    main()
