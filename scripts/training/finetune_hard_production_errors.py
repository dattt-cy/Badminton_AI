"""Targeted Hard-Example Production Fine-Tuning with Match-Level Group Split.

Protocol:
1. Strict match-level split:
   - Train matches: 1, 2, 3, 4, 7, 11, 13, 14, 15, 16, 17
   - Held-out validation matches: 41, 44
2. Clean replay buffer to prevent forgetting.
3. Hard-example mining: samples misclassified by the baseline receive boosted sampling/loss weight.
4. Selection metric: Macro-F1 / Balanced Accuracy on held-out real matches 41 & 44.
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
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from ai_classifier.models import KAPScratchModel, STROKE_CLASSES
from scripts.training.train_kap_scratch import confusion_metrics, load_data

KEYS = ("emb", "s_j", "s_p", "s_s", "s_c", "s_q")
TRAIN_MATCHES = {"1", "2", "3", "4", "7", "11", "13", "14", "15", "16", "17"}
VAL_MATCHES = {"41", "44"}


class HardMiningDataset(Dataset):
    def __init__(self, records: list[dict], hard_boost: float = 2.5):
        self.records = records
        self.weights = []
        for r in records:
            # Boost weight if sample was previously misclassified (hard error)
            w = hard_boost if not r.get("is_correct", True) else 1.0
            self.weights.append(w)
        self.weights = torch.tensor(self.weights, dtype=torch.float)

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        r = self.records[idx]
        return (
            torch.as_tensor(r["emb"]).float().view(512),
            torch.as_tensor(r["s_j"]).float().view(32, 68),
            torch.as_tensor(r["s_p"]).float().view(32, 4),
            torch.as_tensor(r["s_s"]).float().view(32, 2),
            torch.as_tensor(r["s_c"]).float().view(32, 1),
            torch.as_tensor(r["s_q"]).float().view(4),
            torch.tensor(r["gt_idx"], dtype=torch.long),
            self.weights[idx],
        )


def evaluate(model: nn.Module, records: list[dict], device: torch.device) -> dict:
    model.eval()
    confusion = torch.zeros(8, 8, dtype=torch.int64)

    emb = torch.cat([torch.as_tensor(x["emb"]).float().view(1, 512) for x in records], dim=0)
    pose = torch.cat([torch.as_tensor(x["s_j"]).float().view(1, 32, 68) for x in records], dim=0)
    court = torch.cat([torch.as_tensor(x["s_p"]).float().view(1, 32, 4) for x in records], dim=0)
    shuttle = torch.cat([torch.as_tensor(x["s_s"]).float().view(1, 32, 2) for x in records], dim=0)
    contact = torch.cat([torch.as_tensor(x["s_c"]).float().view(1, 32, 1) for x in records], dim=0)
    quality = torch.cat([torch.as_tensor(x["s_q"]).float().view(1, 4) for x in records], dim=0)
    gts = torch.tensor([x["gt_idx"] for x in records], dtype=torch.long)

    batch_size = 128
    with torch.inference_mode():
        for start in range(0, len(records), batch_size):
            e = emb[start : start + batch_size].to(device)
            p = pose[start : start + batch_size].to(device)
            c = court[start : start + batch_size].to(device)
            s = shuttle[start : start + batch_size].to(device)
            ct = contact[start : start + batch_size].to(device)
            q = quality[start : start + batch_size].to(device)
            target = gts[start : start + batch_size]

            os, _, _ = model(e, p, c, s, ct, q)
            preds = os.argmax(1).cpu()

            for t, pred in zip(target, preds):
                confusion[int(t), int(pred)] += 1

    return confusion_metrics(confusion)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, default=ROOT / "src/ai_classifier/models/checkpoints/kap_classifier.pth")
    parser.add_argument("--clean", type=Path, default=ROOT / "work_dirs/cr_gated_fusion/sequence_features.npz")
    parser.add_argument("--production-data", type=Path, default=ROOT / "work_dirs/error_mining/production_all_aligned.pt")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "work_dirs/kap_hard_mined_production")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--learning-rate", type=float, default=6e-6)
    parser.add_argument("--hard-boost", type=float, default=3.0)
    parser.add_argument("--clean-ratio", type=float, default=0.4)
    parser.add_argument("--seed", type=int, default=20261021)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load production data and split by match
    all_data = torch.load(args.production_data, map_location="cpu", weights_only=False)
    train_records = [r for r in all_data if str(r["match_id"]) in TRAIN_MATCHES]
    val_records = [r for r in all_data if str(r["match_id"]) in VAL_MATCHES]

    print(f"Loaded production records: Total={len(all_data)}")
    print(f"Train matches ({','.join(sorted(TRAIN_MATCHES))}): {len(train_records)} samples")
    print(f"Held-out validation matches ({','.join(sorted(VAL_MATCHES))}): {len(val_records)} samples")

    # 2. Load clean replay buffer
    clean_tensors, clean_splits = load_data(args.clean)
    clean_indices = list(clean_splits["train"])
    clean_inputs = [
        clean_tensors["rgb"],
        clean_tensors["pose"],
        clean_tensors["court"],
        clean_tensors["shuttle"],
        clean_tensors["contact_dist"],
        clean_tensors["quality"],
    ]
    clean_targets = clean_tensors["stroke"]

    # 3. Model setup
    model = KAPScratchModel().to(device)
    payload = torch.load(args.base, map_location=device, weights_only=False)
    model.load_state_dict(payload["model"] if "model" in payload else payload)

    # Baseline evaluation on validation set
    base_metrics = evaluate(model, val_records, device)
    print("\n--- BASELINE VALIDATION PERFORMANCE (Matches 41 & 44) ---")
    print(f"Accuracy         : {base_metrics['accuracy']:.2f}%")
    print(f"Balanced Accuracy: {base_metrics['balanced_accuracy']:.2f}%")
    print(f"Macro F1         : {base_metrics['macro_f1']:.2f}%")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss(reduction="none")

    train_ds = HardMiningDataset(train_records, hard_boost=args.hard_boost)
    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)

    best_macro_f1 = base_metrics["macro_f1"]
    best_epoch = 0
    history = []

    clean_step = int(len(train_records) * args.clean_ratio)

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        n_batches = 0

        for emb, pose, court, shuttle, contact, quality, targets, weights in train_loader:
            emb, pose, court = emb.to(device), pose.to(device), court.to(device)
            shuttle, contact, quality = shuttle.to(device), contact.to(device), quality.to(device)
            targets = targets.to(device)
            weights = weights.to(device)

            # Clean replay batch mix
            c_idx = random.sample(clean_indices, k=min(32, len(clean_indices)))
            c_inputs = [clean_inputs[k][c_idx].to(device) for k in range(len(KEYS))]
            c_targets = clean_targets[c_idx].to(device)

            optimizer.zero_grad()

            # Forward production batch
            os_prod, _, _ = model(emb, pose, court, shuttle, contact, quality)
            loss_prod_raw = criterion(os_prod, targets)
            loss_prod = (loss_prod_raw * weights).mean()

            # Forward clean replay batch
            os_clean, _, _ = model(*c_inputs)
            loss_clean = criterion(os_clean, c_targets).mean()

            loss = loss_prod + 0.5 * loss_clean
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        val_metrics = evaluate(model, val_records, device)
        avg_loss = total_loss / max(1, n_batches)

        print(
            f"Epoch {epoch:2d}/{args.epochs}: Loss={avg_loss:.4f} | "
            f"Val Acc={val_metrics['accuracy']:.2f}% | "
            f"BalAcc={val_metrics['balanced_accuracy']:.2f}% | "
            f"Macro-F1={val_metrics['macro_f1']:.2f}%"
        )

        history.append({
            "epoch": epoch,
            "loss": avg_loss,
            "val_acc": val_metrics["accuracy"],
            "val_bal_acc": val_metrics["balanced_accuracy"],
            "val_macro_f1": val_metrics["macro_f1"],
        })

        if val_metrics["macro_f1"] > best_macro_f1:
            best_macro_f1 = val_metrics["macro_f1"]
            best_epoch = epoch
            torch.save(
                {
                    "model": model.state_dict(),
                    "epoch": epoch,
                    "metrics": val_metrics,
                    "history": history,
                },
                args.output_dir / "best.pth",
            )
            print(f"  ==> [NEW BEST] Macro-F1: {best_macro_f1:.2f}% at epoch {epoch}")

    print("\n" + "=" * 60)
    print("FINETUNING FINISHED")
    print(f"Baseline Macro-F1: {base_metrics['macro_f1']:.2f}%")
    print(f"Best Macro-F1    : {best_macro_f1:.2f}% (Epoch {best_epoch})")
    print(f"Checkpoint saved : {args.output_dir / 'best.pth'}")
    print("=" * 60)

    # Save summary report
    with open(args.output_dir / "training_summary.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "baseline": base_metrics,
                "best_epoch": best_epoch,
                "best_macro_f1": best_macro_f1,
                "history": history,
            },
            f,
            indent=2,
        )


if __name__ == "__main__":
    main()
