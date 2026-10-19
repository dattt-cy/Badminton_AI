"""Multi-match real online-domain KAP adaptation with clean replay and held-out validation."""

from __future__ import annotations

import argparse
import csv
import json
import random
import re
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from ai_classifier.models import KAPScratchModel, STROKE_CLASSES
from scripts.training.train_kap_scratch import confusion_metrics, load_data

KEYS = ("emb", "s_j", "s_p", "s_s", "s_c", "s_q")

OFFSETS = {
    # Original 5 pilot matches
    "13": 16300,
    "14": 89539,
    "15": 42240,
    "16": 13792,
    "17": 68475,
    # 6 new rich matches
    "1": 14746,
    "2": 34460,
    "3": 11717,
    "4": 73728,
    "7": 47753,
    "11": 105597,
}


def args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, default=Path("work_dirs/kap_robust_seed20260926/kap_best.pth"))
    parser.add_argument("--clean", type=Path, default=Path("work_dirs/cr_gated_fusion/sequence_features.npz"))
    parser.add_argument("--train-online", type=Path, nargs="+", default=[
        Path("work_dirs/match13_5min_features.pt"),
        *[Path(f"work_dirs/online_train_match{m}_features.pt") for m in (14, 15, 16, 17, 1, 2, 3, 4, 7, 11)],
    ])
    parser.add_argument("--validation-online", type=Path, default=Path("work_dirs/match41_5min_aligned_features.pt"))
    parser.add_argument("--output-dir", type=Path, default=Path("work_dirs/kap_online_multi_matches"))
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--learning-rate", type=float, default=5e-6)
    parser.add_argument("--seed", type=int, default=20260929)
    parser.add_argument(
        "--selection-metric", choices=("accuracy", "balanced_accuracy", "macro_f1"),
        default="accuracy",
        help="Validation metric used to select the checkpoint.",
    )
    return parser.parse_args()


def stack(records: list[dict]) -> list[torch.Tensor]:
    return [torch.cat([row[key].detach().cpu() for row in records]).float() for key in KEYS]


def match_train(records: list[dict], match_id: str, offset: int) -> tuple[list[dict], torch.Tensor]:
    with (ROOT / "data/manifests/shuttleset_rgb.csv").open(encoding="utf-8-sig", newline="") as handle:
        truth = [row for row in csv.DictReader(handle) if row["match_id"] == match_id]
    available = set(range(len(records)))
    matched_records, targets = [], []
    for row in sorted(truth, key=lambda value: int(value["hit_frame"])):
        local = int(row["hit_frame"]) - offset
        choices = [(abs(int(records[index]["hit_frame"]) - local), index)
                   for index in available if abs(int(records[index]["hit_frame"]) - local) <= 12]
        if not choices:
            continue
        _distance, index = min(choices)
        available.remove(index)
        matched_records.append(records[index])
        targets.append(STROKE_CLASSES.index(row["coarse_label"]))
    return matched_records, torch.tensor(targets).long()


@torch.inference_mode()
def evaluate(model: nn.Module, inputs: list[torch.Tensor], targets: torch.Tensor, device: torch.device) -> dict:
    model.eval()
    confusion = torch.zeros(8, 8, dtype=torch.int64)
    for start in range(0, len(targets), 128):
        batch = [value[start:start + 128].to(device) for value in inputs]
        prediction = model(*batch)[0].argmax(1).cpu()
        for truth, pred in zip(targets[start:start + 128], prediction):
            confusion[int(truth), int(pred)] += 1
    return confusion_metrics(confusion)


def main() -> None:
    cfg = args()
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    online_records, target_parts = [], []
    per_match = {}
    for path in cfg.train_online:
        match_id = re.search(r"match(\d+)", path.name).group(1)
        if not path.exists():
            print(f"[!] Warning: {path} does not exist, skipping.")
            continue
        records = torch.load(path, map_location="cpu", weights_only=False)
        records, targets = match_train(records, match_id, OFFSETS[match_id])
        online_records.extend(records)
        target_parts.append(targets)
        per_match[match_id] = len(records)
        print(f"Loaded match {match_id:>2s}: {len(records):3d} matched samples")

    online_targets = torch.cat(target_parts)
    online_inputs = stack(online_records)

    # Class distribution of online data
    counts = np.bincount(online_targets.numpy(), minlength=8)
    print(f"\n--- TOTAL ONLINE SAMPLES: {len(online_targets)} ---")
    for i, c in enumerate(STROKE_CLASSES):
        print(f"  {c:<12s}: {counts[i]:3d} ({counts[i]/len(online_targets)*100:5.2f}%)")

    validation = torch.load(cfg.validation_online, map_location="cpu", weights_only=False)
    validation_inputs = stack(validation)
    validation_targets = torch.tensor([row["gt_stroke_idx"] for row in validation]).long()

    clean, indices = load_data(cfg.clean)
    generator = torch.Generator().manual_seed(cfg.seed)
    replay_ids = torch.from_numpy(indices["train"])[torch.randperm(len(indices["train"]), generator=generator)[:4096]]
    replay = TensorDataset(*[clean[key][replay_ids] for key in
        ("rgb", "pose", "court", "shuttle", "contact_dist", "quality", "stroke")])
    online = TensorDataset(*online_inputs, online_targets)
    replay_loader = DataLoader(replay, batch_size=64, shuffle=True, generator=generator)
    online_loader = DataLoader(online, batch_size=32, shuffle=True, generator=generator)

    payload = torch.load(cfg.base, map_location=device, weights_only=False)
    model = KAPScratchModel().to(device)
    model.load_state_dict(payload["model"])
    baseline = evaluate(model, validation_inputs, validation_targets, device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.03)

    best_score = baseline[cfg.selection_metric]
    best_metrics = baseline
    history = []
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    print(
        f"\nonline_train={len(online)} validation={len(validation_targets)} "
        f"baseline_{cfg.selection_metric}={best_score:.4f}", flush=True,
    )

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        replay_iter = iter(replay_loader)
        total_loss = 0.0

        for _repeat in range(2):
            for batch in online_loader:
                try:
                    clean_batch = next(replay_iter)
                except StopIteration:
                    replay_iter = iter(replay_loader)
                    clean_batch = next(replay_iter)
                optimizer.zero_grad(set_to_none=True)
                online_x = [value.to(device) for value in batch[:6]]
                clean_x = [value.to(device) for value in clean_batch[:6]]
                online_loss = criterion(model(*online_x)[0], batch[6].to(device))
                clean_loss = criterion(model(*clean_x)[0], clean_batch[6].to(device))
                loss = online_loss + 0.5 * clean_loss
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                total_loss += float(loss.detach())

        metrics = evaluate(model, validation_inputs, validation_targets, device)
        row = {"epoch": epoch, "loss": total_loss, "validation": metrics}
        history.append(row)
        print(f"epoch={epoch:02d} val_acc={metrics['accuracy']:.4f} val_bal={metrics['balanced_accuracy']:.4f} val_f1={metrics['macro_f1']:.4f}", flush=True)

        if metrics[cfg.selection_metric] > best_score:
            best_score = metrics[cfg.selection_metric]
            best_metrics = metrics
            torch.save({
                "model": model.state_dict(),
                "epoch": epoch,
                "base_checkpoint": str(cfg.base),
                "online_train_matches": sorted(per_match),
                "online_validation_match": "41",
                "val_metrics": metrics,
                "selection_metric": cfg.selection_metric,
                "selection_score": best_score,
            }, cfg.output_dir / "best.pth")

    result = {
        "protocol": "multi_match_online_adaptation",
        "offsets": OFFSETS,
        "per_match": per_match,
        "online_train_samples": len(online),
        "baseline": baseline,
        "best_score": best_score,
        "history": history,
    }
    (cfg.output_dir / "metrics.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"\n[DONE] Best {cfg.selection_metric}: {best_score:.4f} saved to {cfg.output_dir / 'best.pth'}")


if __name__ == "__main__":
    main()
