"""Pilot online-domain KAP adaptation: train Match 13, validate Match 41."""

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

from ai_classifier.models import KAPScratchModel, STROKE_CLASSES  # noqa: E402
from scripts.training.train_kap_scratch import confusion_metrics, load_data  # noqa: E402


KEYS = ("emb", "s_j", "s_p", "s_s", "s_c", "s_q")


def args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, default=Path("work_dirs/kap_robust_seed20260926/kap_best.pth"))
    parser.add_argument("--clean", type=Path, default=Path("work_dirs/cr_gated_fusion/sequence_features.npz"))
    parser.add_argument("--train-online", type=Path, nargs="+", default=[
        Path("work_dirs/match13_5min_features.pt"),
        *[Path(f"work_dirs/online_train_match{match}_features.pt") for match in range(14, 18)],
    ])
    parser.add_argument("--validation-online", type=Path, default=Path("work_dirs/match41_5min_aligned_features.pt"))
    parser.add_argument("--output-dir", type=Path, default=Path("work_dirs/kap_online_pilot_match13"))
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--learning-rate", type=float, default=5e-6)
    parser.add_argument("--seed", type=int, default=20260929)
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
    model.eval(); confusion = torch.zeros(8, 8, dtype=torch.int64)
    for start in range(0, len(targets), 128):
        batch = [value[start:start + 128].to(device) for value in inputs]
        prediction = model(*batch)[0].argmax(1).cpu()
        for truth, pred in zip(targets[start:start + 128], prediction):
            confusion[int(truth), int(pred)] += 1
    return confusion_metrics(confusion)


def main() -> None:
    cfg = args(); random.seed(cfg.seed); np.random.seed(cfg.seed); torch.manual_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    offsets = {"13": 16300, "14": 89539, "15": 42240, "16": 13792, "17": 68475}
    online_records, target_parts = [], []
    per_match = {}
    for path in cfg.train_online:
        match_id = re.search(r"match(\d+)", path.name).group(1)
        records = torch.load(path, map_location="cpu", weights_only=False)
        records, targets = match_train(records, match_id, offsets[match_id])
        online_records.extend(records); target_parts.append(targets); per_match[match_id] = len(records)
    online_targets = torch.cat(target_parts)
    online_inputs = stack(online_records)
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
    model = KAPScratchModel().to(device); model.load_state_dict(payload["model"])
    baseline = evaluate(model, validation_inputs, validation_targets, device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.03)
    best = baseline["accuracy"]; history = []
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    print(f"online_train={len(online)} validation={len(validation_targets)} baseline={best:.4f}", flush=True)
    for epoch in range(1, cfg.epochs + 1):
        model.train(); replay_iter = iter(replay_loader); total_loss = 0.0
        # Repeat the real online samples while anchoring every update with clean replay.
        for _repeat in range(2):
            for batch in online_loader:
                try: clean_batch = next(replay_iter)
                except StopIteration:
                    replay_iter = iter(replay_loader); clean_batch = next(replay_iter)
                optimizer.zero_grad(set_to_none=True)
                online_x = [value.to(device) for value in batch[:6]]
                clean_x = [value.to(device) for value in clean_batch[:6]]
                online_loss = criterion(model(*online_x)[0], batch[6].to(device))
                clean_loss = criterion(model(*clean_x)[0], clean_batch[6].to(device))
                loss = online_loss + 0.5 * clean_loss
                loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 1.0); optimizer.step()
                total_loss += float(loss.detach())
        metrics = evaluate(model, validation_inputs, validation_targets, device)
        row = {"epoch": epoch, "loss": total_loss, "validation": metrics}; history.append(row)
        print(f"epoch={epoch:02d} val_acc={metrics['accuracy']:.4f} val_f1={metrics['macro_f1']:.4f}", flush=True)
        if metrics["accuracy"] > best:
            best = metrics["accuracy"]
            torch.save({"model": model.state_dict(), "epoch": epoch,
                        "base_checkpoint": str(cfg.base), "online_train_matches": sorted(per_match),
                        "online_validation_match": "41", "val_metrics": metrics},
                       cfg.output_dir / "best.pth")
    result = {"protocol": "online_train_matches13_17_validate_match41", "offsets": offsets,
              "per_match": per_match,
              "online_train_samples": len(online), "baseline": baseline,
              "best_accuracy": best, "history": history}
    (cfg.output_dir / "metrics.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"baseline": baseline["accuracy"], "best": best}, indent=2))


if __name__ == "__main__":
    main()
