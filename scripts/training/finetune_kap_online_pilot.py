"""Online-domain KAP adaptation with clean replay and held-out validation."""

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
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from ai_classifier.models import (  # noqa: E402
    KAPCrossTransModel, KAPCrossTransV2Model, KAPScratchModel, STROKE_CLASSES,
)
from scripts.training.finetune_kap_online_multi_matches import OFFSETS  # noqa: E402
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
    parser.add_argument("--validation-online", type=Path, nargs="+",
                        default=[Path("work_dirs/match41_5min_aligned_features.pt")],
                        help="Labeled records (gt_stroke_idx) used for checkpoint selection; pooled.")
    parser.add_argument("--train-labeled", type=Path, nargs="*", default=[],
                        help="Extra online records that already carry gt_stroke_idx (e.g. ShuttleSet22).")
    parser.add_argument("--output-dir", type=Path, default=Path("work_dirs/kap_online_pilot_match13"))
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--learning-rate", type=float, default=5e-6)
    parser.add_argument("--seed", type=int, default=20260929)
    parser.add_argument(
        "--selection-metric", choices=("accuracy", "balanced_accuracy", "macro_f1"),
        default="balanced_accuracy",
        help="Validation metric used to select the checkpoint.",
    )
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument(
        "--class-balance", choices=("none", "weights", "sampler"), default="none",
        help="Counter online class imbalance with loss weights or a balanced sampler.",
    )
    parser.add_argument(
        "--class-weight-power", type=float, default=0.5,
        help="Loss weight = (N / (K * count)) ** power; 0.5 is square-root inverse frequency.",
    )
    parser.add_argument(
        "--replay-size", type=int, default=4096,
        help="Clean ShuttleSet train samples replayed during adaptation; 0 replays all of them.",
    )
    parser.add_argument("--replay-weight", type=float, default=0.5, help="Weight of the clean replay loss.")
    parser.add_argument(
        "--court-reference", choices=("mat", "lines"), default="mat",
        help="Court quad the online records were extracted with; stored so the pipeline uses the same one.",
    )
    parser.add_argument(
        "--selection-scope", choices=("online", "joint"), default="online",
        help="online: select on the online validation match only. joint: average it with the clean "
             "ShuttleSet validation macro-F1 so adaptation cannot silently forget the clean domain.",
    )
    return parser.parse_args()


MODEL_CLASSES = {
    "baseline": KAPScratchModel,
    "cross": KAPCrossTransModel,
    "cross_v2": KAPCrossTransV2Model,
    "cross_v3": KAPCrossTransV2Model,
}


def stack(records: list[dict]) -> list[torch.Tensor]:
    return [torch.cat([row[key].detach().cpu() for row in records]).float() for key in KEYS]


def load_labeled(path: Path) -> list[dict]:
    """Records with gt_stroke_idx, stored as a list or as {"records": [...]}."""
    payload = torch.load(path, map_location="cpu", weights_only=False)
    return payload["records"] if isinstance(payload, dict) else payload


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
    if cfg.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(cfg.device)
    offsets = OFFSETS
    online_records, target_parts = [], []
    per_match = {}
    for path in cfg.train_online:
        match_id = re.search(r"match(\d+)", path.name).group(1)
        records = torch.load(path, map_location="cpu", weights_only=False)
        records, targets = match_train(records, match_id, offsets[match_id])
        online_records.extend(records); target_parts.append(targets); per_match[match_id] = len(records)
    for path in cfg.train_labeled:
        records = load_labeled(path)
        online_records.extend(records); per_match[path.stem] = len(records)
        target_parts.append(torch.tensor([row["gt_stroke_idx"] for row in records]).long())
    online_targets = torch.cat(target_parts)
    online_inputs = stack(online_records)
    validation = [row for path in cfg.validation_online for row in load_labeled(path)]
    validation_inputs = stack(validation)
    validation_targets = torch.tensor([row["gt_stroke_idx"] for row in validation]).long()

    clean, indices = load_data(cfg.clean)
    generator = torch.Generator().manual_seed(cfg.seed)
    replay_count = cfg.replay_size or len(indices["train"])
    replay_ids = torch.from_numpy(indices["train"])[torch.randperm(len(indices["train"]), generator=generator)[:replay_count]]
    replay = TensorDataset(*[clean[key][replay_ids] for key in
        ("rgb", "pose", "court", "shuttle", "contact_dist", "quality", "stroke")])
    online = TensorDataset(*online_inputs, online_targets)
    clean_val_ids = torch.from_numpy(indices["val"])
    clean_val_inputs = [clean[key][clean_val_ids] for key in
        ("rgb", "pose", "court", "shuttle", "contact_dist", "quality")]
    clean_val_targets = clean["stroke"][clean_val_ids]
    replay_loader = DataLoader(replay, batch_size=64, shuffle=True, generator=generator)
    counts = torch.bincount(online_targets, minlength=len(STROKE_CLASSES)).float().clamp_min(1)
    if cfg.class_balance == "sampler":
        sampler = WeightedRandomSampler(
            (1.0 / counts)[online_targets], num_samples=len(online), replacement=True,
            generator=generator,
        )
        online_loader = DataLoader(online, batch_size=32, sampler=sampler)
    else:
        online_loader = DataLoader(online, batch_size=32, shuffle=True, generator=generator)

    payload = torch.load(cfg.base, map_location=device, weights_only=False)
    model_type = payload.get("model_type", "baseline")
    model = MODEL_CLASSES[model_type]().to(device); model.load_state_dict(payload["model"])
    baseline = evaluate(model, validation_inputs, validation_targets, device)
    clean_baseline = evaluate(model, clean_val_inputs, clean_val_targets, device)

    def selection_score(online_metrics: dict, clean_metrics: dict) -> float:
        if cfg.selection_scope == "online":
            return online_metrics[cfg.selection_metric]
        return 0.5 * online_metrics[cfg.selection_metric] + 0.5 * clean_metrics["macro_f1"]

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.03)
    online_criterion = criterion
    if cfg.class_balance == "weights":
        # Weighting applies to online samples only; clean replay keeps its own prior.
        weights = (len(online_targets) / (len(counts) * counts)) ** cfg.class_weight_power
        online_criterion = nn.CrossEntropyLoss(
            weight=(weights / weights.mean()).to(device), label_smoothing=0.03,
        )
    best_score = selection_score(baseline, clean_baseline)
    best_metrics = baseline
    best_clean_metrics = clean_baseline
    history = []
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    print(
        f"online_train={len(online)} validation={len(validation_targets)} "
        f"baseline_{cfg.selection_metric}={best_score:.4f}", flush=True,
    )
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
                online_loss = online_criterion(model(*online_x)[0], batch[6].to(device))
                clean_loss = criterion(model(*clean_x)[0], clean_batch[6].to(device))
                loss = online_loss + cfg.replay_weight * clean_loss
                loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 1.0); optimizer.step()
                total_loss += float(loss.detach())
        metrics = evaluate(model, validation_inputs, validation_targets, device)
        clean_metrics = evaluate(model, clean_val_inputs, clean_val_targets, device)
        score = selection_score(metrics, clean_metrics)
        row = {"epoch": epoch, "loss": total_loss, "validation": metrics, "clean_validation": clean_metrics,
               "selection_score": score}
        history.append(row)
        print(f"epoch={epoch:02d} val_acc={metrics['accuracy']:.4f} val_f1={metrics['macro_f1']:.4f} "
              f"clean_val_acc={clean_metrics['accuracy']:.4f} clean_val_f1={clean_metrics['macro_f1']:.4f} "
              f"score={score:.4f}", flush=True)
        if score > best_score:
            best_score = score
            best_metrics = metrics
            best_clean_metrics = clean_metrics
            torch.save({"model": model.state_dict(), "epoch": epoch, "model_type": model_type,
                        "base_checkpoint": str(cfg.base), "online_train_matches": sorted(per_match),
                        "online_validation": [str(path) for path in cfg.validation_online], "val_metrics": metrics,
                        "selection_metric": cfg.selection_metric,
                        "selection_score": best_score,
                        "class_balance": cfg.class_balance,
                        "class_weight_power": cfg.class_weight_power,
                        "replay_size": replay_count, "replay_weight": cfg.replay_weight,
                        "selection_scope": cfg.selection_scope, "clean_val_metrics": clean_metrics,
                        "context_window": payload.get("context_window"),
                        "court_reference": cfg.court_reference},
                       cfg.output_dir / "best.pth")
    result = {"protocol": "online_train_matches13_17_validate_match41", "offsets": offsets,
              "per_match": per_match,
              "online_train_samples": len(online), "baseline": baseline,
              "selection_metric": cfg.selection_metric,
              "best_score": best_score, "best_metrics": best_metrics,
              "best_accuracy": best_metrics["accuracy"], "clean_baseline": clean_baseline,
              "best_clean_metrics": best_clean_metrics, "selection_scope": cfg.selection_scope,
              "replay_size": replay_count, "replay_weight": cfg.replay_weight, "history": history}
    (cfg.output_dir / "metrics.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "selection_metric": cfg.selection_metric,
        "baseline": baseline[cfg.selection_metric], "best": best_score,
        "best_accuracy": best_metrics["accuracy"],
    }, indent=2))


if __name__ == "__main__":
    main()
