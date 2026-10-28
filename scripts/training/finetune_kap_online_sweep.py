"""Multi-task (stroke + stroke side) online-domain KAP fine-tuning sweep.

Protocol:
- Train on pipeline-extracted features of ShuttleSet train matches
  1, 2, 3, 4, 7, 11, 13, 14, 15, 16, 17, with clean ShuttleSet replay.
  Both heads are supervised, so the shared trunk cannot drift away from the
  stroke-side head (a stroke-only fine-tune degraded side accuracy).
- Select epoch and run on the 10-minute benchmark clips of validation matches
  36, 38 and 41 (cached by compare_kap_checkpoints_8clips), scoring
  stroke balanced accuracy + SIDE_WEIGHT * side balanced accuracy.
- Test matches 39, 40, 42, 43 and 44 are never read here.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import shutil
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from ai_classifier.models import SIDE_CLASSES, STROKE_CLASSES  # noqa: E402
from scripts.evaluation.benchmark_8_matches_10min import CLIPS_CONFIG  # noqa: E402
from scripts.evaluation.compare_kap_checkpoints_8clips import (  # noqa: E402
    TEST_MATCHES, load_gt, match_gt,
)
from scripts.training.finetune_kap_online_multi_matches import OFFSETS  # noqa: E402
from scripts.training.finetune_kap_online_pilot import MODEL_CLASSES, stack  # noqa: E402
from scripts.training.train_kap_scratch import confusion_metrics, load_data  # noqa: E402

IGNORE = -100


def args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, default=Path("work_dirs/kap_cross_v2_liftboost_seed20260926/kap_best.pth"))
    parser.add_argument("--clean", type=Path, default=Path("work_dirs/cr_gated_fusion/sequence_features.npz"))
    parser.add_argument("--train-matches", nargs="+", default=sorted(OFFSETS, key=int))
    parser.add_argument("--feature-cache", type=Path, default=Path("work_dirs/kap_ab_8clips/feature_cache"))
    parser.add_argument("--manifest", type=Path, default=Path("data/manifests/shuttleset_rgb.csv"))
    parser.add_argument("--learning-rates", type=float, nargs="+", default=[5e-6, 1e-5])
    parser.add_argument("--seeds", type=int, nargs="+", default=[20260929, 20261029, 20261129])
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--class-weight-power", type=float, default=1.0)
    parser.add_argument("--side-loss-weight", type=float, default=0.5)
    parser.add_argument("--output-dir", type=Path, default=Path("work_dirs/kap_v2_online_multitask"))
    return parser.parse_args()


def online_path(match_id: str) -> Path:
    if match_id == "13":
        return ROOT / "work_dirs/match13_5min_features.pt"
    return ROOT / f"work_dirs/online_train_match{match_id}_features.pt"


def side_index(label: str) -> int:
    return SIDE_CLASSES.index(label) if label in SIDE_CLASSES else IGNORE


def match_online(cfg, records: list[dict], match_id: str) -> tuple[list[dict], list[int], list[int]]:
    """Greedy GT matching (±12 frames) as in finetune_kap_online_pilot, keeping both labels."""
    with cfg.manifest.open(encoding="utf-8-sig", newline="") as handle:
        truth = [row for row in csv.DictReader(handle) if row["match_id"] == match_id]
    available = set(range(len(records)))
    matched, strokes, sides = [], [], []
    for row in sorted(truth, key=lambda value: int(value["hit_frame"])):
        local = int(row["hit_frame"]) - OFFSETS[match_id]
        choices = [(abs(int(records[i]["hit_frame"]) - local), i)
                   for i in available if abs(int(records[i]["hit_frame"]) - local) <= 12]
        if not choices:
            continue
        _, index = min(choices)
        available.remove(index)
        matched.append(records[index])
        strokes.append(STROKE_CLASSES.index(row["coarse_label"]))
        sides.append(side_index(row["stroke_side"]))
    return matched, strokes, sides


def load_online(cfg) -> tuple[list[torch.Tensor], torch.Tensor, torch.Tensor, dict]:
    records, strokes, sides, per_match = [], [], [], {}
    for match_id in cfg.train_matches:
        raw = torch.load(online_path(match_id), map_location="cpu", weights_only=False)
        matched, s, d = match_online(cfg, raw, match_id)
        records.extend(matched); strokes.extend(s); sides.extend(d); per_match[match_id] = len(matched)
    return stack(records), torch.tensor(strokes).long(), torch.tensor(sides).long(), per_match


def load_validation(cfg) -> tuple[list[torch.Tensor], torch.Tensor, torch.Tensor]:
    """Matched (feature, stroke, side) triples from the cached validation benchmark clips."""
    records, strokes, sides = [], [], []
    for clip in CLIPS_CONFIG:
        if clip["match_id"] in TEST_MATCHES:
            continue
        cached = torch.load(cfg.feature_cache / f"{clip['video'].stem}.pt", map_location="cpu", weights_only=False)
        predictions = [{"frame": item["frame"], "feature": item["feature"]} for item in cached["features"]]
        gt = load_gt(cfg.manifest, clip["match_id"], clip["raw_start"], 18000)
        for truth, pred in zip(gt, match_gt(gt, predictions, tolerance=12)):
            if pred is not None:
                records.append(pred["feature"])
                strokes.append(STROKE_CLASSES.index(truth["stroke"]))
                sides.append(side_index(truth["stroke_side"]))
    return stack(records), torch.tensor(strokes).long(), torch.tensor(sides).long()


def class_weights(labels: torch.Tensor, classes: int, power: float) -> torch.Tensor:
    valid = labels[labels != IGNORE]
    counts = torch.bincount(valid, minlength=classes).float().clamp_min(1)
    weights = (len(valid) / (classes * counts)) ** power
    return weights / weights.mean()


@torch.inference_mode()
def evaluate(model: nn.Module, inputs, strokes, sides, device) -> dict:
    model.eval()
    stroke_cm = torch.zeros(len(STROKE_CLASSES), len(STROKE_CLASSES), dtype=torch.int64)
    side_cm = torch.zeros(len(SIDE_CLASSES), len(SIDE_CLASSES), dtype=torch.int64)
    for start in range(0, len(strokes), 128):
        stroke_logits, side_logits, _ = model(*[value[start:start + 128].to(device) for value in inputs])
        for truth, pred in zip(strokes[start:start + 128], stroke_logits.argmax(1).cpu()):
            stroke_cm[int(truth), int(pred)] += 1
        for truth, pred in zip(sides[start:start + 128], side_logits.argmax(1).cpu()):
            if int(truth) != IGNORE:
                side_cm[int(truth), int(pred)] += 1
    return {"stroke": confusion_metrics(stroke_cm), "side": confusion_metrics(side_cm)}


def score(metrics: dict, side_weight: float) -> float:
    return metrics["stroke"]["balanced_accuracy"] + side_weight * metrics["side"]["balanced_accuracy"]


def brief(metrics: dict) -> dict:
    return {f"{head}_{key}": metrics[head][key]
            for head in ("stroke", "side") for key in ("accuracy", "balanced_accuracy", "macro_f1")}


def run(cfg, base, model_class, online, replay, validation, lr: float, seed: int, device) -> dict:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    online_inputs, online_strokes, online_sides = online
    generator = torch.Generator().manual_seed(seed)
    replay_loader = DataLoader(replay, batch_size=64, shuffle=True, generator=generator)
    online_loader = DataLoader(
        TensorDataset(*online_inputs, online_strokes, online_sides),
        batch_size=32, shuffle=True, generator=generator,
    )
    online_stroke_loss = nn.CrossEntropyLoss(
        weight=class_weights(online_strokes, len(STROKE_CLASSES), cfg.class_weight_power).to(device),
        label_smoothing=0.03)
    online_side_loss = nn.CrossEntropyLoss(
        weight=class_weights(online_sides, len(SIDE_CLASSES), cfg.class_weight_power).to(device),
        label_smoothing=0.03, ignore_index=IGNORE)
    clean_loss_fn = nn.CrossEntropyLoss(label_smoothing=0.03, ignore_index=IGNORE)

    model = model_class().to(device); model.load_state_dict(base["model"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    best = {"score": -1.0}
    history = []
    for epoch in range(1, cfg.epochs + 1):
        model.train(); replay_iter = iter(replay_loader)
        for _repeat in range(2):
            for batch in online_loader:
                try:
                    clean_batch = next(replay_iter)
                except StopIteration:
                    replay_iter = iter(replay_loader); clean_batch = next(replay_iter)
                optimizer.zero_grad(set_to_none=True)
                stroke, side, _ = model(*[value.to(device) for value in batch[:6]])
                online_loss = (online_stroke_loss(stroke, batch[6].to(device))
                               + cfg.side_loss_weight * online_side_loss(side, batch[7].to(device)))
                clean_stroke, clean_side, _ = model(*[value.to(device) for value in clean_batch[:6]])
                clean_loss = (clean_loss_fn(clean_stroke, clean_batch[6].to(device))
                              + cfg.side_loss_weight * clean_loss_fn(clean_side, clean_batch[7].to(device)))
                (online_loss + 0.5 * clean_loss).backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0); optimizer.step()
        metrics = evaluate(model, *validation, device)
        current = score(metrics, cfg.side_loss_weight)
        history.append({"epoch": epoch, "score": current, **brief(metrics)})
        if current > best["score"]:
            best = {"score": current, "epoch": epoch, "metrics": metrics,
                    "state": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}}
    m = best["metrics"]
    print(f"  lr={lr:g} seed={seed} best_epoch={best['epoch']} score={best['score']:.4f} "
          f"stroke_acc={m['stroke']['accuracy']:.4f} stroke_bal={m['stroke']['balanced_accuracy']:.4f} "
          f"side_acc={m['side']['accuracy']:.4f} side_bal={m['side']['balanced_accuracy']:.4f}", flush=True)
    return {**best, "history": history}


def main() -> None:
    cfg = args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    base = torch.load(cfg.base, map_location="cpu", weights_only=False)
    model_type = base.get("model_type", "baseline")
    model_class = MODEL_CLASSES[model_type]

    online_inputs, online_strokes, online_sides, per_match = load_online(cfg)
    validation = load_validation(cfg)
    print(f"online_train={len(online_strokes)} per_match={per_match}")
    print("  train sides:", dict(zip(SIDE_CLASSES, torch.bincount(online_sides[online_sides != IGNORE], minlength=3).tolist())),
          "ignored:", int((online_sides == IGNORE).sum()))
    print(f"validation(36,38,41 matched)={len(validation[1])}")

    clean, indices = load_data(cfg.clean)
    replay_ids = torch.from_numpy(indices["train"])[
        torch.randperm(len(indices["train"]), generator=torch.Generator().manual_seed(0))[:4096]]
    replay = TensorDataset(*[clean[key][replay_ids] for key in
                             ("rgb", "pose", "court", "shuttle", "contact_dist", "quality", "stroke", "side")])

    model = model_class().to(device); model.load_state_dict(base["model"])
    baseline = evaluate(model, *validation, device)
    print(f"base validation: score={score(baseline, cfg.side_loss_weight):.4f} {brief(baseline)}")

    runs = []
    for lr in cfg.learning_rates:
        for seed in cfg.seeds:
            result = run(cfg, base, model_class, (online_inputs, online_strokes, online_sides),
                         replay, validation, lr, seed, device)
            path = cfg.output_dir / f"lr{lr:g}_seed{seed}.pth"
            path.parent.mkdir(parents=True, exist_ok=True)
            torch.save({"model": result["state"], "model_type": model_type, "epoch": result["epoch"],
                        "learning_rate": lr, "seed": seed, "val_metrics": result["metrics"],
                        "selection_score": result["score"], "base_checkpoint": str(cfg.base),
                        "online_train_matches": sorted(per_match, key=int),
                        "validation_matches": ["36", "38", "41"], "multitask": True,
                        "side_loss_weight": cfg.side_loss_weight,
                        "class_weight_power": cfg.class_weight_power}, path)
            runs.append({"lr": lr, "seed": seed, "epoch": result["epoch"], "score": result["score"],
                         "path": str(path), **brief(result["metrics"]), "history": result["history"]})

    best = max(runs, key=lambda row: row["score"])
    shutil.copyfile(best["path"], cfg.output_dir / "best.pth")
    print(f"best run: lr={best['lr']:g} seed={best['seed']} score={best['score']:.4f} -> best.pth")
    summary = {
        "protocol": "multitask_online_11_matches_select_on_val_clips_36_38_41",
        "side_loss_weight": cfg.side_loss_weight, "per_match": per_match,
        "baseline": brief(baseline), "baseline_score": score(baseline, cfg.side_loss_weight),
        "best": {k: v for k, v in best.items() if k != "history"}, "runs": runs,
    }
    (cfg.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
