"""Online fine-tune of the clean-trained KAP with R(2+1)D or own RGB embeddings (resumable).

Both variants use the same strokes: only records with an own-RGB embedding, with ``emb``
swapped back to the stored R(2+1)D vector for the ``default`` variant.  The recipe matches
the production ShuttleSet22 fine-tune: class-weight power 0.5, clean replay, and selection on
the average of pooled online balanced accuracy and clean validation macro-F1.  last.pth is
written atomically after every epoch and a restart continues from it.

Example:
    python scripts/paper/finetune_kap_ownrgb.py --rgb ownrgb --ss22-matches 5
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO_ROOT), str(REPO_ROOT / "src"), str(REPO_ROOT / "scripts/training"), str(REPO_ROOT / "scripts/paper")]

from embed_pipeline_ownrgb import CLIP_OFFSETS, OLD_RECORDS, balanced_train  # noqa: E402
from finetune_kap_online_pilot import evaluate, load_labeled, match_train, stack  # noqa: E402
from ai_classifier.models import SIDE_CLASSES  # noqa: E402
from train_kap_ablation import KAPAblationModel  # noqa: E402
from train_kap_bstgap import FEATURES  # noqa: E402
from train_kap_scratch import load_data  # noqa: E402

WORK = REPO_ROOT / "work_dirs"
CLEAN_KEYS = ("rgb", "pose", "court", "shuttle", "contact_dist", "quality")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rgb", choices=("default", "ownrgb", "ownrgb_ss22", "ownrgb_x13"), required=True)
    parser.add_argument("--own-records", type=Path, default=None,
                        help="Records holding the own-RGB vectors; defaults by --rgb.")
    parser.add_argument("--ss22-matches", type=int, default=5, help="First N ShuttleSet22 train matches, alternating women and men.")
    parser.add_argument("--base-seed", type=int, default=1, help="Seed of the clean work_dirs/paper_bstgap/rgbcmp_* base.")
    parser.add_argument("--seed", type=int, default=20260929)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=5e-6)
    parser.add_argument("--class-weight-power", type=float, default=0.5)
    parser.add_argument("--replay-size", type=int, default=4096)
    parser.add_argument("--replay-weight", type=float, default=0.5)
    parser.add_argument("--side-weight", type=float, default=0.5,
                        help="Weight of the stroke-side loss; without it the side head drifts during fine-tuning.")
    parser.add_argument("--output-root", type=Path, default=WORK / "paper_ownrgb/finetune")
    parser.add_argument("--init-checkpoint", type=Path, default=None,
                        help="Continue from these KAP weights instead of the clean base.")
    parser.add_argument("--extra-records", type=Path, nargs="*", default=[],
                        help="Own-RGB record files of matches outside the ShuttleSet22 train split.")
    parser.add_argument("--extra-repeat", type=int, default=1, help="Copies of the extra records per epoch.")
    return parser.parse_args()


def own_name(match: str) -> str:
    return re.sub(r"\s+", "_", match)


def paired(rgb: str, stored: list[dict], own_path: Path) -> list[dict]:
    """Records present in the own-RGB file, carrying the embedding of the requested variant."""
    own = load_labeled(own_path)
    if rgb != "default":
        return own
    keyed = {(row["hit_frame"], row["side"]): row for row in stored}
    return [{**row, "emb": keyed[(row["hit_frame"], row["side"])]["emb"]} for row in own]


def save_atomic(payload: dict, path: Path) -> None:
    temporary = path.with_suffix(".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def main() -> None:
    cfg = parse_args()
    OWN = cfg.own_records or (WORK / "paper_ownrgb_ss22/pipeline_records" if cfg.rgb == "ownrgb_ss22" else WORK / "paper_ownrgb/pipeline_records")
    random.seed(cfg.seed); np.random.seed(cfg.seed); torch.manual_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    splits = json.loads((REPO_ROOT / "data/manifests/shuttleset22_splits.json").read_text(encoding="utf-8"))
    out = cfg.output_root / f"{cfg.rgb}_ss22x{cfg.ss22_matches}_base{cfg.base_seed}"
    out.mkdir(parents=True, exist_ok=True)

    records, targets, per_match = [], [], {}
    for match in ("13", "14", "15", "16", "17"):
        stored_path = WORK / OLD_RECORDS[match]
        rows = paired(cfg.rgb, torch.load(stored_path, map_location="cpu", weights_only=False), OWN / stored_path.name)
        rows, truth = match_train(rows, match, CLIP_OFFSETS[match])
        records += rows; targets.append(truth); per_match[match] = len(rows)
    for match in balanced_train(splits)[:cfg.ss22_matches]:
        rows = paired(cfg.rgb, load_labeled(WORK / "shuttleset22_records" / f"{match}.pt"), OWN / f"{own_name(match)}.pt")
        records += rows; per_match[match] = len(rows)
        targets.append(torch.tensor([row["gt_stroke_idx"] for row in rows]).long())
    for path in cfg.extra_records:
        rows = load_labeled(path) * cfg.extra_repeat
        records += rows; per_match[path.stem] = len(rows)
        targets.append(torch.tensor([row["gt_stroke_idx"] for row in rows]).long())
    online_targets = torch.cat(targets)
    # Old online records carry no side label; -1 is ignored by the side loss.
    online_sides = torch.tensor([row.get("gt_side_idx", -1) for row in records]).long()
    online = TensorDataset(*stack(records), online_targets, online_sides)

    validation = paired(cfg.rgb, load_labeled(WORK / OLD_RECORDS["41"]), OWN / f"{Path(OLD_RECORDS['41']).name}")
    for match in splits["val"]:
        validation += paired(cfg.rgb, load_labeled(WORK / "shuttleset22_records" / f"{match}.pt"), OWN / f"{own_name(match)}.pt")
    validation = [row for row in validation if "gt_stroke_idx" in row]
    validation_inputs = stack(validation)
    validation_targets = torch.tensor([row["gt_stroke_idx"] for row in validation]).long()
    validation_sides = torch.tensor([row.get("gt_side_idx", -1) for row in validation]).long()

    clean, indices = load_data(FEATURES[cfg.rgb])
    generator = torch.Generator().manual_seed(cfg.seed)
    train_ids = torch.from_numpy(indices["train"])
    replay_ids = train_ids[torch.randperm(len(train_ids), generator=generator)[:cfg.replay_size]]
    replay = TensorDataset(*[clean[key][replay_ids] for key in (*CLEAN_KEYS, "stroke", "side")])
    clean_val_ids = torch.from_numpy(indices["val"])
    clean_val_inputs = [clean[key][clean_val_ids] for key in CLEAN_KEYS]
    clean_val_targets = clean["stroke"][clean_val_ids]
    online_loader = DataLoader(online, batch_size=32, shuffle=True, generator=generator)
    replay_loader = DataLoader(replay, batch_size=64, shuffle=True, generator=generator)

    base = cfg.init_checkpoint or WORK / f"paper_bstgap/rgbcmp_{cfg.rgb}_seed{cfg.base_seed}/kap_best.pth"
    model = KAPAblationModel(gate="none", rgb_dim=clean["rgb"].shape[1]).to(device)
    model.load_state_dict(torch.load(base, map_location=device, weights_only=False)["model"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=1e-4)
    counts = torch.bincount(online_targets, minlength=8).float().clamp_min(1)
    weights = (len(online_targets) / (len(counts) * counts)) ** cfg.class_weight_power
    online_criterion = nn.CrossEntropyLoss(weight=(weights / weights.mean()).to(device), label_smoothing=0.03)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.03)
    side_criterion = nn.CrossEntropyLoss(label_smoothing=0.03, ignore_index=-1)

    @torch.inference_mode()
    def side_accuracy() -> float:
        """Forehand vs backhand on validation records with a side label, aroundhead counted as forehand."""
        model.eval()
        labeled = validation_sides >= 0
        predicted = torch.cat([model(*[v[i:i + 128].to(device) for v in validation_inputs])[1].argmax(1).cpu()
                               for i in range(0, len(validation_sides), 128)])
        backhand = SIDE_CLASSES.index("backhand")
        return ((predicted[labeled] == backhand) == (validation_sides[labeled] == backhand)).float().mean().item()

    def score() -> tuple[float, dict, dict]:
        online_metrics = evaluate(model, validation_inputs, validation_targets, device)
        clean_metrics = evaluate(model, clean_val_inputs, clean_val_targets, device)
        return 0.5 * online_metrics["balanced_accuracy"] + 0.5 * clean_metrics["macro_f1"], online_metrics, clean_metrics

    start, history = 1, []
    best, baseline, baseline_clean = score()
    best_metrics = baseline
    if (out / "last.pth").exists():
        state = torch.load(out / "last.pth", map_location=device, weights_only=False)
        model.load_state_dict(state["model"]); optimizer.load_state_dict(state["optimizer"])
        generator.set_state(state["generator"].cpu()); torch.set_rng_state(state["torch_rng"].cpu())
        start, history, best, best_metrics = state["epoch"] + 1, state["history"], state["best"], state["best_metrics"]
        print(f"[resume] from epoch {state['epoch']}", flush=True)
    print(f"[{cfg.rgb}] online={len(online)} val={len(validation_targets)} base {base.parent.name} "
          f"val_acc={baseline['accuracy']:.4f} val_bal={baseline['balanced_accuracy']:.4f} clean_f1={baseline_clean['macro_f1']:.4f}", flush=True)

    for epoch in range(start, cfg.epochs + 1):
        model.train(); replay_iter = iter(replay_loader); total = 0.0
        for _repeat in range(2):
            for batch in online_loader:
                try:
                    clean_batch = next(replay_iter)
                except StopIteration:
                    replay_iter = iter(replay_loader); clean_batch = next(replay_iter)
                optimizer.zero_grad(set_to_none=True)
                online_stroke, online_side, _ = model(*[v.to(device) for v in batch[:6]])
                clean_stroke, clean_side, _ = model(*[v.to(device) for v in clean_batch[:6]])
                online_loss = online_criterion(online_stroke, batch[6].to(device))
                clean_loss = criterion(clean_stroke, clean_batch[6].to(device))
                loss = online_loss + cfg.replay_weight * clean_loss
                if cfg.side_weight > 0:
                    side_loss = cfg.replay_weight * side_criterion(clean_side, clean_batch[7].to(device))
                    if (batch[7] >= 0).any():
                        side_loss = side_loss + side_criterion(online_side, batch[7].to(device))
                    loss = loss + cfg.side_weight * side_loss
                loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 1.0); optimizer.step()
                total += float(loss.detach())
        current, metrics, clean_metrics = score()
        val_side = side_accuracy()
        history.append({"epoch": epoch, "loss": total, "validation": metrics, "clean_validation": clean_metrics,
                        "side_accuracy": val_side, "score": current})
        print(f"[{cfg.rgb}] {epoch:02d} val_acc={metrics['accuracy']:.4f} val_bal={metrics['balanced_accuracy']:.4f} "
              f"val_f1={metrics['macro_f1']:.4f} clean_f1={clean_metrics['macro_f1']:.4f} side={val_side:.4f} score={current:.4f}", flush=True)
        if current > best:
            best, best_metrics = current, metrics
            save_atomic({"model": model.state_dict(), "epoch": epoch, "rgb": cfg.rgb, "rgb_dim": clean["rgb"].shape[1],
                         "base_checkpoint": str(base), "val_metrics": metrics, "clean_val_metrics": clean_metrics,
                         "court_reference": "lines", "per_match": per_match}, out / "best.pth")
        save_atomic({"model": model.state_dict(), "optimizer": optimizer.state_dict(), "generator": generator.get_state(),
                     "torch_rng": torch.get_rng_state(), "epoch": epoch, "history": history, "best": best,
                     "best_metrics": best_metrics}, out / "last.pth")
    (out / "metrics.json").write_text(json.dumps({
        "config": {k: str(v) for k, v in vars(cfg).items()}, "per_match": per_match, "baseline": baseline,
        "best_score": best, "best_metrics": best_metrics, "history": history,
    }, indent=2) + "\n", encoding="utf-8")
    print(f"[best] {cfg.rgb} acc={best_metrics['accuracy']:.4f} bal={best_metrics['balanced_accuracy']:.4f} f1={best_metrics['macro_f1']:.4f}", flush=True)


if __name__ == "__main__":
    main()
