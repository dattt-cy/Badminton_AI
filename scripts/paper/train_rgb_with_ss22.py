"""Continue training the own RGB encoder on clean ShuttleSet plus ShuttleSet22 crops (resumable).

Starts from the ShuttleSet-only encoder.  ShuttleSet22 train clips are oversampled to about a third
of each epoch.  Selection averages macro-F1 on the clean validation match and on the ShuttleSet22
validation matches; no test match is read.  last.pth is written atomically after every epoch.

Example:
    python scripts/paper/train_rgb_with_ss22.py --ss22-matches 5
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import ConcatDataset, DataLoader, WeightedRandomSampler

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO_ROOT), str(REPO_ROOT / "src"), str(REPO_ROOT / "scripts/training"), str(REPO_ROOT / "scripts/paper")]

from train_kap_scratch import seed_everything  # noqa: E402
from train_rgb_scratch import ClipDataset, RGBEncoder, evaluate  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clean-cache", type=Path, default=Path(r"C:\Users\ADMIN\Downloads\AI_Classifier_cache\shuttleset_v3_full\frames_16"))
    parser.add_argument("--ss22-cache", type=Path, default=Path(r"C:\Users\ADMIN\Downloads\AI_Classifier_cache\ss22_frames_16"))
    parser.add_argument("--features", type=Path, default=REPO_ROOT / "work_dirs/context_window_pilot/features_w15_30.npz")
    parser.add_argument("--init", type=Path, default=REPO_ROOT / "work_dirs/paper_ownrgb/rgb_encoder_best.pth")
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "work_dirs/paper_ownrgb_ss22")
    parser.add_argument("--ss22-matches", type=int, default=5)
    parser.add_argument("--ss22-share", type=float, default=1 / 3)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=1)
    return parser.parse_args()


def ss22_clips(matches: list[str]) -> tuple[list[str], list[int]]:
    # Imported here so spawned data-loader workers do not load the whole video pipeline.
    from cut_ss22_crops import clip_id
    from embed_pipeline_ownrgb import OWN_RECORDS

    ids, labels = [], []
    for match in matches:
        records = torch.load(OWN_RECORDS / f"{re.sub(r'\s+', '_', match)}.pt", map_location="cpu", weights_only=False)["records"]
        ids += [clip_id(match, row) for row in records]
        labels += [int(row["gt_stroke_idx"]) for row in records]
    return ids, labels


def save_atomic(payload: dict, path: Path) -> None:
    temporary = path.with_suffix(".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def main() -> None:
    from embed_pipeline_ownrgb import balanced_train

    args = parse_args()
    seed_everything(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda")
    raw = np.load(args.features)
    ids, split, stroke = raw["sample_id"], raw["split"], raw["stroke"]
    splits = json.loads((REPO_ROOT / "data/manifests/shuttleset22_splits.json").read_text(encoding="utf-8"))

    def clean(name: str, train: bool) -> ClipDataset:
        mask = split == name
        return ClipDataset(args.clean_cache, [str(x) for x in ids[mask]], stroke[mask].tolist(), train)

    clean_train = clean("train", True)
    new_ids, new_labels = ss22_clips(balanced_train(splits)[:args.ss22_matches])
    new_train = ClipDataset(args.ss22_cache, new_ids, new_labels, True)
    val_ids, val_labels = ss22_clips(splits["val"])
    missing = sum(not (args.ss22_cache / f"{x}.npy").exists() for x in [*new_ids, *val_ids])
    if missing:
        raise RuntimeError(f"{missing} ShuttleSet22 clips are missing from the cache")
    print(f"[data] clean_train={len(clean_train)} ss22_train={len(new_train)} clean_val={(split == 'val').sum()} ss22_val={len(val_ids)}", flush=True)

    weight_new = args.ss22_share / (1 - args.ss22_share) * len(clean_train) / len(new_train)
    sample_weights = torch.cat([torch.ones(len(clean_train)), torch.full((len(new_train),), weight_new)]).double()
    generator = torch.Generator().manual_seed(args.seed)
    sampler = WeightedRandomSampler(sample_weights, num_samples=len(clean_train), replacement=True, generator=generator)
    train_loader = DataLoader(ConcatDataset([clean_train, new_train]), batch_size=args.batch_size, sampler=sampler,
                              num_workers=args.workers, pin_memory=True, drop_last=True, persistent_workers=True)
    clean_val = DataLoader(clean("val", False), batch_size=64, num_workers=1, persistent_workers=True)
    new_val = DataLoader(ClipDataset(args.ss22_cache, val_ids, val_labels, False), batch_size=64, num_workers=1,
                         persistent_workers=True)

    model = RGBEncoder().to(device)
    model.load_state_dict(torch.load(args.init, map_location=device, weights_only=False)["model"])
    counts = np.bincount(stroke[split == "train"], minlength=8) + np.bincount(new_labels, minlength=8) * weight_new
    weights = torch.tensor(counts.sum() / (8 * counts), dtype=torch.float32, device=device)
    loss_fn = nn.CrossEntropyLoss(weight=weights, label_smoothing=0.1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=args.lr, total_steps=args.epochs * len(train_loader), pct_start=0.1)
    scaler = torch.amp.GradScaler()

    def score() -> tuple[float, dict, dict]:
        old, new = evaluate(model, clean_val, device), evaluate(model, new_val, device)
        return 0.5 * old["macro_f1"] + 0.5 * new["macro_f1"], old, new

    start, history = 1, []
    if (args.output_dir / "last.pth").exists():
        state = torch.load(args.output_dir / "last.pth", map_location=device, weights_only=False)
        model.load_state_dict(state["model"]); optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"]); scaler.load_state_dict(state["scaler"])
        generator.set_state(state["generator"].cpu()); torch.set_rng_state(state["torch_rng"].cpu())
        start, history, best = state["epoch"] + 1, state["history"], state["best"]
        print(f"[resume] from epoch {state['epoch']}", flush=True)
    else:
        best, old, new = score()
        print(f"[init] clean_acc={old['accuracy']:.4f} clean_f1={old['macro_f1']:.4f} ss22_acc={new['accuracy']:.4f} ss22_f1={new['macro_f1']:.4f} score={best:.4f}", flush=True)
        save_atomic({"model": model.state_dict(), "epoch": 0}, args.output_dir / "rgb_encoder_best.pth")

    for epoch in range(start, args.epochs + 1):
        model.train(); total, seen = 0.0, 0
        for clips, labels in train_loader:
            clips, labels = clips.to(device, non_blocking=True), labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.float16):
                loss = loss_fn(model(clips), labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer); scaler.update(); scheduler.step()
            total += float(loss.detach()) * len(labels); seen += len(labels)
        current, old, new = score()
        history.append({"epoch": epoch, "loss": total / seen, "clean": old, "ss22": new, "score": current})
        print(f"[rgb] {epoch:02d} loss={total / seen:.4f} clean_acc={old['accuracy']:.4f} clean_f1={old['macro_f1']:.4f} "
              f"ss22_acc={new['accuracy']:.4f} ss22_f1={new['macro_f1']:.4f} score={current:.4f}", flush=True)
        if current > best:
            best = current
            save_atomic({"model": model.state_dict(), "epoch": epoch}, args.output_dir / "rgb_encoder_best.pth")
        save_atomic({"model": model.state_dict(), "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
                     "scaler": scaler.state_dict(), "generator": generator.get_state(), "torch_rng": torch.get_rng_state(),
                     "epoch": epoch, "history": history, "best": best}, args.output_dir / "last.pth")
    (args.output_dir / "metrics.json").write_text(json.dumps({"best_score": best, "history": history}, indent=2) + "\n", encoding="utf-8")
    print(f"[done] best score {best:.4f}", flush=True)


if __name__ == "__main__":
    main()
