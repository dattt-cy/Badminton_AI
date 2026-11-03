"""Search for a KAP configuration that closes the gap to BST-CG-AP (validation only).

Options on top of KAPCrossTransV2 (train_kap_ablation.KAPAblationModel):
  --features      cache with the default (15/30) or long (45/52) context window
  --gate          sigmoid quality gate or none
  --aux-hitter    weight of an auxiliary 16-way head (8 strokes x hitter side), BST-style labels
  --aux-fine      weight of an auxiliary head over the 18 raw ShuttleSet stroke types
  --side-weight   weight of the stroke-side head (0 disables it)
Selection uses validation Match 41 only.  Test predictions are written to disk
but not scored here; scripts/paper/score_test_once.py scores the chosen runs.

Example:
    python scripts/paper/train_kap_bstgap.py --tag long_nogate --features long --gate none --seed 1
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[2]
for path in (REPO_ROOT / "src", REPO_ROOT / "scripts" / "training", REPO_ROOT / "scripts" / "paper"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from train_kap_ablation import KAPAblationModel, predict  # noqa: E402
from train_kap_scratch import (  # noqa: E402
    VideoDomainAugmentor, balanced_weights, confusion_metrics, feature_statistics,
    load_data, seed_everything, symmetric_consistency,
)

FEATURES = {
    "default": REPO_ROOT / "work_dirs/context_window_pilot/features_w15_30.npz",
    "long": REPO_ROOT / "work_dirs/context_window_pilot/features_w45_52.npz",
    # Same events, RGB embedding from the team's own RGB network trained from scratch.
    "ownrgb": REPO_ROOT / "work_dirs/paper_ownrgb/features_w15_30_ownrgb.npz",
    # Own RGB network continued on ShuttleSet22 crops.
    "ownrgb_ss22": REPO_ROOT / "work_dirs/paper_ownrgb_ss22/features_w15_30_ownrgb.npz",
    "ownrgb_x13": REPO_ROOT / "work_dirs/paper_ownrgb_ss22/x13/features_w15_30_ownrgb.npz",
}
RAW_TYPES = [
    "發短球", "發長球", "長球", "殺球", "點扣", "切球", "過渡切球", "放小球", "擋小球",
    "勾球", "挑球", "防守回挑", "平球", "小平球", "後場抽平球", "防守回抽", "推球", "撲球",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--features", choices=sorted(FEATURES), default="long")
    parser.add_argument("--gate", choices=("sigmoid", "none"), default="none")
    parser.add_argument("--aux-hitter", type=float, default=0.0)
    parser.add_argument("--aux-fine", type=float, default=0.0)
    parser.add_argument("--side-weight", type=float, default=0.5)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--output-root", type=Path, default=REPO_ROOT / "work_dirs/paper_bstgap")
    return parser.parse_args()


def auxiliary_labels(sample_ids: np.ndarray, stroke: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    with (REPO_ROOT / "data/manifests/shuttleset_npy.csv").open(encoding="utf-8-sig", newline="") as stream:
        manifest = {row["sample_id"]: row for row in csv.DictReader(stream)}
    hitter = torch.full((len(sample_ids),), -1, dtype=torch.long)
    fine = torch.full((len(sample_ids),), -1, dtype=torch.long)
    for index, sample_id in enumerate(sample_ids):
        row = manifest.get(str(sample_id))
        if row is None:
            continue
        if row["player_side"] in ("top", "bottom"):
            hitter[index] = int(stroke[index]) * 2 + (row["player_side"] == "bottom")
        if row["raw_label"] in RAW_TYPES:
            fine[index] = RAW_TYPES.index(row["raw_label"])
    return hitter, fine


class AuxHeads(nn.Module):
    """Extra linear heads on the fused [FUS] vector, captured with a forward hook."""

    def __init__(self, model: KAPAblationModel, hidden: int = 128) -> None:
        super().__init__()
        self.hitter = nn.Linear(hidden, 16)
        self.fine = nn.Linear(hidden, len(RAW_TYPES))
        self._fused = None
        model.output_norm.register_forward_hook(lambda _m, _i, out: setattr(self, "_fused", out))

    def forward(self) -> tuple[torch.Tensor, torch.Tensor]:
        return self.hitter(self._fused), self.fine(self._fused)


@torch.inference_mode()
def evaluate(model, tensors, ids, device):
    stroke_prob, side_prob = predict(model, tensors, ids, device)
    def cm(truth, pred, classes):
        matrix = torch.zeros(classes, classes, dtype=torch.int64)
        for t, p in zip(truth, pred):
            matrix[int(t), int(p)] += 1
        return confusion_metrics(matrix)
    return (cm(tensors["stroke"][ids].numpy(), stroke_prob.argmax(1), 8),
            cm(tensors["side"][ids].numpy(), side_prob.argmax(1), 3), stroke_prob, side_prob)


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    out = args.output_root / f"{args.tag}_seed{args.seed}"
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    features = FEATURES[args.features]
    tensors, indices = load_data(features)
    raw = np.load(features)
    tensors["hitter16"], tensors["fine"] = auxiliary_labels(raw["sample_id"], tensors["stroke"])
    test_ids = np.flatnonzero(raw["split"] == "test")

    model = KAPAblationModel(gate=args.gate, rgb_dim=tensors["rgb"].shape[1])
    model.set_normalization(**feature_statistics(tensors, indices["train"]))
    heads = AuxHeads(model)
    model.to(device); heads.to(device)
    params = sum(p.numel() for p in model.parameters())
    print(f"[run] {args.tag} seed={args.seed} params={params} (+aux heads, training only)", flush=True)

    stroke_loss = nn.CrossEntropyLoss(weight=balanced_weights(tensors["stroke"], indices["train"], 8, device))
    side_loss = nn.CrossEntropyLoss(weight=balanced_weights(tensors["side"], indices["train"], 3, device))
    optimizer = torch.optim.AdamW(list(model.parameters()) + list(heads.parameters()), lr=3e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    augmentor = VideoDomainAugmentor(2)
    names = ("rgb", "pose", "court", "shuttle", "contact_dist", "quality")
    train_ids = torch.from_numpy(indices["train"])
    best, history = -1.0, []

    def task_loss(stroke, side, gt):
        loss = stroke_loss(stroke, gt["stroke"])
        if args.side_weight > 0:
            loss = loss + args.side_weight * side_loss(side, gt["side"])
        hit_logits, fine_logits = heads()
        if args.aux_hitter > 0:
            loss = loss + args.aux_hitter * F.cross_entropy(hit_logits, gt["hitter16"], ignore_index=-1)
        if args.aux_fine > 0:
            loss = loss + args.aux_fine * F.cross_entropy(fine_logits, gt["fine"], ignore_index=-1)
        return loss

    for epoch in range(1, args.epochs + 1):
        model.train(); heads.train()
        order = train_ids[torch.randperm(len(train_ids))]
        total, seen = 0.0, 0
        for start in range(0, len(order), 128):
            batch = order[start:start + 128]
            inputs = [tensors[n][batch].to(device) for n in names]
            gt = {k: tensors[k][batch].to(device) for k in ("stroke", "side", "hitter16", "fine")}
            optimizer.zero_grad(set_to_none=True)
            stroke, side, _ = model(*inputs)
            clean = task_loss(stroke, side, gt)
            noisy_stroke, noisy_side, _ = model(*augmentor(inputs))
            noisy = task_loss(noisy_stroke, noisy_side, gt)
            consistency = symmetric_consistency(stroke, noisy_stroke)
            if args.side_weight > 0:
                consistency = consistency + args.side_weight * symmetric_consistency(side, noisy_side)
            loss = clean + 0.75 * noisy + 0.35 * consistency
            loss.backward()
            nn.utils.clip_grad_norm_(list(model.parameters()) + list(heads.parameters()), 1.0)
            optimizer.step()
            total += float(loss.detach()) * len(batch); seen += len(batch)
        scheduler.step()
        val_stroke, val_side, _, _ = evaluate(model, tensors, indices["val"], device)
        score = val_stroke["macro_f1"] + (0.5 * val_side["macro_f1"] if args.side_weight > 0 else 0.0)
        history.append({"epoch": epoch, "loss": total / seen, "val_acc": val_stroke["accuracy"],
                        "val_f1": val_stroke["macro_f1"], "val_side_acc": val_side["accuracy"], "score": score})
        print(f"[{args.tag}] {epoch:02d} loss={total / seen:.4f} val_acc={val_stroke['accuracy']:.4f} "
              f"val_f1={val_stroke['macro_f1']:.4f}", flush=True)
        if score > best:
            best = score
            torch.save({"model": model.state_dict(), "epoch": epoch, "config": vars(args) | {"output_root": str(args.output_root)}},
                       out / "kap_best.pth")

    payload = torch.load(out / "kap_best.pth", map_location=device, weights_only=False)
    model.load_state_dict(payload["model"])
    val_stroke, val_side, _, _ = evaluate(model, tensors, indices["val"], device)
    stroke_prob, side_prob = predict(model, tensors, test_ids, device)
    np.savez_compressed(out / "test_predictions.npz", sample_id=raw["sample_id"][test_ids],
                        stroke_prob=stroke_prob, side_prob=side_prob,
                        stroke_true=tensors["stroke"][test_ids].numpy(), side_true=tensors["side"][test_ids].numpy())
    result = {"tag": args.tag, "seed": args.seed, "config": vars(args) | {"output_root": str(args.output_root)},
              "params": params, "best_epoch": int(payload["epoch"]),
              "val_stroke": val_stroke, "val_side": val_side, "history": history}
    (out / "metrics.json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"[val] {args.tag} seed={args.seed} acc={val_stroke['accuracy']:.4f} f1={val_stroke['macro_f1']:.4f} "
          f"side={val_side['accuracy']:.4f} epoch={payload['epoch']}", flush=True)


if __name__ == "__main__":
    main()
