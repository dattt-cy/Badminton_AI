"""Fast trainer for the paper's KAP runs (main model, seeds and ablations).

Same recipe as train_kap_bstgap.py / train_kap_scratch.py (AdamW 3e-4, cosine,
30 epochs, batch 128, clean + 0.75 noisy + 0.35 symmetric-KL consistency,
selection on validation Match 41), but faster:
  * the whole feature cache lives on the GPU, so batches are not copied per step;
  * the domain augmentor is vectorised (same distributions as VideoDomainAugmentor).

Architecture switches cover every ablation in the paper.

Example:
    python scripts/paper/train_kap_paper.py --tag main --features default --seed 1
    python scripts/paper/train_kap_paper.py --tag no_rgb --no-rgb --seed 1
"""

from __future__ import annotations

import argparse
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

from train_kap_ablation import KAPAblationModel  # noqa: E402
from train_kap_bstgap import FEATURES, AuxHeads, auxiliary_labels  # noqa: E402
from train_kap_scratch import (  # noqa: E402
    VideoDomainAugmentor, balanced_weights, confusion_metrics, feature_statistics,
    load_data, seed_everything, symmetric_consistency,
)

NAMES = ("rgb", "pose", "court", "shuttle", "contact_dist", "quality")


class FastAugmentor:
    """Vectorised VideoDomainAugmentor: identical noise distributions, no per-sample Python loops."""

    def __init__(self, max_shift: int = 2, rgb_mean: torch.Tensor | None = None, rgb_std: torch.Tensor | None = None,
                 branch_drop: float = 0.0) -> None:
        self.max_shift = max_shift
        self.rgb_mean, self.rgb_std = rgb_mean, rgb_std
        self.branch_drop = branch_drop

    @staticmethod
    def _drop_frames(rate: torch.Tensor, shape: tuple[int, ...], frames: int) -> torch.Tensor:
        """Frame mask with scattered losses at the given per-event rate, or one contiguous burst of that share."""
        device = rate.device
        batch = len(rate)
        scattered = torch.rand(*shape, device=device) < rate.view(batch, *([1] * (len(shape) - 1)))
        length = (rate * frames).long().clamp_min(1)
        start = (torch.rand(batch, device=device) * (frames - length + 1)).long()
        steps = torch.arange(frames, device=device)[None, :]
        burst = (steps >= start[:, None]) & (steps < (start + length)[:, None])
        burst = burst.view(batch, frames, *([1] * (len(shape) - 2))).expand(shape)
        return torch.where((torch.rand(batch, device=device) < 0.5).view(batch, *([1] * (len(shape) - 1))), scattered, burst)

    def _drop_pose(self, pose: torch.Tensor, hit: torch.Tensor) -> torch.Tensor:
        """Heavy pose loss for the chosen events: joints or whole players, scattered or in a burst."""
        batch, frames = pose.shape[:2]
        rate = 0.2 + 0.6 * torch.rand(batch, device=pose.device)
        joints = pose.reshape(batch, frames, 2, 17, 2)
        by_joint = self._drop_frames(rate, (batch, frames, 2, 17), frames)
        by_player = self._drop_frames(rate, (batch, frames, 2), frames).unsqueeze(-1).expand(-1, -1, -1, 17)
        use_joint = (torch.rand(batch, device=pose.device) < 0.5).view(batch, 1, 1, 1)
        mask = torch.where(use_joint, by_joint, by_player) & hit.view(batch, 1, 1, 1)
        return joints.masked_fill(mask.unsqueeze(-1), 0.0).reshape_as(pose)

    def _drop_shuttle(self, shuttle: torch.Tensor, hit: torch.Tensor) -> torch.Tensor:
        """Heavy shuttle loss for the chosen events: scattered, in a burst, or the whole trajectory."""
        batch, frames = shuttle.shape[:2]
        rate = 0.2 + 0.8 * torch.rand(batch, device=shuttle.device)
        mask = self._drop_frames(rate, (batch, frames), frames) & hit.view(batch, 1)
        return shuttle.masked_fill(mask.unsqueeze(-1), 0.0)

    def _corrupt_rgb(self, rgb: torch.Tensor, hit: torch.Tensor) -> torch.Tensor:
        """Per chosen event: fade to the mean with noise, lose RGB, take another event's, or lose phase tokens."""
        batch, device = len(rgb), rgb.device
        mean = self.rgb_mean.expand_as(rgb)
        amount = torch.rand(batch, 1, 1, device=device)
        faded = rgb + amount * (mean - rgb) + amount * self.rgb_std * torch.randn_like(rgb)
        other = rgb[torch.randperm(batch, device=device)]
        tokens = torch.where(torch.rand(batch, rgb.shape[1], 1, device=device) < 0.5, mean, rgb)
        kinds = torch.stack([faded, mean, other, tokens])
        picked = kinds[torch.randint(0, 4, (batch,), device=device), torch.arange(batch, device=device)]
        return torch.where(hit.view(batch, 1, 1), picked, rgb)

    @staticmethod
    def _shift(value: torch.Tensor, shifts: torch.Tensor) -> torch.Tensor:
        # out[t] = value[clamp(t - shift)]: edge frames repeat, as in the reference loop.
        frames = value.shape[1]
        steps = torch.arange(frames, device=value.device)[None, :]
        source = (steps - shifts[:, None]).clamp(0, frames - 1)
        index = source.view(*source.shape, *([1] * (value.dim() - 2))).expand_as(value)
        return value.gather(1, index)

    def __call__(self, inputs: list[torch.Tensor]) -> list[torch.Tensor]:
        rgb, pose, court, shuttle, contact, _quality = inputs
        batch, frames = pose.shape[:2]
        device = pose.device
        shifts = torch.randint(-self.max_shift, self.max_shift + 1, (batch,), device=device)
        shifts = torch.where(torch.rand(batch, device=device) < 0.65, shifts, torch.zeros_like(shifts))
        pose, court, shuttle = self._shift(pose, shifts), self._shift(court, shifts), self._shift(shuttle, shifts)

        pose = pose + (torch.rand(batch, 1, 1, device=device) < 0.65).float() * torch.randn_like(pose) * 0.012
        court = court + (torch.rand(batch, 1, 1, device=device) < 0.45).float() * torch.randn_like(court) * 0.008
        shuttle = shuttle + (torch.rand(batch, 1, 1, device=device) < 0.75).float() * torch.randn_like(shuttle) * 0.010

        joints = pose.reshape(batch, frames, -1, 2)
        joints = joints.masked_fill(torch.rand(batch, frames, joints.shape[2], 1, device=device) < 0.035, 0.0)
        pose = joints.reshape_as(pose)
        shuttle = shuttle * (1.0 - (torch.rand(batch, 1, 1, device=device) < 0.15).float())

        # TrackNet gap: with p=0.6, a run of 2-6 frames starting in [4, frames - length).
        length = torch.randint(2, 7, (batch,), device=device)
        high = (frames - length).clamp_min(5)
        start = 4 + (torch.rand(batch, device=device) * (high - 4)).long()
        steps = torch.arange(frames, device=device)[None, :]
        gap = (steps >= start[:, None]) & (steps < (start + length)[:, None])
        gap &= (torch.rand(batch, device=device) < 0.60)[:, None]
        shuttle = shuttle.masked_fill(gap.unsqueeze(-1), 0.0)

        if self.branch_drop > 0:
            # One failure per chosen event: RGB, pose, shuttle, or pose and shuttle together.
            chosen = torch.rand(batch, device=device) < self.branch_drop
            kind = torch.randint(0, 4, (batch,), device=device)
            rgb = self._corrupt_rgb(rgb, chosen & (kind == 0))
            pose = self._drop_pose(pose, chosen & ((kind == 1) | (kind == 3)))
            shuttle = self._drop_shuttle(shuttle, chosen & ((kind == 2) | (kind == 3)))

        pose, court, shuttle =pose.clamp(-1.0, 1.0), court.clamp(-0.1, 1.1), shuttle.clamp(0.0, 1.0)
        return [rgb, pose, court, shuttle, contact, VideoDomainAugmentor._quality(pose, shuttle)]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--features", choices=sorted(FEATURES), default="default")
    parser.add_argument("--gate", choices=("sigmoid", "none"), default="none")
    parser.add_argument("--pooling", choices=("three", "mean"), default="three")
    parser.add_argument("--no-cross", action="store_true")
    parser.add_argument("--no-hitter", action="store_true")
    parser.add_argument("--no-rgb", action="store_true")
    parser.add_argument("--no-noise", action="store_true")
    parser.add_argument("--no-pose", action="store_true")
    parser.add_argument("--no-shuttle", action="store_true")
    parser.add_argument("--no-contact", action="store_true", help="zero the contact encoding; needs --pooling mean")
    parser.add_argument("--rgb-phases", action="store_true", help="three RGB tokens from a (N, 4, dim) RGB input")
    parser.add_argument("--branch-drop", type=float, default=0.0, help="share of events losing RGB, pose, shuttle, or pose and shuttle in the noisy branch")
    parser.add_argument("--aux-hitter", type=float, default=0.0)
    parser.add_argument("--aux-fine", type=float, default=0.0)
    parser.add_argument("--side-weight", type=float, default=0.5)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--output-root", type=Path, default=REPO_ROOT / "work_dirs/paper_main")
    return parser.parse_args()


@torch.inference_mode()
def predict(model: nn.Module, tensors: dict, ids: torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    stroke, side = [], []
    for start in range(0, len(ids), 1024):
        batch = ids[start:start + 1024]
        s, d, _ = model(*[tensors[n][batch] for n in NAMES])
        stroke.append(s.softmax(1)); side.append(d.softmax(1))
    return torch.cat(stroke).cpu().numpy(), torch.cat(side).cpu().numpy()


def metrics(truth: np.ndarray, prediction: np.ndarray, classes: int) -> dict:
    matrix = torch.zeros(classes, classes, dtype=torch.int64)
    np.add.at(matrix.numpy(), (truth, prediction), 1)
    return confusion_metrics(matrix)


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
    stats = feature_statistics(tensors, indices["train"])
    tensors = {k: v.to(device) for k, v in tensors.items()}
    train_ids = torch.from_numpy(indices["train"]).to(device)
    val_ids = torch.from_numpy(indices["val"]).to(device)
    test_np = np.flatnonzero(raw["split"] == "test")
    test_ids = torch.from_numpy(test_np).to(device)

    model = KAPAblationModel(gate=args.gate, pooling=args.pooling, use_cross=not args.no_cross,
                             use_hitter=not args.no_hitter, use_rgb=not args.no_rgb,
                             use_pose=not args.no_pose, use_shuttle=not args.no_shuttle,
                             use_contact=not args.no_contact,
                             rgb_dim=int(tensors["rgb"].shape[-1]),
                             rgb_slots=[1, 2, 3] if args.rgb_phases else None)
    model.set_normalization(**stats)
    heads = AuxHeads(model)
    model.to(device); heads.to(device)
    params = sum(p.numel() for p in model.parameters())
    print(f"[run] {args.tag} seed={args.seed} params={params}", flush=True)

    stroke_loss = nn.CrossEntropyLoss(weight=balanced_weights(tensors["stroke"].cpu(), indices["train"], 8, device))
    side_loss = nn.CrossEntropyLoss(weight=balanced_weights(tensors["side"].cpu(), indices["train"], 3, device))
    trainable = list(model.parameters()) + list(heads.parameters())
    optimizer = torch.optim.AdamW(trainable, lr=3e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    train_rgb = tensors["rgb"][train_ids]
    augmentor = FastAugmentor(2, train_rgb.mean(0), train_rgb.std(0), args.branch_drop)

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

    best, history = -1.0, []
    val_truth = tensors["stroke"][val_ids].cpu().numpy()
    val_side_truth = tensors["side"][val_ids].cpu().numpy()
    for epoch in range(1, args.epochs + 1):
        model.train(); heads.train()
        order = train_ids[torch.randperm(len(train_ids), device=device)]
        total = 0.0
        for start in range(0, len(order), 128):
            batch = order[start:start + 128]
            inputs = [tensors[n][batch] for n in NAMES]
            gt = {k: tensors[k][batch] for k in ("stroke", "side", "hitter16", "fine")}
            optimizer.zero_grad(set_to_none=True)
            stroke, side, _ = model(*inputs)
            loss = task_loss(stroke, side, gt)
            if not args.no_noise:
                noisy_stroke, noisy_side, _ = model(*augmentor(inputs))
                consistency = symmetric_consistency(stroke, noisy_stroke)
                if args.side_weight > 0:
                    consistency = consistency + args.side_weight * symmetric_consistency(side, noisy_side)
                loss = loss + 0.75 * task_loss(noisy_stroke, noisy_side, gt) + 0.35 * consistency
            loss.backward()
            nn.utils.clip_grad_norm_(trainable, 1.0)
            optimizer.step()
            total += float(loss.detach()) * len(batch)
        scheduler.step()
        stroke_prob, side_prob = predict(model, tensors, val_ids)
        val_stroke = metrics(val_truth, stroke_prob.argmax(1), 8)
        val_side = metrics(val_side_truth, side_prob.argmax(1), 3)
        score = val_stroke["macro_f1"] + (0.5 * val_side["macro_f1"] if args.side_weight > 0 else 0.0)
        history.append({"epoch": epoch, "loss": total / len(order), "val_acc": val_stroke["accuracy"],
                        "val_f1": val_stroke["macro_f1"], "score": score})
        print(f"[{args.tag}] {epoch:02d} loss={total / len(order):.4f} val_acc={val_stroke['accuracy']:.4f} "
              f"val_f1={val_stroke['macro_f1']:.4f}", flush=True)
        if score > best:
            best = score
            torch.save({"model": model.state_dict(), "epoch": epoch, "config": {**vars(args), "output_root": str(args.output_root)}},
                       out / "kap_best.pth")

    payload = torch.load(out / "kap_best.pth", map_location=device, weights_only=False)
    model.load_state_dict(payload["model"])
    stroke_prob, side_prob = predict(model, tensors, val_ids)
    val_stroke = metrics(val_truth, stroke_prob.argmax(1), 8)
    val_side = metrics(val_side_truth, side_prob.argmax(1), 3)
    stroke_prob, side_prob = predict(model, tensors, test_ids)
    np.savez_compressed(out / "test_predictions.npz", sample_id=raw["sample_id"][test_np],
                        stroke_prob=stroke_prob, side_prob=side_prob,
                        stroke_true=tensors["stroke"][test_ids].cpu().numpy(),
                        side_true=tensors["side"][test_ids].cpu().numpy())
    result = {"tag": args.tag, "seed": args.seed, "config": {**vars(args), "output_root": str(args.output_root)},
              "params": params, "best_epoch": int(payload["epoch"]),
              "val_stroke": val_stroke, "val_side": val_side, "history": history}
    (out / "metrics.json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"[val] {args.tag} seed={args.seed} acc={val_stroke['accuracy']:.4f} f1={val_stroke['macro_f1']:.4f} "
          f"side={val_side['accuracy']:.4f} epoch={payload['epoch']}", flush=True)


if __name__ == "__main__":
    main()
