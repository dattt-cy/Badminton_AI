"""Train KAP with the own-RGB stream split by contact phase, on the clean features (resumable).

``--tokens phases`` uses three RGB tokens (before, contact, after); ``--tokens context_phases``
also keeps the whole-clip vector. Training follows scripts/paper/train_kap_bstgap.py, and the
result is compared with that script's single-token KAP on the same seed.

    python scripts/paper/train_kap_rgb_phases.py --tokens phases --seed 1
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO_ROOT / "src"), str(REPO_ROOT / "scripts/training"), str(REPO_ROOT / "scripts/paper")]

from train_kap_ablation import COURT, CROSS, FUSION, POSE, RGB, SHUTTLE, KAPAblationModel, predict  # noqa: E402
from train_kap_bstgap import evaluate  # noqa: E402
from train_kap_scratch import (  # noqa: E402
    VideoDomainAugmentor, balanced_weights, feature_statistics, load_data, seed_everything, symmetric_consistency,
)

SLOTS = {"phases": [1, 2, 3], "context_phases": [0, 1, 2, 3]}
NAMES = ("rgb", "pose", "court", "shuttle", "contact_dist", "quality")


class KAPRgbPhases(KAPAblationModel):
    """KAP where RGB enters as one token per phase, each marked by a learned phase embedding."""

    def __init__(self, slots: list[int]) -> None:
        super().__init__(gate="none", rgb_dim=224)
        self.slots = slots
        self.phase_embedding = nn.Parameter(torch.zeros(1, len(slots), self.hidden_dim))

    def forward(self, rgb, pose, court, shuttle, contact_distance, quality):
        pose_n = self._normalize(pose, self.pose_mean, self.pose_std)
        court_n = self._normalize(court, self.court_mean, self.court_std)
        shuttle_n = self._normalize(shuttle, self.shuttle_mean, self.shuttle_std)
        seq_p1, tok_p1 = self._encode(self.pose_encoder, pose_n[:, :, :34], contact_distance)
        seq_p2, tok_p2 = self._encode(self.pose_encoder, pose_n[:, :, 34:], contact_distance)
        _, court_token = self._encode(self.court_encoder, court_n, contact_distance)
        shuttle_sequence, shuttle_token = self._encode(self.shuttle_encoder, shuttle_n, contact_distance)
        weights = F.softmax(self.hitter_disambiguation(torch.cat([tok_p1, tok_p2, shuttle_token], dim=-1)), dim=-1)
        w1, w2 = weights[:, 0:1].unsqueeze(1), weights[:, 1:2].unsqueeze(1)
        active_pose_token = weights[:, 0:1] * tok_p1 + weights[:, 1:2] * tok_p2
        shuttle_kv = self.cross_key_value_norm(shuttle_sequence)
        cross_p1, _ = self.p1_to_shuttle(self.cross_query_norm1(seq_p1), shuttle_kv, shuttle_kv, need_weights=False)
        cross_p2, _ = self.p2_to_shuttle(self.cross_query_norm2(seq_p2), shuttle_kv, shuttle_kv, need_weights=False)
        cross_sequence = self.cross_attention_norm(w1 * seq_p1 + w2 * seq_p2 + w1 * cross_p1 + w2 * cross_p2)
        cross_sequence = self.cross_output_norm(cross_sequence + self.cross_feed_forward(cross_sequence))
        cross_token = self._phase_pool(cross_sequence, contact_distance)

        rgb_tokens = self.rgb_encoder(rgb[:, self.slots]) + self.phase_embedding
        others = torch.stack([active_pose_token, court_token, shuttle_token, cross_token], dim=1)
        embedding = torch.cat([self.modality_embedding[:, [FUSION]],
                               self.modality_embedding[:, [RGB]].expand(-1, len(self.slots), -1),
                               self.modality_embedding[:, [POSE, COURT, SHUTTLE, CROSS]]], dim=1)
        fusion_token = self.fusion_token.expand(rgb.shape[0], -1, -1)
        fused = self.output_norm(self.fusion(torch.cat([fusion_token, rgb_tokens, others], dim=1) + embedding)[:, 0])
        return self.stroke_head(fused), self.side_head(fused), None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokens", choices=sorted(SLOTS), required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--features", type=Path, default=REPO_ROOT / "work_dirs/paper_rgb_phases/features_rgb_phases.npz")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--side-weight", type=float, default=0.5)
    parser.add_argument("--output-root", type=Path, default=REPO_ROOT / "work_dirs/paper_rgb_phases")
    args = parser.parse_args()
    seed_everything(args.seed)
    out = args.output_root / f"{args.tokens}_seed{args.seed}"
    out.mkdir(parents=True, exist_ok=True)
    if (out / "metrics.json").exists():
        print(f"[skip] {out.name}", flush=True)
        return
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tensors, indices = load_data(args.features)
    raw = np.load(args.features, allow_pickle=True)
    test_ids = np.flatnonzero(raw["split"] == "test")
    model = KAPRgbPhases(SLOTS[args.tokens])
    model.set_normalization(**feature_statistics(tensors, indices["train"]))
    model.to(device)
    print(f"[run] {out.name} params={sum(p.numel() for p in model.parameters())}", flush=True)

    stroke_loss = nn.CrossEntropyLoss(weight=balanced_weights(tensors["stroke"], indices["train"], 8, device))
    side_loss = nn.CrossEntropyLoss(weight=balanced_weights(tensors["side"], indices["train"], 3, device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    augmentor = VideoDomainAugmentor(2)
    train_ids = torch.from_numpy(indices["train"])
    start, best, history = 1, -1.0, []
    if (out / "last.pth").exists():
        state = torch.load(out / "last.pth", map_location=device, weights_only=False)
        model.load_state_dict(state["model"]); optimizer.load_state_dict(state["optimizer"]); scheduler.load_state_dict(state["scheduler"])
        torch.set_rng_state(state["torch_rng"].cpu())
        start, best, history = state["epoch"] + 1, state["best"], state["history"]
        print(f"[resume] from epoch {state['epoch']}", flush=True)

    def task(stroke, side, gt):
        return stroke_loss(stroke, gt["stroke"]) + args.side_weight * side_loss(side, gt["side"])

    for epoch in range(start, args.epochs + 1):
        model.train()
        order = train_ids[torch.randperm(len(train_ids))]
        total, seen = 0.0, 0
        for first in range(0, len(order), 128):
            batch = order[first:first + 128]
            inputs = [tensors[name][batch].to(device) for name in NAMES]
            gt = {key: tensors[key][batch].to(device) for key in ("stroke", "side")}
            optimizer.zero_grad(set_to_none=True)
            stroke, side, _ = model(*inputs)
            noisy_stroke, noisy_side, _ = model(*augmentor(inputs))
            consistency = symmetric_consistency(stroke, noisy_stroke) + args.side_weight * symmetric_consistency(side, noisy_side)
            loss = task(stroke, side, gt) + 0.75 * task(noisy_stroke, noisy_side, gt) + 0.35 * consistency
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total += float(loss.detach()) * len(batch); seen += len(batch)
        scheduler.step()
        val_stroke, val_side, _, _ = evaluate(model, tensors, indices["val"], device)
        score = val_stroke["macro_f1"] + 0.5 * val_side["macro_f1"]
        history.append({"epoch": epoch, "loss": total / seen, "val_acc": val_stroke["accuracy"],
                        "val_f1": val_stroke["macro_f1"], "val_side_acc": val_side["accuracy"], "score": score})
        print(f"[{out.name}] {epoch:02d} loss={total / seen:.4f} val_acc={val_stroke['accuracy']:.4f} val_f1={val_stroke['macro_f1']:.4f}", flush=True)
        if score > best:
            best = score
            torch.save({"model": model.state_dict(), "epoch": epoch, "tokens": args.tokens}, out / "kap_best.pth")
        temporary = out / "last.tmp"
        torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
                    "torch_rng": torch.get_rng_state(), "epoch": epoch, "best": best, "history": history}, temporary)
        os.replace(temporary, out / "last.pth")

    payload = torch.load(out / "kap_best.pth", map_location=device, weights_only=False)
    model.load_state_dict(payload["model"])
    val_stroke, val_side, _, _ = evaluate(model, tensors, indices["val"], device)
    stroke_prob, side_prob = predict(model, tensors, test_ids, device)
    np.savez_compressed(out / "test_predictions.npz", sample_id=raw["sample_id"][test_ids], stroke_prob=stroke_prob, side_prob=side_prob,
                        stroke_true=tensors["stroke"][test_ids].numpy(), side_true=tensors["side"][test_ids].numpy())
    (out / "metrics.json").write_text(json.dumps({"tokens": args.tokens, "seed": args.seed, "best_epoch": int(payload["epoch"]),
                                                  "val_stroke": val_stroke, "val_side": val_side, "history": history}, indent=2) + "\n", encoding="utf-8")
    (out / "last.pth").unlink(missing_ok=True)
    print(f"[val] {out.name} acc={val_stroke['accuracy']:.4f} f1={val_stroke['macro_f1']:.4f} side={val_side['accuracy']:.4f} epoch={payload['epoch']}", flush=True)


if __name__ == "__main__":
    main()
