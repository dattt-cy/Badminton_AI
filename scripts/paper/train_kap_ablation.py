"""Train KAPCrossTransV2 ablations for the Robust KAP paper.

Each run trains one variant with one seed on the match-disjoint cache
(train: 30 matches, val: match 41, test: matches 39/40/42/43/44) and stores
per-sample test probabilities for paired significance tests.  The training
recipe is the one of scripts/training/train_kap_scratch.py --model-type cross_v2.

Example:
    python scripts/paper/train_kap_ablation.py --variant full --seed 1
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
for path in (REPO_ROOT / "src", REPO_ROOT / "scripts" / "training"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from ai_classifier.models import KAPCrossTransV2Model  # noqa: E402
from train_kap_scratch import (  # noqa: E402
    confusion_metrics, feature_statistics, load_data, make_loader,
    seed_everything, train_kap,
)

VARIANTS = {
    "full": {},
    "no_gate": {"gate": "none"},
    "softmax_gate": {"gate": "softmax"},
    "no_phase": {"pooling": "mean"},
    "no_cross": {"use_cross": False},
    "no_hitter": {"use_hitter": False},
    "no_rgb": {"use_rgb": False},
    "no_noise": {"noise": False},
}
# Index of each token in modality_embedding: fusion, rgb, pose, court, shuttle, cross.
FUSION, RGB, POSE, COURT, SHUTTLE, CROSS = range(6)


class KAPAblationModel(KAPCrossTransV2Model):
    """KAPCrossTransV2 with switchable gate, phase pooling, cross branch, hitter module and RGB."""

    def __init__(
        self, gate: str = "sigmoid", pooling: str = "three", use_cross: bool = True,
        use_hitter: bool = True, use_rgb: bool = True, rgb_dim: int = 512,
    ):
        super().__init__(rgb_dim=rgb_dim)
        self.gate_mode, self.pooling = gate, pooling
        self.use_cross, self.use_hitter, self.use_rgb = use_cross, use_hitter, use_rgb
        hidden = self.hidden_dim
        if gate == "none":
            self.quality_gate = None
        elif gate == "softmax":
            self.quality_gate[3] = nn.Identity()
        if not use_rgb:
            self.rgb_encoder = None
        if not use_hitter:
            self.hitter_disambiguation = None
        if not use_cross:
            for name in (
                "cross_query_norm1", "cross_query_norm2", "cross_key_value_norm",
                "p1_to_shuttle", "p2_to_shuttle", "cross_attention_norm",
                "cross_feed_forward", "cross_output_norm", "cross_phase_proj",
            ):
                setattr(self, name, None)
        if pooling == "mean":
            for encoder in (self.pose_encoder, self.court_encoder, self.shuttle_encoder):
                encoder.phase_proj[0] = nn.Linear(hidden, hidden)
            if use_cross:
                self.cross_phase_proj[0] = nn.Linear(hidden, hidden)

    def _encode(self, encoder, sequence, contact):
        if self.pooling == "three":
            return encoder(sequence, contact)
        projected = encoder.input_proj(sequence) + encoder.contact_proj(contact)
        temporal = encoder.temporal(projected.transpose(1, 2)).transpose(1, 2)
        encoded = F.gelu(projected + temporal)
        return encoded, encoder.phase_proj(encoded.mean(dim=1))

    def _phase_pool(self, sequence, contact_distance):
        if self.pooling == "three":
            return super()._phase_pool(sequence, contact_distance)
        return self.cross_phase_proj(sequence.mean(dim=1))

    def forward(self, rgb, pose, court, shuttle, contact_distance, quality):
        pose_n = self._normalize(pose, self.pose_mean, self.pose_std)
        court_n = self._normalize(court, self.court_mean, self.court_std)
        shuttle_n = self._normalize(shuttle, self.shuttle_mean, self.shuttle_std)
        quality_n = self._normalize(quality, self.quality_mean, self.quality_std)

        seq_p1, tok_p1 = self._encode(self.pose_encoder, pose_n[:, :, :34], contact_distance)
        seq_p2, tok_p2 = self._encode(self.pose_encoder, pose_n[:, :, 34:], contact_distance)
        _, court_token = self._encode(self.court_encoder, court_n, contact_distance)
        shuttle_sequence, shuttle_token = self._encode(self.shuttle_encoder, shuttle_n, contact_distance)

        if self.use_hitter:
            hitter_weights = F.softmax(
                self.hitter_disambiguation(torch.cat([tok_p1, tok_p2, shuttle_token], dim=-1)), dim=-1
            )
        else:
            hitter_weights = torch.full((pose.shape[0], 2), 0.5, device=pose.device)
        w1 = hitter_weights[:, 0:1].unsqueeze(1)
        w2 = hitter_weights[:, 1:2].unsqueeze(1)
        active_pose_token = hitter_weights[:, 0:1] * tok_p1 + hitter_weights[:, 1:2] * tok_p2

        if self.gate_mode == "none":
            gates = torch.ones(pose.shape[0], 4, device=pose.device)
        else:
            gates = self.quality_gate(quality_n)
            if self.gate_mode == "softmax":
                gates = gates.softmax(dim=1) * 4

        tokens = {POSE: active_pose_token, COURT: court_token, SHUTTLE: shuttle_token}
        token_gates = {POSE: gates[:, 1], COURT: gates[:, 2], SHUTTLE: gates[:, 3]}
        if self.use_rgb:
            tokens[RGB], token_gates[RGB] = self.rgb_encoder(rgb), gates[:, 0]
        if self.use_cross:
            shuttle_kv = self.cross_key_value_norm(shuttle_sequence)
            cross_p1, _ = self.p1_to_shuttle(self.cross_query_norm1(seq_p1), shuttle_kv, shuttle_kv, need_weights=False)
            cross_p2, _ = self.p2_to_shuttle(self.cross_query_norm2(seq_p2), shuttle_kv, shuttle_kv, need_weights=False)
            active_pose = w1 * seq_p1 + w2 * seq_p2
            active_cross = w1 * cross_p1 + w2 * cross_p2
            cross_sequence = self.cross_attention_norm(active_pose + gates[:, 3:4].unsqueeze(1) * active_cross)
            cross_sequence = self.cross_output_norm(cross_sequence + self.cross_feed_forward(cross_sequence))
            tokens[CROSS] = self._phase_pool(cross_sequence, contact_distance)
            token_gates[CROSS] = torch.sqrt((gates[:, 1] * gates[:, 3]).clamp_min(1e-8))

        order = sorted(tokens)
        stacked = torch.stack([tokens[i] for i in order], dim=1)
        stacked = stacked * torch.stack([token_gates[i] for i in order], dim=1).unsqueeze(-1)
        embedding = self.modality_embedding[:, [FUSION, *order]]
        fusion_token = self.fusion_token.expand(stacked.shape[0], -1, -1)
        fused = self.output_norm(self.fusion(torch.cat([fusion_token, stacked], dim=1) + embedding)[:, 0])
        return self.stroke_head(fused), self.side_head(fused), gates


def build_model(variant: str) -> KAPAblationModel:
    return KAPAblationModel(**{k: v for k, v in VARIANTS[variant].items() if k != "noise"})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=sorted(VARIANTS), required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument(
        "--features", type=Path,
        default=REPO_ROOT / "work_dirs/context_window_pilot/features_w15_30.npz",
    )
    parser.add_argument("--output-root", type=Path, default=REPO_ROOT / "work_dirs/paper_ablation")
    parser.add_argument("--kap-epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=128)
    return parser.parse_args()


@torch.inference_mode()
def predict(model: nn.Module, tensors: dict, ids: np.ndarray, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    stroke_parts, side_parts = [], []
    names = ("rgb", "pose", "court", "shuttle", "contact_dist", "quality")
    for start in range(0, len(ids), 512):
        batch = torch.from_numpy(ids[start:start + 512]).long()
        stroke, side, _ = model(*[tensors[n][batch].to(device) for n in names])
        stroke_parts.append(stroke.softmax(1).cpu())
        side_parts.append(side.softmax(1).cpu())
    return torch.cat(stroke_parts).numpy(), torch.cat(side_parts).numpy()


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    output_dir = args.output_root / f"{args.variant}_seed{args.seed}"
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tensors, indices = load_data(args.features)
    raw = np.load(args.features)
    # Evaluate on every test event (3,334), including the two events whose
    # structured streams are all zero, so the set matches BST-CG-AP's.
    test_ids = np.flatnonzero(raw["split"] == "test")
    sample_ids = raw["sample_id"][test_ids]

    model = build_model(args.variant)
    model.set_normalization(**feature_statistics(tensors, indices["train"]))
    model.to(device)
    noise = VARIANTS[args.variant].get("noise", True)
    train_args = argparse.Namespace(
        kap_epochs=args.kap_epochs, kap_learning_rate=3e-4, weight_decay=1e-4,
        side_loss_weight=0.5, consistency_weight=0.35 if noise else 0.0,
        noisy_loss_weight=0.75 if noise else 0.0, max_temporal_shift=2,
        lift_loss_weight=1.0, model_type=f"cross_v2_ablation_{args.variant}",
        feature_provenance="frozen_cached_feature_extractors",
    )
    params = sum(p.numel() for p in model.parameters())
    print(f"[run] {args.variant} seed={args.seed} params={params}", flush=True)
    history = train_kap(
        model,
        make_loader(tensors, indices["train"], args.batch_size, shuffle=True),
        make_loader(tensors, indices["val"], args.batch_size, shuffle=False),
        tensors, indices["train"], train_args, device, output_dir / "kap_best.pth",
    )
    payload = torch.load(output_dir / "kap_best.pth", map_location=device, weights_only=False)
    model.load_state_dict(payload["model"])

    stroke_prob, side_prob = predict(model, tensors, test_ids, device)
    stroke_true = tensors["stroke"][test_ids].numpy()
    side_true = tensors["side"][test_ids].numpy()
    np.savez_compressed(
        output_dir / "test_predictions.npz", sample_id=sample_ids,
        stroke_prob=stroke_prob, side_prob=side_prob,
        stroke_true=stroke_true, side_true=side_true,
    )

    def cm(truth, pred, classes):
        matrix = torch.zeros(classes, classes, dtype=torch.int64)
        for t, p in zip(truth, pred):
            matrix[int(t), int(p)] += 1
        return confusion_metrics(matrix)

    result = {
        "variant": args.variant, "seed": args.seed, "params": params,
        "features": str(args.features), "splits": {k: len(v) for k, v in indices.items()},
        "test_samples": len(test_ids), "best_epoch": int(payload["epoch"]),
        "history": [{k: v for k, v in row.items() if k in ("epoch", "train_loss", "selection_score")} for row in history],
        "test_stroke": cm(stroke_true, stroke_prob.argmax(1), stroke_prob.shape[1]),
        "test_side": cm(side_true, side_prob.argmax(1), side_prob.shape[1]),
    }
    (output_dir / "metrics.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(
        f"[test] {args.variant} seed={args.seed} acc={result['test_stroke']['accuracy']:.4f} "
        f"f1={result['test_stroke']['macro_f1']:.4f} side={result['test_side']['accuracy']:.4f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
