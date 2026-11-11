"""What happens to Robust KAP when RGB breaks instead of pose and shuttle?

Corrupts the 16 hitter crops of the 3,334 test events, re-encodes them with MATE and compares,
with pose and shuttle left clean:
  * Robust KAP (main);
  * MATE alone: the encoder's own 8-way head;
  * no-RGB KAP: never sees the crops, so its score is constant;
  * late fusion: equal-weight average of MATE alone and no-RGB KAP probabilities.
The clean row is a control and must reproduce the stored scores.

Example:
    python scripts/paper/rgb_degradation.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO_ROOT / "src"), str(REPO_ROOT / "scripts/training"), str(REPO_ROOT / "scripts/paper")]

import degradation_eval as de  # noqa: E402
from ai_classifier.models.own_rgb_encoder import CONTACT_PHASES as PHASES  # noqa: E402
from train_kap_ablation import KAPAblationModel  # noqa: E402
from train_rgb_scratch import MEAN, STD, RGBEncoder  # noqa: E402

CONDITIONS = [
    ("clean", 0.0), ("shift", 0.15), ("shift", 0.30), ("shift", 0.50), ("blur", 1.5), ("blur", 3.0),
    ("occlude", 0.25), ("occlude", 0.50), ("freeze", 0.50), ("black", 1.0), ("shuffle", 1.0),
]


def corrupt(clips: torch.Tensor, kind: str, level: float, rng: np.random.Generator) -> torch.Tensor:
    """Corrupt a (B, T, C, H, W) batch of crops in [0, 1]."""
    batch, frames, channels, height, width = clips.shape
    device = clips.device
    if kind == "shift":  # Wrong crop box: the whole clip is translated, edge pixels repeat.
        angle = torch.from_numpy(rng.uniform(0, 2 * np.pi, batch)).float().to(device)
        theta = torch.zeros(batch, 2, 3, device=device)
        theta[:, 0, 0] = theta[:, 1, 1] = 1.0
        theta[:, 0, 2], theta[:, 1, 2] = 2 * level * angle.cos(), 2 * level * angle.sin()
        grid = F.affine_grid(theta, (batch, channels, height, width), align_corners=False)
        flat = clips.flatten(0, 1)
        grid = grid.repeat_interleave(frames, dim=0)
        return F.grid_sample(flat, grid, padding_mode="border", align_corners=False).view_as(clips)
    if kind == "blur":  # Motion or low-resolution blur.
        radius = int(3 * level)
        kernel = torch.exp(-0.5 * (torch.arange(-radius, radius + 1, device=device) / level) ** 2)
        kernel = kernel / kernel.sum()
        flat = F.pad(clips.flatten(0, 1), (radius,) * 4, mode="replicate")
        flat = F.conv2d(flat, kernel.view(1, 1, 1, -1).repeat(channels, 1, 1, 1), groups=channels)
        return F.conv2d(flat, kernel.view(1, 1, -1, 1).repeat(channels, 1, 1, 1), groups=channels).view_as(clips)
    if kind == "occlude":  # A static block (other player, net post) covers this share of the crop.
        side = int(round(height * level ** 0.5))
        out = clips.clone()
        for index in range(batch):
            top, left = rng.integers(0, height - side + 1), rng.integers(0, width - side + 1)
            out[index, :, :, top:top + side, left:left + side] = 0.5
        return out
    if kind == "freeze":  # Dropped frames repeat the previous one.
        source = np.tile(np.arange(frames), (batch, 1))
        for t in range(1, frames):
            drop = rng.random(batch) < level
            source[drop, t] = source[drop, t - 1]
        index = torch.from_numpy(source).to(device).view(batch, frames, 1, 1, 1).expand_as(clips)
        return clips.gather(1, index)
    if kind == "black":
        return torch.zeros_like(clips)
    if kind == "shuffle":  # Crops of another event: RGB is confidently misleading.
        return clips[torch.from_numpy(rng.permutation(batch)).to(device)]
    return clips


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, default=REPO_ROOT / "work_dirs/paper_mate/final")
    parser.add_argument("--features", type=Path, default=REPO_ROOT / "work_dirs/paper_rgb_phases/features_rgb_phases.npz")
    parser.add_argument("--encoder", type=Path, default=REPO_ROOT / "work_dirs/paper_ownrgb_ss22/x13/rgb_encoder_best.pth")
    parser.add_argument("--cache", type=Path, default=Path(r"C:\Users\ADMIN\Downloads\AI_Classifier_cache\shuttleset_v3_full\frames_16"))
    parser.add_argument("--mask-seed", type=int, default=20261104)
    parser.add_argument("--family", default="main", help="KAP family to corrupt; no-RGB KAP always comes from --runs")
    parser.add_argument("--family-runs", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--no-rgb-family", default="no_rgb")
    parser.add_argument("--compare", nargs="*", default=[], help="more families from --family-runs, scored per seed under the same crops")
    args = parser.parse_args()
    device = torch.device("cuda")

    tensors, ids, truth = de.kap_inputs(args.features)
    encoder = RGBEncoder().to(device)
    encoder.load_state_dict(torch.load(args.encoder, map_location=device, weights_only=False)["model"])
    encoder.eval()
    steps: list[torch.Tensor] = []
    encoder.net.temporal_tcn2.register_forward_hook(lambda _m, _i, out: steps.append(out))
    mean, std = MEAN.to(device), STD.to(device)

    def family(name: str, root: Path) -> list:
        models = []
        for run in sorted(root.glob(f"{name}_seed*/kap_best.pth")):
            model = KAPAblationModel(**json.loads((run.parent / "model_kwargs.json").read_text(encoding="utf-8")))
            model.load_state_dict(torch.load(run, map_location="cpu", weights_only=False)["model"])
            models.append(model.to(device).eval())
        return models

    main_models, no_rgb_models = family(args.family, args.family_runs or args.runs), family(args.no_rgb_family, args.runs)
    compared = {name: family(name, args.family_runs or args.runs) for name in [args.family, *args.compare]}
    both_lost = de.degrade_kap(tensors, "both", 0.5, de.mask_rng(args.mask_seed, "both", 0.5, "kap"))
    no_rgb = [de.kap_predict(m, tensors, device) for m in no_rgb_models]
    acc = lambda p: float((p.argmax(1) == truth).mean() * 100)  # noqa: E731
    f1 = lambda p: de.scores(truth, p.argmax(1))["macro_f1"] * 100  # noqa: E731
    result = {"events": len(ids), "no_rgb": float(np.mean([acc(p) for p in no_rgb])), "no_rgb_f1": float(np.mean([f1(p) for p in no_rgb])), "conditions": {}}
    print(f"no-RGB KAP (constant) {result['no_rgb']:.2f}", flush=True)
    output = args.output or args.runs / "rgb_degradation.json"

    for kind, level in CONDITIONS:
        rng = np.random.default_rng([args.mask_seed, len(kind), int(level * 100)])
        vectors, alone = [], []
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
            for first in range(0, len(ids), 64):
                batch = []
                for sample in ids[first:first + 64]:
                    file = args.cache / f"{sample}.npy"
                    batch.append(torch.from_numpy(np.load(file)).float().div_(255.0) if file.exists() else torch.zeros(16, 3, 112, 112))
                clips = corrupt(torch.stack(batch).to(device), kind, level, rng).clamp(0, 1)
                steps.clear()
                logits = encoder((clips.permute(0, 2, 1, 3, 4) - mean) / std)
                context = encoder._context
                pooled = [steps[0][:, :, phase].mean(dim=-1) for phase in PHASES]
                vectors.append(torch.stack([context, *pooled], dim=1).float().cpu())
                alone.append(logits.float().softmax(1).cpu())
        rgb = torch.cat(vectors)
        alone = torch.cat(alone).numpy()
        with torch.inference_mode():
            kap = [de.kap_predict(m, {**tensors, "rgb": rgb}, device) for m in main_models]
        late = [0.5 * p + 0.5 * alone for p in no_rgb]
        row = {
            "kap": float(np.mean([acc(p) for p in kap])), "kap_std": float(np.std([acc(p) for p in kap], ddof=1)),
            "kap_f1": float(np.mean([f1(p) for p in kap])), "mate_alone": acc(alone), "late_fusion": float(np.mean([acc(p) for p in late])),
        }
        with torch.inference_mode():
            row["families"] = {
                name: {
                    "per_seed": [acc(de.kap_predict(m, {**tensors, "rgb": rgb}, device)) for m in models],
                    "with_pose_shuttle_50": [acc(de.kap_predict(m, {**both_lost, "rgb": rgb}, device)) for m in models],
                }
                for name, models in compared.items()
            }
        key = f"{kind}@{level:.2f}"
        if kind == "clean":
            row["stored_feature_max_diff"] = float((rgb - tensors["rgb"]).abs().max())
        result["conditions"][key] = row
        print(key, {k: round(v, 3) for k, v in row.items() if k != "families"}, flush=True)
        for name, scores in row["families"].items():
            print(f"  {name}: {np.mean(scores['per_seed']):.2f} | with pose+shuttle 50% {np.mean(scores['with_pose_shuttle_50']):.2f}", flush=True)
        output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
