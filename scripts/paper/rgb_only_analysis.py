"""Is Robust KAP just falling back on the RGB prediction?

Compares, on the 3,334 test events and under the degradation masks of
degradation_eval.py:
  * RGB-only:     an MLP on the frozen 512-d RGB embedding (train split, val-selected);
  * late fusion:  equal-weight average of RGB-only and no-RGB KAP probabilities;
  * Robust KAP:   the main model (joint fusion inside the Transformer).
RGB is never masked, so RGB-only accuracy is constant across conditions.

Example:
    python scripts/paper/rgb_only_analysis.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn

REPO_ROOT = Path(__file__).resolve().parents[2]
for path in (REPO_ROOT / "src", REPO_ROOT / "scripts" / "training", REPO_ROOT / "scripts" / "paper"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import degradation_eval as de  # noqa: E402
from train_kap_ablation import KAPAblationModel  # noqa: E402
from train_kap_scratch import balanced_weights, load_data, seed_everything  # noqa: E402

KEYS = [("both", 0.0), ("both", 0.3), ("both", 0.5), ("pose", 0.5), ("burst", 0.5), ("player", 0.5), ("shuttle", 1.0)]


def train_rgb_only(tensors, indices, device, seed: int) -> nn.Module:
    seed_everything(seed)
    model = nn.Sequential(nn.LayerNorm(512), nn.Linear(512, 128), nn.GELU(), nn.Dropout(0.2), nn.Linear(128, 8)).to(device)
    x = tensors["rgb"].to(device); y = tensors["stroke"].to(device)
    train = torch.from_numpy(indices["train"]).to(device); val = torch.from_numpy(indices["val"]).to(device)
    loss_fn = nn.CrossEntropyLoss(weight=balanced_weights(tensors["stroke"], indices["train"], 8, device))
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    best, state = -1.0, None
    for _ in range(30):
        model.train()
        for start in range(0, len(train), 256):
            batch = train[torch.randperm(len(train), device=device)][:256] if start == 0 else train[start:start + 256]
            opt.zero_grad(); loss_fn(model(x[batch]), y[batch]).backward(); opt.step()
        model.eval()
        with torch.inference_mode():
            acc = (model(x[val]).argmax(1) == y[val]).float().mean().item()
        if acc > best:
            best, state = acc, {k: v.clone() for k, v in model.state_dict().items()}
    model.load_state_dict(state)
    return model


def main() -> None:
    device = torch.device("cuda")
    args = de.parse_args()
    tensors_all, indices = load_data(args.features)
    test_tensors, _, truth = de.kap_inputs(args.features)

    rgb_probs = []
    for seed in (1, 2, 3):
        model = train_rgb_only(tensors_all, indices, device, seed)
        with torch.inference_mode():
            rgb_probs.append(model(test_tensors["rgb"].to(device)).softmax(1).cpu().numpy())
    rgb_prob = np.mean(rgb_probs, axis=0)
    rgb_accs = [de.scores(truth, p.argmax(1))["accuracy"] * 100 for p in rgb_probs]

    def family_probs(family: str):
        out = []
        for run in sorted(args.runs.glob(f"{family}_seed*/kap_best.pth")):
            kwargs = json.loads((run.parent / "model_kwargs.json").read_text(encoding="utf-8"))
            model = KAPAblationModel(**kwargs)
            model.load_state_dict(torch.load(run, map_location="cpu", weights_only=False)["model"])
            model.to(device).eval()
            out.append({f"{c}@{r:.1f}": de.kap_predict(model, de.degrade_kap(test_tensors, c, r, de.mask_rng(args.mask_seed, c, r, "kap")), device)
                        for c, r in KEYS})
        return out

    main_runs, no_rgb_runs = family_probs("main"), family_probs("no_rgb")
    result = {"rgb_only_acc": {"mean": float(np.mean(rgb_accs)), "seeds": rgb_accs}, "conditions": {}}
    print(f"RGB một mình (3 seed): {np.mean(rgb_accs):.2f}  {[round(a, 2) for a in rgb_accs]}")
    print(f"{'điều kiện':14s} {'RGB một mình':>12s} {'KAP không RGB':>14s} {'ghép thô':>9s} {'Robust KAP':>11s}")
    for c, r in KEYS:
        key = f"{c}@{r:.1f}"
        acc = lambda p: de.scores(truth, p.argmax(1))["accuracy"] * 100  # noqa: E731
        no_rgb = np.mean([acc(run[key]) for run in no_rgb_runs])
        fusion = np.mean([acc(0.5 * run[key] + 0.5 * rgb_prob) for run in no_rgb_runs])
        full = np.mean([acc(run[key]) for run in main_runs])
        result["conditions"][key] = {"rgb_only": float(np.mean(rgb_accs)), "no_rgb": float(no_rgb),
                                     "late_fusion": float(fusion), "kap": float(full)}
        print(f"{key:14s} {np.mean(rgb_accs):12.2f} {no_rgb:14.2f} {fusion:9.2f} {full:11.2f}")
    (args.runs / "rgb_only_analysis.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
