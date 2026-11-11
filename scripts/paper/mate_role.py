"""How much of Robust KAP is MATE alone, and what does each RGB phase contribute?

On the 3,334 test events and the degradation masks of degradation_eval.py, compares:
  * MATE alone:   the encoder's own 8-way head (RGB is never masked, so it is constant);
  * late fusion:  equal-weight average of MATE alone and no-RGB KAP probabilities;
  * one token and three phases inside Robust KAP.
Also replaces each RGB phase by its train-set mean (no retraining) and reports the two
hardest confusion pairs for one token against three phases.

Example:
    python scripts/paper/mate_role.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO_ROOT / "src"), str(REPO_ROOT / "scripts/training"), str(REPO_ROOT / "scripts/paper")]

import degradation_eval as de  # noqa: E402
from summarize import mcnemar_exact  # noqa: E402
from train_kap_ablation import KAPAblationModel  # noqa: E402
from train_rgb_scratch import MEAN, STD, RGBEncoder  # noqa: E402

KEYS = [("both", 0.0), ("both", 0.3), ("both", 0.5), ("pose", 0.5), ("burst", 0.5), ("player", 0.5), ("shuttle", 0.5), ("shuttle", 1.0)]
PAIRS = {"lift_net_attack": (5, 7), "drop_smash": (3, 2)}
PHASES = {"before": [1], "contact": [2], "after": [3], "all": [1, 2, 3]}


@torch.inference_mode()
def mate_alone(encoder: Path, cache: Path, ids: np.ndarray, device: torch.device) -> np.ndarray:
    model = RGBEncoder().to(device)
    model.load_state_dict(torch.load(encoder, map_location=device, weights_only=False)["model"])
    model.eval()
    out = []
    for start in range(0, len(ids), 64):
        batch = []
        for sample in ids[start:start + 64]:
            file = cache / f"{sample}.npy"
            clip = torch.from_numpy(np.load(file)).float().div_(255.0) if file.exists() else torch.zeros(16, 3, 112, 112)
            batch.append((clip.permute(1, 0, 2, 3).clamp(0, 1) - MEAN) / STD)
        out.append(model(torch.stack(batch).to(device)).softmax(1).cpu())
    return torch.cat(out).numpy()


def pair_errors(truth: np.ndarray, prediction: np.ndarray, a: int, b: int) -> int:
    return int((((truth == a) & (prediction == b)) | ((truth == b) & (prediction == a))).sum())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, default=REPO_ROOT / "work_dirs/paper_mate/final")
    parser.add_argument("--features", type=Path, default=REPO_ROOT / "work_dirs/paper_rgb_phases/features_rgb_phases.npz")
    parser.add_argument("--encoder", type=Path, default=REPO_ROOT / "work_dirs/paper_ownrgb_ss22/x13/rgb_encoder_best.pth")
    parser.add_argument("--cache", type=Path, default=Path(r"C:\Users\ADMIN\Downloads\AI_Classifier_cache\shuttleset_v3_full\frames_16"))
    parser.add_argument("--mask-seed", type=int, default=20261101)
    args = parser.parse_args()
    device = torch.device("cuda")

    tensors, ids, truth = de.kap_inputs(args.features)
    raw = np.load(args.features)
    train_mean = torch.from_numpy(raw["rgb"][raw["split"] == "train"].mean(axis=0)).float()
    acc = lambda p: float((p.argmax(1) == truth).mean() * 100)  # noqa: E731
    alone = mate_alone(args.encoder, args.cache, ids, device)

    def family(name: str) -> list:
        models = []
        for run in sorted(args.runs.glob(f"{name}_seed*/kap_best.pth")):
            model = KAPAblationModel(**json.loads((run.parent / "model_kwargs.json").read_text(encoding="utf-8")))
            model.load_state_dict(torch.load(run, map_location="cpu", weights_only=False)["model"])
            models.append(model.to(device).eval())
        return models

    main_models, one_models, no_rgb_models = family("main"), family("one_token"), family("no_rgb")
    result = {"mate_alone": acc(alone), "mate_alone_f1": de.scores(truth, alone.argmax(1))["macro_f1"] * 100, "conditions": {}, "phase_removed": {}, "pairs": {}, "paired": {}}
    print(f"MATE alone {result['mate_alone']:.2f}")
    for condition, rate in KEYS:
        key = f"{condition}@{rate:.1f}"
        degraded = de.degrade_kap(tensors, condition, rate, de.mask_rng(args.mask_seed, condition, rate, "kap"))
        no_rgb = [de.kap_predict(m, degraded, device) for m in no_rgb_models]
        row = {
            "no_rgb": float(np.mean([acc(p) for p in no_rgb])),
            "late_fusion": float(np.mean([acc(0.5 * p + 0.5 * alone) for p in no_rgb])),
            "one_token": float(np.mean([acc(de.kap_predict(m, degraded, device)) for m in one_models])),
            "kap": float(np.mean([acc(de.kap_predict(m, degraded, device)) for m in main_models])),
        }
        result["conditions"][key] = row
        print(key, {k: round(v, 2) for k, v in row.items()}, flush=True)

    clean = [de.kap_predict(m, tensors, device) for m in main_models]
    one = [de.kap_predict(m, tensors, device) for m in one_models]
    no_rgb_clean = [de.kap_predict(m, tensors, device) for m in no_rgb_models]
    for name, slots in PHASES.items():
        scores = []
        for model in main_models:
            masked = tensors["rgb"].clone()
            masked[:, slots] = train_mean[slots]
            scores.append(acc(de.kap_predict(model, {**tensors, "rgb": masked}, device)))
        result["phase_removed"][name] = float(np.mean(scores))
    print("phase removed", result["phase_removed"], "full", np.mean([acc(p) for p in clean]), flush=True)

    for name, (a, b) in PAIRS.items():
        result["pairs"][name] = {
            "one_token": float(np.mean([pair_errors(truth, p.argmax(1), a, b) for p in one])),
            "three_phases": float(np.mean([pair_errors(truth, p.argmax(1), a, b) for p in clean])),
        }
    late = [0.5 * p + 0.5 * alone for p in no_rgb_clean]
    result["paired"]["kap_vs_mate"] = [mcnemar_exact(p.argmax(1) == truth, alone.argmax(1) == truth) for p in clean]
    result["paired"]["kap_vs_late"] = [mcnemar_exact(p.argmax(1) == truth, late[i % len(late)].argmax(1) == truth) for i, p in enumerate(clean)]
    result["paired"]["three_vs_one"] = [mcnemar_exact(clean[i].argmax(1) == truth, one[i].argmax(1) == truth) for i in range(min(len(clean), len(one)))]
    print(json.dumps({k: result[k] for k in ("pairs", "paired")}, indent=1), flush=True)
    (args.runs / "mate_role.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
