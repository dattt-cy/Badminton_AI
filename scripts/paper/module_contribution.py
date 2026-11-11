"""Do CATE phases, cross-attention and hitter disambiguation contribute to Robust KAP?

Compares main with each module removed and with all three removed, 4 seeds per family, on the
3,334 test events, clean and under the masks of degradation_eval.py. Reports seed means, the
difference to main with a 95% t interval, and an exact McNemar test between family ensembles.
Seeds 1-2 of the ablations and main come from --final, the other runs from --extra.

Example:
    python scripts/paper/module_contribution.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from scipy import stats

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO_ROOT / "src"), str(REPO_ROOT / "scripts/training"), str(REPO_ROOT / "scripts/paper")]

import degradation_eval as de  # noqa: E402
from summarize import mcnemar_exact  # noqa: E402
from train_kap_ablation import KAPAblationModel  # noqa: E402

BASE = {"gate": "none", "pooling": "three", "use_cross": True, "use_hitter": True, "use_rgb": True,
        "rgb_dim": 224, "rgb_slots": [1, 2, 3]}
FAMILIES = {
    "main": BASE,
    "no_phase": {**BASE, "pooling": "mean"},
    "no_cross": {**BASE, "use_cross": False},
    "no_hitter": {**BASE, "use_hitter": False},
    "no_all3": {**BASE, "pooling": "mean", "use_cross": False, "use_hitter": False},
    "branch_drop": BASE,
    "no_rgb": {**BASE, "use_rgb": False, "rgb_slots": None},
}
KEYS = [("both", 0.0), ("both", 0.5), ("pose", 0.5), ("burst", 0.5), ("player", 0.5), ("shuttle", 0.5), ("shuttle", 1.0)]


def interval(deltas_a: np.ndarray, deltas_b: np.ndarray) -> tuple[float, float, float]:
    """Mean difference a - b and its 95% Welch t interval over seeds."""
    gap = float(deltas_a.mean() - deltas_b.mean())
    va, vb = deltas_a.var(ddof=1) / len(deltas_a), deltas_b.var(ddof=1) / len(deltas_b)
    freedom = (va + vb) ** 2 / (va ** 2 / (len(deltas_a) - 1) + vb ** 2 / (len(deltas_b) - 1))
    half = float(stats.t.ppf(0.975, freedom) * (va + vb) ** 0.5)
    return gap, gap - half, gap + half


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--final", type=Path, default=REPO_ROOT / "work_dirs/paper_mate/final")
    parser.add_argument("--extra", type=Path, default=REPO_ROOT / "work_dirs/paper_mate/module_seeds")
    parser.add_argument("--features", type=Path, default=REPO_ROOT / "work_dirs/paper_rgb_phases/features_rgb_phases.npz")
    parser.add_argument("--mask-seed", type=int, default=20261101)
    args = parser.parse_args()
    device = torch.device("cuda")
    tensors, _ids, truth = de.kap_inputs(args.features)

    runs = {name: sorted([*args.final.glob(f"{name}_seed*"), *args.extra.glob(f"{name}_seed*")], key=lambda p: p.name) for name in FAMILIES}
    runs = {name: found for name, found in runs.items() if found}
    scores = {name: {} for name in runs}
    ensemble = {}
    validation = {}
    for name in runs:
        kwargs = FAMILIES[name]
        probabilities ={f"{c}@{r:.1f}": [] for c, r in KEYS}
        val = []
        for run in runs[name]:
            model = KAPAblationModel(**kwargs)
            model.load_state_dict(torch.load(run / "kap_best.pth", map_location="cpu", weights_only=False)["model"])
            model = model.to(device).eval()
            for condition, rate in KEYS:
                degraded = de.degrade_kap(tensors, condition, rate, de.mask_rng(args.mask_seed, condition, rate, "kap"))
                with torch.inference_mode():
                    probabilities[f"{condition}@{rate:.1f}"].append(de.kap_predict(model, degraded, device))
            source = json.loads((run / "metrics.json").read_text(encoding="utf-8"))
            if "val_stroke" not in source:  # Collected runs keep their validation scores at the source path.
                source = json.loads((Path(source["source"]) / "metrics.json").read_text(encoding="utf-8"))
            val.append(source["val_stroke"]["accuracy"] * 100)
        validation[name] = val
        for key, values in probabilities.items():
            scores[name][key] = {
                "acc": np.array([(p.argmax(1) == truth).mean() * 100 for p in values]),
                "f1": np.array([de.scores(truth, p.argmax(1))["macro_f1"] * 100 for p in values]),
            }
        ensemble[name] = np.mean(probabilities["both@0.0"], axis=0)
        print(name, "seeds", [r.name for r in runs[name]], "val", np.round(val, 2).tolist(), flush=True)

    result = {"families": {}}
    for name in runs:
        reference = scores["main"]
        row = {"seeds": len(runs[name]), "val_acc": float(np.mean(validation[name])), "conditions": {}}
        for key in scores[name]:
            acc, f1 = scores[name][key]["acc"], scores[name][key]["f1"]
            gap, low, high = interval(acc, reference[key]["acc"]) if name != "main" else (0.0, 0.0, 0.0)
            row["conditions"][key] = {"acc": float(acc.mean()), "acc_std": float(acc.std(ddof=1)), "f1": float(f1.mean()), "gap": gap, "low": low, "high": high,
                                      "f1_gap": float(f1.mean() - reference[key]["f1"].mean()), "per_seed": np.round(acc, 2).tolist()}
        row["ensemble_acc"] = float((ensemble[name].argmax(1) == truth).mean() * 100)
        if name != "main":
            row["ensemble_vs_main"] = mcnemar_exact(ensemble[name].argmax(1) == truth, ensemble["main"].argmax(1) == truth)
        result["families"][name] = row
        print(name, json.dumps({k: row[k] for k in ("seeds", "val_acc", "ensemble_acc")}), row.get("ensemble_vs_main"))
        for key, value in row["conditions"].items():
            print(f"  {key:12s} acc {value['acc']:.2f}+-{value['acc_std']:.2f} gap {value['gap']:+.2f} [{value['low']:+.2f},{value['high']:+.2f}] f1 {value['f1']:.2f} ({value['f1_gap']:+.2f})", flush=True)
    (args.extra / "module_contribution.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
