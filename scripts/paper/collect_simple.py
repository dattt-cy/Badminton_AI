"""Collect the runs of the simplified paper model into work_dirs/paper_mate/simple_final/<family>_seed<n>/.

``main`` is KAP without three-phase pooling, cross-attention and hitter weights, trained with per-branch
failure. Families named here with ``full`` start from the model that still has those three modules.

Example:
    python scripts/paper/collect_simple.py
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO_ROOT / "src"), str(REPO_ROOT / "scripts/training")]

from train_kap_scratch import confusion_metrics  # noqa: E402

W = REPO_ROOT / "work_dirs/paper_mate"
BRANCH = W / "branch_drop"
OUT = W / "simple_final"
SIMPLE = {"gate": "none", "pooling": "mean", "use_cross": False, "use_hitter": False, "rgb_dim": 224, "rgb_slots": [1, 2, 3]}
FULL = {"gate": "none", "pooling": "three", "use_cross": True, "use_hitter": True, "rgb_dim": 224, "rgb_slots": [1, 2, 3]}
FAMILIES = {  # family: (kwargs, source root, source tag)
    "main": (SIMPLE, BRANCH, "no_all3"),
    "no_pose": ({**SIMPLE, "use_pose": False}, BRANCH, "simple_no_pose"),
    "no_shuttle": ({**SIMPLE, "use_shuttle": False}, BRANCH, "simple_no_shuttle"),
    "no_pose_shuttle": ({**SIMPLE, "use_pose": False, "use_shuttle": False}, BRANCH, "simple_no_pose_shuttle"),
    "no_rgb": ({**SIMPLE, "use_rgb": False}, BRANCH, "simple_no_rgb"),
    "no_contact": ({**SIMPLE, "use_contact": False}, BRANCH, "simple_no_contact"),
    "no_noise": (SIMPLE, BRANCH, "simple_no_noise"),
    "old_recipe": (SIMPLE, W / "module_seeds", "no_all3"),
    "full": (FULL, BRANCH, "branch_drop"),
    "full_no_phase": ({**FULL, "pooling": "mean"}, BRANCH, "no_phase"),
    "full_no_cross": ({**FULL, "use_cross": False}, BRANCH, "no_cross"),
    "full_no_hitter": ({**FULL, "use_hitter": False}, BRANCH, "no_hitter"),
    "full_no_rgb": ({**FULL, "use_rgb": False, "rgb_slots": None}, BRANCH, "no_rgb"),
    "full_gate": ({**FULL, "gate": "sigmoid"}, BRANCH, "gate"),
    "full_one_token": ({**FULL, "rgb_slots": None}, BRANCH, "one_token"),
}


def metrics(truth: np.ndarray, prediction: np.ndarray, classes: int) -> dict:
    matrix = torch.zeros(classes, classes, dtype=torch.int64)
    np.add.at(matrix.numpy(), (truth, prediction), 1)
    return confusion_metrics(matrix)


def main() -> None:
    for family, (kwargs, root, tag) in FAMILIES.items():
        runs = sorted(root.glob(f"{tag}_seed*"))
        if not runs:
            raise SystemExit(f"no runs for {family}")
        for run in runs:
            source = json.loads((run / "metrics.json").read_text(encoding="utf-8"))
            out = OUT / f"{family}_seed{source['seed']}"
            out.mkdir(parents=True, exist_ok=True)
            shutil.copy2(run / "kap_best.pth", out / "kap_best.pth")
            shutil.copy2(run / "test_predictions.npz", out / "test_predictions.npz")
            (out / "model_kwargs.json").write_text(json.dumps(kwargs) + "\n", encoding="utf-8")
            p = np.load(run / "test_predictions.npz")
            result = {
                "variant": family, "seed": source["seed"], "source": str(run), "params": source["params"],
                "best_epoch": source.get("best_epoch"), "val_stroke": source.get("val_stroke"),
                "test_stroke": metrics(p["stroke_true"], p["stroke_prob"].argmax(1), 8),
                "test_side": metrics(p["side_true"], p["side_prob"].argmax(1), 3),
            }
            (out / "metrics.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        print(f"{family:16s} seeds={len(runs)} params={source['params']:,} last acc={result['test_stroke']['accuracy'] * 100:.2f}")


if __name__ == "__main__":
    main()
