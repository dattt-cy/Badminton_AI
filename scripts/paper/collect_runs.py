"""Collect the final paper runs into work_dirs/paper_mate/final/<family>_seed<n>/.

Each run directory gets kap_best.pth, test_predictions.npz, model_kwargs.json and a
metrics.json with test metrics computed from the saved predictions, so the
evaluation scripts can treat every family the same way.

Families:
  main       gate-free KAPCrossTransV2 with MATE per contact phase (the paper's model)
  one_token  the same model with a single attention-pooled MATE vector
  gate       the same model with the sigmoid quality gate of the previous version
  no_rgb, no_noise, no_phase, no_cross, no_hitter   ablations of ``main``

Example:
    python scripts/paper/collect_runs.py
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
for path in (REPO_ROOT / "src", REPO_ROOT / "scripts" / "training"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from train_kap_scratch import confusion_metrics  # noqa: E402

W = REPO_ROOT / "work_dirs"
RUNS = W / "paper_mate/runs"
FINAL = W / "paper_mate/final"
BASE = {"gate": "none", "pooling": "three", "use_cross": True, "use_hitter": True, "use_rgb": True,
        "rgb_dim": 224, "rgb_slots": [1, 2, 3]}
FAMILIES = {
    "main": (BASE, sorted(RUNS.glob("main_seed*"))),
    "one_token": ({**BASE, "rgb_slots": None}, sorted(RUNS.glob("one_token_seed*"))),
    "gate": ({**BASE, "gate": "sigmoid"}, sorted(RUNS.glob("gate_seed*"))),
    # The structured streams are unchanged, so these runs do not depend on the RGB encoder.
    "no_rgb": ({**BASE, "use_rgb": False, "rgb_slots": None}, sorted(W.glob("paper_main/no_rgb_seed*"))),
    "no_noise": (BASE, sorted(RUNS.glob("no_noise_seed*"))),
    "no_phase": ({**BASE, "pooling": "mean"}, sorted(RUNS.glob("no_phase_seed*"))),
    "no_cross": ({**BASE, "use_cross": False}, sorted(RUNS.glob("no_cross_seed*"))),
    "no_hitter": ({**BASE, "use_hitter": False}, sorted(RUNS.glob("no_hitter_seed*"))),
}


def metrics(truth: np.ndarray, prediction: np.ndarray, classes: int) -> dict:
    matrix = torch.zeros(classes, classes, dtype=torch.int64)
    np.add.at(matrix.numpy(), (truth, prediction), 1)
    return confusion_metrics(matrix)


def main() -> None:
    if FINAL.exists():
        shutil.rmtree(FINAL)
    for family, (kwargs, runs) in FAMILIES.items():
        for run in runs:
            if not (run / "test_predictions.npz").exists() or not (run / "kap_best.pth").exists():
                continue
            source = json.loads((run / "metrics.json").read_text(encoding="utf-8"))
            seed = source["seed"]
            out = FINAL / f"{family}_seed{seed}"
            out.mkdir(parents=True)
            shutil.copy2(run / "kap_best.pth", out / "kap_best.pth")
            shutil.copy2(run / "test_predictions.npz", out / "test_predictions.npz")
            (out / "model_kwargs.json").write_text(json.dumps(kwargs) + "\n", encoding="utf-8")
            p = np.load(run / "test_predictions.npz")
            result = {
                "variant": family, "seed": seed, "source": str(run),
                "params": source["params"], "best_epoch": source.get("best_epoch"),
                "test_stroke": metrics(p["stroke_true"], p["stroke_prob"].argmax(1), 8),
                "test_side": metrics(p["side_true"], p["side_prob"].argmax(1), 3),
            }
            (out / "metrics.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
            print(f"{out.name:22s} acc={result['test_stroke']['accuracy'] * 100:.2f} "
                  f"f1={result['test_stroke']['macro_f1'] * 100:.2f} params={result['params']}")


if __name__ == "__main__":
    main()
