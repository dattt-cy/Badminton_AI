"""Add the two context-window pilot runs as extra seeds of the ``full`` variant.

work_dirs/context_window_pilot/kap_w15_30{,_seed2} were trained by
scripts/training/train_kap_scratch.py --model-type cross_v2 on the same feature
cache, split and recipe as train_kap_ablation.py --variant full, so they are
valid additional seeds.  This copies each checkpoint into
work_dirs/paper_ablation/full_seed<seed>/ and writes test predictions and
metrics in the same format; nothing is retrained.

Example:
    python scripts/paper/import_pilot_runs.py
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
for path in (REPO_ROOT / "src", REPO_ROOT / "scripts" / "training", REPO_ROOT / "scripts" / "paper"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from train_kap_ablation import build_model, predict  # noqa: E402
from train_kap_scratch import confusion_metrics, load_data  # noqa: E402

PILOTS = ("kap_w15_30", "kap_w15_30_seed2")


def metrics(truth: np.ndarray, prediction: np.ndarray, classes: int) -> dict:
    matrix = torch.zeros(classes, classes, dtype=torch.int64)
    for t, p in zip(truth, prediction):
        matrix[int(t), int(p)] += 1
    return confusion_metrics(matrix)


def main() -> None:
    pilot_root = REPO_ROOT / "work_dirs/context_window_pilot"
    features = pilot_root / "features_w15_30.npz"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tensors, indices = load_data(features)
    raw = np.load(features)
    test_ids = np.flatnonzero(raw["split"] == "test")

    for name in PILOTS:
        source = json.loads((pilot_root / name / "metrics.json").read_text(encoding="utf-8"))
        if source["model_type"] != "cross_v2" or Path(source["features"]).name != features.name:
            raise RuntimeError(f"{name} was not trained with the paper recipe")
        output_dir = REPO_ROOT / "work_dirs/paper_ablation" / f"full_seed{source['seed']}"
        output_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(pilot_root / name / "kap_best.pth", output_dir / "kap_best.pth")

        model = build_model("full")
        payload = torch.load(output_dir / "kap_best.pth", map_location="cpu", weights_only=False)
        model.load_state_dict(payload["model"])
        model.to(device)
        stroke_prob, side_prob = predict(model, tensors, test_ids, device)
        stroke_true = tensors["stroke"][test_ids].numpy()
        side_true = tensors["side"][test_ids].numpy()
        np.savez_compressed(
            output_dir / "test_predictions.npz", sample_id=raw["sample_id"][test_ids],
            stroke_prob=stroke_prob, side_prob=side_prob,
            stroke_true=stroke_true, side_true=side_true,
        )
        result = {
            "variant": "full", "seed": source["seed"],
            "params": sum(p.numel() for p in model.parameters()),
            "features": str(features), "splits": {k: len(v) for k, v in indices.items()},
            "test_samples": len(test_ids), "best_epoch": int(payload["epoch"]),
            "imported_from": str(pilot_root / name),
            "history": [
                {k: v for k, v in row.items() if k in ("epoch", "train_loss", "selection_score")}
                for row in source["kap_history"]
            ],
            "test_stroke": metrics(stroke_true, stroke_prob.argmax(1), stroke_prob.shape[1]),
            "test_side": metrics(side_true, side_prob.argmax(1), side_prob.shape[1]),
        }
        (output_dir / "metrics.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        print(f"{name} -> {output_dir.name}: acc={result['test_stroke']['accuracy']:.4f} "
              f"f1={result['test_stroke']['macro_f1']:.4f} side={result['test_side']['accuracy']:.4f}")


if __name__ == "__main__":
    main()
