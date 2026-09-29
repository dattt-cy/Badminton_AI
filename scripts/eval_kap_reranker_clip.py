"""Evaluate robust KAP with and without conservative Top-2 reranking."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from ai_classifier.models import KAPScratchModel, STROKE_CLASSES, Top2Reranker  # noqa: E402
from ai_classifier.models.kap_fusion import kap_reranker_features  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--features", type=Path,
        default=Path("work_dirs/match44_5min_gt_aligned_features.pt"),
    )
    parser.add_argument(
        "--kap-checkpoint", type=Path,
        default=Path("work_dirs/kap_robust_seed20260926/kap_best.pth"),
    )
    parser.add_argument(
        "--reranker-checkpoint", type=Path,
        default=Path("work_dirs/kap_top2_reranker_seed20260927/best.pth"),
    )
    parser.add_argument(
        "--output", type=Path,
        default=Path("work_dirs/match44_kap_reranker_results.json"),
    )
    return parser.parse_args()


def metrics(predictions: torch.Tensor, top2: torch.Tensor, targets: torch.Tensor) -> dict:
    confusion = torch.zeros(8, 8, dtype=torch.int64)
    for truth, prediction in zip(targets.cpu(), predictions.cpu()):
        confusion[int(truth), int(prediction)] += 1
    matrix = confusion.float()
    true_positive = matrix.diag()
    recall = true_positive / matrix.sum(1).clamp_min(1)
    precision = true_positive / matrix.sum(0).clamp_min(1)
    f1 = 2 * precision * recall / (precision + recall).clamp_min(1e-12)
    return {
        "samples": len(targets),
        "top1_accuracy": float((predictions == targets).float().mean()),
        "top2_accuracy": float((top2 == targets[:, None]).any(dim=1).float().mean()),
        "balanced_accuracy": float(recall.mean()),
        "macro_f1": float(f1.mean()),
        "per_class_recall": {
            name: float(recall[index]) for index, name in enumerate(STROKE_CLASSES)
        },
        "confusion_matrix": confusion.tolist(),
    }


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    records = torch.load(args.features, map_location="cpu", weights_only=False)
    required = {"emb", "s_j", "s_p", "s_s", "s_c", "s_q", "gt_stroke_idx"}
    if not records or not required.issubset(records[0]):
        raise ValueError(f"Aligned cache has no compatible ground truth: {args.features}")
    inputs = [
        torch.cat([record[key] for record in records]).float().to(device)
        for key in ("emb", "s_j", "s_p", "s_s", "s_c", "s_q")
    ]
    targets = torch.tensor([record["gt_stroke_idx"] for record in records], device=device)

    kap_payload = torch.load(args.kap_checkpoint, map_location=device, weights_only=False)
    kap = KAPScratchModel().to(device)
    kap.load_state_dict(kap_payload["model"])
    kap.eval()
    reranker_payload = torch.load(
        args.reranker_checkpoint, map_location=device, weights_only=False
    )
    reranker = Top2Reranker(reranker_payload["input_dim"]).to(device)
    reranker.load_state_dict(reranker_payload["model"])
    reranker.eval()

    with torch.inference_mode():
        features, logits, top2 = kap_reranker_features(kap, *inputs)
        swap_probability = reranker(features).sigmoid()
    base_predictions = top2[:, 0]
    swap = swap_probability >= float(reranker_payload["threshold"])
    final_predictions = torch.where(swap, top2[:, 1], top2[:, 0])
    result = {
        "features": str(args.features.resolve()),
        "kap_checkpoint": str(args.kap_checkpoint.resolve()),
        "reranker_checkpoint": str(args.reranker_checkpoint.resolve()),
        "swap_threshold": float(reranker_payload["threshold"]),
        "swap_count": int(swap.sum()),
        "kap": metrics(base_predictions, top2, targets),
        "kap_reranked": metrics(final_predictions, top2, targets),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
