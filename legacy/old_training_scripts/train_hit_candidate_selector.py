"""Train a lightweight HIT-candidate selector from cached ShuttleSet validation scans."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, f1_score, roc_auc_score

from ai_classifier.localization import HitEvent, temporal_nms


FEATURES = [
    "hit_score", "local_contrast", "previous_gap", "next_gap", "side_upper",
    "previous_score", "next_score", "score_mean_30", "score_std_30",
    "candidate_count_30", "opposite_gap", "opposite_score", "peak_balance",
]


def select_f1_threshold(labels: np.ndarray, probabilities: np.ndarray) -> dict[str, float]:
    """Calibrate on out-of-fold probabilities, preferring recall on F1 ties."""
    candidates = np.unique(np.concatenate((np.linspace(0.05, 0.95, 181), probabilities)))
    rows = []
    for threshold in candidates:
        predictions = probabilities >= threshold
        true_positive = int(np.sum((predictions == 1) & (labels == 1)))
        false_positive = int(np.sum((predictions == 1) & (labels == 0)))
        false_negative = int(np.sum((predictions == 0) & (labels == 1)))
        precision = true_positive / max(true_positive + false_positive, 1)
        recall = true_positive / max(true_positive + false_negative, 1)
        f1 = 2.0 * precision * recall / max(precision + recall, 1e-12)
        rows.append((f1, recall, precision, -abs(float(threshold) - 0.5), float(threshold)))
    f1, recall, precision, _distance, threshold = max(rows)
    return {"threshold": threshold, "f1": f1, "precision": precision, "recall": recall}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path("data/manifests/shuttleset_rgb.csv"))
    parser.add_argument("--raw-cache-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "val"), default="val")
    parser.add_argument("--match-ids", nargs="*", help="Optional subset within the selected split.")
    parser.add_argument("--threshold", type=float, default=0.3)
    parser.add_argument("--nms-radius", type=int, default=8)
    parser.add_argument("--tolerance", type=int, default=5)
    parser.add_argument("--model", choices=("logistic", "random_forest"), default="random_forest")
    return parser.parse_args()


def truth_by_match(path: Path, split: str) -> dict[str, list[tuple[int, str]]]:
    output: dict[str, list[tuple[int, str]]] = {}
    with path.open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            if row["split"] != split:
                continue
            side = "upper" if row["player_side"] == "top" else "lower"
            output.setdefault(row["match_id"], []).append((int(row["hit_frame"]), side))
    return output


def candidate_features(candidates: list[HitEvent], index: int) -> list[float]:
    item = candidates[index]
    same = [event for event in candidates if event.side == item.side]
    position = same.index(item)
    previous_gap = item.frame - same[position - 1].frame if position else 120
    next_gap = same[position + 1].frame - item.frame if position + 1 < len(same) else 120
    previous_score = same[position - 1].score if position else 0.0
    next_score = same[position + 1].score if position + 1 < len(same) else 0.0
    local = [event for event in candidates if abs(event.frame - item.frame) <= 30]
    neighbors = [event.score for event in local if event is not item]
    neighbor_score = max(neighbors) if neighbors else 0.0
    opposite = [event for event in candidates if event.side != item.side]
    nearest_opposite = min(opposite, key=lambda event: abs(event.frame - item.frame), default=None)
    opposite_gap = abs(nearest_opposite.frame - item.frame) if nearest_opposite else 120
    opposite_score = nearest_opposite.score if nearest_opposite else 0.0
    return [
        item.score,
        item.score - neighbor_score,
        min(previous_gap, 120) / 120.0,
        min(next_gap, 120) / 120.0,
        1.0 if item.side == "upper" else 0.0,
        previous_score,
        next_score,
        float(np.mean([event.score for event in local])),
        float(np.std([event.score for event in local])),
        min(len(local), 20) / 20.0,
        min(opposite_gap, 120) / 120.0,
        opposite_score,
        item.score - 0.5 * (previous_score + next_score),
    ]


def build_model(kind: str):
    if kind == "logistic":
        return LogisticRegression(class_weight="balanced", max_iter=2000, random_state=20260920)
    return RandomForestClassifier(
        n_estimators=160, max_depth=9, min_samples_leaf=8,
        max_features="sqrt", class_weight="balanced_subsample",
        n_jobs=-1, random_state=20260920,
    )


def serialize_forest(model: RandomForestClassifier) -> list[dict[str, object]]:
    trees = []
    for estimator in model.estimators_:
        tree = estimator.tree_
        values = tree.value[:, 0, :]
        positive = values[:, 1] / np.maximum(values.sum(axis=1), 1e-12)
        trees.append({
            "children_left": tree.children_left.tolist(),
            "children_right": tree.children_right.tolist(),
            "feature": tree.feature.tolist(),
            "threshold": tree.threshold.tolist(),
            "positive_probability": positive.tolist(),
        })
    return trees


def selector_probability(features: list[float], selector: dict[str, object]) -> float:
    if selector.get("model") != "random_forest":
        logit = float(selector["intercept"]) + sum(
            float(weight) * float(value)
            for weight, value in zip(selector["coefficient"], features)
        )
        return 1.0 / (1.0 + math.exp(-max(-50.0, min(50.0, logit))))
    probabilities = []
    for tree in selector["trees"]:
        node = 0
        while int(tree["children_left"][node]) >= 0:
            feature = int(tree["feature"][node])
            node = int(
                tree["children_left"][node]
                if features[feature] <= float(tree["threshold"][node])
                else tree["children_right"][node]
            )
        probabilities.append(float(tree["positive_probability"][node]))
    return float(np.mean(probabilities))


def main() -> None:
    args = parse_args()
    truth = truth_by_match(args.manifest, args.split)
    if args.match_ids:
        requested = set(args.match_ids)
        missing = requested - set(truth)
        if missing:
            raise ValueError(f"Matches not found in split={args.split}: {sorted(missing)}")
        truth = {match_id: truth[match_id] for match_id in requested}
    rows = []
    groups = []
    labels = []
    for match_id in sorted(truth, key=int):
        cache = args.raw_cache_dir / f"match_{match_id}.json"
        if not cache.is_file():
            raise FileNotFoundError(f"Missing raw cache: {cache}")
        raw = [HitEvent(**item) for item in json.loads(cache.read_text(encoding="utf-8"))]
        candidates = temporal_nms(
            raw, threshold=args.threshold, radius=args.nms_radius, side_aware=True
        )
        for index, candidate in enumerate(candidates):
            label = any(
                side == candidate.side and abs(frame - candidate.frame) <= args.tolerance
                for frame, side in truth[match_id]
            )
            rows.append(candidate_features(candidates, index))
            labels.append(int(label))
            groups.append(match_id)
    x = np.asarray(rows, dtype=np.float64)
    y = np.asarray(labels, dtype=np.int64)
    model = build_model(args.model)
    cross_validation = []
    out_of_fold_probabilities = np.zeros(len(y), dtype=np.float64)
    group_array = np.asarray(groups)
    for held_out in sorted(set(groups), key=int):
        train_mask = group_array != held_out
        test_mask = group_array == held_out
        fold = build_model(args.model)
        fold.fit(x[train_mask], y[train_mask])
        fold_probabilities = fold.predict_proba(x[test_mask])[:, 1]
        out_of_fold_probabilities[test_mask] = fold_probabilities
        fold_predictions = fold_probabilities >= 0.5
        cross_validation.append({
            "held_out_match": held_out,
            "candidates": int(test_mask.sum()),
            "positives": int(y[test_mask].sum()),
            "roc_auc": float(roc_auc_score(y[test_mask], fold_probabilities)),
            "f1": float(f1_score(y[test_mask], fold_predictions)),
        })
    model.fit(x, y)
    calibration = select_f1_threshold(y, out_of_fold_probabilities)
    probabilities = model.predict_proba(x)[:, 1]
    predictions = probabilities >= calibration["threshold"]
    result = {
        "model": "random_forest" if args.model == "random_forest" else "logistic_regression",
        "features": FEATURES,
        "threshold": calibration["threshold"],
        "out_of_fold_calibration": calibration,
        "training_candidates": len(y),
        "training_positives": int(y.sum()),
        "matches": sorted(set(groups), key=int),
        "roc_auc_training": float(roc_auc_score(y, probabilities)),
        "leave_one_match_out": cross_validation,
        "leave_one_match_out_mean_auc": float(np.mean([row["roc_auc"] for row in cross_validation])),
        "leave_one_match_out_mean_f1": float(np.mean([row["f1"] for row in cross_validation])),
        "classification_report_training": classification_report(
            y, predictions, output_dict=True, zero_division=0
        ),
        "candidate_generation": {
            "hit_threshold": args.threshold,
            "nms_radius": args.nms_radius,
            "truth_tolerance": args.tolerance,
            "side_aware": True,
        },
    }
    if args.model == "random_forest":
        result["trees"] = serialize_forest(model)
        result["feature_importances"] = model.feature_importances_.tolist()
    else:
        result["coefficient"] = model.coef_[0].tolist()
        result["intercept"] = float(model.intercept_[0])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: result[key] for key in (
        "training_candidates", "training_positives", "roc_auc_training"
    )}, indent=2))


if __name__ == "__main__":
    main()
