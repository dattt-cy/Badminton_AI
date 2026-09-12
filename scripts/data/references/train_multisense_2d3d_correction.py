"""Train and holdout-test conservative MultiSense elbow correction models."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

from ai_classifier.biomechanics.correction import (
    CORRECTION_FEATURES,
    FRAME_CORRECTION_FEATURES,
)


TARGET = "elbow_3d_same_time_median"
QUALITY_GATES = {
    "elbow_2d_min": 30.0,
    "elbow_2d_max": 180.0,
    "pose_confidence_min": 0.30,
    "peak_confidence_min": 0.10,
}
VIEW_PROJECTION_GATES = {
    "front": {"projected_ratio_min": 0.45, "projected_ratio_max": 2.50},
    "side": {"projected_ratio_min": 0.03, "projected_ratio_max": 0.30},
}


def subject_number(subject: str) -> int:
    digits = "".join(character for character in subject if character.isdigit())
    if not digits:
        raise ValueError(f"Subject ID has no number: {subject}")
    return int(digits)


def quality_reason(row: dict[str, str]) -> str | None:
    try:
        values = {name: float(row[name]) for name in (*CORRECTION_FEATURES, TARGET)}
    except (KeyError, TypeError, ValueError):
        return "missing_or_non_numeric"
    if not all(math.isfinite(value) for value in values.values()):
        return "non_finite"
    if not QUALITY_GATES["elbow_2d_min"] <= values["elbow_2d_median"] <= QUALITY_GATES["elbow_2d_max"]:
        return "implausible_2d_angle"
    projection_gate = VIEW_PROJECTION_GATES.get(row.get("view", ""))
    if projection_gate is None:
        return "unknown_view"
    if not (
        projection_gate["projected_ratio_min"]
        <= values["projected_ratio_median"]
        <= projection_gate["projected_ratio_max"]
    ):
        return "body_orientation_mismatch"
    if values["pose_confidence_median"] < QUALITY_GATES["pose_confidence_min"]:
        return "low_pose_confidence"
    if row.get("sample_kind", "contact") != "frame":
        try:
            peak_confidence = float(row["peak_confidence"])
        except (KeyError, TypeError, ValueError):
            return "missing_or_non_numeric"
        if not math.isfinite(peak_confidence):
            return "non_finite"
        if peak_confidence < QUALITY_GATES["peak_confidence_min"]:
            return "low_contact_confidence"
    return None


def _matrix(
    rows: list[dict[str, str]], feature_names: tuple[str, ...]
) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(
        [[float(row[name]) for name in feature_names] for row in rows],
        dtype=np.float64,
    )
    y = np.asarray([float(row[TARGET]) for row in rows], dtype=np.float64)
    return x, y


def _basis(z: np.ndarray, quadratic: bool) -> np.ndarray:
    if not quadratic:
        return z
    columns = [z]
    columns.append(z ** 2)
    columns.append(np.column_stack([
        z[:, first] * z[:, second]
        for first in range(z.shape[1])
        for second in range(first + 1, z.shape[1])
    ]))
    return np.column_stack(columns)


def _fit_ridge(
    x: np.ndarray, y: np.ndarray, ridge: float, *, quadratic: bool = False
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    mean = np.mean(x, axis=0)
    scale = np.std(x, axis=0)
    scale[scale < 1e-6] = 1.0
    z = (x - mean) / scale
    expanded = _basis(z, quadratic)
    design = np.column_stack([np.ones(len(expanded)), expanded])
    penalty = np.diag([0.0] + [ridge] * expanded.shape[1])
    fitted = np.linalg.solve(design.T @ design + penalty, design.T @ y)
    return mean, scale, fitted[1:], float(fitted[0])


def _predict(
    x: np.ndarray, mean: np.ndarray, scale: np.ndarray,
    coefficients: np.ndarray, intercept: float, *, quadratic: bool = False,
) -> np.ndarray:
    return np.clip(
        intercept + _basis((x - mean) / scale, quadratic) @ coefficients,
        0.0, 180.0,
    )


def _metrics(observed: np.ndarray, expected: np.ndarray) -> dict[str, float | int]:
    error = np.abs(observed - expected)
    return {
        "sample_count": len(expected),
        "mae": float(np.mean(error)),
        "median_absolute_error": float(np.median(error)),
        "absolute_error_p90": float(np.percentile(error, 90)),
        "within_10_degrees_ratio": float(np.mean(error <= 10.0)),
    }


def train_models(
    rows: list[dict[str, str]], *, split_subject: int = 16,
    ridge: float = 10.0, min_train_samples: int = 8,
    min_validation_samples: int = 8,
    group_by_technique: bool = False,
    quadratic: bool = False,
    max_validation_mae: float = 15.0,
    max_validation_p90: float = 30.0,
    min_within_10_ratio: float = 0.50,
) -> dict[str, object]:
    """Fit by camera view on lower-numbered subjects and validate on the rest."""
    clean = [row for row in rows if quality_reason(row) is None]
    use_frame_features = bool(clean) and all(
        row.get("sample_kind") == "frame"
        and all(name in row for name in FRAME_CORRECTION_FEATURES)
        for row in clean
    )
    feature_names = FRAME_CORRECTION_FEATURES if use_frame_features else CORRECTION_FEATURES
    clean = [
        row for row in clean
        if all(
            name in row and math.isfinite(float(row[name]))
            for name in feature_names
        )
    ]
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in clean:
        key = f"{row['technique']}:{row['view']}" if group_by_technique else row["view"]
        grouped[key].append(row)

    models: dict[str, object] = {}
    for model_key, view_rows in sorted(grouped.items()):
        training = [row for row in view_rows if subject_number(row["subject"]) < split_subject]
        validation = [row for row in view_rows if subject_number(row["subject"]) >= split_subject]
        if len(training) < min_train_samples or len(validation) < min_validation_samples:
            models[model_key] = {
                "status": "rejected",
                "reason": "too_few_split_samples",
                "training_sample_count": len(training),
                "validation_sample_count": len(validation),
            }
            continue
        x_train, y_train = _matrix(training, feature_names)
        x_valid, y_valid = _matrix(validation, feature_names)
        mean, scale, coefficients, intercept = _fit_ridge(
            x_train, y_train, ridge, quadratic=quadratic
        )
        corrected = _predict(
            x_valid, mean, scale, coefficients, intercept, quadratic=quadratic
        )
        raw = x_valid[:, 0]
        prior = np.full(len(y_valid), np.median(y_train))
        corrected_metrics = _metrics(corrected, y_valid)
        raw_metrics = _metrics(raw, y_valid)
        prior_metrics = _metrics(prior, y_valid)
        improves_raw = corrected_metrics["mae"] <= raw_metrics["mae"] * 0.90
        beats_prior = corrected_metrics["mae"] <= prior_metrics["mae"] * 0.95
        improves_tail = corrected_metrics["absolute_error_p90"] < raw_metrics["absolute_error_p90"]
        acceptable_absolute_error = (
            corrected_metrics["mae"] <= max_validation_mae
            and corrected_metrics["absolute_error_p90"] <= max_validation_p90
            and corrected_metrics["within_10_degrees_ratio"] >= min_within_10_ratio
        )
        accepted = improves_raw and beats_prior and improves_tail and acceptable_absolute_error
        models[model_key] = {
            "status": "ready" if accepted else "rejected",
            "reason": None if accepted else "holdout_acceptance_failed",
            "feature_names": list(feature_names),
            "basis": "quadratic" if quadratic else "linear",
            "feature_mean": mean.tolist(),
            "feature_scale": scale.tolist(),
            "coefficients": coefficients.tolist(),
            "intercept": intercept,
            "quality_gates": {
                **QUALITY_GATES,
                **VIEW_PROJECTION_GATES[view_rows[0]["view"]],
            },
            "training_subjects": sorted({row["subject"] for row in training}),
            "validation_subjects": sorted({row["subject"] for row in validation}),
            "training_sample_count": len(training),
            "validation_sample_count": len(validation),
            "validation_metrics": corrected_metrics,
            "raw_2d_metrics": raw_metrics,
            "training_median_prior_metrics": prior_metrics,
            "acceptance": {
                "improves_raw_mae_by_10_percent": improves_raw,
                "beats_training_median_prior_by_5_percent": beats_prior,
                "improves_raw_p90": improves_tail,
                "mae_at_most": max_validation_mae,
                "p90_at_most": max_validation_p90,
                "within_10_degrees_ratio_at_least": min_within_10_ratio,
                "acceptable_absolute_error": acceptable_absolute_error,
            },
        }
    return {
        "version": 1,
        "model_kind": "multisense_elbow_2d_to_3d_ridge",
        "target": TARGET,
        "split_rule": f"train subject number < {split_subject}; validate >= {split_subject}",
        "ridge": ridge,
        "group_by_technique": group_by_technique,
        "basis": "quadratic" if quadratic else "linear",
        "runtime_acceptance_limits": {
            "max_validation_mae": max_validation_mae,
            "max_validation_p90": max_validation_p90,
            "min_within_10_degrees_ratio": min_within_10_ratio,
        },
        "input_row_count": len(rows),
        "quality_gated_row_count": len(clean),
        "models": models,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_csv", type=Path)
    parser.add_argument("output_json", type=Path)
    parser.add_argument("--split-subject", type=int, default=16)
    parser.add_argument("--ridge", type=float, default=10.0)
    parser.add_argument("--group-by-technique", action="store_true")
    parser.add_argument("--quadratic", action="store_true")
    parser.add_argument("--max-validation-mae", type=float, default=15.0)
    parser.add_argument("--max-validation-p90", type=float, default=30.0)
    parser.add_argument("--min-within-10-ratio", type=float, default=0.50)
    args = parser.parse_args()
    with args.input_csv.open(newline="", encoding="utf-8-sig") as source:
        rows = list(csv.DictReader(source))
    document = train_models(
        rows, split_subject=args.split_subject, ridge=args.ridge,
        group_by_technique=args.group_by_technique,
        quadratic=args.quadratic,
        max_validation_mae=args.max_validation_mae,
        max_validation_p90=args.max_validation_p90,
        min_within_10_ratio=args.min_within_10_ratio,
    )
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")
    for model_key, model in document["models"].items():
        if "validation_metrics" not in model:
            print(f"{model_key}: {model['status']} ({model['reason']})")
            continue
        print(
            f"{model_key}: {model['status']} corrected MAE={model['validation_metrics']['mae']:.2f} deg, "
            f"raw={model['raw_2d_metrics']['mae']:.2f} deg, "
            f"prior={model['training_median_prior_metrics']['mae']:.2f} deg"
        )
    print(f"Wrote correction audit model: {args.output_json}")


if __name__ == "__main__":
    main()
