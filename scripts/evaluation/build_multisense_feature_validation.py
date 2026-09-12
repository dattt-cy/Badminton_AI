"""Build holdout 2D-vs-3D reliability gates for biomechanics features."""

from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
import yaml


FEATURES = {
    "elbow_angle": {
        "column_2d": "elbow_2d_median", "column_3d": "elbow_3d_same_time_median",
        "unit": "degree", "max_mae": 12.0, "max_p90": 25.0,
    },
    "wrist_shoulder_distance": {
        "column_2d": "wrist_shoulder_distance_2d", "column_3d": "wrist_shoulder_distance_3d",
        "unit": "body_scale_ratio", "max_mae": 0.20, "max_p90": 0.40,
    },
    "wrist_height": {
        "column_2d": "wrist_height_2d", "column_3d": "wrist_height_3d",
        "unit": "body_scale_ratio", "max_mae": 0.25, "max_p90": 0.50,
    },
    "elbow_torso_distance": {
        "column_2d": "elbow_torso_distance_2d", "column_3d": "elbow_torso_distance_3d",
        "unit": "body_scale_ratio", "max_mae": 0.20, "max_p90": 0.40,
    },
    "torso_lean": {
        "column_2d": "torso_lean_2d", "column_3d": "torso_lean_3d",
        "unit": "degree", "max_mae": 8.0, "max_p90": 18.0,
    },
    "stance_width": {
        "column_2d": "stance_width_2d", "column_3d": "stance_width_3d",
        "unit": "body_scale_ratio", "max_mae": 0.20, "max_p90": 0.40,
    },
}


def subject_number(subject: str) -> int:
    digits = "".join(character for character in subject if character.isdigit())
    return int(digits) if digits else -1


def projected_row_torso_scale(row: dict[str, str]) -> float:
    left_shoulder = np.asarray([
        float(row["left_shoulder_yolo_x"]), float(row["left_shoulder_yolo_y"])
    ])
    right_shoulder = np.asarray([
        float(row["right_shoulder_yolo_x"]), float(row["right_shoulder_yolo_y"])
    ])
    left_hip = np.asarray([
        float(row["left_hip_yolo_x"]), float(row["left_hip_yolo_y"])
    ])
    right_hip = np.asarray([
        float(row["right_hip_yolo_x"]), float(row["right_hip_yolo_y"])
    ])
    return float(np.linalg.norm((left_shoulder + right_shoulder - left_hip - right_hip) / 2))


def build_validation(
    rows: list[dict[str, str]], *, holdout_subject: int = 16,
    min_samples: int = 100, min_subjects: int = 3,
    min_correlation: float = 0.40,
    min_alignment_correlation: float = 0.50,
    max_camera_median_error: float = 0.15,
    max_camera_p90_error: float = 0.30,
    min_pose_scale_px: float = 0.0,
) -> dict[str, object]:
    """Audit each technique/view/feature only on held-out subjects."""
    projected_mode = bool(rows) and "elbow_angle_projected_3d_2d" in rows[0]
    holdout = [row for row in rows if subject_number(row["subject"]) >= holdout_subject]
    if projected_mode:
        holdout = [
            row for row in holdout
            if float(row.get("motion_alignment_correlation", "-inf"))
            >= min_alignment_correlation
            and float(row.get("camera_median_error_ratio", "inf"))
            <= max_camera_median_error
            and float(row.get("camera_p90_error_ratio", "inf"))
            <= max_camera_p90_error
            and projected_row_torso_scale(row) >= min_pose_scale_px
        ]
    grouped: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in holdout:
        grouped[(row["technique"], row["view"])].append(row)

    techniques: dict[str, object] = {}
    status_counts: defaultdict[str, int] = defaultdict(int)
    for (technique, view), items in sorted(grouped.items()):
        features = {}
        for feature, specification in FEATURES.items():
            column_2d = (
                f"{feature}_yolo_2d" if projected_mode else specification["column_2d"]
            )
            column_3d = (
                f"{feature}_projected_3d_2d"
                if projected_mode else specification["column_3d"]
            )
            pairs = []
            for row in items:
                try:
                    value_2d = float(row[column_2d])
                    value_3d = float(row[column_3d])
                except (KeyError, TypeError, ValueError):
                    continue
                if math.isfinite(value_2d) and math.isfinite(value_3d):
                    pairs.append((value_2d, value_3d, row["subject"]))
            if not pairs:
                continue
            values_2d = np.asarray([pair[0] for pair in pairs])
            values_3d = np.asarray([pair[1] for pair in pairs])
            errors = np.abs(values_2d - values_3d)
            subjects = sorted({pair[2] for pair in pairs})
            correlation = (
                float(np.corrcoef(values_2d, values_3d)[0, 1])
                if np.std(values_2d) > 1e-9 and np.std(values_3d) > 1e-9
                else 0.0
            )
            mae = float(np.mean(errors))
            p90 = float(np.percentile(errors, 90))
            enough_data = len(pairs) >= min_samples and len(subjects) >= min_subjects
            passes_error = mae <= specification["max_mae"] and p90 <= specification["max_p90"]
            passes_correlation = correlation >= min_correlation
            if enough_data and passes_error and passes_correlation:
                status = "validated"
            elif (
                len(pairs) >= min_samples // 2 and len(subjects) >= 2
                and mae <= specification["max_mae"] * 2
                and p90 <= specification["max_p90"] * 2
                and correlation >= 0.15
            ):
                status = "review"
            else:
                status = "rejected"
            status_counts[status] += 1
            features[feature] = {
                "status": status,
                "unit": specification["unit"],
                "sample_count": len(pairs),
                "subject_count": len(subjects),
                "subjects": subjects,
                "mae": mae,
                "median_absolute_error": float(np.median(errors)),
                "p90_absolute_error": p90,
                "pearson_correlation": correlation,
                "limits": {
                    "max_mae": specification["max_mae"],
                    "max_p90": specification["max_p90"],
                    "min_correlation": min_correlation,
                },
            }
        techniques.setdefault(technique, {"views": {}})["views"][view] = {
            "features": features
        }
    return {
        "version": 1,
        "validation_kind": (
            "yolo_2d_against_camera_projected_multisense_3d_subject_holdout"
            if projected_mode else "multisense_2d_against_3d_subject_holdout"
        ),
        "holdout_rule": f"subject number >= {holdout_subject}",
        "runtime_policy": {
            "validated": "allow_pass_or_deviation",
            "review": "force_review",
            "rejected": "insufficient_data",
            "missing": "insufficient_data",
        },
        "minimum_samples": min_samples,
        "minimum_subjects": min_subjects,
        "projection_quality_gates": (
            {
                "min_alignment_correlation": min_alignment_correlation,
                "max_camera_median_error": max_camera_median_error,
                "max_camera_p90_error": max_camera_p90_error,
                "min_pose_scale_px": min_pose_scale_px,
            }
            if projected_mode else None
        ),
        "status_counts": dict(sorted(status_counts.items())),
        "techniques": techniques,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_csv", type=Path)
    parser.add_argument("output_yaml", type=Path)
    parser.add_argument("--holdout-subject", type=int, default=16)
    parser.add_argument("--min-alignment-correlation", type=float, default=0.50)
    parser.add_argument("--max-camera-median-error", type=float, default=0.15)
    parser.add_argument("--max-camera-p90-error", type=float, default=0.30)
    parser.add_argument(
        "--min-pose-scale-px", type=float, default=0.0,
        help="Optional capture-scale audit gate; disabled by default",
    )
    args = parser.parse_args()
    with args.input_csv.open(newline="", encoding="utf-8-sig") as source:
        rows = list(csv.DictReader(source))
    document = build_validation(
        rows, holdout_subject=args.holdout_subject,
        min_alignment_correlation=args.min_alignment_correlation,
        max_camera_median_error=args.max_camera_median_error,
        max_camera_p90_error=args.max_camera_p90_error,
        min_pose_scale_px=args.min_pose_scale_px,
    )
    args.output_yaml.parent.mkdir(parents=True, exist_ok=True)
    args.output_yaml.write_text(
        yaml.safe_dump(document, sort_keys=False), encoding="utf-8"
    )
    print(f"Wrote feature validation registry: {args.output_yaml}")
    print(f"Status counts: {document['status_counts']}")
    for technique, technique_entry in document["techniques"].items():
        for view, view_entry in technique_entry["views"].items():
            rendered = ", ".join(
                f"{name}={stats['status']} (MAE={stats['mae']:.3f})"
                for name, stats in view_entry["features"].items()
            )
            print(f"  {technique}/{view}: {rendered}")


if __name__ == "__main__":
    main()
