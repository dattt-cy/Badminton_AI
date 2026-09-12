"""Build validation-only 3D reference ranges from MultiSense expert strokes.

The input is one or more CSV files produced by
``export_multisense_correction_features.py``.  Only expert strokes with a
valid phase layout, an accepted contact label, and sufficient phase confidence
are included by default.  The resulting ranges stay in MultiSense 3D space;
they must not be compared directly with raw 2D video measurements.
"""

from __future__ import annotations

import argparse
import csv
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable

import numpy as np
import yaml


IDENTITY_COLUMNS = ("subject", "recording", "stroke_number", "technique")
REQUIRED_COLUMNS = {
    *IDENTITY_COLUMNS,
    "skill_level",
    "phase_valid",
    "hitting_sound",
    "contact_confidence",
}


def _is_true(value: object) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes"}


def _source_name(path: Path) -> str:
    match = re.search(r"part\s*([0-9]+)", path.stem, flags=re.IGNORECASE)
    return f"part{match.group(1)}" if match else path.stem


def read_feature_csvs(paths: Iterable[Path]) -> list[dict[str, str]]:
    """Read feature rows and retain a compact source-split label."""
    rows: list[dict[str, str]] = []
    for path in paths:
        with path.open(newline="", encoding="utf-8-sig") as source:
            reader = csv.DictReader(source)
            missing = REQUIRED_COLUMNS.difference(reader.fieldnames or ())
            if missing:
                raise ValueError(f"{path} is missing columns: {sorted(missing)}")
            for row in reader:
                row["__source"] = _source_name(path)
                rows.append(row)
    if not rows:
        raise ValueError("No rows were found in the input feature CSV files")
    return rows


def select_reference_rows(
    rows: Iterable[dict[str, str]],
    *,
    skill_level: str = "Expert",
    accepted_hitting_sounds: tuple[str, ...] = ("Good",),
    min_contact_confidence: float = 0.5,
) -> tuple[list[dict[str, str]], dict[str, int]]:
    """Apply explicit quality gates and remove duplicate stroke records."""
    accepted = {value.casefold() for value in accepted_hitting_sounds}
    selected: list[dict[str, str]] = []
    seen: set[tuple[str, ...]] = set()
    excluded: Counter[str] = Counter()

    for row in rows:
        if row.get("skill_level", "").casefold() != skill_level.casefold():
            excluded["non_expert"] += 1
            continue
        if not _is_true(row.get("phase_valid")):
            excluded["invalid_phase"] += 1
            continue
        if accepted and row.get("hitting_sound", "").casefold() not in accepted:
            excluded["unaccepted_hitting_sound"] += 1
            continue
        try:
            confidence = float(row.get("contact_confidence", "nan"))
        except (TypeError, ValueError):
            confidence = math.nan
        if not math.isfinite(confidence) or confidence < min_contact_confidence:
            excluded["low_contact_confidence"] += 1
            continue
        identity = tuple(row.get(column, "") for column in IDENTITY_COLUMNS)
        if identity in seen:
            excluded["duplicate_stroke"] += 1
            continue
        seen.add(identity)
        selected.append(row)

    return selected, dict(sorted(excluded.items()))


def _feature_unit(name: str) -> str:
    feature = name.split("__", 1)[-1]
    if feature.endswith("angle") or feature == "torso_lean":
        return "degree"
    return "shoulder_width_ratio"


def build_reference_document(
    rows: list[dict[str, str]],
    *,
    source_files: Iterable[Path] = (),
    low_percentile: float = 10.0,
    high_percentile: float = 90.0,
    min_samples: int = 10,
    skill_level: str = "Expert",
    accepted_hitting_sounds: tuple[str, ...] = ("Good",),
    min_contact_confidence: float = 0.5,
) -> dict[str, object]:
    """Create percentile ranges grouped by technique from filtered 3D rows."""
    if not 0 <= low_percentile < high_percentile <= 100:
        raise ValueError("percentiles must satisfy 0 <= low < high <= 100")
    if min_samples < 1:
        raise ValueError("min_samples must be positive")

    selected, excluded = select_reference_rows(
        rows,
        skill_level=skill_level,
        accepted_hitting_sounds=accepted_hitting_sounds,
        min_contact_confidence=min_contact_confidence,
    )
    if not selected:
        raise ValueError("No strokes passed the MultiSense expert quality gates")

    feature_columns = sorted(
        name for name in selected[0] if "__" in name and not name.startswith("__")
    )
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in selected:
        grouped[row["technique"]].append(row)

    techniques: dict[str, object] = {}
    for technique, technique_rows in sorted(grouped.items()):
        features: dict[str, object] = {}
        for name in feature_columns:
            samples = []
            for row in technique_rows:
                try:
                    value = float(row.get(name, "nan"))
                except (TypeError, ValueError):
                    continue
                if math.isfinite(value):
                    samples.append(value)
            if len(samples) < min_samples:
                continue
            values = np.asarray(samples, dtype=np.float64)
            low, median, high = np.percentile(
                values, [low_percentile, 50.0, high_percentile]
            )
            phase, feature = name.split("__", 1)
            features[name] = {
                "phase": phase,
                "feature": feature,
                "unit": _feature_unit(name),
                "low": float(low),
                "median": float(median),
                "high": float(high),
                "mean": float(np.mean(values)),
                "std": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
                "sample_count": len(values),
            }

        sources = Counter(row.get("__source", "unknown") for row in technique_rows)
        subjects = sorted({row["subject"] for row in technique_rows})
        techniques[technique] = {
            "stroke_count": len(technique_rows),
            "subject_count": len(subjects),
            "subjects": subjects,
            "stroke_count_by_source": dict(sorted(sources.items())),
            "features": features,
        }

    return {
        "version": 1,
        "reference_kind": "expert_multisense_3d",
        "reference_status": "provisional",
        "coordinate_space": "multisense_global_3d",
        "usage": "validation_and_2d_to_3d_corrected_features_only",
        "warning": "Do not compare these ranges directly with raw 2D video angles.",
        "source_files": [str(path) for path in source_files],
        "filters": {
            "skill_level": skill_level,
            "phase_valid": True,
            "accepted_hitting_sounds": list(accepted_hitting_sounds),
            "min_contact_confidence": min_contact_confidence,
            "low_percentile": low_percentile,
            "high_percentile": high_percentile,
            "min_samples_per_feature": min_samples,
        },
        "selected_stroke_count": len(selected),
        "excluded_stroke_count_by_reason": excluded,
        "techniques": techniques,
    }


def validate_holdout(
    document: dict[str, object], rows: list[dict[str, str]]
) -> dict[str, object]:
    """Measure how often independently selected expert strokes fit the ranges."""
    filters = document["filters"]
    assert isinstance(filters, dict)
    selected, excluded = select_reference_rows(
        rows,
        skill_level=str(filters["skill_level"]),
        accepted_hitting_sounds=tuple(filters["accepted_hitting_sounds"]),
        min_contact_confidence=float(filters["min_contact_confidence"]),
    )
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in selected:
        grouped[row["technique"]].append(row)

    validation: dict[str, object] = {}
    techniques = document["techniques"]
    assert isinstance(techniques, dict)
    for technique, reference_summary in techniques.items():
        assert isinstance(reference_summary, dict)
        technique_rows = grouped.get(technique, [])
        features: dict[str, object] = {}
        for name, interval in reference_summary["features"].items():
            values = []
            for row in technique_rows:
                try:
                    value = float(row.get(name, "nan"))
                except (TypeError, ValueError):
                    continue
                if math.isfinite(value):
                    values.append(value)
            if not values:
                continue
            low, high = float(interval["low"]), float(interval["high"])
            below = sum(value < low for value in values)
            above = sum(value > high for value in values)
            within = len(values) - below - above
            features[name] = {
                "sample_count": len(values),
                "within_count": within,
                "coverage": within / len(values),
                "below_count": below,
                "above_count": above,
                "median": float(np.median(values)),
            }
        validation[technique] = {
            "stroke_count": len(technique_rows),
            "subject_count": len({row["subject"] for row in technique_rows}),
            "subjects": sorted({row["subject"] for row in technique_rows}),
            "features": features,
        }
    coverage_target = 0.70
    below_target = [
        f"{technique}.{name}"
        for technique, summary in validation.items()
        for name, stats in summary["features"].items()
        if stats["coverage"] < coverage_target
    ]
    return {
        "status": "passed" if not below_target else "failed",
        "coverage_target": coverage_target,
        "features_below_target": below_target,
        "selected_stroke_count": len(selected),
        "excluded_stroke_count_by_reason": excluded,
        "techniques": validation,
    }


def write_flat_csv(document: dict[str, object], path: Path) -> None:
    """Write a review-friendly flat copy of all reference intervals."""
    rows = []
    techniques = document["techniques"]
    assert isinstance(techniques, dict)
    for technique, summary in techniques.items():
        assert isinstance(summary, dict)
        for name, stats in summary["features"].items():
            rows.append({
                "technique": technique,
                "phase_feature": name,
                **stats,
                "subject_count": summary["subject_count"],
                "stroke_count": summary["stroke_count"],
            })
    if not rows:
        raise ValueError("Reference document contains no feature ranges")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as output:
        writer = csv.DictWriter(output, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_csv", type=Path, nargs="+")
    parser.add_argument(
        "--holdout-csv", type=Path, nargs="+",
        help="Independent feature CSV(s) used only to validate range coverage",
    )
    parser.add_argument("--output-yaml", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path)
    parser.add_argument("--skill-level", default="Expert")
    parser.add_argument("--hitting-sound", nargs="+", default=["Good"])
    parser.add_argument("--min-contact-confidence", type=float, default=0.5)
    parser.add_argument("--low-percentile", type=float, default=10.0)
    parser.add_argument("--high-percentile", type=float, default=90.0)
    parser.add_argument("--min-samples", type=int, default=10)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = read_feature_csvs(args.input_csv)
    document = build_reference_document(
        rows,
        source_files=args.input_csv,
        low_percentile=args.low_percentile,
        high_percentile=args.high_percentile,
        min_samples=args.min_samples,
        skill_level=args.skill_level,
        accepted_hitting_sounds=tuple(args.hitting_sound),
        min_contact_confidence=args.min_contact_confidence,
    )
    if args.holdout_csv:
        holdout_rows = read_feature_csvs(args.holdout_csv)
        document["holdout_source_files"] = [str(path) for path in args.holdout_csv]
        document["holdout_validation"] = validate_holdout(document, holdout_rows)
    args.output_yaml.parent.mkdir(parents=True, exist_ok=True)
    args.output_yaml.write_text(
        yaml.safe_dump(document, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    if args.output_csv:
        write_flat_csv(document, args.output_csv)

    print(f"Selected {document['selected_stroke_count']} expert 3D strokes")
    for technique, summary in document["techniques"].items():
        print(
            f"  {technique}: {summary['stroke_count']} strokes, "
            f"{summary['subject_count']} subjects, {len(summary['features'])} features"
        )
    if args.holdout_csv:
        holdout = document["holdout_validation"]
        print(f"Validated on {holdout['selected_stroke_count']} holdout expert strokes")
        for technique, summary in holdout["techniques"].items():
            elbow = summary["features"].get("contact_estimated__elbow_angle")
            coverage = f", contact elbow coverage={elbow['coverage']:.1%}" if elbow else ""
            print(
                f"  {technique}: {summary['stroke_count']} strokes, "
                f"{summary['subject_count']} subjects{coverage}"
            )
    print(f"Wrote 3D reference: {args.output_yaml}")
    if args.output_csv:
        print(f"Wrote flat audit: {args.output_csv}")


if __name__ == "__main__":
    main()
