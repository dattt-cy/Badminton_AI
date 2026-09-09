"""Audit exported MultiSense 3D correction features."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


NON_FEATURE_COLUMNS = {
    "subject", "start_time", "stop_time", "stroke_number", "technique",
    "skill_level", "landing_horizontal", "landing_vertical", "hitting_location",
    "hitting_sound", "recording", "annotation_duration_seconds",
    "phase_window_start_seconds", "phase_window_end_seconds", "phase_valid",
    "phase_invalid_reason", "contact_candidate_seconds", "contact_confidence",
}


def audit_rows(rows: list[dict[str, str]]) -> dict[str, object]:
    """Summarize phase usability and valid feature distributions by technique."""
    if not rows:
        raise ValueError("Feature CSV contains no rows")
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[row["technique"]].append(row)

    feature_names = [name for name in rows[0] if name not in NON_FEATURE_COLUMNS]
    techniques = {}
    for technique, items in sorted(grouped.items()):
        valid = [row for row in items if row["phase_valid"].lower() == "true"]
        invalid_reasons = Counter(
            row["phase_invalid_reason"] or "unspecified"
            for row in items if row["phase_valid"].lower() != "true"
        )
        distributions = {}
        for name in feature_names:
            values = []
            for row in valid:
                try:
                    value = float(row[name])
                except (KeyError, TypeError, ValueError):
                    continue
                if math.isfinite(value):
                    values.append(value)
            if values:
                p10, median, p90 = np.percentile(values, [10, 50, 90])
                distributions[name] = {
                    "count": len(values), "p10": float(p10),
                    "median": float(median), "p90": float(p90),
                }
        techniques[technique] = {
            "rows": len(items),
            "valid_phase_rows": len(valid),
            "valid_phase_ratio": len(valid) / len(items),
            "skill_levels": dict(sorted(Counter(row["skill_level"] for row in items).items())),
            "invalid_phase_reasons": dict(sorted(invalid_reasons.items())),
            "feature_distributions_valid_phases": distributions,
        }
    return {
        "row_count": len(rows),
        "subjects": len({row["subject"] for row in rows}),
        "warning": (
            "Dataset distributions are not coaching thresholds. Build reference "
            "profiles only from expert-approved strokes."
        ),
        "techniques": techniques,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_csv", type=Path)
    parser.add_argument("output_json", type=Path)
    args = parser.parse_args()
    with args.input_csv.open(newline="", encoding="utf-8-sig") as source:
        report = audit_rows(list(csv.DictReader(source)))
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Audited {report['row_count']} rows across {report['subjects']} subjects")
    for technique, summary in report["techniques"].items():
        print(
            f"  {technique}: {summary['valid_phase_rows']}/{summary['rows']} "
            f"valid phases ({summary['valid_phase_ratio']:.1%})"
        )
    print(f"Wrote audit report to {args.output_json}")


if __name__ == "__main__":
    main()
