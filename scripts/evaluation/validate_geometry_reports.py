"""Validate geometry JSON reports against a small manually labelled CSV.

CSV columns:
    report,stroke_id,true_swing_peak_frame,rule.<rule_name>,...

Rule labels must be ``pass`` or ``deviation``. Blank labels are ignored.
``review`` and ``insufficient_data`` predictions are reported as abstentions,
so a system cannot appear accurate merely by declining difficult examples.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def validate_manifest(manifest_path: Path) -> dict:
    rows = list(csv.DictReader(manifest_path.open(encoding="utf-8-sig", newline="")))
    if not rows:
        raise ValueError("Validation manifest is empty")

    peak_errors: list[int] = []
    counts = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0, "tn": 0, "abstain": 0})
    resolved_correct = 0
    resolved_total = 0
    labelled_total = 0

    for row in rows:
        report_path = Path(row["report"])
        if not report_path.is_absolute():
            report_path = manifest_path.parent / report_path
        report = json.loads(report_path.read_text(encoding="utf-8"))
        stroke_id = int(row.get("stroke_id") or 1)
        stroke = next(
            (item for item in report.get("strokes", []) if int(item["stroke_id"]) == stroke_id),
            None,
        )
        if stroke is None:
            raise ValueError(f"Stroke {stroke_id} not found in {report_path}")

        truth_peak = row.get("true_swing_peak_frame", "").strip()
        if truth_peak:
            phases = stroke.get("phases", {})
            predicted_peak = stroke.get(
                "estimated_swing_peak_frame",
                phases.get("estimated_swing_peak_frame", phases.get("contact_frame")),
            )
            if predicted_peak is not None:
                peak_errors.append(abs(int(predicted_peak) - int(truth_peak)))

        predicted_rules = {
            item["rule_name"]: item["status"] for item in stroke.get("rules", [])
        }
        for column, expected in row.items():
            if not column.startswith("rule.") or not expected or not expected.strip():
                continue
            rule_name = column.removeprefix("rule.")
            expected = expected.strip().lower()
            if expected not in {"pass", "deviation"}:
                raise ValueError(
                    f"{column} must be pass, deviation, or blank; got {expected!r}"
                )
            predicted = predicted_rules.get(rule_name, "insufficient_data")
            labelled_total += 1
            if predicted in {"review", "insufficient_data"}:
                counts[rule_name]["abstain"] += 1
                if expected == "deviation":
                    counts[rule_name]["fn"] += 1
                else:
                    counts[rule_name]["tn"] += 1
                continue
            resolved_total += 1
            resolved_correct += int(predicted == expected)
            if expected == "deviation" and predicted == "deviation":
                counts[rule_name]["tp"] += 1
            elif expected == "pass" and predicted == "deviation":
                counts[rule_name]["fp"] += 1
            elif expected == "deviation" and predicted == "pass":
                counts[rule_name]["fn"] += 1
            else:
                counts[rule_name]["tn"] += 1

    per_rule = {}
    aggregate = {name: 0 for name in ("tp", "fp", "fn", "tn", "abstain")}
    for rule_name, rule_counts in sorted(counts.items()):
        for name, value in rule_counts.items():
            aggregate[name] += value
        per_rule[rule_name] = {**rule_counts, **_classification_metrics(rule_counts)}

    peak = {
        "labelled_count": len(peak_errors),
        "mae_frames": float(np.mean(peak_errors)) if peak_errors else None,
        "median_error_frames": float(np.median(peak_errors)) if peak_errors else None,
        "within_3_frames_ratio": (
            float(np.mean(np.asarray(peak_errors) <= 3)) if peak_errors else None
        ),
    }
    return {
        "manifest": str(manifest_path),
        "report_count": len(rows),
        "swing_peak": peak,
        "rules": {
            "labelled_count": labelled_total,
            "coverage": resolved_total / labelled_total if labelled_total else None,
            "accuracy_on_resolved": (
                resolved_correct / resolved_total if resolved_total else None
            ),
            "aggregate": {**aggregate, **_classification_metrics(aggregate)},
            "per_rule": per_rule,
        },
    }


def _classification_metrics(counts: dict[str, int]) -> dict[str, float | None]:
    tp, fp, fn = counts["tp"], counts["fp"], counts["fn"]
    precision = tp / (tp + fp) if tp + fp else None
    recall = tp / (tp + fn) if tp + fn else None
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision is not None and recall is not None and precision + recall
        else None
    )
    return {"precision": precision, "recall": recall, "f1": f1}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    result = validate_manifest(args.manifest)
    rendered = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    print(rendered, end="")
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")


if __name__ == "__main__":
    main()
