"""Learn interpretable 2D thresholds on Part2 and validate them on Part3."""

from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
import yaml


FEATURES = {
    "elbow_extension_delta": {"direction": "higher", "minimum_threshold": 10.0},
    "contact_reach_gain": {"direction": "higher", "minimum_threshold": 0.0},
    "contact_wrist_height": {"direction": "higher", "minimum_threshold": -0.20},
    "contact_wrist_height_gain": {"direction": "higher", "minimum_threshold": 0.0},
    "followthrough_wrist_drop": {"direction": "higher", "minimum_threshold": 0.0},
    "wrist_vertical_excursion": {"direction": "higher", "minimum_threshold": 0.30},
    "wrist_path_length": {"direction": "higher", "minimum_threshold": 1.0},
    "elbow_angle_excursion": {"direction": "higher", "minimum_threshold": 20.0},
    "peak_elbow_extension": {"direction": "higher", "minimum_threshold": 120.0},
    "reach_excursion": {"direction": "higher", "minimum_threshold": 0.20},
    "peak_reach": {"direction": "higher", "minimum_threshold": 0.70},
}


def subject_number(subject: str) -> int:
    digits = "".join(character for character in subject if character.isdigit())
    return int(digits) if digits else -1


def _metrics(values: np.ndarray, labels: np.ndarray, threshold: float, direction: str) -> dict:
    predicted_expert = values >= threshold if direction == "higher" else values <= threshold
    expert = labels == 1
    non_expert = labels == 0
    expert_recall = float(np.mean(predicted_expert[expert])) if expert.any() else math.nan
    non_expert_recall = (
        float(np.mean(~predicted_expert[non_expert])) if non_expert.any() else math.nan
    )
    balanced = (expert_recall + non_expert_recall) / 2
    return {
        "expert_recall": expert_recall,
        "non_expert_recall": non_expert_recall,
        "balanced_accuracy": balanced,
    }


def learn_threshold(rows: list[dict[str, str]], feature: str) -> dict | None:
    samples = [
        (float(row[feature]), 1 if row["skill_level"].casefold() == "expert" else 0)
        for row in rows if row["skill_level"].casefold()
        in {"expert", "intermediate", "beginner"}
        and math.isfinite(float(row[feature]))
    ]
    if not samples:
        return None
    values = np.asarray([sample[0] for sample in samples])
    labels = np.asarray([sample[1] for sample in samples])
    if len(np.unique(labels)) < 2:
        return None
    expert_count, non_expert_count = int((labels == 1).sum()), int((labels == 0).sum())
    if expert_count < 10 or non_expert_count < 10:
        return None
    candidates = np.unique(np.percentile(values, np.linspace(5, 95, 91)))
    best = None
    specification = FEATURES[feature]
    direction = specification["direction"]
    for threshold in candidates:
        if threshold < specification["minimum_threshold"]:
            continue
        metrics = _metrics(values, labels, float(threshold), direction)
        score = (
            metrics["balanced_accuracy"], metrics["expert_recall"],
            -abs(float(threshold) - float(np.median(values[labels == 1]))),
        )
        if metrics["expert_recall"] >= 0.70 and (best is None or score > best[0]):
            best = score, float(threshold), direction, metrics
    if best is None:
        return None
    _, threshold, direction, metrics = best
    return {"threshold": threshold, "direction": direction, "train_metrics": metrics}


def build_rules(rows: list[dict[str, str]], *, split_subject: int = 16) -> dict:
    training = [row for row in rows if subject_number(row["subject"]) < split_subject]
    holdout = [row for row in rows if subject_number(row["subject"]) >= split_subject]
    groups: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in training:
        groups[(row["technique"], row["view"])].append(row)
    output, status_counts = {}, defaultdict(int)
    for (technique, view), train_rows in sorted(groups.items()):
        test_rows = [
            row for row in holdout
            if row["technique"] == technique and row["view"] == view
        ]
        rules = {}
        for feature in FEATURES:
            learned = learn_threshold(train_rows, feature)
            if learned is None:
                continue
            test_samples = [
                (float(row[feature]), 1 if row["skill_level"].casefold() == "expert" else 0)
                for row in test_rows if row["skill_level"].casefold()
                in {"expert", "intermediate", "beginner"}
                and math.isfinite(float(row[feature]))
            ]
            test_expert_count = sum(sample[1] == 1 for sample in test_samples)
            test_non_expert_count = sum(sample[1] == 0 for sample in test_samples)
            if test_expert_count < 5 or test_non_expert_count < 10:
                status, test_metrics = "rejected", None
            else:
                values = np.asarray([sample[0] for sample in test_samples])
                labels = np.asarray([sample[1] for sample in test_samples])
                test_metrics = _metrics(
                    values, labels, learned["threshold"], learned["direction"]
                )
                status = (
                    "validated" if learned["train_metrics"]["balanced_accuracy"] >= 0.60
                    and test_metrics["expert_recall"] >= 0.70
                    and test_metrics["balanced_accuracy"] >= 0.60
                    else "review" if learned["train_metrics"]["balanced_accuracy"] >= 0.52
                    and test_metrics["expert_recall"] >= 0.60
                    and test_metrics["balanced_accuracy"] >= 0.52
                    else "rejected"
                )
            status_counts[status] += 1
            rules[feature] = {
                "status": status, **learned,
                "test_metrics": test_metrics,
                "train_sample_count": len(train_rows),
                "test_sample_count": len(test_rows),
                "train_subject_count": len({row["subject"] for row in train_rows}),
                "test_subject_count": len({row["subject"] for row in test_rows}),
                "train_expert_count": sum(
                    row["skill_level"].casefold() == "expert" for row in train_rows
                ),
                "train_non_expert_count": sum(
                    row["skill_level"].casefold() != "expert" for row in train_rows
                ),
                "test_expert_count": test_expert_count,
                "test_non_expert_count": test_non_expert_count,
            }
        output.setdefault(technique, {"views": {}})["views"][view] = {"rules": rules}
    return {
        "version": 1,
        "rule_kind": "relative_2d_skill_level_thresholds",
        "split_rule": f"train subject number < {split_subject}; test >= {split_subject}",
        "status_counts": dict(sorted(status_counts.items())),
        "techniques": output,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_csv", type=Path)
    parser.add_argument("output_yaml", type=Path)
    args = parser.parse_args()
    with args.input_csv.open(newline="", encoding="utf-8-sig") as source:
        rows = list(csv.DictReader(source))
    document = build_rules(rows)
    args.output_yaml.parent.mkdir(parents=True, exist_ok=True)
    args.output_yaml.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")
    print(f"Wrote relative 2D rule registry: {args.output_yaml}")
    print(f"Status counts: {document['status_counts']}")
    for technique, entry in document["techniques"].items():
        for view, view_entry in entry["views"].items():
            rendered = ", ".join(
                f"{name}={rule['status']} ({rule['test_metrics']['balanced_accuracy']:.1%})"
                if rule["test_metrics"] else f"{name}=rejected (no holdout)"
                for name, rule in view_entry["rules"].items()
            )
            print(f"  {technique}/{view}: {rendered}")


if __name__ == "__main__":
    main()
