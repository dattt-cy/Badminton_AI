"""Export per-stroke relative 2D geometry from paired MultiSense pose clips."""

from __future__ import annotations

import argparse
import csv
import math
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from ai_classifier.biomechanics import (
    detect_stroke_phases,
    extract_geometry_features,
    summarize_stroke_geometry,
)
from ai_classifier.error_detection import phase_feature_value
from build_multisense_2d3d_pairs import load_pose


def annotation_index(paths: list[Path]) -> dict[tuple[str, str, int], dict[str, str]]:
    result = {}
    for path in paths:
        with path.open(newline="", encoding="utf-8-sig") as source:
            for row in csv.DictReader(source):
                result[(row["subject"], row["technique"], int(row["stroke_number"]))] = row
    return result


def extract_sample(
    clip: Path, subject: str, technique: str, view: str,
    stroke: int, annotation: dict[str, str],
) -> dict[str, object] | None:
    sequence = load_pose(clip)
    features = extract_geometry_features(sequence)
    phases = detect_stroke_phases(features)
    if not phases.valid:
        return None
    summary = summarize_stroke_geometry(features, phases)
    prep_reach = phase_feature_value(features, phases.preparation, "wrist_shoulder_distance")
    contact_reach = phase_feature_value(
        features, phases.contact_estimated, "wrist_shoulder_distance"
    )
    prep_height = phase_feature_value(features, phases.preparation, "wrist_height")
    contact_height = phase_feature_value(features, phases.contact_estimated, "wrist_height")
    follow_height = phase_feature_value(features, phases.follow_through, "wrist_height")
    follow_elbow = phase_feature_value(features, phases.follow_through, "elbow_angle")
    values = (
        summary.elbow_extension_delta, prep_reach, contact_reach,
        prep_height, contact_height, follow_height, follow_elbow,
    )
    if not all(math.isfinite(value) for value in values):
        return None
    elbow_values = features.column("elbow_angle")
    reach_values = features.column("wrist_shoulder_distance")
    finite_elbow = elbow_values[np.isfinite(elbow_values)]
    finite_reach = reach_values[np.isfinite(reach_values)]
    if len(finite_elbow) < 10 or len(finite_reach) < 10:
        return None
    elbow_p10, elbow_p90 = np.percentile(finite_elbow, [10, 90])
    reach_p10, reach_p90 = np.percentile(finite_reach, [10, 90])
    return {
        "clip": str(clip), "subject": subject, "technique": technique,
        "view": view, "stroke": stroke,
        "skill_level": annotation["skill_level"],
        "hitting_sound": annotation["hitting_sound"],
        "contact_confidence": phases.contact_confidence,
        "elbow_extension_delta": summary.elbow_extension_delta,
        "contact_reach_gain": contact_reach - prep_reach,
        "contact_wrist_height": contact_height,
        "contact_wrist_height_gain": contact_height - prep_height,
        "followthrough_wrist_drop": contact_height - follow_height,
        "followthrough_elbow_angle": follow_elbow,
        "wrist_vertical_excursion": summary.wrist_vertical_excursion,
        "wrist_path_length": summary.wrist_path_length,
        "elbow_angle_excursion": elbow_p90 - elbow_p10,
        "peak_elbow_extension": elbow_p90,
        "reach_excursion": reach_p90 - reach_p10,
        "peak_reach": reach_p90,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_csv", type=Path)
    parser.add_argument("--paired-csv", type=Path)
    parser.add_argument("--annotation-csv", type=Path, nargs="+", required=True)
    parser.add_argument("--pose-root", type=Path)
    args = parser.parse_args()
    annotations = annotation_index(args.annotation_csv)
    if args.pose_root:
        labels: dict[tuple[str, str], Counter[str]] = defaultdict(Counter)
        for (subject, technique, _), annotation in annotations.items():
            labels[(subject, technique)][annotation["skill_level"]] += 1
        unique = {}
        for clip in sorted(args.pose_root.rglob("*.npz")):
            relative = clip.relative_to(args.pose_root)
            if len(relative.parts) < 5:
                continue
            _, technique, view, subject = relative.parts[:4]
            if (subject, technique) not in labels or view not in {"front", "side"}:
                continue
            match = re.search(r"seg(\d+)", clip.stem)
            stroke = int(match.group(1)) if match else len(unique) + 1
            skill_level = labels[(subject, technique)].most_common(1)[0][0]
            unique[(str(clip), subject, technique, view, stroke)] = {
                "skill_level": skill_level, "hitting_sound": "unknown"
            }
    else:
        if args.paired_csv is None:
            parser.error("paired_csv or --pose-root is required")
        with args.paired_csv.open(newline="", encoding="utf-8-sig") as source:
            paired_rows = list(csv.DictReader(source))
        unique = {}
        for row in paired_rows:
            key = (row["clip"], row["subject"], row["technique"], row["view"], int(row["stroke"]))
            unique[key] = annotations.get(
                (row["subject"], row["technique"], int(row["stroke"]))
            )
    rows, skipped = [], 0
    for clip_name, subject, technique, view, stroke in sorted(unique):
        annotation = unique[(clip_name, subject, technique, view, stroke)]
        if annotation is None:
            skipped += 1
            continue
        sample = extract_sample(
            Path(clip_name), subject, technique, view, stroke, annotation
        )
        if sample is None:
            skipped += 1
        else:
            rows.append(sample)
    if not rows:
        raise RuntimeError("No clips produced valid relative 2D geometry")
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.open("w", newline="", encoding="utf-8-sig") as output:
        writer = csv.DictWriter(output, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Exported {len(rows)} relative-feature strokes; skipped {skipped}: {args.output_csv}")


if __name__ == "__main__":
    main()
