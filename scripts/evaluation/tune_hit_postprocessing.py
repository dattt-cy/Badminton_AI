"""Tune hit-score threshold and temporal NMS without rescanning video.

Each case is ``NAME|SCAN|MANIFEST|MATCH_ID|OFFSET``.  Parameters are selected
on --tune-case inputs, then reported unchanged on --holdout-case inputs.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from ai_classifier.localization import HitEvent, match_hit_events, temporal_nms


@dataclass(frozen=True)
class Case:
    name: str
    raw_events: list[HitEvent]
    truth: list[HitEvent]


def parse_case(spec: str) -> Case:
    name, scan_text, manifest_text, match_id, offset_text = spec.split("|", 4)
    scan_path, manifest_path, offset = Path(scan_text), Path(manifest_text), int(offset_text)
    payload = json.loads(scan_path.read_text(encoding="utf-8"))
    if not payload.get("raw_events"):
        raise ValueError(f"{scan_path} has no raw_events")
    raw = [HitEvent(int(row["frame"]), row["side"], float(row["score"]))
           for row in payload["raw_events"]]
    truth = []
    with manifest_path.open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            frame = int(float(row["hit_frame"]))
            row_match = row.get("match_id", "")
            video_name = Path(row.get("video_path", "")).name
            if (row_match == match_id or video_name.startswith(match_id + " ")) and offset <= frame < offset + 9000:
                side = "upper" if row.get("player_side") == "top" else "lower"
                truth.append(HitEvent(frame - offset, side, 1.0))
    return Case(name, raw, truth)


def aggregate(cases: list[Case], threshold: float, radius: int, tolerance: int) -> dict:
    totals = {key: 0 for key in ("truth", "predicted", "true_positives", "false_positives", "false_negatives")}
    per_case = {}
    for case in cases:
        predictions = temporal_nms(case.raw_events, threshold=threshold, radius=radius)
        metrics = match_hit_events(predictions, case.truth, tolerance=tolerance)
        per_case[case.name] = {key: metrics[key] for key in (*totals, "precision", "recall", "f1")}
        for key in totals:
            totals[key] += metrics[key]
    precision = totals["true_positives"] / totals["predicted"] if totals["predicted"] else 0.0
    recall = totals["true_positives"] / totals["truth"] if totals["truth"] else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {**totals, "precision": precision, "recall": recall, "f1": f1, "per_case": per_case}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tune-case", action="append", required=True)
    parser.add_argument("--holdout-case", action="append", default=[])
    parser.add_argument("--thresholds", default="0.50,0.55,0.60,0.65,0.70,0.75,0.80")
    parser.add_argument("--radii", default="8,12,15,18,21,24,30")
    parser.add_argument("--min-recall", type=float, default=0.97)
    parser.add_argument("--tolerance", type=int, default=12)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    tune = [parse_case(spec) for spec in args.tune_case]
    holdout = [parse_case(spec) for spec in args.holdout_case]
    rows = []
    for threshold in (float(value) for value in args.thresholds.split(",")):
        for radius in (int(value) for value in args.radii.split(",")):
            metrics = aggregate(tune, threshold, radius, args.tolerance)
            rows.append({"threshold": threshold, "radius": radius, **metrics})
    eligible = [row for row in rows if row["recall"] >= args.min_recall]
    choices = eligible or rows
    # Prefer the established 15-frame NMS radius when metrics tie; changing one
    # post-processing dimension at a time makes the production rollout auditable.
    best = max(
        choices,
        key=lambda row: (
            row["f1"], row["recall"], -abs(row["radius"] - 15), row["threshold"]
        ),
    )
    result = {
        "selection": "maximum tuning F1 subject to minimum recall",
        "minimum_recall": args.min_recall,
        "tolerance": args.tolerance,
        "best": best,
        "holdout": aggregate(holdout, best["threshold"], best["radius"], args.tolerance) if holdout else None,
        "grid": rows,
    }
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"best": best, "holdout": result["holdout"]}, indent=2))


if __name__ == "__main__":
    main()
