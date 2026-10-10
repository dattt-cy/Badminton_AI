"""Evaluate a CoachAI pipeline JSON against an annotated ShuttleSet manifest."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prediction", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--offset", type=int, required=True)
    parser.add_argument("--match-id", help="Optional manifest match_id filter")
    parser.add_argument("--frames", type=int, default=9000)
    parser.add_argument("--tolerance", type=int, default=12)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def load_ground_truth(
    manifest: Path, *, offset: int, frames: int, match_id: str | None = None
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with manifest.open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            if match_id is not None and str(row.get("match_id", "")) != match_id:
                continue
            frame = int(row["hit_frame"]) - offset
            if 0 <= frame < frames:
                rows.append(
                    {
                        "frame": frame,
                        "stroke": row["coarse_label"],
                        "hitter": {"top": "upper", "bottom": "lower"}.get(
                            row["player_side"], row["player_side"]
                        ),
                        "rally": int(row["rally"]),
                    }
                )
    return sorted(rows, key=lambda row: row["frame"])


def evaluate(
    predictions: list[dict[str, Any]],
    ground_truth: list[dict[str, Any]],
    *,
    tolerance: int = 12,
) -> dict[str, Any]:
    used: set[int] = set()
    matches: list[tuple[dict[str, Any], dict[str, Any], int]] = []
    for truth in ground_truth:
        choices = [
            (abs(int(pred["frame"]) - truth["frame"]), index, pred)
            for index, pred in enumerate(predictions)
            if index not in used
            and abs(int(pred["frame"]) - truth["frame"]) <= tolerance
        ]
        if choices:
            distance, index, prediction = min(choices)
            used.add(index)
            matches.append((truth, prediction, distance))

    matched = len(matches)
    top1 = sum(truth["stroke"] == pred["stroke"] for truth, pred, _ in matches)
    top2 = sum(truth["stroke"] in pred.get("top2", []) for truth, pred, _ in matches)
    hitter_correct = sum(
        truth["hitter"] == pred.get("hitter") for truth, pred, _ in matches
    )

    per_class: dict[str, dict[str, int]] = defaultdict(
        lambda: {"total": 0, "matched": 0, "top1": 0}
    )
    for truth in ground_truth:
        per_class[truth["stroke"]]["total"] += 1
    for truth, pred, _ in matches:
        stats = per_class[truth["stroke"]]
        stats["matched"] += 1
        stats["top1"] += int(truth["stroke"] == pred["stroke"])

    boundary_truth: list[bool] = []
    boundary_prediction: list[bool] = []
    for index in range(1, len(matches)):
        boundary_truth.append(matches[index][0]["rally"] != matches[index - 1][0]["rally"])
        boundary_prediction.append(
            matches[index][1].get("rally") != matches[index - 1][1].get("rally")
        )
    boundary_tp = sum(a and b for a, b in zip(boundary_truth, boundary_prediction))
    boundary_fp = sum(not a and b for a, b in zip(boundary_truth, boundary_prediction))
    boundary_fn = sum(a and not b for a, b in zip(boundary_truth, boundary_prediction))

    gt_count = len(ground_truth)
    pred_count = len(predictions)
    return {
        "ground_truth_hits": gt_count,
        "predicted_hits": pred_count,
        "matched_hits": matched,
        "false_positives": pred_count - matched,
        "missed_hits": gt_count - matched,
        "hit_recall": matched / gt_count if gt_count else 0.0,
        "hit_precision": matched / pred_count if pred_count else 0.0,
        "end_to_end_top1": top1 / gt_count if gt_count else 0.0,
        "end_to_end_top2": top2 / gt_count if gt_count else 0.0,
        "conditional_top1": top1 / matched if matched else 0.0,
        "conditional_top2": top2 / matched if matched else 0.0,
        "hitter_accuracy": hitter_correct / matched if matched else 0.0,
        "mean_frame_error": (
            sum(distance for _, _, distance in matches) / matched if matched else None
        ),
        "ground_truth_rallies": len({row["rally"] for row in ground_truth}),
        "predicted_rallies": len({row.get("rally") for row in predictions}),
        "rally_boundary_precision": (
            boundary_tp / (boundary_tp + boundary_fp)
            if boundary_tp + boundary_fp
            else 0.0
        ),
        "rally_boundary_recall": (
            boundary_tp / (boundary_tp + boundary_fn)
            if boundary_tp + boundary_fn
            else 0.0
        ),
        "per_class": dict(sorted(per_class.items())),
    }


def main() -> None:
    args = parse_args()
    payload = json.loads(args.prediction.read_text(encoding="utf-8"))
    ground_truth = load_ground_truth(
        args.manifest,
        offset=args.offset,
        frames=args.frames,
        match_id=args.match_id,
    )
    result = evaluate(payload["predictions"], ground_truth, tolerance=args.tolerance)
    rendered = json.dumps(result, ensure_ascii=False, indent=2)
    print(rendered)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
