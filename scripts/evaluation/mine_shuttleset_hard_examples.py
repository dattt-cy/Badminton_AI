"""Mine high-value ShuttleSet classifier errors for focused audit/fine-tuning."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path


CONFUSER_CLASSES = {"drive", "lift", "net_attack", "net_shot", "drop", "clear", "smash"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=Path("data/manifests/shuttleset_rgb.csv"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=500)
    parser.add_argument("--audit-limit", type=int, default=160)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    predictions = json.loads(args.predictions.read_text(encoding="utf-8"))["predictions"]
    with args.manifest.open(encoding="utf-8-sig", newline="") as handle:
        metadata = {row["sample_id"]: row for row in csv.DictReader(handle)}

    candidates = []
    for prediction in predictions:
        if prediction["expected_stroke"] == prediction["predicted_stroke"]:
            continue
        if not {
            prediction["expected_stroke"], prediction["predicted_stroke"]
        }.issubset(CONFUSER_CLASSES):
            continue
        item = metadata[prediction["sample_id"]]
        confidence = float(prediction["stroke_probability"])
        priority = confidence
        priority += 0.20 if item["player_side"] == "top" else 0.0
        priority += 0.20 if item["coarse_label"] == "drive" else 0.0
        priority += 0.10 if item["stroke_side"] == "forehand" else 0.0
        candidates.append({
            "sample_id": item["sample_id"],
            "video_path": item["video_path"],
            "start_frame": int(item["start_frame"]),
            "hit_frame": int(item["hit_frame"]),
            "end_frame": int(item["end_frame"]),
            "player_side": item["player_side"],
            "stroke_side": item["stroke_side"],
            "raw_label": item["raw_label"],
            "expected_stroke": prediction["expected_stroke"],
            "predicted_stroke": prediction["predicted_stroke"],
            "stroke_probability": confidence,
            "predicted_side": prediction["predicted_side"],
            "priority": priority,
        })
    candidates.sort(key=lambda row: (-row["priority"], row["sample_id"]))
    selected = candidates[: args.limit]

    # Balanced audit queue: round-robin over true->predicted confusion pairs.
    buckets = defaultdict(list)
    for row in selected:
        buckets[(row["expected_stroke"], row["predicted_stroke"])].append(row)
    audit = []
    active = sorted(buckets)
    while active and len(audit) < args.audit_limit:
        remaining = []
        for key in active:
            if len(audit) >= args.audit_limit:
                break
            audit.append(buckets[key].pop(0))
            if buckets[key]:
                remaining.append(key)
        active = remaining

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "prediction_samples": len(predictions),
        "eligible_errors": len(candidates),
        "selected": len(selected),
        "audit_queue": len(audit),
        "confusions": dict(Counter(
            f"{row['expected_stroke']}->{row['predicted_stroke']}" for row in candidates
        )),
        "selected_top_player": sum(row["player_side"] == "top" for row in selected),
        "selected_forehand": sum(row["stroke_side"] == "forehand" for row in selected),
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    for name, rows in (("hard_examples.json", selected), ("audit_queue.json", audit)):
        (args.output_dir / name).write_text(
            json.dumps(rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
    fieldnames = list(selected[0]) if selected else []
    if fieldnames:
        with (args.output_dir / "hard_examples.csv").open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(selected)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
