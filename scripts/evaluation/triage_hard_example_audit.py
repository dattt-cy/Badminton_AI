"""Automatically triage audit samples before human label review."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("queue", type=Path)
    parser.add_argument("output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = json.loads(args.queue.read_text(encoding="utf-8"))
    output = []
    for row in rows:
        confidence = float(row["stroke_probability"])
        if confidence >= 0.90:
            tier = "label_or_taxonomy_conflict"
        elif confidence >= 0.70:
            tier = "high_value_hard_negative"
        else:
            tier = "classifier_uncertain"
        output.append({**row, "auto_tier": tier})
    output.sort(key=lambda row: (
        {"label_or_taxonomy_conflict": 0, "high_value_hard_negative": 1,
         "classifier_uncertain": 2}[row["auto_tier"]],
        -float(row["priority"]),
    ))
    result = {
        "summary": dict(Counter(row["auto_tier"] for row in output)),
        "items": output,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result["summary"], indent=2))


if __name__ == "__main__":
    main()
