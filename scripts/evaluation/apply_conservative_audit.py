"""Apply conservative taxonomy audit decisions to high-confidence conflicts."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


UNCLEAR_RAW_LABELS = {"小平球", "過渡切球"}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("triage", type=Path)
    parser.add_argument("decisions", type=Path)
    parser.add_argument("clean_output", type=Path)
    return parser.parse_args()


def main():
    args = parse_args()
    doc = json.loads(args.triage.read_text(encoding="utf-8"))
    conflicts = [row for row in doc["items"] if row["auto_tier"] == "label_or_taxonomy_conflict"]
    decisions = {}
    clean = []
    for row in conflicts:
        decision = "unclear" if row["raw_label"] in UNCLEAR_RAW_LABELS else "keep"
        decisions[row["sample_id"]] = decision
        if decision == "keep":
            clean.append(row)
    args.decisions.write_text(json.dumps(decisions, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    args.clean_output.write_text(json.dumps(clean, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"decisions": dict(Counter(decisions.values())), "clean": len(clean)}, indent=2))


if __name__ == "__main__":
    main()
