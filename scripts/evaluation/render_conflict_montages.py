"""Render auto-triaged taxonomy conflicts into labeled montage pages."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("triage", type=Path)
    parser.add_argument("image_dir", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--per-page", type=int, default=12)
    return parser.parse_args()


def main():
    args = parse_args()
    doc = json.loads(args.triage.read_text(encoding="utf-8"))
    items = [row for row in doc["items"] if row["auto_tier"] == "label_or_taxonomy_conflict"]
    images = {path.name.split("_", 1)[1].rsplit(".", 1)[0]: path for path in args.image_dir.glob("*.jpg")}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    pages = []
    for page_start in range(0, len(items), args.per_page):
        tiles = []
        for item in items[page_start:page_start + args.per_page]:
            path = images[item["sample_id"]]
            frame_strip = cv2.imread(str(path))
            frame_strip = cv2.resize(frame_strip, (560, 112), interpolation=cv2.INTER_AREA)
            tile = np.zeros((152, 560, 3), dtype=np.uint8)
            tile[40:] = frame_strip
            line1 = f"{item['sample_id']}  raw={item['raw_label']}  court={item['player_side']}"
            line2 = f"GT {item['stroke_side']} {item['expected_stroke']} -> PRED {item['predicted_side']} {item['predicted_stroke']} {item['stroke_probability']:.2f}"
            cv2.putText(tile, line1, (5, 15), cv2.FONT_HERSHEY_SIMPLEX, .38, (220, 220, 220), 1)
            cv2.putText(tile, line2, (5, 34), cv2.FONT_HERSHEY_SIMPLEX, .42, (80, 220, 255), 1)
            tiles.append(tile)
        blank = np.zeros_like(tiles[0])
        while len(tiles) % 2:
            tiles.append(blank)
        rows = [np.concatenate(tiles[i:i + 2], axis=1) for i in range(0, len(tiles), 2)]
        page = np.concatenate(rows, axis=0)
        page_path = args.output_dir / f"page_{page_start // args.per_page + 1:02d}.jpg"
        cv2.imwrite(str(page_path), page, [cv2.IMWRITE_JPEG_QUALITY, 94])
        pages.append(str(page_path))
    print(json.dumps({"items": len(items), "pages": pages}, indent=2))


if __name__ == "__main__":
    main()
