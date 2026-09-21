"""Render every HIT event crop from an archive comparison report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path)
    parser.add_argument("--label", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = json.loads(args.report.read_text(encoding="utf-8"))
    tiles = []
    for row in report["results"]:
        if row["label"] != args.label or "error" in row["with_hit"]:
            continue
        video = Path(row["video"])
        capture = cv2.VideoCapture(str(video))
        try:
            for event in row["with_hit"]["event_predictions"]:
                frame_index = int(event["event_frame"])
                capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
                ok, frame = capture.read()
                if not ok:
                    continue
                x1, y1, x2, y2 = (int(value) for value in event["crop_box"])
                preview = frame.copy()
                cv2.rectangle(preview, (x1, y1), (x2, y2), (0, 255, 255), 4)
                crop = frame[y1:y2, x1:x2]
                if crop.size == 0:
                    continue
                crop = cv2.resize(crop, (320, 240), interpolation=cv2.INTER_CUBIC)
                preview = cv2.resize(preview, (427, 240), interpolation=cv2.INTER_AREA)
                tile = np.concatenate((preview, crop), axis=1)
                text = (
                    f"{video.name} f={frame_index} hit={event['hit_score']:.2f} "
                    f"-> {event['predicted_side']} {event['predicted_stroke']} "
                    f"({event['stroke_confidence']:.2f})"
                )
                cv2.rectangle(tile, (0, 0), (tile.shape[1], 30), (0, 0, 0), -1)
                cv2.putText(tile, text, (7, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1)
                tiles.append(tile)
        finally:
            capture.release()
    if not tiles:
        raise ValueError(f"No event crops found for label={args.label}")
    sheet = np.concatenate(tiles, axis=0)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(args.output), sheet):
        raise ValueError(f"Cannot write {args.output}")
    print(f"Rendered {len(tiles)} event crops to {args.output}")


if __name__ == "__main__":
    main()
