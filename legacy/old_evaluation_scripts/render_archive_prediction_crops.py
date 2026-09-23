"""Render selected action-window crops from archive prediction JSON files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_dir", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--columns", type=int, default=2)
    return parser.parse_args()


def render_tile(document: dict) -> np.ndarray:
    metadata = document["video_metadata"]
    window = metadata["windows"][int(metadata["selected_window_index"])]
    frame_index = (int(window["start_frame"]) + int(window["end_frame"])) // 2
    capture = cv2.VideoCapture(document["video"])
    try:
        capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ok, frame = capture.read()
    finally:
        capture.release()
    if not ok:
        raise ValueError(f"Cannot decode frame {frame_index}: {document['video']}")

    x1, y1, x2, y2 = (int(value) for value in window["crop_box"])
    preview = frame.copy()
    cv2.rectangle(preview, (x1, y1), (x2, y2), (0, 255, 255), 4)
    crop = cv2.resize(frame[y1:y2, x1:x2], (320, 320), interpolation=cv2.INTER_AREA)
    preview = cv2.resize(preview, (568, 320), interpolation=cv2.INTER_AREA)
    tile = np.concatenate((preview, crop), axis=1)
    label = f"{Path(document['video']).parent.name}/{Path(document['video']).name} -> {document['prediction']['combined_label']}"
    cv2.rectangle(tile, (0, 0), (tile.shape[1], 34), (0, 0, 0), -1)
    cv2.putText(tile, label, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.56, (255, 255, 255), 2)
    return tile


def main() -> None:
    args = parse_args()
    documents = [
        json.loads(path.read_text(encoding="utf-8-sig"))
        for path in sorted(args.input_dir.glob("*.json"))
        if path.name != "summary.json"
    ]
    tiles = [render_tile(document) for document in documents]
    blank = np.zeros_like(tiles[0])
    while len(tiles) % args.columns:
        tiles.append(blank)
    rows = [
        np.concatenate(tiles[index:index + args.columns], axis=1)
        for index in range(0, len(tiles), args.columns)
    ]
    sheet = np.concatenate(rows, axis=0)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(args.output), sheet):
        raise ValueError(f"Cannot write {args.output}")
    print(f"Rendered {len(documents)} samples to {args.output}")


if __name__ == "__main__":
    main()
