"""Render a small diagnostic contact sheet for automatic hitter crops."""

from __future__ import annotations

import argparse
import csv
import math
import random
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest", type=Path, default=Path("data/manifests/shuttleset_rgb.csv")
    )
    parser.add_argument(
        "--model", type=Path, default=Path("models/checkpoints/pose/yolov8n-pose.pt")
    )
    parser.add_argument(
        "--output", type=Path, default=Path("outputs/shuttleset_crop_pilot/contact_sheet.jpg")
    )
    parser.add_argument("--per-group", type=int, default=2)
    parser.add_argument("--page-size", type=int, default=20)
    parser.add_argument("--columns", type=int, default=4)
    parser.add_argument("--padding", type=float, default=0.30)
    parser.add_argument("--image-size", type=int, default=960)
    parser.add_argument("--seed", type=int, default=20260917)
    return parser.parse_args()


def read_frame(path: str, frame_index: int) -> np.ndarray:
    capture = cv2.VideoCapture(path)
    try:
        capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ok, frame = capture.read()
    finally:
        capture.release()
    if not ok:
        raise ValueError(f"Cannot decode frame {frame_index}: {path}")
    return frame


def select_hitter_box(result, player_side: str, width: int, height: int) -> np.ndarray | None:
    if result.boxes is None or result.keypoints is None:
        return None
    boxes = result.boxes.xyxy.detach().cpu().numpy()
    classes = result.boxes.cls.detach().cpu().numpy()
    keypoints = result.keypoints.data.detach().cpu().numpy()
    candidates = []
    for box, class_id, pose in zip(boxes, classes, keypoints):
        if int(class_id) != 0:
            continue
        visible = pose[:, 2] >= 0.25
        center = pose[visible, :2].mean(axis=0) if visible.any() else np.array([(box[0] + box[2]) / 2, (box[1] + box[3]) / 2])
        x, y = center
        # Exclude spectators, line judges and coaches around the court.
        min_x, max_x = (0.32, 0.68) if player_side == "top" else (0.22, 0.78)
        min_y = 0.25 if player_side == "top" else 0.42
        if not (min_x * width <= x <= max_x * width and min_y * height <= y <= 0.92 * height):
            continue
        if player_side == "top" and y >= 0.58 * height:
            continue
        candidates.append((float(y), box))
    if not candidates:
        return None
    candidates.sort(key=lambda item: item[0])
    return candidates[0][1] if player_side == "top" else candidates[-1][1]


def padded_square(box: np.ndarray, width: int, height: int, padding: float) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = (float(value) for value in box)
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    side = max(x2 - x1, y2 - y1) * (1.0 + 2.0 * padding)
    # Add extra headroom/arm space for racket motions.
    side = max(side, min(width, height) * 0.22)
    return (
        max(0, int(round(cx - side / 2))),
        max(0, int(round(cy - side * 0.58))),
        min(width, int(round(cx + side / 2))),
        min(height, int(round(cy + side * 0.42))),
    )


def tile(frame: np.ndarray, crop_box: tuple[int, int, int, int] | None, label: str) -> np.ndarray:
    preview = frame.copy()
    if crop_box is None:
        crop = np.zeros((320, 320, 3), dtype=np.uint8)
        cv2.putText(crop, "NO HITTER", (45, 170), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
    else:
        x1, y1, x2, y2 = crop_box
        cv2.rectangle(preview, (x1, y1), (x2, y2), (0, 255, 255), 4)
        crop = frame[y1:y2, x1:x2]
        crop = cv2.resize(crop, (320, 320), interpolation=cv2.INTER_AREA)
    preview = cv2.resize(preview, (568, 320), interpolation=cv2.INTER_AREA)
    combined = np.concatenate((preview, crop), axis=1)
    cv2.rectangle(combined, (0, 0), (combined.shape[1], 34), (0, 0, 0), -1)
    cv2.putText(combined, label, (8, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.68, (255, 255, 255), 2)
    return combined


def sample_diverse(
    candidates: list[dict[str, str]], count: int, rng: random.Random
) -> list[dict[str, str]]:
    """Round-robin matches so a group is not dominated by one source video."""
    by_video: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in candidates:
        by_video[row["video_path"]].append(row)
    buckets = list(by_video.values())
    rng.shuffle(buckets)
    for bucket in buckets:
        rng.shuffle(bucket)
    selected = []
    while buckets and len(selected) < count:
        next_buckets = []
        for bucket in buckets:
            if len(selected) >= count:
                break
            selected.append(bucket.pop())
            if bucket:
                next_buckets.append(bucket)
        buckets = next_buckets
    return selected


def write_pages(rendered: list[np.ndarray], output: Path, page_size: int, columns: int) -> list[Path]:
    paths = []
    page_count = math.ceil(len(rendered) / page_size)
    for page_index in range(page_count):
        items = rendered[page_index * page_size : (page_index + 1) * page_size]
        blank = np.zeros_like(items[0])
        while len(items) % columns:
            items.append(blank)
        rows = [np.concatenate(items[i : i + columns], axis=1) for i in range(0, len(items), columns)]
        sheet = np.concatenate(rows, axis=0)
        page_path = output.with_name(f"{output.stem}_{page_index + 1:02d}{output.suffix}")
        if not cv2.imwrite(str(page_path), sheet):
            raise ValueError(f"Cannot write {page_path}")
        paths.append(page_path)
    return paths


def main() -> None:
    args = parse_args()
    with args.manifest.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    grouped: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        if row["split"] == "train":
            grouped[(row["player_side"], row["stroke_side"])].append(row)
    rng = random.Random(args.seed)
    selected = []
    for player_side in ("top", "bottom"):
        for stroke_side in ("forehand", "backhand", "aroundhead"):
            candidates = grouped[(player_side, stroke_side)]
            selected.extend(sample_diverse(candidates, min(args.per_group, len(candidates)), rng))

    frames = [read_frame(row["video_path"], int(row["hit_frame"])) for row in selected]
    model = YOLO(str(args.model))
    results = model.predict(frames, imgsz=args.image_size, conf=0.20, verbose=False)
    rendered = []
    report_rows = []
    missing = 0
    for row, frame, result in zip(selected, frames, results):
        height, width = frame.shape[:2]
        box = select_hitter_box(result, row["player_side"], width, height)
        crop_box = padded_square(box, width, height, args.padding) if box is not None else None
        missing += crop_box is None
        report_rows.append(
            {
                "sample_id": row["sample_id"],
                "video_path": row["video_path"],
                "hit_frame": row["hit_frame"],
                "player_side": row["player_side"],
                "stroke_side": row["stroke_side"],
                "coarse_label": row["coarse_label"],
                "hitter_detected": int(crop_box is not None),
                "crop_box": "" if crop_box is None else ",".join(map(str, crop_box)),
            }
        )
        rendered.append(
            tile(
                frame,
                crop_box,
                f"{row['sample_id']} | {row['player_side']} | {row['stroke_side']} | {row['coarse_label']}",
            )
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    paths = write_pages(rendered, args.output, args.page_size, args.columns)
    report_path = args.output.with_suffix(".csv")
    with report_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=report_rows[0].keys())
        writer.writeheader()
        writer.writerows(report_rows)
    unique_videos = len({row["video_path"] for row in selected})
    print(
        f"samples={len(selected)} unique_videos={unique_videos} missing_hitter={missing} "
        f"pages={len(paths)} report={report_path}"
    )


if __name__ == "__main__":
    main()
