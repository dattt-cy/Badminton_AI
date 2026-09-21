"""Detect contact candidates from a tracked shuttlecock trajectory."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class ShuttleEvent:
    frame: int
    x: float
    y: float
    score: float


def load_tracknet_csv(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    frames = np.asarray([int(row["Frame"]) for row in rows], dtype=np.int64)
    points = np.asarray(
        [
            (float(row["X"]), float(row["Y"]))
            if int(row["Visibility"]) else (np.nan, np.nan)
            for row in rows
        ],
        dtype=np.float32,
    )
    return frames, points


def direction_change_events(
    frames: np.ndarray,
    points: np.ndarray,
    *,
    width: int,
    height: int,
    window: int = 5,
    min_score: float = 0.0015,
    nms_radius: int = 8,
) -> list[ShuttleEvent]:
    """Find strong trajectory reversals using normalized pre/post velocities."""
    if len(frames) != len(points):
        raise ValueError("frames and points must have equal length")
    if window <= 0 or nms_radius < 0 or width <= 0 or height <= 0:
        raise ValueError("invalid trajectory detector configuration")
    if len(points) < 2 * window + 1:
        return []
    scale = np.asarray([width, height], dtype=np.float32)
    normalized = points / scale
    candidates: list[ShuttleEvent] = []
    for index in range(window, len(points) - window):
        triple = normalized[[index - window, index, index + window]]
        if not np.isfinite(triple).all():
            continue
        before = (triple[1] - triple[0]) / window
        after = (triple[2] - triple[1]) / window
        before_speed, after_speed = float(np.linalg.norm(before)), float(np.linalg.norm(after))
        if before_speed <= 1e-8 or after_speed <= 1e-8:
            continue
        cosine = float(np.dot(before, after) / (before_speed * after_speed))
        score = max(0.0, -cosine) * min(before_speed, after_speed)
        if score >= min_score:
            candidates.append(
                ShuttleEvent(
                    int(frames[index]), float(points[index, 0]),
                    float(points[index, 1]), score,
                )
            )
    candidates.sort(key=lambda event: (-event.score, event.frame))
    kept: list[ShuttleEvent] = []
    for event in candidates:
        if all(abs(event.frame - selected.frame) > nms_radius for selected in kept):
            kept.append(event)
    return sorted(kept, key=lambda event: event.frame)
