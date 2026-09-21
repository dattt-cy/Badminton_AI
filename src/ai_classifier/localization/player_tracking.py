"""Short-window pose tracking for selecting the active badminton player."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class PoseDetection:
    frame: int
    box: np.ndarray
    center: np.ndarray
    wrists: np.ndarray


@dataclass
class PoseTrack:
    detections: list[PoseDetection] = field(default_factory=list)

    @property
    def last(self) -> PoseDetection:
        return self.detections[-1]


def box_iou(first: np.ndarray, second: np.ndarray) -> float:
    x1, y1 = np.maximum(first[:2], second[:2])
    x2, y2 = np.minimum(first[2:], second[2:])
    intersection = max(0.0, float(x2 - x1)) * max(0.0, float(y2 - y1))
    first_area = max(0.0, float(first[2] - first[0])) * max(0.0, float(first[3] - first[1]))
    second_area = max(0.0, float(second[2] - second[0])) * max(0.0, float(second[3] - second[1]))
    union = first_area + second_area - intersection
    return intersection / union if union > 0 else 0.0


def build_tracks(
    detections_by_frame: list[list[PoseDetection]], *, max_center_distance: float
) -> list[PoseTrack]:
    tracks: list[PoseTrack] = []
    for detections in detections_by_frame:
        available = set(range(len(tracks)))
        for detection in detections:
            choices = []
            for index in available:
                previous = tracks[index].last
                distance = float(np.linalg.norm(detection.center - previous.center))
                if distance <= max_center_distance or box_iou(detection.box, previous.box) > 0.05:
                    choices.append((distance, -box_iou(detection.box, previous.box), index))
            if choices:
                _, _, index = min(choices)
                tracks[index].detections.append(detection)
                available.remove(index)
            else:
                tracks.append(PoseTrack([detection]))
    return tracks


def track_motion(track: PoseTrack) -> float:
    if len(track.detections) < 2:
        return 0.0
    points = []
    for detection in track.detections:
        valid = detection.wrists[np.isfinite(detection.wrists).all(axis=1)]
        points.append(valid.mean(axis=0) if len(valid) else detection.center)
    return float(sum(np.linalg.norm(current - previous) for previous, current in zip(points, points[1:])))


def select_active_track(
    tracks: list[PoseTrack], *, player_side: str, width: int, height: int
) -> PoseTrack | None:
    candidates = []
    for track in tracks:
        if len(track.detections) < 2:
            continue
        centers = np.stack([item.center for item in track.detections])
        mean_x, mean_y = centers.mean(axis=0)
        if not 0.12 * width <= mean_x <= 0.88 * width:
            continue
        if player_side == "top" and mean_y >= 0.58 * height:
            continue
        if player_side == "bottom" and mean_y < 0.42 * height:
            continue
        expected_y = 0.40 * height if player_side == "top" else 0.68 * height
        court_penalty = abs(mean_x - 0.50 * width) + 0.35 * abs(mean_y - expected_y)
        motion = track_motion(track)
        coverage = len(track.detections)
        score = motion + 12.0 * coverage - 0.12 * court_penalty
        candidates.append((score, track))
    return max(candidates, key=lambda item: item[0])[1] if candidates else None


def union_box(track: PoseTrack, *, width: int, height: int, padding: float) -> tuple[int, int, int, int]:
    boxes = np.stack([item.box for item in track.detections])
    x1, y1 = boxes[:, :2].min(axis=0)
    x2, y2 = boxes[:, 2:].max(axis=0)
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    side = max(x2 - x1, y2 - y1) * (1.0 + 2.0 * padding)
    side = max(side, min(width, height) * 0.22)
    return (
        max(0, int(round(cx - side / 2))),
        max(0, int(round(cy - side * 0.58))),
        min(width, int(round(cx + side / 2))),
        min(height, int(round(cy + side * 0.42))),
    )
