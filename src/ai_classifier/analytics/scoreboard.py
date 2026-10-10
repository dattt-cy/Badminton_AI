"""Optional scoreboard OCR for obtaining observed score-change evidence.

The ROI preprocessing approach is adapted from HAO-xianmi's
``badminton-analysis-system`` (MIT License, Copyright (c) 2026 HAO-xianmi).
The state machine and JSON schema are specific to this project.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2


Score = tuple[int, int]


def read_score_from_frame(
    frame: Any,
    roi: tuple[int, int, int, int],
    reader: Any,
) -> Score | None:
    """Read two scoreboard numbers from a calibrated ``(y1,y2,x1,x2)`` ROI."""
    y1, y2, x1, x2 = roi
    height, width = frame.shape[:2]
    y1, y2 = max(0, y1), min(height, y2)
    x1, x2 = max(0, x1), min(width, x2)
    if y1 >= y2 or x1 >= x2:
        return None

    crop = frame[y1:y2, x1:x2]
    enlarged = cv2.resize(crop, None, fx=4, fy=4, interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(enlarged, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 160, 255, cv2.THRESH_BINARY)
    results = reader.readtext(
        binary, allowlist="0123456789", detail=1, paragraph=False
    )
    numbers: list[str] = []
    for box, text, _confidence in sorted(results, key=lambda row: min(p[0] for p in row[0])):
        del box
        numbers.extend(re.findall(r"\d+", str(text)))
    if len(numbers) >= 2:
        return int(numbers[0]), int(numbers[1])
    if len(numbers) == 1 and len(numbers[0]) == 2:
        return int(numbers[0][0]), int(numbers[0][1])
    return None


def _plausible_change(previous: Score, current: Score) -> bool:
    delta = current[0] - previous[0], current[1] - previous[1]
    return delta in {(1, 0), (0, 1)}


@dataclass
class ScoreChangeTracker:
    confirmations_required: int = 2
    initial: Score | None = None
    current: Score | None = None
    candidate: Score | None = None
    confirmations: int = 0
    events: list[dict[str, Any]] = field(default_factory=list)

    def observe(self, score: Score | None, frame: int, fps: float) -> None:
        if score is None:
            return
        if self.current is None:
            self.initial = score
            self.current = score
            return
        if score == self.current:
            self.candidate = None
            self.confirmations = 0
            return
        if not _plausible_change(self.current, score):
            return
        if score != self.candidate:
            self.candidate = score
            self.confirmations = 1
            return
        self.confirmations += 1
        if self.confirmations < self.confirmations_required:
            return

        previous = self.current
        winner = "first" if score[0] > previous[0] else "second"
        self.events.append(
            {
                "frame": frame,
                "time_seconds": round(frame / fps, 3) if fps > 0 else None,
                "previous_score": list(previous),
                "score": list(score),
                "point_winner": winner,
            }
        )
        self.current = score
        self.candidate = None
        self.confirmations = 0


def extract_score_events(
    video_path: Path | str,
    roi: tuple[int, int, int, int],
    *,
    sample_interval_frames: int = 30,
    confirmations_required: int = 2,
) -> dict[str, Any]:
    """Scan a video ROI and return confirmed score changes."""
    try:
        import easyocr
        import torch
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise RuntimeError("Scoreboard OCR requires the optional 'ocr' dependency") from exc

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Cannot open video for scoreboard OCR: {video_path}")
    fps = float(capture.get(cv2.CAP_PROP_FPS)) or 30.0
    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    # EasyOCR's Unicode progress bar can fail on Windows cp1258 consoles.
    reader = easyocr.Reader(["en"], gpu=torch.cuda.is_available(), verbose=False)
    tracker = ScoreChangeTracker(confirmations_required=confirmations_required)
    reads = 0
    try:
        for frame_index in range(0, total_frames, max(1, sample_interval_frames)):
            capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            success, frame = capture.read()
            if not success or frame is None:
                continue
            score = read_score_from_frame(frame, roi, reader)
            reads += score is not None
            tracker.observe(score, frame_index, fps)
    finally:
        capture.release()

    return {
        "enabled": True,
        "method": "calibrated_roi_easyocr",
        "roi": {"y1": roi[0], "y2": roi[1], "x1": roi[2], "x2": roi[3]},
        "sample_interval_frames": sample_interval_frames,
        "successful_reads": reads,
        "initial_score": list(tracker.initial) if tracker.initial is not None else None,
        "events": tracker.events,
        "observed_score": list(tracker.current) if tracker.current is not None else None,
        "is_official_score": False,
        "limitations": [
            "OCR evidence depends on a manually calibrated scoreboard ROI.",
            "Player-to-score-column mapping must be confirmed by the frontend or user.",
        ],
    }
