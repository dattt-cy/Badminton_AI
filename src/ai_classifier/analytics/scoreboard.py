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

from ai_classifier.model_cache import cached


Score = tuple[int, int]


def calibrate_bwf_scoreboard_roi(height: int, width: int) -> tuple[int, int, int, int]:
    """Calculate calibrated BWF broadcast scoreboard score digits ROI (y1, y2, x1, x2).

    From the second game on, the scoreboard keeps each finished game's score in a column
    and shows the current one to its right, so the ROI spans three columns and
    :func:`read_score_from_frame` reads the rightmost number of each row.
    """
    return (
        int(round(45 * height / 720)),
        int(round(90 * height / 720)),
        int(round(410 * width / 1280)),
        int(round(500 * width / 1280)),
    )


def crop_score_roi(frame: Any, roi: tuple[int, int, int, int]) -> Any:
    """The ROI pixels :func:`read_score_from_frame` reads, clamped to the frame the same way."""
    y1, y2, x1, x2 = roi
    height, width = frame.shape[:2]
    y1, y2 = max(0, y1), min(height, y2)
    x1, x2 = max(0, x1), min(width, x2)
    if y1 >= y2 or x1 >= x2:
        return frame[0:0, 0:0].copy()
    return frame[y1:y2, x1:x2].copy()


def read_score_from_frame(
    frame: Any,
    roi: tuple[int, int, int, int],
    reader: Any,
) -> Score | None:
    """Read the current-game score: the rightmost number of each of the ROI's two rows."""
    y1, y2, x1, x2 = roi
    height, width = frame.shape[:2]
    y1, y2 = max(0, y1), min(height, y2)
    x1, x2 = max(0, x1), min(width, x2)
    if y1 >= y2 or x1 >= x2:
        return None

    crop = frame[y1:y2, x1:x2]
    score = _rightmost_per_row(crop, reader)
    if score is None:
        # EasyOCR sometimes misses a lone digit in the wide crop that it finds in the
        # first column alone, which is the whole score during the first game.
        score = _rightmost_per_row(
            crop[:, : max(1, round(crop.shape[1] * FIRST_COLUMN_SHARE))], reader, single_column=True
        )
    return score


# Share of the ROI width taken by the first score column (32 of 90 px at 1280x720).
FIRST_COLUMN_SHARE = 32 / 90


def _rightmost_per_row(crop: Any, reader: Any, single_column: bool = False) -> Score | None:
    enlarged = cv2.resize(crop, None, fx=4, fy=4, interpolation=cv2.INTER_CUBIC)
    results = reader.readtext(
        enlarged, allowlist="0123456789", detail=1, paragraph=False
    )
    middle = enlarged.shape[0] / 2
    # Row 0 is the first (upper) scoreboard row; each holds (left x, number) pairs.
    rows: list[list[tuple[float, int]]] = [[], []]
    for box, text, _confidence in results:
        xs = [p[0] for p in box]
        ys = [p[1] for p in box]
        row = rows[0 if (min(ys) + max(ys)) / 2 < middle else 1]
        numbers = re.findall(r"\d+", str(text))
        # A box that spans two columns ("24 17") ends with the current score.
        for k, num_str in enumerate(numbers):
            if int(num_str) <= 30:
                row.append((min(xs) + k * (max(xs) - min(xs)) / len(numbers), int(num_str)))
    if rows[0] and rows[1]:
        return max(rows[0])[1], max(rows[1])[1]
    if not single_column:
        return None
    # One column read as a single box holds both rows' digits ("14").
    single = [n for row in rows for _, n in row]
    if len(single) == 1 and len(str(single[0])) == 2:
        return int(str(single[0])[0]), int(str(single[0])[1])
    return None


def _plausible_change(previous: Score, current: Score) -> bool:
    delta = current[0] - previous[0], current[1] - previous[1]
    return delta in {(1, 0), (0, 1)}


@dataclass
class ScoreChangeTracker:
    """Follows the scoreboard through one video and records each confirmed point.

    A reading that is not one point more than the current score is normally a misread
    and ignored. The same such reading seen ``resync_confirmations`` times in a row is
    real instead: a misread first reading, points missed while the scoreboard was
    hidden, or the 0-0 of a new game. The tracker then jumps to it without recording a
    point; a jump to a lower total after some points were recorded starts a new game.
    """

    confirmations_required: int = 2
    resync_confirmations: int = 4
    initial: Score | None = None
    initial_frame: int | None = None
    current: Score | None = None
    candidate: Score | None = None
    confirmations: int = 0
    resync_candidate: Score | None = None
    resync_count: int = 0
    game: int = 1
    game_points: int = 0
    events: list[dict[str, Any]] = field(default_factory=list)
    resyncs: list[dict[str, Any]] = field(default_factory=list)

    def _resync(self, score: Score, frame: int, fps: float) -> None:
        if self.resync_candidate != score:
            self.resync_candidate = score
            self.resync_count = 1
            return
        self.resync_count += 1
        if self.resync_count < self.resync_confirmations:
            return
        previous = self.current
        if self.game_points and sum(score) < sum(previous):
            self.game += 1
            self.game_points = 0
        elif not self.events:
            # Nothing was recorded from the first reading, so it was the misread.
            self.initial = score
            self.initial_frame = frame
        self.resyncs.append({
            "frame": frame,
            "time_seconds": round(frame / fps, 3) if fps > 0 else None,
            "previous_score": list(previous),
            "score": list(score),
            "game": self.game,
        })
        self.current = score
        self.candidate = None
        self.confirmations = 0
        self.resync_candidate = None
        self.resync_count = 0

    def observe(self, score: Score | None, frame: int, fps: float) -> None:
        if score is None:
            return
        if self.current is None:
            self.initial = score
            self.initial_frame = frame
            self.current = score
            return
        if score == self.current:
            self.candidate = None
            self.confirmations = 0
            self.resync_candidate = None
            self.resync_count = 0
            return
        if not _plausible_change(self.current, score):
            self._resync(score, frame, fps)
            return
        self.resync_candidate = None
        self.resync_count = 0
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
                "game": self.game,
            }
        )
        self.game_points += 1
        self.current = score
        self.candidate = None
        self.confirmations = 0


def extract_score_events(
    video_path: Path | str,
    roi: tuple[int, int, int, int],
    *,
    sample_interval_frames: int = 25,
    confirmations_required: int = 2,
    roi_crops: dict[int, Any] | None = None,
) -> dict[str, Any]:
    """Scan a video ROI and return confirmed score changes.

    ``roi_crops`` maps sampled frame indices to ROI crops already taken from a
    sequential decode (see :func:`crop_score_roi`); the video is then not seeked.
    """
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
    gpu = torch.cuda.is_available()
    reader = cached("easyocr", ("en", gpu), lambda: easyocr.Reader(["en"], gpu=gpu, verbose=False))
    tracker = ScoreChangeTracker(confirmations_required=confirmations_required)
    reads = 0
    try:
        for frame_index in range(0, total_frames, max(1, sample_interval_frames)):
            if roi_crops is not None:
                crop = roi_crops.get(frame_index)
                if crop is None:
                    continue
                score = read_score_from_frame(crop, (0, crop.shape[0], 0, crop.shape[1]), reader)
            else:
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
        "initial_time_seconds": (
            round(tracker.initial_frame / fps, 3) if tracker.initial_frame is not None and fps > 0 else None
        ),
        "events": tracker.events,
        "resyncs": tracker.resyncs,
        "observed_score": list(tracker.current) if tracker.current is not None else None,
        "is_official_score": False,
        "limitations": [
            "OCR evidence depends on a manually calibrated scoreboard ROI.",
            "Player-to-score-column mapping must be confirmed by the frontend or user.",
        ],
    }
