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
import numpy as np

from ai_classifier.model_cache import cached


Score = tuple[int, int]


# Score-digit ROI (y1, y2, x1, x2) at 1280x720 of each BWF broadcast scoreboard graphic.
# From the second game on, the scoreboard keeps each finished game's score in a column and
# shows the current one to its right, so each ROI spans three columns and
# :func:`read_score_from_frame` reads the rightmost number of each row.
BWF_SCOREBOARD_LAYOUTS: dict[str, tuple[int, int, int, int]] = {
    # Older World Tour graphics, the score columns right of a wide name plate.
    "bwf_classic": (45, 90, 410, 500),
    # 2025-2026 World Tour graphics: a compact plate in the top-left corner, names at
    # x 50-220 and the score columns from x 225. Also used by HYLO Open 2026.
    "bwf_2025": (32, 94, 218, 330),
    # BWF World Championships 2026: a wider name plate, the score columns from x 258.
    "bwf_wc_2026": (38, 92, 255, 360),
    # Tokyo 2020 Olympics (OBS feed, e.g. NHK): country, then a games-won column at x 237-262
    # and the current game's points at x 265-305.
    "olympic_tokyo_2020": (33, 99, 235, 308),
}

# Distance between two score columns as a share of the ROI width, for the layouts whose
# column positions were measured: the column a score is read from then tells its game.
BWF_SCOREBOARD_COLUMN_SHARE: dict[str, float] = {
    # Columns centred at x 240, 272 and 304 of an ROI from x 218 to 330.
    "bwf_2025": 32 / 112,
    # Columns centred at x 275, 308 and 341 of an ROI from x 255 to 360.
    "bwf_wc_2026": 33 / 105,
    # Games won centred at x 250, points at x 285, of an ROI from x 235 to 308.
    "olympic_tokyo_2020": 35 / 73,
}

# Layouts whose first column holds the games each player has won, not a finished game's
# score: the game number is then their sum plus one.
GAMES_WON_COLUMN_LAYOUTS = {"olympic_tokyo_2020"}

# Layouts tried first when picking: one whose reads reach this share of the best layout's
# wins. The Olympic graphic needs both its columns, so it reads nothing on other videos,
# while the wide World Championships ROI also reads the Olympic points column.
PREFERRED_LAYOUTS = ("olympic_tokyo_2020",)
PREFERRED_LAYOUT_SHARE = 0.6

# A digit box this close to the ROI's left edge (px at 1280x720) is cut off: the ROI sits on
# another layout's columns, not on its own.
LEFT_EDGE_PX = 2


def _scale_roi(roi: tuple[int, int, int, int], height: int, width: int) -> tuple[int, int, int, int]:
    y1, y2, x1, x2 = roi
    return (
        int(round(y1 * height / 720)),
        int(round(y2 * height / 720)),
        int(round(x1 * width / 1280)),
        int(round(x2 * width / 1280)),
    )


def calibrate_bwf_scoreboard_rois(height: int, width: int) -> dict[str, tuple[int, int, int, int]]:
    """Every known layout's score-digit ROI scaled to the frame size; see :func:`pick_scoreboard_layout`."""
    return {name: _scale_roi(roi, height, width) for name, roi in BWF_SCOREBOARD_LAYOUTS.items()}


def calibrate_bwf_scoreboard_roi(height: int, width: int) -> tuple[int, int, int, int]:
    """The classic layout's ROI, for callers that cannot tell the layouts apart."""
    return _scale_roi(BWF_SCOREBOARD_LAYOUTS["bwf_classic"], height, width)


def _ocr_reader() -> Any:
    import easyocr
    import torch

    # EasyOCR's Unicode progress bar can fail on Windows cp1258 consoles.
    gpu = torch.cuda.is_available()
    return cached("easyocr", ("en", gpu), lambda: easyocr.Reader(["en"], gpu=gpu, verbose=False))


def pick_scoreboard_layout(crops_by_layout: dict[str, dict[int, Any]], samples: int = 40) -> str:
    """The layout whose ROI yields a score on the most of ``samples`` evenly spread frames.

    The ROI of a layout the video does not use shows background or ads and reads nothing.
    A tie keeps the first layout, so a video without a scoreboard behaves as before.

    A wide ROI can also read another layout's digits: the 2025 ROI covers the whole Tokyo
    2020 scoreboard, whose points sit where the 2025 graphic puts a third game. So a reading
    only counts when no digit is cut by the ROI's left edge and, for a layout with known
    columns, when its columns run from the first without a gap and the rightmost is at most
    one past the furthest already seen (the second from the start), as games add columns
    one at a time.
    """
    reader = _ocr_reader()
    best_name, best_reads = next(iter(crops_by_layout)), -1
    reads_by_layout: dict[str, int] = {}
    for name, crops in crops_by_layout.items():
        frames = sorted(crops)
        step = max(1, len(frames) // samples)
        column_share = BWF_SCOREBOARD_COLUMN_SHARE.get(name)
        furthest_column = 0
        reads = 0
        for frame_index in frames[::step][:samples]:
            crop = crops[frame_index]
            reading = read_score_and_column(
                crop, (0, crop.shape[0], 0, crop.shape[1]), reader, column_share, strict=True,
                games_won=name in GAMES_WON_COLUMN_LAYOUTS,
            )
            if reading is None:
                continue
            column = reading[1]
            if column is not None and name not in GAMES_WON_COLUMN_LAYOUTS:
                if column > furthest_column + 1:
                    continue
                furthest_column = max(furthest_column, column)
            reads += 1
        reads_by_layout[name] = reads
        if reads > best_reads:
            best_name, best_reads = name, reads
    for name in PREFERRED_LAYOUTS:
        if best_reads > 0 and reads_by_layout.get(name, 0) >= PREFERRED_LAYOUT_SHARE * best_reads:
            return name
    return best_name


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
    reading = read_score_and_column(frame, roi, reader)
    return reading[0] if reading else None


def read_score_and_column(
    frame: Any,
    roi: tuple[int, int, int, int],
    reader: Any,
    column_share: float | None = None,
    strict: bool = False,
    games_won: bool = False,
) -> tuple[Score, int | None] | None:
    """The current-game score and, given the layout's ``column_share``, the 0-based column
    it was read from, which is the game number minus one.

    ``strict`` also rejects a reading whose digits do not sit in this layout's columns (see
    :func:`pick_scoreboard_layout`). With ``games_won`` the first column holds games won,
    so the reading needs both columns and its column is the games played so far."""
    y1, y2, x1, x2 = roi
    height, width = frame.shape[:2]
    y1, y2 = max(0, y1), min(height, y2)
    x1, x2 = max(0, x1), min(width, x2)
    if y1 >= y2 or x1 >= x2:
        return None

    crop = frame[y1:y2, x1:x2]
    reading = _rightmost_per_row(crop, reader)
    if reading is not None:
        score, position, positions = reading
        if games_won and column_share:
            columns = [int(p / column_share) for p, _, _ in positions]
            won = [n for p, _, n in positions if int(p / column_share) == 0]
            if set(columns) != {0, 1} or len(won) != 2 or max(won) > 2:
                return None
            return score, sum(won)
        if strict:
            edge = LEFT_EDGE_PX / crop.shape[1]
            if min(p - w / 2 for p, w, _ in positions) <= edge:
                return None
            if column_share:
                columns = {int(p / column_share) for p, _, _ in positions}
                if columns != set(range(max(columns) + 1)):
                    return None
        return score, int(position / column_share) if column_share else None
    # EasyOCR sometimes misses a lone digit in the wide crop that it finds in the
    # first column alone, which is the whole score during the first game.
    reading = _rightmost_per_row(
        crop[:, : max(1, round(crop.shape[1] * FIRST_COLUMN_SHARE))], reader, single_column=True
    )
    if reading is None or strict:
        return None
    return reading[0], 0 if column_share else None


# Share of the ROI width taken by the first score column (32 of 90 px at 1280x720).
FIRST_COLUMN_SHARE = 32 / 90


# Half the distance between two score columns, as a share of the ROI width: the columns are
# about 32 px apart in an ROI 90 (classic) or 112 (2025) px wide at 1280x720.
CURRENT_COLUMN_TOLERANCE = 0.12

# EasyOCR readings below this are mostly a digit misread as another (an 8 as 0, a 1 as 4).
MIN_DIGIT_CONFIDENCE = 0.4


def _whiteness(crop: Any) -> Any:
    """Brightness times (1 - saturation): white digits on a gold or red score column stand out
    as on a black one, which EasyOCR reads far more reliably (13 instead of 18, 8 instead of 0),
    while black digits on the white plate keep their contrast."""
    if crop.ndim != 3:
        return crop
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV).astype(np.float32)
    return (hsv[..., 2] * (1.0 - hsv[..., 1] / 255.0)).astype(np.uint8)


def _rightmost_per_row(
    crop: Any, reader: Any, single_column: bool = False
) -> tuple[Score, float, list[tuple[float, float, int]]] | None:
    """The score, the centre of its column and the (centre, width, number) of every number read,
    positions as shares of the crop width."""
    enlarged = cv2.resize(_whiteness(crop), None, fx=4, fy=4, interpolation=cv2.INTER_CUBIC)
    results = reader.readtext(
        enlarged, allowlist="0123456789", detail=1, paragraph=False
    )
    middle = enlarged.shape[0] / 2
    # Row 0 is the first (upper) scoreboard row; each holds (left x, number) pairs.
    rows: list[list[tuple[float, int]]] = [[], []]
    widths: dict[float, float] = {}
    for box, text, confidence in results:
        if confidence < MIN_DIGIT_CONFIDENCE:
            continue
        xs = [p[0] for p in box]
        ys = [p[1] for p in box]
        row = rows[0 if (min(ys) + max(ys)) / 2 < middle else 1]
        numbers = re.findall(r"\d+", str(text))
        # A box that spans two columns ("24 17") ends with the current score; each number
        # is placed at the centre of its share of the box.
        for k, num_str in enumerate(numbers):
            if int(num_str) <= 30:
                share = (max(xs) - min(xs)) / len(numbers)
                row.append((min(xs) + (k + 0.5) * share, int(num_str)))
                widths[row[-1][0]] = share
    if rows[0] and rows[1]:
        # The current game is the rightmost column of both rows. A row whose digit there was
        # missed would otherwise give the previous game's score (21 instead of 3), so the
        # frame is skipped instead.
        column = max(x for row in rows for x, _ in row)
        tolerance = enlarged.shape[1] * CURRENT_COLUMN_TOLERANCE
        current = [[item for item in row if item[0] >= column - tolerance] for row in rows]
        if current[0] and current[1]:
            width = enlarged.shape[1]
            positions = [(x / width, widths[x] / width, n) for row in rows for x, n in row]
            return (max(current[0])[1], max(current[1])[1]), column / width, positions
        return None
    if not single_column:
        return None
    # One column read as a single box holds both rows' digits ("14").
    single = [n for row in rows for _, n in row]
    if len(single) == 1 and len(str(single[0])) == 2:
        return (int(str(single[0])[0]), int(str(single[0])[1])), 0.0, []
    return None


# The first reading of a new game, taken a point or two after its 0-0 at most.
NEW_GAME_MAX_POINTS = 2

# Readings in a row from a column right of the current game's before it is the next game. The
# 2025 graphic briefly adds a "Challenges Remaining" column (2 / 1) of the same look for about
# five seconds; a new game's column stays until the game ends.
NEW_COLUMN_CONFIRMATIONS = 12


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

    When the reader also gives the scoreboard column of each reading (the game number
    minus one), the column decides instead: a reading from a finished game's column is
    ignored and a reading from a new column starts that game.
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
    # Final scores of the finished games, still shown in their columns. When the scoreboard
    # fades during a replay only those dark digits stay readable, so reading one of them
    # again means the current column was missed.
    finished_games: list[Score] = field(default_factory=list)
    new_column_count: int = 0

    def _resync(self, score: Score, frame: int, fps: float, column: int | None = None) -> None:
        previous = self.current
        lowered = score[0] < previous[0] or score[1] < previous[1]
        if column is not None:
            misread = lowered and bool(self.events)
        else:
            # A misread digit lowers one player's score (7-8 read as 1-8); a new game lowers both,
            # even when it is first read a few points in (an edited video skips its 0-0).
            both_lowered = score[0] < previous[0] and score[1] < previous[1]
            misread = bool(self.game_points) and lowered and not both_lowered and sum(score) > NEW_GAME_MAX_POINTS
        if misread:
            # Scores never go down within a game, and a new game starts near 0-0, so this is
            # a misread that stayed on screen (a 7 read as 1 for as long as the score shows).
            self.resync_candidate = None
            self.resync_count = 0
            return
        if self.resync_candidate != score:
            self.resync_candidate = score
            self.resync_count = 1
            return
        self.resync_count += 1
        if self.resync_count < self.resync_confirmations:
            return
        if column is None and self.game_points and sum(score) < sum(previous):
            self.finished_games.append(previous)
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

    def _new_column(self, score: Score, column: int, frame: int, fps: float) -> None:
        self.new_column_count += 1
        if self.new_column_count < NEW_COLUMN_CONFIRMATIONS:
            return
        previous = self.current
        self.finished_games.append(previous)
        self.game = column + 1
        self.game_points = 0
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
        self.new_column_count = 0

    def observe(self, score: Score | None, frame: int, fps: float, column: int | None = None) -> None:
        if score is None or score in self.finished_games:
            return
        if column is not None and self.current is not None:
            if column + 1 < self.game:
                return
            if column + 1 > self.game:
                self._new_column(score, column, frame, fps)
                return
            self.new_column_count = 0
        if self.current is None:
            self.initial = score
            self.initial_frame = frame
            self.current = score
            if column is not None:
                # A video can start in a later game.
                self.game = column + 1
            return
        if score == self.current:
            self.candidate = None
            self.confirmations = 0
            self.resync_candidate = None
            self.resync_count = 0
            return
        if not _plausible_change(self.current, score):
            self._resync(score, frame, fps, column)
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
    layout: str | None = None,
) -> dict[str, Any]:
    """Scan a video ROI and return confirmed score changes.

    ``roi_crops`` maps sampled frame indices to ROI crops already taken from a
    sequential decode (see :func:`crop_score_roi`); the video is then not seeked.
    """
    try:
        reader = _ocr_reader()
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise RuntimeError("Scoreboard OCR requires the optional 'ocr' dependency") from exc

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Cannot open video for scoreboard OCR: {video_path}")
    fps = float(capture.get(cv2.CAP_PROP_FPS)) or 30.0
    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    tracker = ScoreChangeTracker(confirmations_required=confirmations_required)
    column_share = BWF_SCOREBOARD_COLUMN_SHARE.get(layout or "")
    games_won = layout in GAMES_WON_COLUMN_LAYOUTS
    reads = 0
    try:
        for frame_index in range(0, total_frames, max(1, sample_interval_frames)):
            if roi_crops is not None:
                crop = roi_crops.get(frame_index)
                if crop is None:
                    continue
                reading = read_score_and_column(
                    crop, (0, crop.shape[0], 0, crop.shape[1]), reader, column_share, games_won=games_won
                )
            else:
                capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
                success, frame = capture.read()
                if not success or frame is None:
                    continue
                reading = read_score_and_column(frame, roi, reader, column_share, games_won=games_won)
            reads += reading is not None
            if reading is not None:
                tracker.observe(reading[0], frame_index, fps, reading[1])
    finally:
        capture.release()

    return {
        "enabled": True,
        "method": "calibrated_roi_easyocr",
        "layout": layout,
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
