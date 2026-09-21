"""Fine-Badminton virtual clips backed by full-match MP4 files."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class FineBadmintonRecord:
    sample_id: str
    match_id: str
    video_relpath: str
    start_frame: int
    end_frame: int
    player_side: str
    raw_label: str
    canonical_label: str
    coarse_label: str
    split: str
    fps: float
    frame_width: int
    frame_height: int

    @property
    def duration_frames(self) -> int:
        return self.end_frame - self.start_frame + 1


def load_fine_badminton_manifest(
    path: str | Path, *, split: str | None = None, valid_only: bool = True
) -> list[FineBadmintonRecord]:
    """Load records while keeping source videos external to the repository."""
    source = Path(path)
    records: list[FineBadmintonRecord] = []
    with source.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            if split is not None and row["split"] != split:
                continue
            if valid_only and row.get("valid", "True").lower() not in {"true", "1"}:
                continue
            record = FineBadmintonRecord(
                sample_id=row["sample_id"],
                match_id=row["match_id"],
                video_relpath=row["video_relpath"],
                start_frame=int(row["start_frame"]),
                end_frame=int(row["end_frame"]),
                player_side=row["player_side"],
                raw_label=row["raw_label"],
                canonical_label=row["canonical_label"],
                coarse_label=row["coarse_label"],
                split=row["split"],
                fps=float(row["fps"]),
                frame_width=int(row["frame_width"]),
                frame_height=int(row["frame_height"]),
            )
            if record.start_frame < 0 or record.end_frame < record.start_frame:
                raise ValueError(f"Invalid interval for {record.sample_id}")
            records.append(record)
    if not records:
        qualifier = f" split={split!r}" if split else ""
        raise RuntimeError(f"No records found in {source}{qualifier}")
    return records


def sample_frame_indices(
    start_frame: int, end_frame: int, frame_count: int
) -> NDArray[np.int64]:
    """Uniformly sample an inclusive interval, repeating only when necessary."""
    if start_frame < 0 or end_frame < start_frame:
        raise ValueError("Expected 0 <= start_frame <= end_frame")
    if frame_count <= 0:
        raise ValueError("frame_count must be positive")
    return np.rint(
        np.linspace(start_frame, end_frame, num=frame_count, dtype=np.float64)
    ).astype(np.int64)


def decode_virtual_clip(
    record: FineBadmintonRecord,
    dataset_root: str | Path,
    *,
    frame_count: int,
) -> NDArray[np.uint8]:
    """Decode sampled RGB frames with shape ``(T, H, W, 3)``."""
    video_path = Path(dataset_root) / Path(record.video_relpath)
    if not video_path.is_file():
        raise FileNotFoundError(f"Video not found for {record.sample_id}: {video_path}")
    indices = sample_frame_indices(record.start_frame, record.end_frame, frame_count)
    needed = set(int(index) for index in indices)
    decoded: dict[int, NDArray[np.uint8]] = {}

    capture = cv2.VideoCapture(str(video_path))
    try:
        if not capture.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")
        capture.set(cv2.CAP_PROP_POS_FRAMES, record.start_frame)
        for index in range(record.start_frame, record.end_frame + 1):
            ok, frame = capture.read()
            if not ok:
                raise ValueError(
                    f"Could not decode frame {index} for {record.sample_id}"
                )
            if index in needed:
                decoded[index] = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    finally:
        capture.release()

    missing = sorted(needed.difference(decoded))
    if missing:
        raise ValueError(f"Missing sampled frames for {record.sample_id}: {missing}")
    return np.stack([decoded[int(index)] for index in indices]).astype(
        np.uint8, copy=False
    )
