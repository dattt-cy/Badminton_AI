"""Propose candidate racket-swing intervals from geometric motion signals."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from ai_classifier.biomechanics import GeometryFeatures


@dataclass(frozen=True)
class MotionProposal:
    start_frame: int
    end_frame: int
    peak_frame: int
    peak_score: float
    source_count: int = 1


def motion_score(features: GeometryFeatures, *, smooth_frames: int = 5) -> NDArray[np.float32]:
    """Combine robustly normalized wrist and elbow velocities."""
    if smooth_frames <= 0:
        raise ValueError("smooth_frames must be positive")
    wrist = _robust_normalize(features.column("wrist_speed"))
    elbow = _robust_normalize(features.column("elbow_angular_speed"))
    score = np.nan_to_num(0.7 * wrist + 0.3 * elbow, nan=0.0).astype(np.float32)
    if smooth_frames > 1 and len(score):
        kernel = np.ones(smooth_frames, dtype=np.float32) / smooth_frames
        score = np.convolve(score, kernel, mode="same").astype(np.float32)
    return score


def find_motion_proposals(
    features: GeometryFeatures,
    *,
    fps: float,
    percentile: float = 75.0,
    min_duration_seconds: float = 0.15,
    max_gap_seconds: float = 0.25,
    padding_seconds: float = 0.30,
) -> list[MotionProposal]:
    """Group high-motion frames into padded candidate intervals."""
    if fps <= 0:
        raise ValueError("fps must be positive")
    if not 0 < percentile < 100:
        raise ValueError("percentile must be between 0 and 100")
    score = motion_score(features)
    positive = score[score > 0]
    if not len(positive):
        return []
    threshold = float(np.percentile(positive, percentile))
    active = np.flatnonzero(score >= threshold)
    if not len(active):
        return []

    max_gap = max(1, round(max_gap_seconds * fps))
    min_frames = max(1, round(min_duration_seconds * fps))
    padding = max(0, round(padding_seconds * fps))
    raw_groups: list[tuple[int, int]] = []
    start = previous = int(active[0])
    for frame in active[1:]:
        frame = int(frame)
        if frame - previous > max_gap:
            raw_groups.append((start, previous))
            start = frame
        previous = frame
    raw_groups.append((start, previous))

    proposals: list[MotionProposal] = []
    for start, end in raw_groups:
        if end - start + 1 < min_frames:
            continue
        padded_start = max(0, start - padding)
        padded_end = min(len(score) - 1, end + padding)
        local_peak = int(np.argmax(score[padded_start:padded_end + 1])) + padded_start
        proposals.append(
            MotionProposal(padded_start, padded_end, local_peak, float(score[local_peak]))
        )
    return proposals


def merge_motion_proposals(
    proposals: list[MotionProposal],
    features: GeometryFeatures,
    *,
    fps: float,
    max_pause_seconds: float = 1.0,
    max_stroke_seconds: float = 3.5,
) -> list[MotionProposal]:
    """Merge nearby motion bursts separated by a short tutorial-style pause.

    A badminton stroke can contain a deliberate pause between demonstration
    phases. The higher-scoring peak is preserved and the combined duration is
    bounded so separate rallies are not joined indefinitely.
    """
    if not proposals:
        return []
    max_pause = max(0, round(max_pause_seconds * fps))
    max_length = max(1, round(max_stroke_seconds * fps))
    ordered = sorted(proposals, key=lambda item: item.start_frame)
    merged: list[MotionProposal] = [ordered[0]]
    for current in ordered[1:]:
        previous = merged[-1]
        gap = current.start_frame - previous.end_frame - 1
        combined_length = current.end_frame - previous.start_frame + 1
        if gap <= max_pause and combined_length <= max_length:
            peak = previous if previous.peak_score >= current.peak_score else current
            merged[-1] = MotionProposal(
                previous.start_frame,
                max(previous.end_frame, current.end_frame),
                peak.peak_frame,
                peak.peak_score,
                previous.source_count + current.source_count,
            )
        else:
            merged.append(current)
    return merged


def _robust_normalize(values: NDArray[np.float32]) -> NDArray[np.float32]:
    output = np.zeros(len(values), dtype=np.float32)
    valid = np.isfinite(values)
    if not valid.any():
        return output
    observed = values[valid]
    median = float(np.median(observed))
    mad = float(np.median(np.abs(observed - median)))
    denominator = max(1.4826 * mad, 1e-6)
    output[valid] = np.maximum((observed - median) / denominator, 0)
    return output
