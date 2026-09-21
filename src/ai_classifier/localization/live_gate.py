"""Utilities for gating hit events to live-play video segments."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence


@dataclass(frozen=True)
class GateSample:
    frame: int
    live: bool
    score: float


def samples_to_segments(
    samples: Sequence[GateSample], *, total_frames: int, max_gap_frames: int
) -> list[dict[str, object]]:
    """Convert sampled live decisions into merged inclusive frame segments."""
    live = sorted((sample for sample in samples if sample.live), key=lambda item: item.frame)
    if not live:
        return []
    groups: list[list[GateSample]] = [[live[0]]]
    for sample in live[1:]:
        if sample.frame - groups[-1][-1].frame <= max_gap_frames:
            groups[-1].append(sample)
        else:
            groups.append([sample])
    half = max(1, max_gap_frames // 2)
    return [
        {
            "start_frame": max(0, group[0].frame - half),
            "end_frame": min(total_frames - 1, group[-1].frame + half),
            "state": "LIVE",
            "mean_score": sum(item.score for item in group) / len(group),
            "samples": len(group),
        }
        for group in groups
    ]


def frame_is_live(frame: int, segments: Sequence[dict[str, object]]) -> bool:
    return any(
        int(segment["start_frame"]) <= frame <= int(segment["end_frame"])
        and segment.get("state") == "LIVE"
        for segment in segments
    )
