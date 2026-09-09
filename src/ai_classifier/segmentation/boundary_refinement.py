"""Refine the coarse start/end boundaries of a Motion Proposal (mục 8).

Motion Proposal adds generous padding (~0.3 s) so that no part of the stroke
is clipped.  This module tightens those boundaries by walking backward and
forward from the wrist-speed peak until a stable low-motion region is found,
giving ``detect_stroke_phases`` a clean single-stroke clip to work on.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
from numpy.typing import NDArray

from ai_classifier.biomechanics import GeometryFeatures
from ai_classifier.pose import PoseSequence

from .motion_proposal import MotionProposal


@dataclass(frozen=True)
class RefinedStroke:
    """Tightly bounded stroke segment within the original video.

    All frame indices are relative to the **original video** (not the clip).
    Use :meth:`clip` to extract the corresponding :class:`PoseSequence`.

    ``technique`` is ``None`` until an action-recognition step fills it in
    (boundary refinement itself is purely geometric and technique-agnostic).
    """

    start_frame: int
    end_frame: int
    peak_frame: int
    proposal: MotionProposal
    technique: str | None = None

    def clip(self, sequence: PoseSequence) -> PoseSequence:
        """Return a :class:`PoseSequence` slice covering this stroke.

        The returned sequence shares the same fps/width/height as *sequence*
        but contains only frames ``[start_frame, end_frame]`` (inclusive).
        """
        keypoints = np.asarray(sequence.keypoints)[self.start_frame : self.end_frame + 1]
        return PoseSequence(
            keypoints,
            sequence.fps,
            sequence.frame_width,
            sequence.frame_height,
        )

    def with_technique(self, technique: str) -> RefinedStroke:
        """Return a copy of this stroke with the technique label set."""
        return replace(self, technique=technique)

    def local_frame(self, video_frame: int) -> int:
        """Convert a video-level frame index to a clip-local index."""
        return video_frame - self.start_frame


def refine_boundary(
    proposal: MotionProposal,
    features: GeometryFeatures,
    *,
    fps: float,
    stable_ratio: float = 0.20,
    stable_window: int = 3,
    fine_padding: int = 3,
) -> RefinedStroke:
    """Tighten a Motion Proposal's boundaries using the wrist-speed profile.

    Parameters
    ----------
    proposal:
        Coarse candidate interval from :func:`find_motion_proposals`.
    features:
        Geometry features computed from the **full video** (not the clip).
    fps:
        Video frame rate (used only for validation; not consumed directly).
    stable_ratio:
        Speed below ``stable_ratio × peak_speed`` is considered "at rest".
        Default 0.20 (20 % of peak).
    stable_window:
        Number of consecutive "at rest" frames required to confirm stability.
        Prevents a single low-speed frame from prematurely terminating the
        search.
    fine_padding:
        Frames added to both sides of the refined boundary.  Smaller than the
        Motion Proposal padding; keeps a few frames of context without
        reintroducing noise.  Clipped to the video extent.

    Returns
    -------
    RefinedStroke
        ``technique`` is ``None``; callers should fill it in after running the
        action-recognition step.
    """
    if fps <= 0:
        raise ValueError("fps must be positive")
    if not 0.0 < stable_ratio < 1.0:
        raise ValueError("stable_ratio must be between 0 and 1 (exclusive)")
    if stable_window < 1:
        raise ValueError("stable_window must be >= 1")
    if fine_padding < 0:
        raise ValueError("fine_padding must be >= 0")

    speed = features.column("wrist_speed")
    total_frames = len(speed)

    start = max(0, proposal.start_frame)
    end = min(total_frames - 1, proposal.end_frame)

    # A merged proposal intentionally contains a low-motion pause between
    # motion bursts. Refining around one peak would stop at that pause and
    # silently discard the other half, undoing the merge.
    if proposal.source_count > 1:
        return RefinedStroke(start, end, proposal.peak_frame, proposal)

    # --- Step 1: find peak within the proposal window -------------------
    window_speed = speed[start : end + 1].copy()
    window_speed = np.where(np.isfinite(window_speed), window_speed, 0.0)
    local_peak = int(np.argmax(window_speed))
    peak_frame = start + local_peak
    peak_speed = float(window_speed[local_peak])

    # If the entire window is zero/NaN we cannot refine — fall back to
    # the proposal boundaries and the proposal's peak frame.
    if peak_speed <= 0:
        refined_start = max(0, start - fine_padding)
        refined_end = min(total_frames - 1, end + fine_padding)
        return RefinedStroke(refined_start, refined_end, proposal.peak_frame, proposal)

    stable_threshold = stable_ratio * peak_speed

    # --- Step 2: walk backward from peak to find refined start ----------
    refined_start = start  # fallback
    consecutive = 0
    for frame in range(peak_frame, start - 1, -1):
        s = speed[frame] if np.isfinite(speed[frame]) else 0.0
        if s < stable_threshold:
            consecutive += 1
            if consecutive >= stable_window:
                # frame is the last of the stable run; mark one frame ahead
                # as the boundary so the stable segment itself is excluded.
                refined_start = min(frame + stable_window - 1, peak_frame)
                break
        else:
            consecutive = 0

    # --- Step 3: walk forward from peak to find refined end -------------
    refined_end = end  # fallback
    consecutive = 0
    for frame in range(peak_frame, end + 1):
        s = speed[frame] if np.isfinite(speed[frame]) else 0.0
        if s < stable_threshold:
            consecutive += 1
            if consecutive >= stable_window:
                refined_end = max(frame - stable_window + 1, peak_frame)
                break
        else:
            consecutive = 0

    # --- Step 4: add fine padding and clip to video extent --------------
    refined_start = max(0, refined_start - fine_padding)
    refined_end = min(total_frames - 1, refined_end + fine_padding)

    return RefinedStroke(refined_start, refined_end, peak_frame, proposal)


def refine_all_proposals(
    proposals: list[MotionProposal],
    features: GeometryFeatures,
    *,
    fps: float,
    stable_ratio: float = 0.20,
    stable_window: int = 3,
    fine_padding: int = 3,
) -> list[RefinedStroke]:
    """Refine every proposal in *proposals*.

    A convenience wrapper around :func:`refine_boundary` that processes a
    list of proposals in one call.  The order of the returned list matches
    the order of *proposals*.
    """
    return [
        refine_boundary(
            proposal,
            features,
            fps=fps,
            stable_ratio=stable_ratio,
            stable_window=stable_window,
            fine_padding=fine_padding,
        )
        for proposal in proposals
    ]
