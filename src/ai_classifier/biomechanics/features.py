"""Confidence-aware geometric features derived from COCO-17 poses."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from ai_classifier.pose import PoseSequence


# COCO-17 joint indices.
LEFT_SHOULDER, RIGHT_SHOULDER = 5, 6
LEFT_ELBOW, RIGHT_ELBOW = 7, 8
LEFT_WRIST, RIGHT_WRIST = 9, 10
LEFT_HIP, RIGHT_HIP = 11, 12
LEFT_KNEE, RIGHT_KNEE = 13, 14
LEFT_ANKLE, RIGHT_ANKLE = 15, 16


@dataclass(frozen=True)
class GeometryFeatures:
    """Per-frame features. Invalid measurements are represented by NaN."""

    names: tuple[str, ...]
    values: NDArray[np.float32]
    confidence: NDArray[np.float32]
    timestamps: NDArray[np.float32]

    def column(self, name: str) -> NDArray[np.float32]:
        return self.values[:, self.names.index(name)]


@dataclass(frozen=True)
class GeometryQuality:
    valid_ratio: float
    scale_px: float
    scale_cv: float
    speed_spike_ratio: float
    phases_valid: bool

    @property
    def suitable_for_reference(self) -> bool:
        return (
            self.valid_ratio >= 0.90
            and self.scale_px >= 30.0
            and self.scale_cv <= 0.25
            and self.speed_spike_ratio <= 3.0
            and self.phases_valid
        )


# Force-transfer order for the racket-arm kinetic chain (knee -> ... -> wrist).
KINETIC_CHAIN = (
    "knee_angular_speed",
    "hip_angular_speed",
    "shoulder_angular_speed",
    "elbow_angular_speed",
    "wrist_speed",
)


@dataclass(frozen=True)
class KineticChainTiming:
    """Peak-timing summary of the knee-hip-shoulder-elbow-wrist chain for one stroke."""

    peak_frames: dict[str, float]
    lags: dict[str, float]
    order_correct: bool
    transfer_time: float
    valid: bool


def extract_kinetic_chain_timing(features: GeometryFeatures) -> KineticChainTiming:
    """Summarize kinetic-chain peak timing from one stroke's `GeometryFeatures`.

    Requires all `KINETIC_CHAIN` columns to have at least one finite value;
    otherwise returns an invalid result instead of guessing a peak frame.
    """
    peak_frames: dict[str, float] = {}
    for name in KINETIC_CHAIN:
        column = features.column(name)
        valid = np.isfinite(column)
        if not valid.any():
            peak_frames[name] = float("nan")
            continue
        masked = np.where(valid, column, -np.inf)
        peak_frames[name] = float(np.argmax(masked))

    if any(np.isnan(frame) for frame in peak_frames.values()):
        return KineticChainTiming(peak_frames, {}, False, float("nan"), False)

    def time_at(frame: float) -> float:
        return float(features.timestamps[int(frame)])

    lags = {
        f"{_joint_prefix(after)}_after_{_joint_prefix(before)}": (
            time_at(peak_frames[after]) - time_at(peak_frames[before])
        )
        for before, after in zip(KINETIC_CHAIN, KINETIC_CHAIN[1:])
    }
    order_correct = all(
        peak_frames[before] <= peak_frames[after]
        for before, after in zip(KINETIC_CHAIN, KINETIC_CHAIN[1:])
    )
    transfer_time = time_at(peak_frames[KINETIC_CHAIN[-1]]) - time_at(peak_frames[KINETIC_CHAIN[0]])
    return KineticChainTiming(peak_frames, lags, order_correct, transfer_time, True)


def _joint_prefix(column_name: str) -> str:
    return column_name.split("_", 1)[0]


@dataclass(frozen=True)
class PhaseBoundaries:
    """Frame ranges `[start, end)` for the 5 stroke phases (mục 10.2)."""

    preparation: tuple[int, int]
    backswing: tuple[int, int]
    forward_swing: tuple[int, int]
    contact_estimated: tuple[int, int]
    follow_through: tuple[int, int]
    contact_frame: int
    valid: bool
    contact_confidence: float = 0.0
    contact_candidates: tuple[int, ...] = ()
    invalid_reason: str | None = None


def detect_stroke_phases(
    features: GeometryFeatures,
    *,
    prep_speed_ratio: float = 0.15,
    contact_window_frames: int = 2,
) -> PhaseBoundaries:
    """Propose phase boundaries from robust contact candidates.

    This is a semi-automatic proposal for the reference set (mục 10.2): the
    contact frame combines wrist and elbow motion, the backswing/forward-swing split
    is the speed valley (direction reversal) before that peak, and the
    preparation/backswing split is where speed first exceeds
    `prep_speed_ratio` of the peak. Review and adjust by hand before
    freezing a reference profile; this is not a ground-truth labeler.
    """
    speed = _median_filter(features.column("wrist_speed"), window=5)
    frame_count = len(speed)
    valid = np.isfinite(speed)
    if frame_count < 10 or valid.mean() < 0.7:
        empty = (0, 0)
        return PhaseBoundaries(
            empty, empty, empty, empty, empty, 0, False,
            invalid_reason="too_short_or_missing_wrist_speed",
        )

    # Ignore edge peaks: a peak at the beginning/end normally means that the
    # clip is truncated or that a one-frame pose spike won over the stroke.
    edge = max(contact_window_frames + 1, int(round(frame_count * 0.08)))
    candidates = np.arange(frame_count)
    interior = valid & (candidates >= edge) & (candidates < frame_count - edge)
    if not interior.any():
        empty = (0, 0)
        return PhaseBoundaries(
            empty, empty, empty, empty, empty, 0, False,
            invalid_reason="no_interior_contact_candidate",
        )

    contact_score = _contact_motion_score(features, speed)
    # A racket contact normally occurs in the latter part of a complete stroke.
    # This soft prior prevents an early draw-back acceleration from defeating
    # a nearly equal forward-swing peak; it cannot create motion where none exists.
    position = candidates / max(frame_count - 1, 1)
    temporal_prior = 0.40 + 0.60 * np.exp(-0.5 * ((position - 0.70) / 0.24) ** 2)
    ranked_score = np.where(interior, contact_score * temporal_prior, -np.inf)
    contact_frame = int(np.argmax(ranked_score))
    candidate_frames = _separated_contact_candidates(
        ranked_score, min_separation=max(contact_window_frames * 2 + 1, int(frame_count * 0.12))
    )
    top_score = float(ranked_score[contact_frame])
    second_score = float(ranked_score[candidate_frames[1]]) if len(candidate_frames) > 1 else 0.0
    contact_confidence = (
        (top_score - second_score) / max(top_score, 1e-6) if top_score > 0 else 0.0
    )
    peak_speed = speed[contact_frame]

    # Scalar speed does not encode the direction reversal of a backswing.
    # Use the last sustained low-speed valley before the peak when available;
    # otherwise use conservative temporal fractions and mark implausible
    # layouts invalid below.
    threshold = peak_speed * prep_speed_ratio
    low = valid[:contact_frame] & (speed[:contact_frame] <= threshold)
    low_frames = np.flatnonzero(low)
    reversal_frame = int(low_frames[-1] + 1) if low_frames.size else max(2, contact_frame // 2)
    prep_end = max(1, reversal_frame // 2)

    contact_start = max(reversal_frame, contact_frame - contact_window_frames)
    contact_end = min(frame_count, contact_frame + contact_window_frames + 1)

    ranges = (
        (0, prep_end),
        (prep_end, reversal_frame),
        (reversal_frame, contact_start),
        (contact_start, contact_end),
        (contact_end, frame_count),
    )
    min_phase = max(1, int(round(frame_count * 0.03)))
    contact_ratio = contact_frame / max(frame_count - 1, 1)
    follow_ratio = (frame_count - contact_end) / frame_count
    plausible = (
        all(end - start >= min_phase for start, end in ranges)
        and 0.35 <= contact_ratio <= 0.90
        and follow_ratio <= 0.45
        and np.isfinite(peak_speed)
        and top_score > 0
    )
    reason = None
    if not plausible:
        reason = "implausible_phase_layout"
    elif len(candidate_frames) > 1 and contact_confidence < 0.10:
        plausible = False
        reason = "ambiguous_contact_peaks"
    return PhaseBoundaries(
        preparation=(0, prep_end),
        backswing=(prep_end, reversal_frame),
        forward_swing=(reversal_frame, contact_start),
        contact_estimated=(contact_start, contact_end),
        follow_through=(contact_end, frame_count),
        contact_frame=contact_frame,
        valid=plausible,
        contact_confidence=contact_confidence,
        contact_candidates=tuple(candidate_frames),
        invalid_reason=reason,
    )


def _contact_motion_score(
    features: GeometryFeatures, wrist_speed: NDArray[np.float32]
) -> NDArray[np.float32]:
    """Composite contact evidence without depending on the segmentation package."""
    wrist = _robust_positive_normalize(features.column("wrist_speed"))
    if "elbow_angular_speed" in features.names:
        elbow = _robust_positive_normalize(features.column("elbow_angular_speed"))
        score = (0.70 * wrist + 0.30 * elbow).astype(np.float32)
    else:
        score = wrist
    if len(score):
        score = np.convolve(score, np.ones(5, dtype=np.float32) / 5, mode="same")
    return score.astype(np.float32)


def _robust_positive_normalize(values: NDArray[np.float32]) -> NDArray[np.float32]:
    output = np.zeros(len(values), dtype=np.float32)
    valid = np.isfinite(values)
    if not valid.any():
        return output
    observed = values[valid]
    median = float(np.median(observed))
    mad = float(np.median(np.abs(observed - median)))
    output[valid] = np.maximum((observed - median) / max(1.4826 * mad, 1e-6), 0)
    return output


def _separated_contact_candidates(
    score: NDArray[np.float32], *, min_separation: int, limit: int = 3
) -> list[int]:
    """Return strongest non-adjacent peaks, best candidate first."""
    candidates: list[int] = []
    for frame in np.argsort(score)[::-1]:
        frame = int(frame)
        if not np.isfinite(score[frame]) or score[frame] <= 0:
            break
        if all(abs(frame - chosen) >= min_separation for chosen in candidates):
            candidates.append(frame)
            if len(candidates) == limit:
                break
    return candidates


def extract_geometry_features(
    sequence: PoseSequence,
    *,
    handedness: str = "right",
    min_confidence: float = 0.3,
) -> GeometryFeatures:
    """Extract scale-normalized geometry and motion for the racket arm."""
    if handedness not in {"left", "right"}:
        raise ValueError("handedness must be left or right")
    if not 0.0 <= min_confidence <= 1.0:
        raise ValueError("min_confidence must be between 0 and 1")

    keypoints = np.asarray(sequence.keypoints, dtype=np.float32)
    if keypoints.ndim != 3 or keypoints.shape[1:] != (17, 3):
        raise ValueError(f"Expected keypoints with shape (T, 17, 3), got {keypoints.shape}")
    if sequence.fps <= 0:
        raise ValueError("fps must be positive")

    shoulder = RIGHT_SHOULDER if handedness == "right" else LEFT_SHOULDER
    elbow = RIGHT_ELBOW if handedness == "right" else LEFT_ELBOW
    wrist = RIGHT_WRIST if handedness == "right" else LEFT_WRIST
    hip = RIGHT_HIP if handedness == "right" else LEFT_HIP
    knee = RIGHT_KNEE if handedness == "right" else LEFT_KNEE
    ankle = RIGHT_ANKLE if handedness == "right" else LEFT_ANKLE

    frame_scale, scale_confidence = _body_scale(keypoints, min_confidence)
    finite_scale = frame_scale[np.isfinite(frame_scale)]
    # A single scale for the whole stroke prevents camera/pose jitter and the
    # shoulder-to-torso fallback from leaking into normalized velocities.
    stable_scale = float(np.median(finite_scale)) if finite_scale.size else float("nan")
    scale = np.where(np.isfinite(frame_scale), stable_scale, np.nan).astype(np.float32)
    mid_shoulder = (keypoints[:, LEFT_SHOULDER, :2] + keypoints[:, RIGHT_SHOULDER, :2]) / 2
    mid_hip = (keypoints[:, LEFT_HIP, :2] + keypoints[:, RIGHT_HIP, :2]) / 2

    angle_specs = (
        ("elbow_angle", shoulder, elbow, wrist),
        ("shoulder_angle", hip, shoulder, elbow),
        ("hip_angle", shoulder, hip, knee),
        ("knee_angle", hip, knee, ankle),
    )
    columns: list[NDArray[np.float32]] = []
    confidences: list[NDArray[np.float32]] = []
    names: list[str] = []
    for name, first, center, last in angle_specs:
        value, confidence = _joint_angle(
            keypoints, first, center, last, min_confidence
        )
        if name == "elbow_angle":
            # Near-zero projected elbow angles occur when the arm points into
            # the camera or the detector collapses/swaps elbow and wrist. Such
            # frames contain no defensible 2D anatomical angle.
            implausible_projection = np.isfinite(value) & (value < 10.0)
            value[implausible_projection] = np.nan
            confidence[implausible_projection] = 0.0
        names.append(name)
        columns.append(value)
        confidences.append(confidence)

    torso_vector = mid_shoulder - mid_hip
    torso_confidence = np.min(
        keypoints[:, [LEFT_SHOULDER, RIGHT_SHOULDER, LEFT_HIP, RIGHT_HIP], 2],
        axis=1,
    )
    torso_angle = np.degrees(
        np.arctan2(np.abs(torso_vector[:, 0]), np.abs(torso_vector[:, 1]))
    ).astype(np.float32)
    torso_valid = (torso_confidence >= min_confidence) & (scale > 0)
    torso_angle[~torso_valid] = np.nan
    names.append("torso_lean")
    columns.append(torso_angle)
    confidences.append(np.where(torso_valid, torso_confidence, 0).astype(np.float32))

    wrist_xy = keypoints[:, wrist, :2]
    shoulder_xy = keypoints[:, shoulder, :2]
    elbow_xy = keypoints[:, elbow, :2]
    arm_confidence = np.minimum(keypoints[:, wrist, 2], keypoints[:, shoulder, 2])
    arm_valid = (arm_confidence >= min_confidence) & (scale > 0)
    elbow_shoulder_confidence = np.minimum(keypoints[:, elbow, 2], keypoints[:, shoulder, 2])
    elbow_valid = (elbow_shoulder_confidence >= min_confidence) & (scale > 0)

    wrist_distance = _safe_scale(np.linalg.norm(wrist_xy - shoulder_xy, axis=1), scale)
    wrist_height = _safe_scale(shoulder_xy[:, 1] - wrist_xy[:, 1], scale)
    elbow_height = _safe_scale(shoulder_xy[:, 1] - elbow_xy[:, 1], scale)
    elbow_torso_distance = _point_to_line_distance(
        elbow_xy, mid_shoulder, mid_hip
    )
    elbow_torso_distance = _safe_scale(elbow_torso_distance, scale)
    wrist_distance[~arm_valid] = np.nan
    wrist_height[~arm_valid] = np.nan
    elbow_height[~elbow_valid] = np.nan
    elbow_torso_valid = elbow_valid & torso_valid
    elbow_torso_distance[~elbow_torso_valid] = np.nan
    for name, value, valid, confidence in (
        ("wrist_shoulder_distance", wrist_distance, arm_valid, arm_confidence),
        ("wrist_height", wrist_height, arm_valid, arm_confidence),
        ("elbow_height", elbow_height, elbow_valid, elbow_shoulder_confidence),
        (
            "elbow_torso_distance", elbow_torso_distance,
            elbow_torso_valid, np.minimum(elbow_shoulder_confidence, torso_confidence),
        ),
    ):
        names.append(name)
        columns.append(value.astype(np.float32))
        confidences.append(np.where(valid, confidence, 0).astype(np.float32))

    ankle_confidence = np.minimum(keypoints[:, LEFT_ANKLE, 2], keypoints[:, RIGHT_ANKLE, 2])
    stance_valid = (ankle_confidence >= min_confidence) & (scale > 0)
    stance_width = _safe_scale(
        np.linalg.norm(keypoints[:, LEFT_ANKLE, :2] - keypoints[:, RIGHT_ANKLE, :2], axis=1),
        scale,
    )
    stance_width[~stance_valid] = np.nan
    names.append("stance_width")
    columns.append(stance_width.astype(np.float32))
    confidences.append(np.where(stance_valid, ankle_confidence, 0).astype(np.float32))

    wrist_speed = _point_speed(wrist_xy, scale, sequence.fps, arm_valid)
    elbow_speed = _angular_speed(columns[0], sequence.fps)
    shoulder_speed = _angular_speed(columns[1], sequence.fps)
    hip_speed = _angular_speed(columns[2], sequence.fps)
    knee_speed = _angular_speed(columns[3], sequence.fps)
    for name, value, confidence in (
        ("wrist_speed", wrist_speed, arm_confidence),
        ("elbow_angular_speed", elbow_speed, confidences[0]),
        ("shoulder_angular_speed", shoulder_speed, confidences[1]),
        ("hip_angular_speed", hip_speed, confidences[2]),
        ("knee_angular_speed", knee_speed, confidences[3]),
    ):
        names.append(name)
        columns.append(value)
        confidences.append(np.where(np.isfinite(value), confidence, 0).astype(np.float32))

    frame_count = keypoints.shape[0]
    timestamps = np.arange(frame_count, dtype=np.float32) / np.float32(sequence.fps)
    return GeometryFeatures(
        tuple(names),
        np.column_stack(columns).astype(np.float32),
        np.column_stack(confidences).astype(np.float32),
        timestamps,
    )


def estimate_pose_scale_px(
    sequence: PoseSequence,
    *,
    min_confidence: float = 0.3,
) -> float:
    """Median per-frame body scale in raw pixels (shoulder width, or torso
    length fallback) — the same scale used to normalize other features.

    Use as a data-quality gate: a small pixel scale means keypoint jitter of
    a few pixels dominates the normalized signal (mục 9.1). NaN if no frame
    has a usable scale.
    """
    keypoints = np.asarray(sequence.keypoints, dtype=np.float32)
    scale, _ = _body_scale(keypoints, min_confidence)
    finite = scale[np.isfinite(scale)]
    if finite.size == 0:
        return float("nan")
    return float(np.median(finite))


def assess_geometry_quality(
    sequence: PoseSequence,
    *,
    handedness: str = "right",
    min_confidence: float = 0.3,
) -> GeometryQuality:
    """Measure trajectory stability, not merely keypoint availability."""
    features = extract_geometry_features(
        sequence, handedness=handedness, min_confidence=min_confidence
    )
    keypoints = np.asarray(sequence.keypoints, dtype=np.float32)
    raw_scale, _ = _body_scale(keypoints, min_confidence)
    # Shoulder width naturally collapses as a player rotates during a stroke,
    # and switching between shoulder/torso fallbacks creates artificial scale
    # jumps. Prefer torso length for the stability gate when it is available.
    torso_scale = _torso_length(keypoints, min_confidence)
    if np.isfinite(torso_scale).mean() >= 0.70:
        raw_scale = torso_scale
    finite_scale = raw_scale[np.isfinite(raw_scale)]
    scale_px = float(np.median(finite_scale)) if finite_scale.size else float("nan")
    scale_cv = (
        float(np.std(finite_scale) / np.mean(finite_scale))
        if finite_scale.size and np.mean(finite_scale) > 0
        else float("inf")
    )
    essential = ("elbow_angle", "shoulder_angle", "wrist_speed")
    valid_ratio = min(float(np.isfinite(features.column(name)).mean()) for name in essential)
    speed = features.column("wrist_speed")
    finite_speed = speed[np.isfinite(speed)]
    p95 = float(np.percentile(finite_speed, 95)) if finite_speed.size else 0.0
    spike_ratio = float(np.max(finite_speed) / p95) if p95 > 0 else float("inf")
    return GeometryQuality(
        valid_ratio=valid_ratio,
        scale_px=scale_px,
        scale_cv=scale_cv,
        speed_spike_ratio=spike_ratio,
        phases_valid=detect_stroke_phases(features).valid,
    )


def _joint_angle(
    keypoints: NDArray[np.float32],
    first: int,
    center: int,
    last: int,
    min_confidence: float,
) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
    ba = keypoints[:, first, :2] - keypoints[:, center, :2]
    bc = keypoints[:, last, :2] - keypoints[:, center, :2]
    denominator = np.linalg.norm(ba, axis=1) * np.linalg.norm(bc, axis=1)
    confidence = np.min(keypoints[:, [first, center, last], 2], axis=1)
    valid = (confidence >= min_confidence) & (denominator > 1e-6)
    cosine = np.zeros(len(keypoints), dtype=np.float32)
    cosine[valid] = np.sum(ba[valid] * bc[valid], axis=1) / denominator[valid]
    angle = np.degrees(np.arccos(np.clip(cosine, -1, 1))).astype(np.float32)
    angle[~valid] = np.nan
    return angle, np.where(valid, confidence, 0).astype(np.float32)


def _body_scale(
    keypoints: NDArray[np.float32], min_confidence: float
) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
    """Pick a per-frame body scale, preferring shoulder width but falling
    back to torso length when the shoulders are foreshortened by rotation
    (e.g. turning side-on for a backhand), which collapses shoulder width
    to an implausibly small value even at high keypoint confidence.
    """
    # Typical shoulder-width / torso-length ratio is ~0.4-0.6; below this,
    # shoulder width is more likely a rotation artifact than a true measurement.
    min_shoulder_torso_ratio = 0.35

    shoulder_confidence = np.min(keypoints[:, [LEFT_SHOULDER, RIGHT_SHOULDER], 2], axis=1)
    shoulder_width = np.linalg.norm(
        keypoints[:, LEFT_SHOULDER, :2] - keypoints[:, RIGHT_SHOULDER, :2], axis=1
    )
    hip_confidence = np.min(keypoints[:, [LEFT_HIP, RIGHT_HIP], 2], axis=1)
    torso_confidence = np.minimum(shoulder_confidence, hip_confidence)
    mid_shoulder = (keypoints[:, LEFT_SHOULDER, :2] + keypoints[:, RIGHT_SHOULDER, :2]) / 2
    mid_hip = (keypoints[:, LEFT_HIP, :2] + keypoints[:, RIGHT_HIP, :2]) / 2
    torso_length = np.linalg.norm(mid_shoulder - mid_hip, axis=1)

    shoulder_ok = (shoulder_confidence >= min_confidence) & (shoulder_width > 1e-6)
    torso_ok = (torso_confidence >= min_confidence) & (torso_length > 1e-6)
    shoulder_foreshortened = shoulder_ok & torso_ok & (
        shoulder_width < min_shoulder_torso_ratio * torso_length
    )

    use_shoulder = shoulder_ok & ~shoulder_foreshortened
    use_torso = (~use_shoulder) & torso_ok
    scale = np.full(len(keypoints), np.nan, dtype=np.float32)
    confidence = np.zeros(len(keypoints), dtype=np.float32)
    scale[use_shoulder] = shoulder_width[use_shoulder]
    confidence[use_shoulder] = shoulder_confidence[use_shoulder]
    scale[use_torso] = torso_length[use_torso]
    confidence[use_torso] = torso_confidence[use_torso]
    return scale, confidence


def _torso_length(
    keypoints: NDArray[np.float32], min_confidence: float
) -> NDArray[np.float32]:
    confidence = np.min(
        keypoints[:, [LEFT_SHOULDER, RIGHT_SHOULDER, LEFT_HIP, RIGHT_HIP], 2], axis=1
    )
    mid_shoulder = (
        keypoints[:, LEFT_SHOULDER, :2] + keypoints[:, RIGHT_SHOULDER, :2]
    ) / 2
    mid_hip = (keypoints[:, LEFT_HIP, :2] + keypoints[:, RIGHT_HIP, :2]) / 2
    length = np.linalg.norm(mid_shoulder - mid_hip, axis=1).astype(np.float32)
    length[(confidence < min_confidence) | (length <= 1e-6)] = np.nan
    return length


def _safe_scale(value: NDArray[np.float32], scale: NDArray[np.float32]) -> NDArray[np.float32]:
    output = np.full(len(value), np.nan, dtype=np.float32)
    valid = np.isfinite(scale) & (scale > 1e-6)
    output[valid] = value[valid] / scale[valid]
    return output


def _point_to_line_distance(
    point: NDArray[np.float32],
    line_start: NDArray[np.float32],
    line_end: NDArray[np.float32],
) -> NDArray[np.float32]:
    """Perpendicular distance to an infinite body-axis line per frame."""
    axis = line_end - line_start
    offset = point - line_start
    denominator = np.linalg.norm(axis, axis=1)
    result = np.full(len(point), np.nan, dtype=np.float32)
    valid = denominator > 1e-6
    result[valid] = np.abs(
        axis[valid, 0] * offset[valid, 1]
        - axis[valid, 1] * offset[valid, 0]
    ) / denominator[valid]
    return result


def _point_speed(
    points: NDArray[np.float32],
    scale: NDArray[np.float32],
    fps: float,
    valid: NDArray[np.bool_],
) -> NDArray[np.float32]:
    output = np.full(len(points), np.nan, dtype=np.float32)
    if len(points) < 2:
        return output
    smoothed_x = _median_filter(points[:, 0], window=5)
    smoothed_y = _median_filter(points[:, 1], window=5)
    smoothed = np.column_stack((smoothed_x, smoothed_y))
    pair_valid = valid[1:] & valid[:-1] & np.isfinite(scale[1:]) & np.isfinite(scale[:-1])
    pair_scale = (scale[1:] + scale[:-1]) / 2
    delta = np.linalg.norm(smoothed[1:] - smoothed[:-1], axis=1)
    values = delta * np.float32(fps) / pair_scale
    output[1:][pair_valid] = values[pair_valid]
    return output


def _angular_speed(angle: NDArray[np.float32], fps: float) -> NDArray[np.float32]:
    output = np.full(len(angle), np.nan, dtype=np.float32)
    if len(angle) < 2:
        return output
    smoothed = _median_filter(angle, window=5)
    valid = np.isfinite(smoothed[1:]) & np.isfinite(smoothed[:-1])
    delta = np.abs(smoothed[1:] - smoothed[:-1]) * np.float32(fps)
    output[1:][valid] = delta[valid]
    return output


def _median_filter(values: NDArray[np.float32], *, window: int) -> NDArray[np.float32]:
    """NaN-aware centered median filter without a SciPy dependency."""
    values = np.asarray(values, dtype=np.float32)
    output = np.full(values.shape, np.nan, dtype=np.float32)
    radius = window // 2
    for index in range(len(values)):
        chunk = values[max(0, index - radius) : min(len(values), index + radius + 1)]
        finite = chunk[np.isfinite(chunk)]
        if finite.size:
            output[index] = np.median(finite)
    return output
