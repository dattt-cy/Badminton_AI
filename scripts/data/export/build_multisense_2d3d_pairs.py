"""Pair MultiSense 2D pose clips with 3D joints after motion alignment."""

from __future__ import annotations

import argparse
import csv
import math
import re
from collections import Counter
from pathlib import Path

import h5py
import cv2
import numpy as np

from ai_classifier.biomechanics import detect_stroke_phases, extract_geometry_features
from ai_classifier.error_detection import phase_feature_value
from ai_classifier.pose import PoseSequence


POSE_PATTERN = re.compile(
    r"^(Sub\d+)_(forehand_clear|backhand_drive)_(front|side)_(\d+)_pose$"
)
MANUAL_TIME_PATTERN = re.compile(
    r"-(\d{2}\.\d{2}\.\d{2}\.\d{3})-(\d{2}\.\d{2}\.\d{2}\.\d{3})-seg\d+$"
)
VIDEO_NAMES = {
    "forehand_clear": "Forehand Clear",
    "backhand_drive": "Backhand Drive",
}


def parse_video_time(value: str) -> float:
    """Convert HH.MM.SS.mmm from a manual clip name to elapsed seconds."""
    hours, minutes, seconds, milliseconds = (int(part) for part in value.split("."))
    return hours * 3600 + minutes * 60 + seconds + milliseconds / 1000


def load_pose(path: Path) -> PoseSequence:
    with np.load(path) as source:
        return PoseSequence(
            np.asarray(source["keypoints"], dtype=np.float32),
            float(source["fps"]), int(source["frame_width"]), int(source["frame_height"]),
        )


def read_annotations(paths: list[Path]) -> list[dict[str, str]]:
    output = []
    for path in paths:
        with path.open(newline="", encoding="utf-8-sig") as source:
            for row in csv.DictReader(source):
                output.append(row)
    return output


def index_recordings(roots: list[Path]) -> dict[str, Path]:
    recordings = {}
    for root in roots:
        for path in root.rglob("*.hdf5"):
            recordings[path.stem] = path
    return recordings


def index_source_videos(roots: list[Path]) -> dict[tuple[str, str, str], Path]:
    videos = {}
    expected = {
        (technique, view): f"{label} {view.title()} Video.mov"
        for technique, label in VIDEO_NAMES.items()
        for view in ("front", "side")
    }
    for root in roots:
        for path in root.rglob("*.mov"):
            subject = path.parent.name
            for key, filename in expected.items():
                if path.name.casefold() == filename.casefold():
                    videos[(subject, *key)] = path
    return videos


def video_duration(path: Path) -> float:
    capture = cv2.VideoCapture(str(path))
    try:
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        frames = float(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    finally:
        capture.release()
    if fps <= 0 or frames <= 0:
        raise ValueError(f"Cannot read source video duration: {path}")
    return frames / fps


def estimate_video_epoch_start(
    clip_intervals: list[tuple[float, float]],
    annotations: list[dict[str, str]],
    prior_epoch_start: float,
    *,
    max_prior_error_seconds: float = 120.0,
) -> float:
    """Align manual clip intervals to annotation intervals with an end-time prior."""
    if not clip_intervals or not annotations:
        return prior_epoch_start
    annotation_intervals = [
        (float(row["start_time"]), float(row["stop_time"])) for row in annotations
    ]
    candidates = {prior_epoch_start}
    for clip_start, clip_stop in clip_intervals:
        clip_center = (clip_start + clip_stop) / 2
        for annotation_start, annotation_stop in annotation_intervals:
            annotation_center = (annotation_start + annotation_stop) / 2
            candidate = annotation_center - clip_center
            if abs(candidate - prior_epoch_start) <= max_prior_error_seconds:
                candidates.add(candidate)

    def score(epoch_start: float) -> tuple[int, float, float]:
        matched, overlap_sum = 0, 0.0
        for clip_start, clip_stop in clip_intervals:
            absolute_start = epoch_start + clip_start
            absolute_stop = epoch_start + clip_stop
            best_overlap = max(
                max(0.0, min(absolute_stop, stop) - max(absolute_start, start))
                for start, stop in annotation_intervals
            )
            if best_overlap >= 0.20:
                matched += 1
                overlap_sum += best_overlap
        return matched, overlap_sum, -abs(epoch_start - prior_epoch_start)

    return float(max(candidates, key=score))


def manual_clip_metadata(
    pose_path: Path, pose_root: Path
) -> tuple[str, str, str, float, float] | None:
    """Read subject/action/view and source-video times from a mirrored pose path."""
    try:
        relative = pose_path.relative_to(pose_root)
    except ValueError:
        return None
    parts = relative.parts
    if len(parts) < 5:
        return None
    _, technique, view, subject = parts[:4]
    if technique not in VIDEO_NAMES or view not in {"front", "side"}:
        return None
    match = MANUAL_TIME_PATTERN.search(pose_path.stem)
    if match is None:
        return None
    start, stop = (parse_video_time(value) for value in match.groups())
    return subject, technique, view, start, stop


def match_manual_annotation(
    metadata: tuple[str, str, str, float, float],
    annotations: list[dict[str, str]],
    recordings: dict[str, Path],
    source_videos: dict[tuple[str, str, str], Path],
    recording_cache: dict[str, tuple[np.ndarray, dict[str, np.ndarray]]],
    video_epoch_starts: dict[tuple[str, str, str], float],
) -> tuple[dict[str, str], float] | None:
    """Map source-video elapsed time to the best-overlapping HDF5 annotation."""
    subject, technique, view, clip_start, clip_stop = metadata
    source_video = source_videos.get((subject, technique, view))
    if source_video is None:
        return None
    candidates = [
        row for row in annotations
        if row["subject"] == subject and row["technique"] == technique
        and row["recording"] in recordings
    ]
    best: tuple[float, dict[str, str], float] | None = None
    for recording in {row["recording"] for row in candidates}:
        if recording not in recording_cache:
            recording_cache[recording] = load_3d_recording(recordings[recording])
        timestamps, _ = recording_cache[recording]
        video_epoch_start = video_epoch_starts.get(
            (subject, technique, view),
            float(timestamps[-1]) - video_duration(source_video),
        )
        epoch_start = video_epoch_start + clip_start
        epoch_stop = video_epoch_start + clip_stop
        for row in candidates:
            if row["recording"] != recording:
                continue
            overlap = max(
                0.0,
                min(epoch_stop, float(row["stop_time"]))
                - max(epoch_start, float(row["start_time"])),
            )
            if best is None or overlap > best[0]:
                best = overlap, row, epoch_start
    if best is None or best[0] < 0.20:
        return None
    return best[1], best[2]


def _rolling_median(values: np.ndarray, window: int = 5) -> np.ndarray:
    if len(values) < window:
        return values
    padded = np.pad(values, (window // 2, window // 2), mode="edge")
    chunks = np.lib.stride_tricks.sliding_window_view(padded, window)
    return np.nanmedian(chunks, axis=1)


def load_3d_recording(path: Path) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    with h5py.File(path, "r") as source:
        stream = source["pns-joint"]["global-position"]
        timestamps = np.asarray(stream["time_s"], dtype=np.float64).reshape(-1)
        positions = np.asarray(stream["data"], dtype=np.float32)
    xyz = positions.reshape(-1, 21, 3)
    shoulder, elbow, wrist = xyz[:, 14], xyz[:, 15], xyz[:, 16]
    first, last = shoulder - elbow, wrist - elbow
    denominator = np.linalg.norm(first, axis=1) * np.linalg.norm(last, axis=1)
    cosine = np.sum(first * last, axis=1) / np.maximum(denominator, 1e-9)
    elbow_angle = np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))
    elbow_angle[denominator <= 1e-9] = np.nan
    scale = np.linalg.norm(xyz[:, 18] - xyz[:, 14], axis=1)
    hips, neck = xyz[:, 0], xyz[:, 10]
    torso = neck - hips
    torso_norm = np.linalg.norm(torso, axis=1)
    torso_lean = np.degrees(np.arccos(np.clip(
        np.abs(torso[:, 1]) / np.maximum(torso_norm, 1e-9), 0.0, 1.0
    )))

    def normalized(values: np.ndarray) -> np.ndarray:
        result = np.full(len(values), np.nan, dtype=np.float64)
        usable = np.isfinite(values) & np.isfinite(scale) & (scale > 1e-6)
        result[usable] = values[usable] / scale[usable]
        return result

    torso_axis = hips - neck
    torso_denominator = np.linalg.norm(torso_axis, axis=1)
    elbow_torso = np.full(len(xyz), np.nan, dtype=np.float64)
    usable_torso = torso_denominator > 1e-9
    elbow_torso[usable_torso] = (
        np.linalg.norm(
            np.cross(elbow[usable_torso] - neck[usable_torso], torso_axis[usable_torso]),
            axis=1,
        ) / torso_denominator[usable_torso]
    )
    wrist_speed = np.full(len(xyz), np.nan, dtype=np.float64)
    delta_t = np.diff(timestamps)
    pair_scale = (scale[1:] + scale[:-1]) / 2
    valid = (delta_t > 0) & np.isfinite(pair_scale) & (pair_scale > 1e-6)
    raw_speed = np.linalg.norm(np.diff(wrist, axis=0), axis=1) / delta_t / pair_scale
    wrist_speed[1:][valid] = raw_speed[valid]
    if np.isfinite(wrist_speed).sum() < 15:
        raise ValueError(f"Recording has insufficient finite 3D wrist speed: {path}")
    elbow_angular_speed = np.full(len(xyz), np.nan, dtype=np.float64)
    angle_valid = (delta_t > 0) & np.isfinite(elbow_angle[1:]) & np.isfinite(elbow_angle[:-1])
    elbow_angular_speed[1:][angle_valid] = (
        np.abs(np.diff(_rolling_median(elbow_angle))) / delta_t
    )[angle_valid]
    return timestamps, {
        "joints_xyz": xyz,
        "elbow_angle": elbow_angle,
        "wrist_speed": _rolling_median(wrist_speed),
        "elbow_angular_speed": _rolling_median(elbow_angular_speed),
        "wrist_shoulder_distance": normalized(np.linalg.norm(wrist - shoulder, axis=1)),
        "wrist_height": normalized(wrist[:, 1] - shoulder[:, 1]),
        "elbow_torso_distance": normalized(elbow_torso),
        "torso_lean": torso_lean,
        "stance_width": normalized(np.linalg.norm(xyz[:, 6] - xyz[:, 3], axis=1)),
    }


def _standardize(values: np.ndarray) -> np.ndarray:
    finite = np.isfinite(values)
    output = np.full(values.shape, np.nan, dtype=np.float64)
    if finite.sum() < 3:
        return output
    center = np.median(values[finite])
    scale = np.percentile(np.abs(values[finite] - center), 75)
    if scale <= 1e-9:
        scale = np.std(values[finite])
    if scale > 1e-9:
        output[finite] = (values[finite] - center) / scale
    return output


def motion_alignment_offset(
    pose_time: np.ndarray,
    speed_2d: np.ndarray,
    timestamps_3d: np.ndarray,
    speed_3d: np.ndarray,
    base_epoch: float,
    *,
    max_offset_seconds: float = 0.5,
    step_seconds: float = 1 / 60,
) -> tuple[float, float]:
    """Find the offset maximizing normalized 2D/3D wrist-speed correlation."""
    pose_time = np.asarray(pose_time, dtype=np.float64)
    speed_2d = _standardize(np.asarray(speed_2d, dtype=np.float64))
    speed_3d = _standardize(np.asarray(speed_3d, dtype=np.float64))
    offsets = np.arange(
        -max_offset_seconds, max_offset_seconds + step_seconds / 2, step_seconds
    )
    best_offset, best_score = 0.0, -math.inf
    valid_3d = np.isfinite(timestamps_3d) & np.isfinite(speed_3d)
    if valid_3d.sum() < 15:
        return best_offset, best_score
    for offset in offsets:
        sample_time = base_epoch + pose_time + offset
        sampled = np.interp(
            sample_time, timestamps_3d[valid_3d], speed_3d[valid_3d],
            left=np.nan, right=np.nan,
        )
        valid = np.isfinite(speed_2d) & np.isfinite(sampled)
        if valid.sum() < 15:
            continue
        if np.std(speed_2d[valid]) <= 1e-9 or np.std(sampled[valid]) <= 1e-9:
            continue
        score = float(np.corrcoef(speed_2d[valid], sampled[valid])[0, 1])
        if math.isfinite(score) and score > best_score:
            best_offset, best_score = float(offset), score
    if not math.isfinite(best_score):
        sample_time = base_epoch + pose_time
        sampled = np.interp(
            sample_time, timestamps_3d[valid_3d], speed_3d[valid_3d],
            left=np.nan, right=np.nan,
        )
        raise ValueError(
            "Cannot correlate motion: "
            f"pose_time=({sample_time[0]}, {sample_time[-1]}), "
            f"3d_time=({timestamps_3d[valid_3d][0]}, {timestamps_3d[valid_3d][-1]}), "
            f"finite_2d={np.isfinite(speed_2d).sum()}, "
            f"finite_sampled={np.isfinite(sampled).sum()}"
        )
    return best_offset, best_score


def multi_joint_motion_alignment_offset(
    pose_time: np.ndarray,
    signals_2d: tuple[np.ndarray, ...],
    timestamps_3d: np.ndarray,
    signals_3d: tuple[np.ndarray, ...],
    base_epoch: float,
    *,
    max_offset_seconds: float = 0.5,
    step_seconds: float = 1 / 60,
) -> tuple[float, float]:
    """Align one global clock offset using multiple independent motion signals."""
    if len(signals_2d) != len(signals_3d) or not signals_2d:
        raise ValueError("2D and 3D motion signal collections must have equal length")
    pose_time = np.asarray(pose_time, dtype=np.float64)
    standardized_2d = tuple(_standardize(np.asarray(signal)) for signal in signals_2d)
    standardized_3d = tuple(_standardize(np.asarray(signal)) for signal in signals_3d)
    offsets = np.arange(
        -max_offset_seconds, max_offset_seconds + step_seconds / 2, step_seconds
    )
    best_offset, best_score = 0.0, -math.inf
    for offset in offsets:
        sample_time = base_epoch + pose_time + offset
        correlations = []
        for signal_2d, signal_3d in zip(standardized_2d, standardized_3d):
            valid_3d = np.isfinite(timestamps_3d) & np.isfinite(signal_3d)
            if valid_3d.sum() < 15:
                continue
            sampled = np.interp(
                sample_time, timestamps_3d[valid_3d], signal_3d[valid_3d],
                left=np.nan, right=np.nan,
            )
            valid = np.isfinite(signal_2d) & np.isfinite(sampled)
            if valid.sum() < 15 or np.std(signal_2d[valid]) <= 1e-9:
                continue
            correlation = float(np.corrcoef(signal_2d[valid], sampled[valid])[0, 1])
            if math.isfinite(correlation):
                correlations.append(correlation)
        if correlations:
            score = float(np.mean(correlations))
            if score > best_score:
                best_offset, best_score = float(offset), score
    if not math.isfinite(best_score):
        raise ValueError("Cannot correlate the available multi-joint motion signals")
    return best_offset, best_score


def projected_ratio(sequence: PoseSequence, frame_range: tuple[int, int]) -> float:
    start, end = frame_range
    points = sequence.keypoints[start:end]
    shoulder = np.linalg.norm(points[:, 5, :2] - points[:, 6, :2], axis=1)
    mid_shoulder = (points[:, 5, :2] + points[:, 6, :2]) / 2
    mid_hip = (points[:, 11, :2] + points[:, 12, :2]) / 2
    torso = np.linalg.norm(mid_shoulder - mid_hip, axis=1)
    confidence = np.min(points[:, [5, 6, 11, 12], 2], axis=1)
    valid = (confidence >= 0.3) & (torso > 1e-6)
    return float(np.median(shoulder[valid] / torso[valid])) if valid.any() else math.nan


def arm_confidence(sequence: PoseSequence, frame_range: tuple[int, int]) -> float:
    start, end = frame_range
    points = sequence.keypoints[start:end]
    confidence = np.min(points[:, [6, 8, 10], 2], axis=1)
    finite = confidence[np.isfinite(confidence)]
    return float(np.median(finite)) if finite.size else math.nan


def frame_projected_ratio(sequence: PoseSequence) -> np.ndarray:
    points = sequence.keypoints
    shoulder = np.linalg.norm(points[:, 5, :2] - points[:, 6, :2], axis=1)
    mid_shoulder = (points[:, 5, :2] + points[:, 6, :2]) / 2
    mid_hip = (points[:, 11, :2] + points[:, 12, :2]) / 2
    torso = np.linalg.norm(mid_shoulder - mid_hip, axis=1)
    confidence = np.min(points[:, [5, 6, 11, 12], 2], axis=1)
    result = np.full(len(points), np.nan, dtype=np.float64)
    valid = (confidence >= 0.3) & (torso > 1e-6)
    result[valid] = shoulder[valid] / torso[valid]
    return result


def frame_arm_confidence(sequence: PoseSequence) -> np.ndarray:
    return np.min(sequence.keypoints[:, [6, 8, 10], 2], axis=1).astype(np.float64)


def build_pair(
    pose_path: Path,
    annotation: dict[str, str],
    timestamps_3d: np.ndarray,
    features_3d: dict[str, np.ndarray],
    *,
    max_offset_seconds: float,
    base_epoch: float | None = None,
    identity: tuple[str, str, str] | None = None,
) -> dict[str, object] | None:
    sequence = load_pose(pose_path)
    features_2d = extract_geometry_features(sequence)
    phases_2d = detect_stroke_phases(features_2d)
    if not phases_2d.valid:
        return None
    duration = len(sequence.keypoints) / sequence.fps
    annotation_duration = float(annotation["stop_time"]) - float(annotation["start_time"])
    padding = max(0.0, (duration - annotation_duration) / 2) if base_epoch is None else 0.0
    base_epoch = float(annotation["start_time"]) - padding if base_epoch is None else base_epoch
    offset, correlation = motion_alignment_offset(
        features_2d.timestamps,
        features_2d.column("wrist_speed"),
        timestamps_3d,
        features_3d["wrist_speed"],
        base_epoch,
        max_offset_seconds=max_offset_seconds,
    )
    contact_start, contact_end = phases_2d.contact_estimated
    contact_time = (
        base_epoch
        + np.asarray(features_2d.timestamps[contact_start:contact_end], dtype=np.float64)
        + offset
    )
    elbow_3d = np.interp(
        contact_time, timestamps_3d, features_3d["elbow_angle"],
        left=np.nan, right=np.nan,
    )
    finite_3d = elbow_3d[np.isfinite(elbow_3d)]
    elbow_2d = phase_feature_value(
        features_2d, phases_2d.contact_estimated, "elbow_angle"
    )
    if not finite_3d.size or not math.isfinite(elbow_2d):
        return None
    match = POSE_PATTERN.match(pose_path.stem)
    if identity is None:
        assert match is not None
        subject, technique, view, _ = match.groups()
    else:
        subject, technique, view = identity
    return {
        "clip": str(pose_path),
        "subject": subject,
        "technique": technique,
        "view": view,
        "stroke": int(annotation["stroke_number"]),
        "peak_frame": phases_2d.contact_frame,
        "peak_confidence": phases_2d.contact_confidence,
        "elbow_2d_median": elbow_2d,
        "elbow_3d_same_time_median": float(np.median(finite_3d)),
        "absolute_error_deg": abs(elbow_2d - float(np.median(finite_3d))),
        "pose_confidence_median": arm_confidence(sequence, phases_2d.contact_estimated),
        "projected_ratio_median": projected_ratio(sequence, phases_2d.contact_estimated),
        "padding_estimate_s": padding,
        "motion_alignment_offset_s": offset,
        "motion_alignment_correlation": correlation,
        "sample_kind": "contact",
    }


def build_frame_pairs(
    pose_path: Path,
    annotation: dict[str, str],
    timestamps_3d: np.ndarray,
    features_3d: dict[str, np.ndarray],
    *,
    max_offset_seconds: float,
    base_epoch: float,
    identity: tuple[str, str, str],
    frame_stride: int,
) -> tuple[list[dict[str, object]], float]:
    """Return synchronized in-stroke frame samples and clip alignment quality."""
    sequence = load_pose(pose_path)
    features_2d = extract_geometry_features(sequence)
    offset, correlation = motion_alignment_offset(
        features_2d.timestamps,
        features_2d.column("wrist_speed"),
        timestamps_3d,
        features_3d["wrist_speed"],
        base_epoch,
        max_offset_seconds=max_offset_seconds,
    )
    subject, technique, view = identity
    absolute_times = (
        base_epoch + np.asarray(features_2d.timestamps, dtype=np.float64) + offset
    )
    elbow_2d = features_2d.column("elbow_angle").astype(np.float64)
    elbow_3d = np.interp(
        absolute_times, timestamps_3d, features_3d["elbow_angle"],
        left=np.nan, right=np.nan,
    )
    ratios = frame_projected_ratio(sequence)
    confidence = frame_arm_confidence(sequence)
    extra_2d = {
        "shoulder_angle_2d": features_2d.column("shoulder_angle"),
        "wrist_shoulder_distance_2d": features_2d.column("wrist_shoulder_distance"),
        "elbow_torso_distance_2d": features_2d.column("elbow_torso_distance"),
        "wrist_height_2d": features_2d.column("wrist_height"),
        "torso_lean_2d": features_2d.column("torso_lean"),
        "stance_width_2d": features_2d.column("stance_width"),
    }
    paired_features = (
        "wrist_shoulder_distance", "wrist_height", "elbow_torso_distance",
        "torso_lean", "stance_width",
    )
    extra_3d = {
        f"{name}_3d": np.interp(
            absolute_times, timestamps_3d, features_3d[name],
            left=np.nan, right=np.nan,
        )
        for name in paired_features
    }
    annotation_start = float(annotation["start_time"])
    annotation_stop = float(annotation["stop_time"])
    rows = []
    for frame in range(0, len(sequence.keypoints), frame_stride):
        if not annotation_start <= absolute_times[frame] <= annotation_stop:
            continue
        values = (
            elbow_2d[frame], elbow_3d[frame], ratios[frame], confidence[frame],
            *(values[frame] for values in extra_2d.values()),
        )
        if not all(math.isfinite(value) for value in values):
            continue
        row = {
            "clip": str(pose_path),
            "subject": subject,
            "technique": technique,
            "view": view,
            "stroke": int(annotation["stroke_number"]),
            "frame": frame,
            "timestamp_3d": absolute_times[frame],
            "peak_frame": "",
            "peak_confidence": "",
            "elbow_2d_median": elbow_2d[frame],
            "elbow_3d_same_time_median": elbow_3d[frame],
            "absolute_error_deg": abs(elbow_2d[frame] - elbow_3d[frame]),
            "pose_confidence_median": confidence[frame],
            "projected_ratio_median": ratios[frame],
            "padding_estimate_s": 0.0,
            "motion_alignment_offset_s": offset,
            "motion_alignment_correlation": correlation,
            "sample_kind": "frame",
        }
        row.update({name: float(values[frame]) for name, values in extra_2d.items()})
        row.update({name: float(values[frame]) for name, values in extra_3d.items()})
        rows.append(row)
    return rows, correlation


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pose_root", type=Path)
    parser.add_argument("output_csv", type=Path)
    parser.add_argument("--feature-csv", type=Path, nargs="+", required=True)
    parser.add_argument("--archive-root", type=Path, nargs="+", required=True)
    parser.add_argument("--max-offset-seconds", type=float, default=0.5)
    parser.add_argument("--min-alignment-correlation", type=float, default=0.15)
    parser.add_argument(
        "--pair-mode", choices=("contact", "frames"), default="contact",
        help="Pair one estimated-contact summary or multiple synchronized frames",
    )
    parser.add_argument("--frame-stride", type=int, default=3)
    args = parser.parse_args()
    if args.frame_stride < 1:
        raise ValueError("--frame-stride must be positive")
    annotations = read_annotations(args.feature_csv)
    recordings = index_recordings(args.archive_root)
    source_videos = index_source_videos(args.archive_root)
    cache: dict[str, tuple[np.ndarray, dict[str, np.ndarray]]] = {}
    rows = []
    skipped: Counter[str] = Counter()
    alignment_scores: list[float] = []
    pose_paths = sorted(args.pose_root.rglob("*.npz"))
    manual_metadata = [
        metadata for path in pose_paths
        if (metadata := manual_clip_metadata(path, args.pose_root)) is not None
    ]
    video_epoch_starts: dict[tuple[str, str, str], float] = {}
    for key in sorted({metadata[:3] for metadata in manual_metadata}):
        subject, technique, view = key
        source_video = source_videos.get(key)
        candidate_rows = [
            row for row in annotations
            if row["subject"] == subject and row["technique"] == technique
            and row["recording"] in recordings
        ]
        if source_video is None or not candidate_rows:
            continue
        recording = candidate_rows[0]["recording"]
        if recording not in cache:
            cache[recording] = load_3d_recording(recordings[recording])
        timestamps, _ = cache[recording]
        prior = float(timestamps[-1]) - video_duration(source_video)
        intervals = [metadata[3:] for metadata in manual_metadata if metadata[:3] == key]
        video_epoch_starts[key] = estimate_video_epoch_start(
            intervals, candidate_rows, prior
        )
    for pose_path in pose_paths:
        match = POSE_PATTERN.match(pose_path.stem)
        base_epoch = None
        identity = None
        if match is not None:
            subject, technique, view, stroke = match.groups()
            annotation = next((
                row for row in annotations
                if row["subject"] == subject and row["technique"] == technique
                and int(row["stroke_number"]) == int(stroke)
            ), None)
        else:
            metadata = manual_clip_metadata(pose_path, args.pose_root)
            if metadata is None:
                skipped["unrecognized_pose_name"] += 1
                continue
            matched = match_manual_annotation(
                metadata, annotations, recordings, source_videos, cache,
                video_epoch_starts,
            )
            if matched is None:
                skipped["annotation_not_found"] += 1
                continue
            annotation, base_epoch = matched
            subject, technique, view, _, _ = metadata
            identity = subject, technique, view
        if annotation is None:
            skipped["annotation_not_found"] += 1
            continue
        if annotation["recording"] not in recordings:
            skipped["recording_not_found"] += 1
            continue
        recording = annotation["recording"]
        if recording not in cache:
            cache[recording] = load_3d_recording(recordings[recording])
        timestamps, features_3d = cache[recording]
        try:
            if args.pair_mode == "frames":
                assert base_epoch is not None and identity is not None
                paired_rows, correlation = build_frame_pairs(
                    pose_path, annotation, timestamps, features_3d,
                    max_offset_seconds=args.max_offset_seconds,
                    base_epoch=base_epoch, identity=identity,
                    frame_stride=args.frame_stride,
                )
                pair = None
            else:
                pair = build_pair(
                    pose_path, annotation, timestamps, features_3d,
                    max_offset_seconds=args.max_offset_seconds,
                    base_epoch=base_epoch,
                    identity=identity,
                )
                if pair is None:
                    skipped["invalid_2d_phase_or_angle"] += 1
                    continue
                paired_rows = [pair]
                correlation = float(pair["motion_alignment_correlation"])
        except ValueError:
            skipped["insufficient_motion_for_alignment"] += 1
            continue
        alignment_scores.append(correlation)
        if correlation < args.min_alignment_correlation:
            skipped["low_alignment_correlation"] += 1
            continue
        if not paired_rows:
            skipped["no_valid_in_stroke_frames"] += 1
            continue
        rows.extend(paired_rows)
    if not rows:
        finite_scores = [score for score in alignment_scores if math.isfinite(score)]
        raise RuntimeError(
            "No pose/3D pairs passed phase and alignment quality gates: "
            f"{dict(skipped)}; finite alignment scores={len(finite_scores)}, "
            f"range={((min(finite_scores), max(finite_scores)) if finite_scores else None)}"
        )
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.open("w", newline="", encoding="utf-8-sig") as output:
        writer = csv.DictWriter(output, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(
        f"Exported {len(rows)} motion-aligned pairs; "
        f"skipped {dict(sorted(skipped.items()))}: {args.output_csv}"
    )


if __name__ == "__main__":
    main()
