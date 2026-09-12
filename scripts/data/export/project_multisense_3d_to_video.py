"""Project synchronized MultiSense 3D joints into each manual video camera."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter
from pathlib import Path

import numpy as np

from ai_classifier.biomechanics import (
    extract_geometry_features,
    fit_perspective_camera,
    fit_weak_perspective_camera,
)
from ai_classifier.pose import PoseSequence
from build_multisense_2d3d_pairs import (
    estimate_video_epoch_start,
    index_recordings,
    index_source_videos,
    load_3d_recording,
    load_pose,
    manual_clip_metadata,
    match_manual_annotation,
    multi_joint_motion_alignment_offset,
    read_annotations,
    video_duration,
)


# COCO joint -> MultiSense joint. Arm joints are projected for evaluation but
# excluded from camera fitting so the target movement cannot tune its own camera.
JOINT_MAP = {
    5: 18, 6: 14, 7: 19, 8: 15, 9: 20, 10: 16,
    11: 4, 12: 1, 13: 5, 14: 2, 15: 6, 16: 3,
}
CAMERA_FIT_JOINTS = (5, 6, 11, 12, 13, 14, 15, 16)
FEATURES = (
    "elbow_angle", "wrist_shoulder_distance", "wrist_height",
    "elbow_torso_distance", "torso_lean", "stance_width",
)
JOINT_NAMES = {
    5: "left_shoulder", 6: "right_shoulder",
    7: "left_elbow", 8: "right_elbow",
    9: "left_wrist", 10: "right_wrist",
    11: "left_hip", 12: "right_hip",
    13: "left_knee", 14: "right_knee",
    15: "left_ankle", 16: "right_ankle",
}


def interpolate_joints(
    sample_times: np.ndarray, timestamps: np.ndarray, joints_xyz: np.ndarray
) -> np.ndarray:
    flattened = joints_xyz.reshape(len(joints_xyz), -1)
    sampled = np.column_stack([
        np.interp(sample_times, timestamps, flattened[:, column], left=np.nan, right=np.nan)
        for column in range(flattened.shape[1])
    ])
    return sampled.reshape(len(sample_times), 21, 3)


def project_clip(
    pose_path: Path,
    annotation: dict[str, str],
    base_epoch: float,
    identity: tuple[str, str, str],
    timestamps_3d: np.ndarray,
    recording_3d: dict[str, np.ndarray],
    *,
    max_offset_seconds: float,
    frame_stride: int,
    alignment_signal: str = "wrist",
    residual_offset_seconds: float = 0.0,
    camera_model: str = "perspective",
) -> tuple[list[dict[str, object]], dict[str, float]]:
    sequence = load_pose(pose_path)
    yolo_features = extract_geometry_features(sequence)
    if alignment_signal == "multi":
        offset, correlation = multi_joint_motion_alignment_offset(
            yolo_features.timestamps,
            (yolo_features.column("wrist_speed"), yolo_features.column("elbow_angular_speed")),
            timestamps_3d,
            (recording_3d["wrist_speed"], recording_3d["elbow_angular_speed"]),
            base_epoch,
            max_offset_seconds=max_offset_seconds,
        )
    else:
        from build_multisense_2d3d_pairs import motion_alignment_offset
        offset, correlation = motion_alignment_offset(
            yolo_features.timestamps, yolo_features.column("wrist_speed"),
            timestamps_3d, recording_3d["wrist_speed"], base_epoch,
            max_offset_seconds=max_offset_seconds,
        )
    absolute_times = (
        base_epoch + np.asarray(yolo_features.timestamps, dtype=np.float64)
        + offset + residual_offset_seconds
    )
    joints_3d = interpolate_joints(
        absolute_times, timestamps_3d, recording_3d["joints_xyz"]
    )

    calibration_3d, calibration_2d, weights = [], [], []
    for frame in range(len(sequence.keypoints)):
        for coco_joint in CAMERA_FIT_JOINTS:
            confidence = float(sequence.keypoints[frame, coco_joint, 2])
            if confidence < 0.30:
                continue
            point_3d = joints_3d[frame, JOINT_MAP[coco_joint]]
            point_2d = sequence.keypoints[frame, coco_joint, :2]
            if np.isfinite(point_3d).all() and np.isfinite(point_2d).all():
                calibration_3d.append(point_3d)
                calibration_2d.append(point_2d)
                weights.append(confidence)
    camera_fitter = (
        fit_perspective_camera if camera_model == "perspective"
        else fit_weak_perspective_camera
    )
    camera = camera_fitter(
        np.asarray(calibration_3d), np.asarray(calibration_2d), np.asarray(weights)
    )

    projected = np.zeros_like(sequence.keypoints, dtype=np.float32)
    for coco_joint, multisense_joint in JOINT_MAP.items():
        projected[:, coco_joint, :2] = camera.project(
            joints_3d[:, multisense_joint]
        ).astype(np.float32)
        projected[:, coco_joint, 2] = 1.0
    projected_sequence = PoseSequence(
        projected, sequence.fps, sequence.frame_width, sequence.frame_height
    )
    projected_features = extract_geometry_features(projected_sequence)

    mid_shoulder = (
        sequence.keypoints[:, 5, :2] + sequence.keypoints[:, 6, :2]
    ) / 2
    mid_hip = (
        sequence.keypoints[:, 11, :2] + sequence.keypoints[:, 12, :2]
    ) / 2
    torso_scale = np.linalg.norm(mid_shoulder - mid_hip, axis=1)
    finite_scale = torso_scale[np.isfinite(torso_scale) & (torso_scale > 1e-6)]
    scale = float(np.median(finite_scale)) if finite_scale.size else math.nan
    median_error_ratio = camera.median_error / scale if scale > 0 else math.inf
    p90_error_ratio = camera.p90_error / scale if scale > 0 else math.inf

    subject, technique, view = identity
    start, stop = float(annotation["start_time"]), float(annotation["stop_time"])
    rows = []
    for frame in range(0, len(sequence.keypoints), frame_stride):
        if not start <= absolute_times[frame] <= stop:
            continue
        row: dict[str, object] = {
            "clip": str(pose_path), "subject": subject,
            "technique": technique, "view": view,
            "stroke": int(annotation["stroke_number"]), "frame": frame,
            "timestamp_3d": absolute_times[frame],
            "motion_alignment_offset_s": offset,
            "residual_calibration_offset_s": residual_offset_seconds,
            "motion_alignment_correlation": correlation,
            "camera_scale": getattr(camera, "scale", ""),
            "camera_fit_points": camera.point_count,
            "camera_median_error_ratio": median_error_ratio,
            "camera_p90_error_ratio": p90_error_ratio,
        }
        usable = True
        for feature in FEATURES:
            value_yolo = float(yolo_features.column(feature)[frame])
            value_projected = float(projected_features.column(feature)[frame])
            row[f"{feature}_yolo_2d"] = value_yolo
            row[f"{feature}_projected_3d_2d"] = value_projected
            row[f"{feature}_absolute_error"] = abs(value_yolo - value_projected)
            usable &= math.isfinite(value_yolo) and math.isfinite(value_projected)
        for joint, name in JOINT_NAMES.items():
            yolo_xy = sequence.keypoints[frame, joint, :2]
            projected_xy = projected[frame, joint, :2]
            row[f"{name}_yolo_x"] = float(yolo_xy[0])
            row[f"{name}_yolo_y"] = float(yolo_xy[1])
            row[f"{name}_projected_x"] = float(projected_xy[0])
            row[f"{name}_projected_y"] = float(projected_xy[1])
            row[f"{name}_error_ratio"] = float(
                np.linalg.norm(yolo_xy - projected_xy) / scale
            ) if scale > 0 else math.nan
        if usable:
            rows.append(row)
    return rows, {
        "correlation": correlation,
        "median_error_ratio": median_error_ratio,
        "p90_error_ratio": p90_error_ratio,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pose_root", type=Path)
    parser.add_argument("output_csv", type=Path)
    parser.add_argument("--feature-csv", type=Path, nargs="+", required=True)
    parser.add_argument("--archive-root", type=Path, nargs="+", required=True)
    parser.add_argument("--min-alignment-correlation", type=float, default=0.30)
    parser.add_argument("--max-camera-median-error", type=float, default=0.20)
    parser.add_argument("--max-camera-p90-error", type=float, default=0.40)
    parser.add_argument("--max-offset-seconds", type=float, default=0.5)
    parser.add_argument("--frame-stride", type=int, default=3)
    parser.add_argument(
        "--alignment-signal", choices=("wrist", "multi"), default="wrist",
        help="Wrist-only is the validated default; multi is retained for A/B audits",
    )
    parser.add_argument(
        "--camera-model", choices=("weak", "perspective"), default="perspective",
    )
    parser.add_argument(
        "--residual-offset-audit", type=Path,
        help="Training-only projection audit containing residual time offsets",
    )
    parser.add_argument("--min-residual-calibration-clips", type=int, default=10)
    args = parser.parse_args()
    residual_offsets = {}
    if args.residual_offset_audit:
        audit = json.loads(args.residual_offset_audit.read_text(encoding="utf-8"))
        residual_offsets = audit.get("training_residual_offset_calibration", {})

    annotations = read_annotations(args.feature_csv)
    recordings = index_recordings(args.archive_root)
    source_videos = index_source_videos(args.archive_root)
    cache: dict[str, tuple[np.ndarray, dict[str, np.ndarray]]] = {}
    pose_paths = sorted(args.pose_root.rglob("*.npz"))
    metadata_by_path = {
        path: manual_clip_metadata(path, args.pose_root) for path in pose_paths
    }
    metadata_items = [item for item in metadata_by_path.values() if item is not None]

    video_epoch_starts = {}
    for key in sorted({metadata[:3] for metadata in metadata_items}):
        subject, technique, view = key
        video = source_videos.get(key)
        candidates = [
            row for row in annotations
            if row["subject"] == subject and row["technique"] == technique
            and row["recording"] in recordings
        ]
        if video is None or not candidates:
            continue
        recording = candidates[0]["recording"]
        if recording not in cache:
            cache[recording] = load_3d_recording(recordings[recording])
        timestamps, _ = cache[recording]
        prior = float(timestamps[-1]) - video_duration(video)
        intervals = [item[3:] for item in metadata_items if item[:3] == key]
        video_epoch_starts[key] = estimate_video_epoch_start(intervals, candidates, prior)

    rows = []
    skipped: Counter[str] = Counter()
    accepted_clips = 0
    for pose_path, metadata in metadata_by_path.items():
        if metadata is None or metadata[:3] not in video_epoch_starts:
            skipped["metadata_or_video_missing"] += 1
            continue
        matched = match_manual_annotation(
            metadata, annotations, recordings, source_videos, cache,
            video_epoch_starts,
        )
        if matched is None:
            skipped["annotation_not_found"] += 1
            continue
        annotation, base_epoch = matched
        recording = annotation["recording"]
        try:
            calibration = residual_offsets.get(metadata[1], {}).get(metadata[2], {})
            residual_offset = (
                float(calibration.get("residual_offset_seconds", 0.0))
                if int(calibration.get("calibration_clip_count", 0))
                >= args.min_residual_calibration_clips else 0.0
            )
            projected_rows, quality = project_clip(
                pose_path, annotation, base_epoch, metadata[:3],
                *cache[recording], max_offset_seconds=args.max_offset_seconds,
                frame_stride=args.frame_stride,
                alignment_signal=args.alignment_signal,
                residual_offset_seconds=residual_offset,
                camera_model=args.camera_model,
            )
        except (ValueError, np.linalg.LinAlgError):
            skipped["projection_failed"] += 1
            continue
        if quality["correlation"] < args.min_alignment_correlation:
            skipped["low_alignment_correlation"] += 1
            continue
        if (
            quality["median_error_ratio"] > args.max_camera_median_error
            or quality["p90_error_ratio"] > args.max_camera_p90_error
        ):
            skipped["poor_camera_fit"] += 1
            continue
        if not projected_rows:
            skipped["no_valid_frames"] += 1
            continue
        rows.extend(projected_rows)
        accepted_clips += 1

    if not rows:
        raise RuntimeError(f"No projected pairs passed quality gates: {dict(skipped)}")
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.open("w", newline="", encoding="utf-8-sig") as output:
        writer = csv.DictWriter(output, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(
        f"Projected {len(rows)} frames from {accepted_clips} clips; "
        f"skipped {dict(sorted(skipped.items()))}: {args.output_csv}"
    )


if __name__ == "__main__":
    main()
