"""Recenter MultiSense trials on 2D racket-arm motion for action training."""

from __future__ import annotations

import argparse
import csv
import subprocess
from pathlib import Path

import numpy as np

from ai_classifier.biomechanics import (
    assess_geometry_quality,
    detect_stroke_phases,
    extract_geometry_features,
)
from ai_classifier.pose import PoseSequence


def load_pose(path: Path) -> PoseSequence:
    with np.load(path) as data:
        return PoseSequence(np.asarray(data["keypoints"], dtype=np.float32), float(data["fps"]),
                            int(data["frame_width"]), int(data["frame_height"]))


def robust_range(values: np.ndarray) -> float:
    finite = values[np.isfinite(values)]
    if not len(finite):
        return 0.0
    low, high = np.percentile(finite, [10, 90])
    return float(high - low)


def stroke_motion_rejection_reason(features) -> str | None:
    """Reject relative pose jitter and stationary preparation as positive strokes."""
    wrist_speed_range = robust_range(features.column("wrist_speed"))
    wrist_excursion = max(
        robust_range(features.column("wrist_height")),
        robust_range(features.column("wrist_shoulder_distance")),
    )
    arm_angle_range = max(
        robust_range(features.column("elbow_angle")),
        robust_range(features.column("shoulder_angle")),
    )
    if wrist_speed_range < 2.0:
        return "insufficient_wrist_speed"
    if wrist_excursion < 0.35:
        return "insufficient_wrist_excursion"
    if arm_angle_range < 25.0:
        return "insufficient_arm_articulation"
    return None


def best_window(sequence: PoseSequence, before: float = 1.4, after: float = 1.2):
    features = extract_geometry_features(sequence)
    speed = features.column("wrist_speed")
    finite = np.isfinite(speed)
    if finite.mean() < 0.9:
        return None, "missing_wrist_speed"
    score = np.where(finite, speed, -np.inf)
    edge_before, edge_after = int(before * sequence.fps), int(after * sequence.fps)
    allowed = np.arange(len(score))
    score[(allowed < edge_before) | (allowed >= len(score) - edge_after)] = -np.inf
    if not np.isfinite(score).any():
        return None, "motion_peak_outside_complete_window"
    peak = int(np.argmax(score))
    start, end = peak - edge_before, peak + edge_after
    clip = PoseSequence(sequence.keypoints[start:end], sequence.fps,
                        sequence.frame_width, sequence.frame_height)
    clip_features = extract_geometry_features(clip)
    quality = assess_geometry_quality(clip)
    if quality.valid_ratio < 0.80:
        return None, "low_pose_valid_ratio"
    if quality.scale_px < 30.0:
        return None, "player_too_small"
    if quality.scale_cv > 0.25:
        return None, "unstable_body_scale"
    if quality.speed_spike_ratio > 3.0:
        return None, "keypoint_speed_spike"
    if not detect_stroke_phases(clip_features).valid:
        return None, "incomplete_stroke_phases"
    motion_reason = stroke_motion_rejection_reason(clip_features)
    if motion_reason:
        return None, motion_reason
    return (start, end, peak, clip), None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("candidate_root", type=Path)
    parser.add_argument("output_root", type=Path)
    parser.add_argument("--max-per-subject-class", type=int, default=20)
    args = parser.parse_args()
    rows = []
    rejected = []
    for technique in ("backhand_drive", "forehand_clear"):
        grouped = {}
        for pose_path in sorted((args.candidate_root / technique).rglob("*_pose.npz")):
            subject = pose_path.parent.name
            grouped.setdefault(subject, []).append(pose_path)
        for subject, paths in grouped.items():
            accepted = []
            for pose_path in paths:
                result, reason = best_window(load_pose(pose_path))
                if result is not None:
                    accepted.append((pose_path, result))
                else:
                    rejected.append({
                        "pose": str(pose_path), "label": technique,
                        "subject": subject, "reason": reason,
                    })
            if len(accepted) > args.max_per_subject_class:
                indices = np.linspace(0, len(accepted)-1, args.max_per_subject_class, dtype=int)
                accepted = [accepted[i] for i in indices]
            for pose_path, (start, end, peak, clip) in accepted:
                video = pose_path.with_name(pose_path.name.removesuffix("_pose.npz") + ".mp4")
                view = pose_path.parent.parent.name
                destination_dir = args.output_root / technique / view / subject
                destination_dir.mkdir(parents=True, exist_ok=True)
                stem = video.stem
                output_video = destination_dir / f"{stem}_action.mp4"
                output_pose = destination_dir / f"{stem}_action.npz"
                subprocess.run(["ffmpeg", "-y", "-v", "error", "-ss", f"{start/clip.fps:.3f}",
                                "-i", str(video), "-t", f"{(end-start)/clip.fps:.3f}", "-an",
                                "-c:v", "copy", str(output_video)], check=True)
                clip.save(output_pose)
                rows.append({"video": str(output_video), "pose": str(output_pose), "label": technique,
                             "subject": subject, "view": view, "source": str(video),
                             "source_start_frame": start, "motion_peak_frame": peak})
    manifest = args.output_root / "positive_manifest.csv"
    with manifest.open("w", newline="", encoding="utf-8-sig") as target:
        fields = (
            "video", "pose", "label", "subject", "view", "source",
            "source_start_frame", "motion_peak_frame",
        )
        writer = csv.DictWriter(target, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    rejection_report = args.output_root / "rejected_positive_candidates.csv"
    with rejection_report.open("w", newline="", encoding="utf-8-sig") as target:
        writer = csv.DictWriter(
            target, fieldnames=("pose", "label", "subject", "reason")
        )
        writer.writeheader()
        writer.writerows(rejected)
    print(f"Materialized {len(rows)} positive clips to {args.output_root}")
    print(f"Rejected {len(rejected)} candidates; report: {rejection_report}")


if __name__ == "__main__":
    main()
