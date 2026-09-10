"""Create balanced, fixed-length motion-onset variants of manual clips."""

from __future__ import annotations

import argparse
import csv
import hashlib
import subprocess
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ai_classifier.biomechanics import extract_geometry_features
from ai_classifier.pose import PoseSequence
from ai_classifier.segmentation import find_motion_proposals

try:
    from scripts.data.augment_manual_clips_temporal import (
        VIDEO_EXTENSIONS,
        load_pose,
        validate_output_roots,
    )
except ImportError:  # Direct execution adds scripts/data rather than the repo root.
    from augment_manual_clips_temporal import (
        VIDEO_EXTENSIONS,
        load_pose,
        validate_output_roots,
    )


@dataclass(frozen=True)
class MotionOnsetJob:
    source_video: Path
    source_pose: Path
    relative_path: Path
    split: str
    action: str
    view: str
    subject: str
    start_frame: int
    end_frame: int
    peak_frame: int
    fps: float


def discover_motion_onset_candidates(
    source_root: Path,
    source_pose_root: Path,
    *,
    window_seconds: float = 1.6,
    min_peak_position: float = 0.10,
    max_peak_position: float = 0.40,
    min_detected_ratio: float = 0.80,
    min_mean_confidence: float = 0.30,
) -> list[MotionOnsetJob]:
    candidates: list[MotionOnsetJob] = []
    for video in sorted(source_root.rglob("*")):
        if not video.is_file() or video.suffix.lower() not in VIDEO_EXTENSIONS:
            continue
        relative = video.relative_to(source_root)
        if len(relative.parts) < 5:
            continue
        split, action, view, subject = relative.parts[:4]
        pose_path = (source_pose_root / relative).with_suffix(".npz")
        if not pose_path.is_file():
            continue
        sequence = load_pose(pose_path)
        if sequence.fps <= 0:
            continue
        proposals = find_motion_proposals(
            extract_geometry_features(sequence),
            fps=sequence.fps,
            min_duration_seconds=0.10,
            padding_seconds=0.30,
        )
        if not proposals:
            continue
        proposal = max(proposals, key=lambda item: item.peak_score)
        window_frames = max(2, round(window_seconds * sequence.fps))
        start = proposal.start_frame
        end = start + window_frames
        peak_position = (proposal.peak_frame - start) / window_frames
        if (
            end > len(sequence.keypoints)
            or peak_position < min_peak_position
            or peak_position > max_peak_position
        ):
            continue
        sampled = resample_pose(sequence.keypoints[start:end], 64)
        confidence = np.clip(sampled[..., 2], 0.0, 1.0)
        detected_ratio = float((confidence.max(axis=1) > 0).mean())
        mean_confidence = float(confidence.mean())
        if (
            detected_ratio < min_detected_ratio
            or mean_confidence < min_mean_confidence
        ):
            continue
        candidates.append(
            MotionOnsetJob(
                video,
                pose_path,
                relative,
                split,
                action,
                view,
                subject,
                start,
                end,
                proposal.peak_frame,
                sequence.fps,
            )
        )
    return candidates


def select_paired_jobs(
    candidates: list[MotionOnsetJob],
    *,
    split_ratios: dict[str, float],
    seed: int,
) -> list[MotionOnsetJob]:
    """Select equal class counts within every split/view/subject group."""
    for split, ratio in split_ratios.items():
        if not 0 < ratio <= 1:
            raise ValueError(f"Ratio for {split} must be in (0, 1]")
    grouped: dict[tuple[str, str, str], dict[str, list[MotionOnsetJob]]] = (
        defaultdict(lambda: defaultdict(list))
    )
    for job in candidates:
        grouped[(job.split, job.view, job.subject)][job.action].append(job)

    selected: list[MotionOnsetJob] = []
    for (split, _view, _subject), by_action in grouped.items():
        if split not in split_ratios:
            continue
        pair_count = min(
            len(by_action.get("backhand_drive", [])),
            len(by_action.get("forehand_clear", [])),
        )
        if not pair_count:
            continue
        count = max(1, round(pair_count * split_ratios[split]))
        for action in ("backhand_drive", "forehand_clear"):
            ordered = sorted(
                by_action[action],
                key=lambda job: hashlib.sha256(
                    f"{seed}:{job.relative_path.as_posix()}".encode("utf-8")
                ).digest(),
            )
            selected.extend(ordered[:count])
    return sorted(selected, key=lambda job: job.relative_path.as_posix())


def resample_pose(keypoints: np.ndarray, target_frames: int = 64) -> np.ndarray:
    if len(keypoints) < 2:
        raise ValueError("At least two source frames are required")
    if target_frames <= 0:
        raise ValueError("target_frames must be positive")
    indices = np.linspace(0, len(keypoints) - 1, target_frames)
    lower = np.floor(indices).astype(int)
    upper = np.ceil(indices).astype(int)
    weight = (indices - lower).astype(np.float32)[:, None, None]
    return (
        keypoints[lower] * (1.0 - weight) + keypoints[upper] * weight
    ).astype(np.float32)


def output_relative_path(relative: Path) -> Path:
    return relative.with_name(f"{relative.stem}__motion_onset.mp4")


def materialize(
    job: MotionOnsetJob,
    video_output_root: Path,
    pose_output_root: Path,
    *,
    target_frames: int,
    overwrite: bool,
) -> tuple[Path, Path]:
    relative = output_relative_path(job.relative_path)
    video_output = video_output_root / relative
    pose_output = (pose_output_root / relative).with_suffix(".npz")
    if not overwrite and (video_output.exists() or pose_output.exists()):
        raise FileExistsError(f"Output exists; use --overwrite: {video_output}")
    video_output.parent.mkdir(parents=True, exist_ok=True)
    pose_output.parent.mkdir(parents=True, exist_ok=True)
    duration = (job.end_frame - job.start_frame) / job.fps
    subprocess.run(
        [
            "ffmpeg", "-y", "-v", "error",
            "-ss", f"{job.start_frame / job.fps:.6f}",
            "-i", str(job.source_video),
            "-t", f"{duration:.6f}",
            "-map", "0:v:0", "-an",
            "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
            "-movflags", "+faststart", str(video_output),
        ],
        check=True,
    )
    source = load_pose(job.source_pose)
    sampled = resample_pose(source.keypoints[job.start_frame : job.end_frame], target_frames)
    PoseSequence(
        sampled,
        target_frames / duration,
        source.frame_width,
        source.frame_height,
    ).save(pose_output)
    return video_output, pose_output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=Path("data/manual_clips"))
    parser.add_argument(
        "--source-pose-root", type=Path,
        default=Path("data/processed/manual_clips_2class_poses"),
    )
    parser.add_argument(
        "--video-output-root", type=Path,
        default=Path("data/manual_clips_motion_onset_balanced"),
    )
    parser.add_argument(
        "--pose-output-root", type=Path,
        default=Path("data/processed/manual_clips_2class_motion_onset_balanced_poses"),
    )
    parser.add_argument("--window-seconds", type=float, default=1.6)
    parser.add_argument("--target-frames", type=int, default=64)
    parser.add_argument("--train-ratio", type=float, default=0.75)
    parser.add_argument("--val-ratio", type=float, default=0.50)
    parser.add_argument("--seed", type=int, default=20260910)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.window_seconds <= 0 or args.target_frames <= 0:
        raise ValueError("window-seconds and target-frames must be positive")
    validate_output_roots(args.source_root, args.video_output_root, args.pose_output_root)
    candidates = discover_motion_onset_candidates(
        args.source_root,
        args.source_pose_root,
        window_seconds=args.window_seconds,
    )
    jobs = select_paired_jobs(
        candidates,
        split_ratios={"train": args.train_ratio, "val": args.val_ratio},
        seed=args.seed,
    )
    print(f"Eligible={len(candidates)}, selected={len(jobs)}", flush=True)
    rows: list[dict] = []
    for index, job in enumerate(jobs, 1):
        video_output, pose_output = materialize(
            job,
            args.video_output_root,
            args.pose_output_root,
            target_frames=args.target_frames,
            overwrite=args.overwrite,
        )
        rows.append(
            {
                "source_video": str(job.source_video),
                "output_video": str(video_output),
                "output_pose": str(pose_output),
                "split": job.split,
                "class": job.action,
                "view": job.view,
                "subject": job.subject,
                "source_start_frame": job.start_frame,
                "source_end_frame": job.end_frame,
                "motion_peak_frame": job.peak_frame,
                "peak_position": (job.peak_frame - job.start_frame)
                / (job.end_frame - job.start_frame),
                "output_frames": args.target_frames,
            }
        )
        print(f"[{index}/{len(jobs)}] {video_output}", flush=True)
    manifest = args.video_output_root / "augmentation_manifest.csv"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0]) if rows else [
        "source_video", "output_video", "output_pose", "split", "class",
        "view", "subject", "source_start_frame", "source_end_frame",
        "motion_peak_frame", "peak_position", "output_frames",
    ]
    with manifest.open("w", newline="", encoding="utf-8") as target:
        writer = csv.DictWriter(target, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Created {len(rows)} motion-onset variants; manifest={manifest}")


if __name__ == "__main__":
    main()
