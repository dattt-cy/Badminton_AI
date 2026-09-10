"""Create isolated start-at-swing variants of the manual action clips."""

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
from ai_classifier.segmentation import motion_score


VIDEO_EXTENSIONS = {".avi", ".mkv", ".mov", ".mp4"}


@dataclass(frozen=True)
class AugmentationJob:
    source_video: Path
    source_pose: Path
    relative_path: Path
    split: str
    action: str
    view: str
    subject: str
    start_frame: int
    peak_frame: int
    total_frames: int
    fps: float


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True


def validate_output_roots(
    source_root: Path, video_output_root: Path, pose_output_root: Path
) -> None:
    """Prevent an augmentation run from writing into the source dataset."""
    source = source_root.resolve()
    video_output = video_output_root.resolve()
    pose_output = pose_output_root.resolve()
    for output in (video_output, pose_output):
        if output == source or _is_relative_to(output, source):
            raise ValueError(f"Output root must be outside the source dataset: {output}")
    if video_output == pose_output:
        raise ValueError("Video and pose output roots must be different")


def load_pose(path: Path) -> PoseSequence:
    with np.load(path) as data:
        return PoseSequence(
            keypoints=np.asarray(data["keypoints"], dtype=np.float32),
            fps=float(data["fps"]),
            frame_width=int(data["frame_width"]),
            frame_height=int(data["frame_height"]),
        )


def swing_peak(sequence: PoseSequence) -> int | None:
    """Use the same racket-arm motion score as aligned inference."""
    if len(sequence.keypoints) < 2:
        return None
    scores = motion_score(extract_geometry_features(sequence))
    if not len(scores) or float(scores.max()) <= 0:
        return None
    return int(np.argmax(scores))


def discover_candidates(
    source_root: Path,
    source_pose_root: Path,
    *,
    pre_roll_seconds: float,
    min_output_seconds: float,
    min_peak_ratio: float,
    max_peak_ratio: float,
) -> list[AugmentationJob]:
    candidates: list[AugmentationJob] = []
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
        peak = swing_peak(sequence)
        total = len(sequence.keypoints)
        if peak is None or total < 2:
            continue
        peak_ratio = peak / (total - 1)
        start = max(0, peak - round(pre_roll_seconds * sequence.fps))
        min_output_frames = max(2, round(min_output_seconds * sequence.fps))
        if (
            peak_ratio < min_peak_ratio
            or peak_ratio > max_peak_ratio
            or total - start < min_output_frames
        ):
            continue
        candidates.append(
            AugmentationJob(
                source_video=video,
                source_pose=pose_path,
                relative_path=relative,
                split=split,
                action=action,
                view=view,
                subject=subject,
                start_frame=start,
                peak_frame=peak,
                total_frames=total,
                fps=sequence.fps,
            )
        )
    return candidates


def select_balanced_jobs(
    candidates: list[AugmentationJob], ratio: float, seed: int
) -> list[AugmentationJob]:
    """Select the same fraction independently for every subject/class/view."""
    if not 0 < ratio <= 1:
        raise ValueError("ratio must be in (0, 1]")
    grouped: dict[tuple[str, str, str, str], list[AugmentationJob]] = defaultdict(list)
    for job in candidates:
        grouped[(job.split, job.action, job.view, job.subject)].append(job)

    selected: list[AugmentationJob] = []
    for group in grouped.values():
        ordered = sorted(
            group,
            key=lambda job: hashlib.sha256(
                f"{seed}:{job.relative_path.as_posix()}".encode("utf-8")
            ).digest(),
        )
        count = max(1, round(len(ordered) * ratio))
        selected.extend(ordered[:count])
    return sorted(selected, key=lambda job: job.relative_path.as_posix())


def output_relative_path(relative: Path) -> Path:
    return relative.with_name(f"{relative.stem}__start_at_swing.mp4")


def materialize_job(
    job: AugmentationJob,
    video_output_root: Path,
    pose_output_root: Path,
    *,
    overwrite: bool,
) -> tuple[Path, Path]:
    relative_output = output_relative_path(job.relative_path)
    video_output = video_output_root / relative_output
    pose_output = (pose_output_root / relative_output).with_suffix(".npz")
    if not overwrite and (video_output.exists() or pose_output.exists()):
        raise FileExistsError(
            f"Output already exists; use --overwrite to replace it: {video_output}"
        )

    video_output.parent.mkdir(parents=True, exist_ok=True)
    pose_output.parent.mkdir(parents=True, exist_ok=True)
    start_seconds = job.start_frame / job.fps
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-v",
            "error",
            "-ss",
            f"{start_seconds:.6f}",
            "-i",
            str(job.source_video),
            "-map",
            "0:v:0",
            "-an",
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-crf",
            "20",
            "-movflags",
            "+faststart",
            str(video_output),
        ],
        check=True,
    )
    source_sequence = load_pose(job.source_pose)
    PoseSequence(
        keypoints=source_sequence.keypoints[job.start_frame :].copy(),
        fps=source_sequence.fps,
        frame_width=source_sequence.frame_width,
        frame_height=source_sequence.frame_height,
    ).save(pose_output)
    return video_output, pose_output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=Path("data/manual_clips"))
    parser.add_argument(
        "--source-pose-root",
        type=Path,
        default=Path("data/processed/manual_clips_2class_poses"),
    )
    parser.add_argument(
        "--video-output-root",
        type=Path,
        default=Path("data/manual_clips_temporal_augmented"),
    )
    parser.add_argument(
        "--pose-output-root",
        type=Path,
        default=Path("data/processed/manual_clips_2class_temporal_augmented_poses"),
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=1.0,
        help="Fraction of eligible train clips to augment (default: 1.0)",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.30,
        help="Fraction of eligible validation clips to augment (default: 0.30)",
    )
    parser.add_argument("--seed", type=int, default=20260909)
    parser.add_argument("--pre-roll-seconds", type=float, default=0.10)
    parser.add_argument("--min-output-seconds", type=float, default=1.20)
    parser.add_argument("--min-peak-ratio", type=float, default=0.25)
    parser.add_argument("--max-peak-ratio", type=float, default=0.75)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if args.pre_roll_seconds < 0 or args.min_output_seconds <= 0:
        raise ValueError("pre-roll must be non-negative and min-output must be positive")
    if not 0 <= args.min_peak_ratio < args.max_peak_ratio <= 1:
        raise ValueError("peak ratios must satisfy 0 <= min < max <= 1")
    validate_output_roots(
        args.source_root, args.video_output_root, args.pose_output_root
    )
    candidates = discover_candidates(
        args.source_root,
        args.source_pose_root,
        pre_roll_seconds=args.pre_roll_seconds,
        min_output_seconds=args.min_output_seconds,
        min_peak_ratio=args.min_peak_ratio,
        max_peak_ratio=args.max_peak_ratio,
    )
    jobs: list[AugmentationJob] = []
    for split, ratio in (("train", args.train_ratio), ("val", args.val_ratio)):
        split_candidates = [job for job in candidates if job.split == split]
        if split_candidates:
            jobs.extend(select_balanced_jobs(split_candidates, ratio, args.seed))
    jobs.sort(key=lambda job: job.relative_path.as_posix())
    print(f"Eligible clips: {len(candidates)}; selected variants: {len(jobs)}")

    rows: list[dict[str, str | int | float]] = []
    for index, job in enumerate(jobs, start=1):
        video_output, pose_output = materialize_job(
            job,
            args.video_output_root,
            args.pose_output_root,
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
                "source_frames": job.total_frames,
                "start_frame": job.start_frame,
                "motion_peak_frame": job.peak_frame,
                "output_frames": job.total_frames - job.start_frame,
                "peak_position_in_output": (
                    job.peak_frame - job.start_frame
                ) / max(1, job.total_frames - job.start_frame - 1),
            }
        )
        print(f"[{index}/{len(jobs)}] {video_output}")

    manifest = args.video_output_root / "augmentation_manifest.csv"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0]) if rows else [
        "source_video", "output_video", "output_pose", "split", "class",
        "view", "subject", "source_frames", "start_frame",
        "motion_peak_frame", "output_frames", "peak_position_in_output",
    ]
    with manifest.open("w", newline="", encoding="utf-8") as target:
        writer = csv.DictWriter(target, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Created {len(rows)} isolated variants; manifest: {manifest}")


if __name__ == "__main__":
    main()
