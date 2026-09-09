"""Cut a bounded set of MultiSense reference clips without copying source videos."""

from __future__ import annotations

import argparse
import csv
import subprocess
from pathlib import Path

import cv2
import h5py
import numpy as np


SOURCE_LABELS = {
    "backhand_drive": "Backhand Driving",
    "forehand_clear": "Forehand Clear",
}
VIDEO_LABELS = {
    "backhand_drive": "Backhand Drive",
    "forehand_clear": "Forehand Clear",
}


def parse_crop(value: str | None) -> tuple[float, float, float, float] | None:
    """Parse normalized x,y,width,height crop coordinates."""
    if value is None:
        return None
    try:
        crop = tuple(float(part) for part in value.split(","))
    except ValueError as exc:
        raise ValueError("crop must contain four comma-separated numbers") from exc
    if len(crop) != 4:
        raise ValueError("crop must contain x,y,width,height")
    x, y, width, height = crop
    if min(crop) < 0 or width <= 0 or height <= 0 or x + width > 1 or y + height > 1:
        raise ValueError("crop coordinates must define a box inside normalized [0,1]")
    return x, y, width, height


def crop_filter(crop: tuple[float, float, float, float], output_height: int) -> str:
    x, y, width, height = crop
    return (
        f"crop=trunc(iw*{width}/2)*2:trunc(ih*{height}/2)*2:"
        f"trunc(iw*{x}/2)*2:trunc(ih*{y}/2)*2,"
        f"scale=-2:{output_height}:flags=lanczos"
    )


def video_duration(path: Path) -> float:
    capture = cv2.VideoCapture(str(path))
    try:
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        frames = float(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    finally:
        capture.release()
    if fps <= 0 or frames <= 0:
        raise ValueError(f"Cannot read video duration: {path}")
    return frames / fps


def select_rows(
    rows: list[dict[str, str]], *, subject: str, technique: str, limit: int
) -> list[dict[str, str]]:
    candidates = [
        row for row in rows
        if row["subject"] == subject
        and row["technique"] == technique
        and row["phase_valid"].lower() == "true"
        and row["hitting_sound"].lower() == "good"
    ]
    # Spread samples through the recording instead of taking adjacent repetitions.
    if len(candidates) <= limit:
        return candidates
    indices = np.linspace(0, len(candidates) - 1, limit, dtype=int)
    return [candidates[index] for index in indices]


def matching_hdf5(subject_dir: Path, rows: list[dict[str, str]]) -> tuple[Path, float]:
    start = min(float(row["start_time"]) for row in rows)
    stop = max(float(row["stop_time"]) for row in rows)
    for path in sorted(subject_dir.glob("*.hdf5")):
        with h5py.File(path, "r") as source:
            if "pns-joint" not in source:
                continue
            timestamps = np.asarray(
                source["pns-joint"]["global-position"]["time_s"], dtype=np.float64
            ).reshape(-1)
            first = float(timestamps[0])
            last = float(timestamps[-1])
        if first <= stop and last >= start:
            return path, last
    raise FileNotFoundError("No HDF5 recording overlaps the selected annotations")


def compact_epoch_window(row: dict[str, str], padding: float) -> tuple[float, float]:
    """Convert the PNS-derived compact motion window to epoch seconds."""
    annotation_start = float(row["start_time"])
    start = annotation_start + float(row["phase_window_start_seconds"]) - padding
    stop = annotation_start + float(row["phase_window_end_seconds"]) + padding
    return start, stop


def full_epoch_window(row: dict[str, str], padding: float) -> tuple[float, float]:
    """Return the complete expert annotation window for one stroke trial."""
    return float(row["start_time"]) - padding, float(row["stop_time"]) + padding


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive_root", type=Path)
    parser.add_argument("feature_csv", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--subject", required=True)
    parser.add_argument("--technique", choices=tuple(SOURCE_LABELS), required=True)
    parser.add_argument("--view", choices=("front", "side"), required=True)
    parser.add_argument("--max-clips", type=int, default=20)
    parser.add_argument("--max-output-mb", type=float, default=100.0)
    parser.add_argument("--padding-seconds", type=float, default=0.25)
    parser.add_argument(
        "--window", choices=("full", "compact"), default="full",
        help="Use the full annotated trial for references; compact is contact research only.",
    )
    parser.add_argument(
        "--crop", help="Normalized x,y,width,height crop, for example 0.25,0.10,0.50,0.85"
    )
    parser.add_argument("--output-height", type=int, default=960)
    args = parser.parse_args()
    if args.max_clips <= 0 or args.max_output_mb <= 0 or args.output_height <= 0:
        raise ValueError("max-clips, max-output-mb, and output-height must be positive")
    crop = parse_crop(args.crop)
    window_fn = full_epoch_window if args.window == "full" else compact_epoch_window

    subject_dir = args.archive_root / args.subject
    video = subject_dir / f"{VIDEO_LABELS[args.technique]} {args.view.title()} Video.mov"
    if not video.is_file():
        raise FileNotFoundError(f"Source video not found: {video}")
    with args.feature_csv.open(newline="", encoding="utf-8-sig") as source:
        all_rows = list(csv.DictReader(source))
    rows = select_rows(
        all_rows, subject=args.subject, technique=args.technique, limit=len(all_rows)
    )
    if not rows:
        raise RuntimeError("No phase-valid rows with hitting_sound=Good matched")

    _, hdf_end = matching_hdf5(subject_dir, rows)
    duration = video_duration(video)
    video_epoch_start = hdf_end - duration
    rows = [
        row for row in rows
        if window_fn(row, args.padding_seconds)[0] >= video_epoch_start
        and window_fn(row, args.padding_seconds)[1] <= hdf_end
    ]
    rows = select_rows(
        rows, subject=args.subject, technique=args.technique, limit=args.max_clips
    )
    output = args.output_dir / args.technique / args.view / args.subject
    output.mkdir(parents=True, exist_ok=True)
    manifest = []
    byte_budget = int(args.max_output_mb * 1024 * 1024)
    bytes_written = 0
    for row in rows:
        epoch_start, epoch_stop = window_fn(row, args.padding_seconds)
        start = max(0.0, epoch_start - video_epoch_start)
        stop = min(duration, epoch_stop - video_epoch_start)
        if stop <= start or stop - start > 8.0:
            continue
        name = f"{args.subject}_{args.technique}_{args.view}_{int(row['stroke_number']):03d}.mp4"
        destination = output / name
        command = [
            "ffmpeg", "-y", "-v", "error", "-ss", f"{start:.3f}", "-i", str(video),
            "-t", f"{stop-start:.3f}", "-an",
        ]
        if crop is not None:
            command.extend(["-vf", crop_filter(crop, args.output_height)])
        command.extend([
            "-c:v", "libx264", "-preset", "veryfast", "-crf", "23", str(destination),
        ])
        subprocess.run(command, check=True)
        size = destination.stat().st_size
        if bytes_written + size > byte_budget:
            destination.unlink()
            break
        bytes_written += size
        manifest.append({
            "clip": destination.name, "source_video": str(video),
            "subject": args.subject, "technique": args.technique, "view": args.view,
            "stroke_number": int(row["stroke_number"]), "video_start_seconds": start,
            "video_stop_seconds": stop, "contact_confidence": float(row["contact_confidence"]),
            "crop_normalized": args.crop or "", "output_height": args.output_height if crop else "",
            "window": args.window,
        })

    manifest_path = output / "manifest.csv"
    if manifest:
        with manifest_path.open("w", newline="", encoding="utf-8-sig") as target:
            writer = csv.DictWriter(target, fieldnames=list(manifest[0]))
            writer.writeheader()
            writer.writerows(manifest)
    print(
        f"Cut {len(manifest)} clips ({bytes_written / 1024 / 1024:.1f} MB) to {output}"
    )
    print(f"End-aligned source time: {video_epoch_start:.3f}; manifest: {manifest_path}")


if __name__ == "__main__":
    main()
