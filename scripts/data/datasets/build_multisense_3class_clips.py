"""Build synchronized background/backhand/forehand-clear clips from MultiSense."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
from pathlib import Path

import cv2
import h5py
import numpy as np

from cut_multisense_reference_clips import crop_filter, parse_crop, video_duration


LABELS = {"Backhand Driving": "backhand_drive", "Forehand Clear": "forehand_clear"}
VIDEO_NAMES = {"backhand_drive": "Backhand Drive", "forehand_clear": "Forehand Clear"}


def motion_signal(video: Path, crop, *, sample_fps: float = 10.0):
    capture = cv2.VideoCapture(str(video))
    source_fps = float(capture.get(cv2.CAP_PROP_FPS))
    stride = max(1, int(round(source_fps / sample_fps)))
    times, scores, previous = [], [], None
    frame_index = 0
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            if frame_index % stride:
                frame_index += 1
                continue
            if crop is not None:
                x, y, width, height = crop
                h, w = frame.shape[:2]
                frame = frame[int(y*h):int((y+height)*h), int(x*w):int((x+width)*w)]
            gray = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (120, 160))
            score = 0.0 if previous is None else float(np.mean(cv2.absdiff(gray, previous)))
            times.append(frame_index / source_fps)
            scores.append(score)
            previous = gray
            frame_index += 1
    finally:
        capture.release()
    values = np.asarray(scores, dtype=np.float32)
    if len(values) >= 5:
        values = np.convolve(values, np.ones(5, dtype=np.float32) / 5, mode="same")
    return np.asarray(times), values


def estimate_video_epoch_start(
    times: np.ndarray, motion: np.ndarray, contacts_epoch: np.ndarray,
    initial_start: float, *, search_seconds: float = 8.0,
) -> tuple[float, float]:
    """Align synchronized PNS contacts to video motion peaks around an initial offset."""
    initial_video_times = contacts_epoch - initial_start
    overlap = (initial_video_times >= search_seconds) & (
        initial_video_times <= times[-1] - search_seconds
    )
    contacts_epoch = contacts_epoch[overlap]
    if len(contacts_epoch) < 10:
        raise ValueError("Too few PNS contacts overlap the video for synchronization")
    deltas = np.arange(-search_seconds, search_seconds + 0.025, 0.025)
    objectives = []
    for delta in deltas:
        video_contacts = contacts_epoch - (initial_start + delta)
        valid = (video_contacts >= times[0]) & (video_contacts <= times[-1])
        if valid.mean() < 0.95:
            objectives.append(-np.inf)
            continue
        sampled = np.interp(video_contacts[valid], times, motion)
        objectives.append(float(np.median(sampled) + 0.25 * np.mean(sampled)))
    best = int(np.argmax(objectives))
    if best in {0, len(deltas) - 1} or not np.isfinite(objectives[best]):
        raise RuntimeError("Video/PNS synchronization has no reliable interior optimum")
    return initial_start + float(deltas[best]), float(objectives[best])


def read_rows(feature_csv: Path, subject: str, technique: str):
    with feature_csv.open(newline="", encoding="utf-8-sig") as source:
        return [
            row for row in csv.DictReader(source)
            if row["subject"] == subject and row["technique"] == technique
            and row["phase_valid"].lower() == "true"
        ]


def hdf_end(subject_dir: Path, rows) -> float:
    start, stop = min(float(r["start_time"]) for r in rows), max(float(r["stop_time"]) for r in rows)
    for path in subject_dir.glob("*.hdf5"):
        with h5py.File(path, "r") as source:
            t = np.asarray(source["pns-joint"]["global-position"]["time_s"]).reshape(-1)
        if float(t[0]) <= stop and float(t[-1]) >= start:
            return float(t[-1])
    raise FileNotFoundError("No matching HDF5 recording")


def spread(items, limit):
    if len(items) <= limit:
        return items
    return [items[i] for i in np.linspace(0, len(items)-1, limit, dtype=int)]


def refine_contact_time(expected: float, times: np.ndarray, motion: np.ndarray) -> float:
    """Snap a synchronized PNS candidate to the strongest local video motion."""
    local = (times >= expected - 1.2) & (times <= expected + 1.2)
    if not local.any():
        return expected
    local_times, local_motion = times[local], motion[local]
    return float(local_times[int(np.argmax(local_motion))])


def cut(video, destination, start, duration, crop, output_height):
    command = ["ffmpeg", "-y", "-v", "error", "-ss", f"{start:.3f}", "-i", str(video),
               "-t", f"{duration:.3f}", "-an"]
    if crop:
        command += ["-vf", crop_filter(crop, output_height)]
    command += ["-c:v", "libx264", "-preset", "veryfast", "-crf", "24", str(destination)]
    subprocess.run(command, check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive_root", type=Path)
    parser.add_argument("feature_csv", type=Path)
    parser.add_argument("output_root", type=Path)
    parser.add_argument("--subject", required=True)
    parser.add_argument("--technique", choices=("backhand_drive", "forehand_clear"), required=True)
    parser.add_argument("--view", choices=("front", "side"), default="front")
    parser.add_argument("--positive-count", type=int, default=20)
    parser.add_argument("--background-count", type=int, default=20)
    parser.add_argument("--crop", default="0.25,0.10,0.50,0.85")
    parser.add_argument("--output-height", type=int, default=960)
    args = parser.parse_args()
    crop = parse_crop(args.crop)
    rows = read_rows(args.feature_csv, args.subject, args.technique)
    subject_dir = args.archive_root / args.subject
    video = subject_dir / f"{VIDEO_NAMES[args.technique]} {args.view.title()} Video.mov"
    duration = video_duration(video)
    contacts = np.asarray([
        float(row["start_time"]) + float(row["contact_candidate_seconds"]) for row in rows
    ])
    initial = hdf_end(subject_dir, rows) - duration
    times, motion = motion_signal(video, crop)
    video_start, alignment_score = estimate_video_epoch_start(times, motion, contacts, initial)

    video_contacts = contacts - video_start
    refined_contacts = np.asarray([
        refine_contact_time(expected, times, motion) for expected in video_contacts
    ])
    positive = []
    for row, time, expected in zip(rows, refined_contacts, video_contacts):
        if 1.4 <= time <= duration - 1.2 and row["hitting_sound"].lower() == "good":
            positive.append((row, time, expected))
    positive = spread(positive, args.positive_count)

    contact_times = np.sort(refined_contacts)
    contact_times = contact_times[(contact_times >= 0) & (contact_times <= duration)]
    backgrounds = []
    for first, second in zip(contact_times, contact_times[1:]):
        midpoint = (first + second) / 2
        if second - first >= 5.6 and midpoint - 1.3 >= first + 1.5 and midpoint + 1.3 <= second - 1.5:
            backgrounds.append(midpoint)
    backgrounds = spread(backgrounds, args.background_count)

    manifest = []
    for label, selected in ((args.technique, positive), ("background", backgrounds)):
        output = args.output_root / label / args.view / args.subject
        output.mkdir(parents=True, exist_ok=True)
        if label == "background":
            specs = [(f"gap_{index:03d}", time - 1.3, time) for index, time in enumerate(selected, 1)]
        else:
            specs = [(f"stroke_{int(row['stroke_number']):03d}", time - 1.4, time, expected)
                     for row, time, expected in selected]
        if label == "background":
            specs = [(name, start, center, center) for name, start, center in specs]
        for name, start, center, expected in specs:
            destination = output / f"{args.subject}_{args.technique}_{args.view}_{name}.mp4"
            cut(video, destination, start, 2.6, crop, args.output_height)
            manifest.append({"clip": str(destination), "label": label, "subject": args.subject,
                             "source_technique": args.technique, "view": args.view,
                             "video_start_seconds": start, "expected_center_seconds": center,
                             "pns_contact_seconds": expected,
                             "local_refinement_seconds": center - expected,
                             "sync_offset_from_end_alignment": video_start-initial})
    manifest_path = args.output_root / f"manifest_{args.subject}_{args.technique}_{args.view}.csv"
    with manifest_path.open("w", newline="", encoding="utf-8-sig") as target:
        writer = csv.DictWriter(target, fieldnames=list(manifest[0])); writer.writeheader(); writer.writerows(manifest)
    sync_path = args.output_root / f"sync_{args.subject}_{args.technique}_{args.view}.json"
    sync_path.write_text(json.dumps({"initial_epoch_start": initial, "video_epoch_start": video_start,
                                    "offset_seconds": video_start-initial,
                                    "alignment_score": alignment_score}, indent=2)+"\n")
    print(f"sync offset={video_start-initial:+.3f}s; positive={len(positive)} background={len(backgrounds)}")


if __name__ == "__main__":
    main()
