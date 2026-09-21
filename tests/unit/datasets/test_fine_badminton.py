import csv
from pathlib import Path

import cv2
import numpy as np

from ai_classifier.datasets import (
    decode_virtual_clip,
    load_fine_badminton_manifest,
    sample_frame_indices,
)


def test_uniform_sampling_repeats_short_intervals():
    assert sample_frame_indices(10, 12, 5).tolist() == [10, 10, 11, 12, 12]


def test_manifest_and_virtual_clip_decode(tmp_path: Path):
    video_dir = tmp_path / "videos"
    video_dir.mkdir()
    video = video_dir / "sample.mp4"
    writer = cv2.VideoWriter(
        str(video), cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (32, 24)
    )
    for index in range(8):
        writer.write(np.full((24, 32, 3), index * 20, dtype=np.uint8))
    writer.release()

    manifest = tmp_path / "manifest.csv"
    row = {
        "sample_id": "sample_1",
        "match_id": "match_1",
        "video_relpath": "videos/sample.mp4",
        "start_frame": 2,
        "end_frame": 6,
        "player_side": "upper",
        "raw_label": "Smash(Full-length)",
        "canonical_label": "Smash(Full-length)",
        "coarse_label": "smash",
        "split": "train",
        "fps": 10.0,
        "frame_width": 32,
        "frame_height": 24,
        "valid": True,
    }
    with manifest.open("w", encoding="utf-8", newline="") as handle:
        writer_csv = csv.DictWriter(handle, fieldnames=list(row))
        writer_csv.writeheader()
        writer_csv.writerow(row)

    record = load_fine_badminton_manifest(manifest, split="train")[0]
    clip = decode_virtual_clip(record, tmp_path, frame_count=4)

    assert record.duration_frames == 5
    assert clip.shape == (4, 24, 32, 3)
    assert clip.dtype == np.uint8
    assert float(clip[-1].mean()) > float(clip[0].mean())
