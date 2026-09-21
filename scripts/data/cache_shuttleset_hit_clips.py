"""Sequentially cache RGB windows for the ShuttleSet hit verifier."""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from torchvision.transforms import functional as VF

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.training.train_shuttleset_rgb_hit_detector import build_samples, load_matches


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path("data/manifests/shuttleset_rgb.csv"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "val"), required=True)
    parser.add_argument("--match-ids", nargs="+", required=True)
    parser.add_argument("--frames", type=int, default=16)
    parser.add_argument("--positive-radius", type=int, default=2)
    parser.add_argument("--negative-exclusion", type=int, default=12)
    parser.add_argument("--positives-per-match", type=int, default=200)
    parser.add_argument("--negative-ratio", type=float, default=2.0)
    parser.add_argument("--seed", type=int, default=20260917)
    return parser.parse_args()


def cache_video(video_path: Path, samples, output: Path, frames: int) -> int:
    needed = sorted({sample.center for sample in samples})
    if not needed:
        return 0
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(video_path)
    half = frames // 2
    wanted = {center: [] for center in needed}
    frame_index = 0
    active = {center: [] for center in needed}
    try:
        while active:
            ok, frame = cap.read()
            if not ok:
                break
            rgb = None
            finished = []
            for center, clip in active.items():
                if max(0, center - half) <= frame_index <= center + half:
                    if rgb is None:
                        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    clip.append(rgb)
                if frame_index >= center + half:
                    finished.append(center)
            for center in finished:
                wanted[center] = active.pop(center)
            frame_index += 1
    finally:
        cap.release()
    written = 0
    for sample in samples:
        clip = wanted.get(sample.center, [])
        if not clip:
            continue
        while len(clip) < frames:
            clip.append(clip[-1])
        tensor = torch.from_numpy(np.stack(clip[:frames])).permute(0, 3, 1, 2)
        tensor = VF.resize(tensor, [128, 171], antialias=False)
        tensor = VF.center_crop(tensor, [112, 112]).contiguous().numpy()
        path = output / f"{sample.match_id}_{sample.center}_{frames}.npy"
        path.parent.mkdir(parents=True, exist_ok=True)
        np.save(path, tensor)
        written += 1
    return written


def main() -> None:
    args = parse_args()
    matches = load_matches(args.manifest, args.split)
    rng = random.Random(args.seed)
    samples = build_samples(
        {key: matches[key] for key in args.match_ids},
        args.positives_per_match,
        args.negative_ratio,
        args.negative_exclusion,
        args.positive_radius if args.split == "train" else 0,
        args.seed,
    )
    grouped = {}
    for sample in samples:
        grouped.setdefault(sample.video_path, []).append(sample)
    total = 0
    for video_path, video_samples in grouped.items():
        count = cache_video(video_path, video_samples, args.output, args.frames)
        total += count
        print(f"cached video={video_path.name} samples={count}", flush=True)
    print(f"cached_total={total}", flush=True)


if __name__ == "__main__":
    main()
