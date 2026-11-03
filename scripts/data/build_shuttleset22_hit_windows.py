"""Cut HitNet training windows from ShuttleSet22 videos and write a combined hit manifest.

Windows use the hit scanner's preprocessing (squash to 171x128, centre crop 112x112,
16 frames starting at centre-8) and the window-cache naming of
train_badminton_hitnet_v2.py (<split>/<match>_<centre>_16.npy), so the trainer labels
them from the manifest: within 2 frames of a hit -> that hitter's class, more than 12
frames from every hit -> no_hit. Each video is decoded once, forward.

    python scripts/data/build_shuttleset22_hit_windows.py --root C:/Users/ADMIN/Downloads/shuttleset22
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from pathlib import Path

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO_ROOT), str(REPO_ROOT / "src")]

from scripts.data.extract_shuttleset22_records import court_corners, labeled_strokes  # noqa: E402

FRAMES = 16
MIN_GAP = FRAMES // 2 + 4  # same no_hit exclusion as the trainer


def windows_for_match(hits: list[int], rng: random.Random, negatives_per_hit: float) -> list[int]:
    """One jittered positive per hit (offset learning) plus random clear negatives."""
    centres = [hit + rng.randint(-2, 2) for hit in hits]
    lo, hi = min(hits) - 150, max(hits) + 150
    wanted = int(len(hits) * negatives_per_hit)
    negatives: set[int] = set()
    for _ in range(wanted * 20):
        if len(negatives) >= wanted:
            break
        centre = rng.randint(lo, hi)
        if all(abs(centre - hit) > MIN_GAP for hit in hits):
            negatives.add(centre)
    return sorted(set(centres) | negatives)


def cut_windows(video: Path, centres: list[int], out_dir: Path, match_id: str) -> int:
    cap = cv2.VideoCapture(str(video))
    targets = {centre - FRAMES // 2: centre for centre in centres}  # window start -> centre
    starts = sorted(targets)
    recent: list[np.ndarray] = []
    written, frame_index, next_start = 0, 0, 0
    last_needed = starts[-1] + FRAMES - 1
    while frame_index <= last_needed:
        ok, frame = cap.read()
        if not ok:
            break
        resized = cv2.resize(frame, (171, 128), interpolation=cv2.INTER_LINEAR)
        recent.append(cv2.cvtColor(resized[8:120, 30:142], cv2.COLOR_BGR2RGB))
        if len(recent) > FRAMES:
            recent.pop(0)
        while next_start < len(starts) and starts[next_start] + FRAMES - 1 == frame_index:
            if len(recent) == FRAMES:
                np.save(out_dir / f"{match_id}_{targets[starts[next_start]]}_{FRAMES}.npy", np.stack(recent))
                written += 1
            next_start += 1
        frame_index += 1
    cap.release()
    return written


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--splits", type=Path, default=REPO_ROOT / "data/manifests/shuttleset22_splits.json")
    parser.add_argument("--cache", type=Path,
                        default=Path(r"C:\Users\ADMIN\Downloads\AI_Classifier_cache\shuttleset_hit_full"))
    parser.add_argument("--base-manifest", type=Path, default=REPO_ROOT / "data/manifests/shuttleset_rgb.csv")
    parser.add_argument("--out-manifest", type=Path,
                        default=REPO_ROOT / "data/manifests/hit_shuttleset_plus_ss22.csv")
    parser.add_argument("--negatives-per-hit", type=float, default=0.7)
    args = parser.parse_args()
    splits = json.loads(args.splits.read_text(encoding="utf-8"))
    downloads = {item["video"]: item for item in json.loads((args.root / "download_list.json").read_text(encoding="utf-8"))}
    rows = list(csv.DictReader(args.base_manifest.open(encoding="utf-8-sig", newline="")))
    fields = list(rows[0].keys())
    rng = random.Random(22)
    for split in ("train", "val"):
        (args.cache / split).mkdir(parents=True, exist_ok=True)
        for index, video in enumerate(splits[split]):
            match_id = f"s22{split[0]}{index:02d}"  # no underscores: the trainer splits file stems on "_"
            video_path = args.root / "videos" / f"{downloads[video]['id']}.mp4"
            strokes = labeled_strokes(args.root, video, court_corners(args.root, video))
            for stroke in strokes:
                rows.append({**{key: "" for key in fields}, "sample_id": f"{match_id}_{stroke['hit_frame']}",
                             "dataset": "shuttleset22", "split": split, "match_id": match_id,
                             "video_path": str(video_path), "hit_frame": stroke["hit_frame"],
                             "player_side": "top" if stroke["side"] == "upper" else "bottom"})
            if any((args.cache / split).glob(f"{match_id}_*_{FRAMES}.npy")):
                print(f"[skip] {match_id} {video[:50]} windows exist", flush=True)
                continue
            centres = windows_for_match([s["hit_frame"] for s in strokes], rng, args.negatives_per_hit)
            written = cut_windows(video_path, centres, args.cache / split, match_id)
            print(f"[+] {match_id} {video[:50]}: {len(strokes)} hits, {written} windows", flush=True)
    with args.out_manifest.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(f"[+] manifest {args.out_manifest} ({len(rows)} rows)", flush=True)


if __name__ == "__main__":
    main()
