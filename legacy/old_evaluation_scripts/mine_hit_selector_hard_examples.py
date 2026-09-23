"""Mine selector training candidates around labelled ShuttleSet rallies.

Unlike a full-match dense scan, this scans merged ranges around known training
hits. It preserves within-rally positives and hard negatives while avoiding
long breaks, replays, and dead time that dominate full-match runtime.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ai_classifier.localization import HitEvent
from scripts.inference.scan_video_hit_rgb import load_model, scan_video


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path("data/manifests/shuttleset_rgb.csv"))
    parser.add_argument("--checkpoint", type=Path, default=Path("work_dirs/r2plus1d18_hit_full/best.pth"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--match-ids", nargs="+", required=True)
    parser.add_argument("--split", choices=("train", "val"), default="train")
    parser.add_argument("--margin-frames", type=int, default=60)
    parser.add_argument("--stride", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=16)
    return parser.parse_args()


def load_jobs(manifest: Path, split: str, match_ids: set[str]):
    jobs = defaultdict(lambda: {"frames": [], "video": None})
    with manifest.open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            if row["split"] != split or row["match_id"] not in match_ids:
                continue
            jobs[row["match_id"]]["frames"].append(int(row["hit_frame"]))
            jobs[row["match_id"]]["video"] = Path(row["video_path"])
    missing = match_ids - set(jobs)
    if missing:
        raise ValueError(f"Matches not found in split={split}: {sorted(missing)}")
    return jobs


def merged_ranges(frames: list[int], margin: int) -> list[tuple[int, int]]:
    ranges: list[tuple[int, int]] = []
    for frame in sorted(frames):
        start, end = max(0, frame - margin), frame + margin + 1
        if ranges and start <= ranges[-1][1]:
            ranges[-1] = (ranges[-1][0], max(ranges[-1][1], end))
        else:
            ranges.append((start, end))
    return ranges


def main() -> None:
    args = parse_args()
    jobs = load_jobs(args.manifest, args.split, set(args.match_ids))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, checkpoint = load_model(args.checkpoint)
    model.to(device).eval()
    frames_per_clip = int(checkpoint.get("frames", 16))

    for match_id in sorted(jobs, key=int):
        output = args.output_dir / f"match_{match_id}.json"
        if output.is_file():
            print(f"[resume] match {match_id}: {output}", flush=True)
            continue
        job = jobs[match_id]
        ranges = merged_ranges(job["frames"], args.margin_frames)
        coverage = sum(end - start for start, end in ranges)
        print(
            f"[match {match_id}] hits={len(job['frames'])} ranges={len(ranges)} "
            f"frames={coverage} evals~{coverage // args.stride}", flush=True,
        )
        events: list[HitEvent] = []
        for index, (start, end) in enumerate(ranges, 1):
            events.extend(scan_video(
                job["video"], model, frames_per_clip,
                stride=args.stride, batch_size=args.batch_size, device=device,
                use_amp=device.type == "cuda", start_frame=start, max_frames=end,
            ))
            if index % 10 == 0 or index == len(ranges):
                print(f"  ranges {index}/{len(ranges)}", flush=True)
        unique = {(event.frame, event.side): event for event in events}
        payload = [event.__dict__ for event in sorted(unique.values(), key=lambda item: item.frame)]
        temporary = output.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(payload) + "\n", encoding="utf-8")
        temporary.replace(output)
        print(f"[saved] match {match_id}: events={len(payload)} {output}", flush=True)


if __name__ == "__main__":
    main()
