"""Add ShuttleSet22 matches to the HitNet window cache, with the detector's own false alarms as negatives.

Each match gets the usual jittered positives and random negatives, plus windows the given
detector scores highly more than the trainer's exclusion gap away from every labeled hit.
A match with existing windows is not cut again, so the script can be rerun after a stop.

    python scripts/data/add_hit_windows_for_matches.py --root C:/Users/ADMIN/Downloads/shuttleset22 \
        --match s22t13=CpeAGY7xwtI=Viktor_AXELSEN_Kodai_NARAOKA_YONEX_French_Open_2022_Semifinals \
        --detector work_dirs/hitnet_soup_7/best.pth --out-manifest data/manifests/hit_plus_french.csv
"""

from __future__ import annotations

import argparse
import csv
import random
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO_ROOT), str(REPO_ROOT / "src")]

from scripts.data.build_shuttleset22_hit_windows import MIN_GAP, cut_windows, windows_for_match  # noqa: E402
from scripts.data.extract_shuttleset22_records import court_corners, labeled_strokes  # noqa: E402
from scripts.inference.scan_video_hit_rgb import load_model, scan_video  # noqa: E402


def false_alarms(video: Path, detector: Path, hits: list[int], min_score: float, limit: int) -> list[int]:
    """Centres the detector scores at least min_score, clear of every hit and inside the labeled span."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, meta = load_model(detector)
    model.to(device).eval()
    raw = scan_video(video, model, int(meta.get("frames", 16)), stride=4, batch_size=8, device=device,
                     use_amp=device.type == "cuda")
    del model
    torch.cuda.empty_cache()
    lo, hi = min(hits) - 150, max(hits) + 150
    alarms = [e for e in raw if e.score >= min_score and lo <= e.frame <= hi
              and all(abs(e.frame - hit) > MIN_GAP for hit in hits)]
    alarms.sort(key=lambda e: -e.score)
    return [e.frame for e in alarms[:limit]]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--match", action="append", required=True, help="match_id=youtube_id=folder; match_id has no underscore")
    parser.add_argument("--detector", type=Path, required=True)
    parser.add_argument("--cache", type=Path,
                        default=Path(r"C:\Users\ADMIN\Downloads\AI_Classifier_cache\shuttleset_hit_full"))
    parser.add_argument("--base-manifest", type=Path, default=REPO_ROOT / "data/manifests/hit_shuttleset_plus_ss22.csv")
    parser.add_argument("--out-manifest", type=Path, required=True)
    parser.add_argument("--negatives-per-hit", type=float, default=0.7)
    parser.add_argument("--alarms-per-hit", type=float, default=1.0)
    parser.add_argument("--min-alarm-score", type=float, default=0.5)
    args = parser.parse_args()
    rows = list(csv.DictReader(args.base_manifest.open(encoding="utf-8-sig", newline="")))
    fields = list(rows[0].keys())
    rng = random.Random(23)
    out_dir = args.cache / "train"
    for spec in args.match:
        match_id, youtube_id, folder = spec.split("=", 2)
        video = args.root / "videos" / f"{youtube_id}.mp4"
        strokes = labeled_strokes(args.root, folder, court_corners(args.root, folder))
        rows.extend({**{key: "" for key in fields}, "sample_id": f"{match_id}_{s['hit_frame']}",
                     "dataset": "shuttleset22", "split": "train", "match_id": match_id, "video_path": str(video),
                     "hit_frame": s["hit_frame"], "player_side": "top" if s["side"] == "upper" else "bottom"}
                    for s in strokes)
        if any(out_dir.glob(f"{match_id}_*_16.npy")):
            print(f"[skip] {match_id} windows exist", flush=True)
            continue
        hits = [s["hit_frame"] for s in strokes]
        alarms = false_alarms(video, args.detector, hits, args.min_alarm_score, int(len(hits) * args.alarms_per_hit))
        centres = sorted(set(windows_for_match(hits, rng, args.negatives_per_hit)) | set(alarms))
        written = cut_windows(video, centres, out_dir, match_id)
        print(f"[+] {match_id} {folder[:50]}: {len(hits)} hits, {len(alarms)} false alarms, {written} windows", flush=True)
    with args.out_manifest.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(f"[+] manifest {args.out_manifest} ({len(rows)} rows)", flush=True)


if __name__ == "__main__":
    main()
