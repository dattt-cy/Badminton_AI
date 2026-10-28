"""Mine HitNet false positives on ShuttleSet train matches as hard negatives.

For each train match, a window around the middle of its annotated range is
scanned with the current hit detector. Detected peaks (temporal NMS at a low
threshold, so borderline peaks are included) that lie more than TOLERANCE
frames from every annotated hit are false positives: replays, close-ups,
inter-rally motion and duplicate peaks. Only peaks inside the annotated range
are kept, so unannotated footage cannot inject mislabelled negatives.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
for path in (REPO_ROOT, REPO_ROOT / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from ai_classifier.localization import two_stage_temporal_nms  # noqa: E402
from scripts.inference.scan_video_hit_rgb import load_model, scan_video  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path,
                        default=REPO_ROOT / "src/ai_classifier/models/checkpoints/hit_detector.pth")
    parser.add_argument("--manifest", type=Path, default=REPO_ROOT / "data/manifests/shuttleset_rgb.csv")
    parser.add_argument("--split", default="train")
    parser.add_argument("--window-frames", type=int, default=9000, help="Frames scanned per match (5 min at 30 fps).")
    parser.add_argument("--threshold", type=float, default=0.50)
    parser.add_argument("--tolerance", type=int, default=12)
    parser.add_argument("--output", type=Path, default=REPO_ROOT / "work_dirs/hitnet_hard_negatives.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, meta = load_model(args.checkpoint)
    model.to(device).eval()
    n_frames = int(meta.get("frames", 16))

    matches: dict[str, dict] = {}
    with args.manifest.open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            if row["split"] != args.split:
                continue
            entry = matches.setdefault(row["match_id"], {"video": row["video_path"], "hits": []})
            entry["hits"].append(int(float(row["hit_frame"])))

    result = json.loads(args.output.read_text(encoding="utf-8")) if args.output.exists() else {}
    for match_id in sorted(matches, key=int):
        if match_id in result:
            continue
        hits = sorted(matches[match_id]["hits"])
        middle = hits[len(hits) // 2]
        start = max(hits[0], middle - args.window_frames // 2)
        stop = min(hits[-1], start + args.window_frames)
        started = time.time()
        raw = scan_video(Path(matches[match_id]["video"]), model, n_frames, stride=4, batch_size=8,
                         device=device, use_amp=device.type == "cuda",
                         start_frame=start - n_frames // 2, max_frames=stop + n_frames // 2)
        peaks = two_stage_temporal_nms(raw, threshold=args.threshold, radius=8, same_side_radius=18)
        window_hits = [h for h in hits if start <= h <= stop]
        false_positives = [
            {"frame": p.frame, "score": round(p.score, 4)}
            for p in peaks
            if start <= p.frame <= stop and all(abs(p.frame - h) > args.tolerance for h in hits)
        ]
        matched = sum(any(abs(p.frame - h) <= args.tolerance for p in peaks) for h in window_hits)
        result[match_id] = {"start": start, "stop": stop, "gt_hits": len(window_hits),
                            "recalled": matched, "peaks": len(peaks), "false_positives": false_positives}
        print(f"match {match_id:>2}: gt={len(window_hits)} peaks={len(peaks)} recalled={matched} "
              f"fp={len(false_positives)} ({time.time() - started:.0f}s)", flush=True)
        # Saved after every match so an interrupted run resumes.
        args.output.write_text(json.dumps(result, indent=1) + "\n", encoding="utf-8")

    total_fp = sum(len(v["false_positives"]) for v in result.values())
    total_gt = sum(v["gt_hits"] for v in result.values())
    print(f"[done] matches={len(result)} gt={total_gt} hard_negatives={total_fp}")


if __name__ == "__main__":
    main()
