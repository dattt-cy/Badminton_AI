"""Build balanced match-disjoint manifest for Rally-State Gate (LIVE vs NON_LIVE).

Slices video into temporal windows (default 32 frames ~ 1.07s) labeled:
  1 = LIVE_RALLY (inside active rallies with +/- 1.0s margin)
  0 = NON_LIVE   (inter-rally rest intervals, walking, picking shuttle, hard negatives)

Strict match-disjoint separation:
  Train: 30 matches (1..34 excluding val/test)
  Val:   5 matches (35, 36, 37, 38, 41)
  Test:  5 matches (39, 40, 42, 43, 44) strictly held out!
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import cv2

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

DEFAULT_TRAIN_MATCHES = [
    "1", "2", "3", "4", "5", "6", "7", "8", "11", "13",
    "14", "15", "16", "17", "18", "19", "20", "21", "22", "23",
    "24", "25", "26", "28", "29", "30", "31", "32", "33", "34",
]
DEFAULT_VAL_MATCHES = ["35", "36", "37", "38", "41"]
DEFAULT_TEST_MATCHES = ["39", "40", "42", "43", "44"]


@dataclass(frozen=True)
class WindowSample:
    sample_id: str
    match_id: str
    set_id: str
    rally_id: str
    video_path: str
    start_frame: int
    center_frame: int
    end_frame: int
    fps: float
    label: int  # 1=LIVE_RALLY, 0=NON_LIVE
    source: str  # rally_window, rest_window, pre_match, post_match, hard_negative


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path("data/manifests/shuttleset_rgb.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/manifests"))
    parser.add_argument("--window-size", type=int, default=32, help="Number of frames per window (default: 32 ~ 1s)")
    parser.add_argument("--live-stride", type=int, default=24, help="Stride in frames when sampling live rallies")
    parser.add_argument("--nonlive-stride", type=int, default=48, help="Stride in frames when sampling rest intervals")
    parser.add_argument("--margin-buffer", type=int, default=30, help="Frames added before first hit and after last hit of rally")
    parser.add_argument("--exclusion-buffer", type=int, default=60, help="Minimum frame distance from any hit for non-live samples")
    parser.add_argument("--max-live-per-match", type=int, default=350, help="Max live samples per match to balance dataset")
    parser.add_argument("--max-nonlive-per-match", type=int, default=350, help="Max non-live samples per match")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def load_match_rallies(manifest_path: Path) -> dict[str, dict]:
    """Return {match_id: {'video_path': Path, 'fps': float, 'split': str, 'rallies': {(set_id, rally): [hits]}}}."""
    matches: dict[str, dict] = {}
    with manifest_path.open("r", encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            mid = row["match_id"]
            if mid not in matches:
                matches[mid] = {
                    "video_path": Path(row["video_path"]),
                    "fps": float(row.get("fps") or 30.0),
                    "split": row["split"],
                    "rallies": {},
                    "all_hits": [],
                }
            hf = int(float(row["hit_frame"]))
            key = (row["set_id"], row["rally"])
            matches[mid]["rallies"].setdefault(key, []).append(hf)
            matches[mid]["all_hits"].append(hf)

    for mid, item in matches.items():
        item["all_hits"].sort()
        for k in item["rallies"]:
            item["rallies"][k].sort()
    return matches


def find_hard_negatives(match_id: str, all_hits: list[int], exclusion: int) -> list[int]:
    """Find candidate hits outside rallies from existing scans in work_dirs/."""
    candidates = []
    scan_paths = [
        REPO_ROOT / f"work_dirs/online_train_match{match_id}_scan.json",
        REPO_ROOT / f"work_dirs/scan_match{match_id}_5min.json",
        REPO_ROOT / f"work_dirs/shuttleset22_match{match_id}_5min_scan.json",
    ]
    for sp in scan_paths:
        if not sp.exists():
            continue
        try:
            data = json.loads(sp.read_text(encoding="utf-8"))
            for ev in data.get("events", []):
                frame = int(ev["frame"])
                score = float(ev["score"])
                if score >= 0.50:
                    min_dist = min(abs(frame - h) for h in all_hits) if all_hits else 9999
                    if min_dist >= exclusion:
                        candidates.append(frame)
        except Exception:
            pass
    return sorted(set(candidates))


def build_samples_for_match(
    mid: str,
    match_data: dict,
    *,
    window_size: int,
    live_stride: int,
    nonlive_stride: int,
    margin_buffer: int,
    exclusion_buffer: int,
    max_live: int,
    max_nonlive: int,
    rng: random.Random,
) -> tuple[list[WindowSample], list[WindowSample]]:
    vpath = match_data["video_path"]
    if not vpath.exists():
        print(f"  [WARN] Video missing for match {mid}: {vpath}", file=sys.stderr)
        return [], []

    cap = cv2.VideoCapture(str(vpath))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = match_data["fps"]
    cap.release()

    if total_frames <= 0:
        return [], []

    rallies = match_data["rallies"]
    all_hits = match_data["all_hits"]
    half_w = window_size // 2

    # 1. LIVE SAMPLES
    live_samples: list[WindowSample] = []
    rally_spans: list[tuple[int, int, tuple[str, str]]] = []

    for (set_id, rally), hits in rallies.items():
        if not hits:
            continue
        start_rally = max(0, min(hits) - margin_buffer)
        end_rally = min(total_frames - 1, max(hits) + margin_buffer)
        rally_spans.append((start_rally, end_rally, (set_id, rally)))

        span_len = end_rally - start_rally + 1
        if span_len <= window_size:
            center = (start_rally + end_rally) // 2
            w_start = max(0, center - half_w)
            w_end = min(total_frames - 1, w_start + window_size - 1)
            sample_id = f"m{mid}_s{set_id}_r{rally}_{center}_live"
            live_samples.append(
                WindowSample(sample_id, mid, set_id, rally, str(vpath), w_start, center, w_end, fps, 1, "rally_window")
            )
        else:
            for w_start in range(start_rally, end_rally - window_size + 1, live_stride):
                w_end = w_start + window_size - 1
                center = w_start + half_w
                sample_id = f"m{mid}_s{set_id}_r{rally}_{center}_live"
                live_samples.append(
                    WindowSample(sample_id, mid, set_id, rally, str(vpath), w_start, center, w_end, fps, 1, "rally_window")
                )

    rally_spans.sort(key=lambda s: s[0])

    # 2. NON-LIVE SAMPLES (Rest gaps + Pre/Post match + Hard Negatives)
    nonlive_samples: list[WindowSample] = []

    # Hard negatives first
    hard_negs = find_hard_negatives(mid, all_hits, exclusion_buffer)
    for hf in hard_negs:
        w_start = max(0, hf - half_w)
        w_end = min(total_frames - 1, w_start + window_size - 1)
        if w_end - w_start + 1 == window_size:
            sample_id = f"m{mid}_hardneg_{hf}"
            nonlive_samples.append(
                WindowSample(sample_id, mid, "0", "0", str(vpath), w_start, hf, w_end, fps, 0, "hard_negative")
            )

    # Inter-rally rest intervals
    for i in range(len(rally_spans) - 1):
        gap_start = rally_spans[i][1] + exclusion_buffer
        gap_end = rally_spans[i + 1][0] - exclusion_buffer
        if gap_end - gap_start >= window_size:
            for w_start in range(gap_start, gap_end - window_size + 1, nonlive_stride):
                w_end = w_start + window_size - 1
                center = w_start + half_w
                sample_id = f"m{mid}_rest_{center}"
                nonlive_samples.append(
                    WindowSample(sample_id, mid, "0", "0", str(vpath), w_start, center, w_end, fps, 0, "rest_window")
                )

    # Pre-match warmup
    if rally_spans and rally_spans[0][0] - exclusion_buffer > window_size + 60:
        pre_start = 60
        pre_end = rally_spans[0][0] - exclusion_buffer
        for w_start in range(pre_start, min(pre_start + 600, pre_end - window_size + 1), nonlive_stride * 2):
            w_end = w_start + window_size - 1
            center = w_start + half_w
            sample_id = f"m{mid}_pre_{center}"
            nonlive_samples.append(
                WindowSample(sample_id, mid, "0", "0", str(vpath), w_start, center, w_end, fps, 0, "pre_match")
            )

    # Post-match
    if rally_spans and total_frames - (rally_spans[-1][1] + exclusion_buffer) > window_size + 60:
        post_start = rally_spans[-1][1] + exclusion_buffer
        post_end = min(total_frames - 60, post_start + 900)
        for w_start in range(post_start, post_end - window_size + 1, nonlive_stride * 2):
            w_end = w_start + window_size - 1
            center = w_start + half_w
            sample_id = f"m{mid}_post_{center}"
            nonlive_samples.append(
                WindowSample(sample_id, mid, "0", "0", str(vpath), w_start, center, w_end, fps, 0, "post_match")
            )

    # Balance per match
    rng.shuffle(live_samples)
    rng.shuffle(nonlive_samples)

    live_capped = live_samples[:max_live]
    nonlive_capped = nonlive_samples[:max_nonlive]

    return live_capped, nonlive_capped


def write_manifest(samples: Sequence[WindowSample], output_file: Path) -> None:
    output_file.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "sample_id", "match_id", "set_id", "rally_id", "video_path",
        "start_frame", "center_frame", "end_frame", "fps", "label", "source",
    ]
    with output_file.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for s in sorted(samples, key=lambda item: (int(item.match_id) if item.match_id.isdigit() else item.match_id, item.center_frame)):
            writer.writerow(s.__dict__)


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)

    print(f"[*] Loading matches from {args.manifest}...")
    all_matches = load_match_rallies(args.manifest)

    train_match_ids = set(DEFAULT_TRAIN_MATCHES)
    val_match_ids = set(DEFAULT_VAL_MATCHES)
    test_match_ids = set(DEFAULT_TEST_MATCHES)

    # Sanity check: strictly disjoint!
    assert train_match_ids.isdisjoint(val_match_ids), "Train and Val must be disjoint!"
    assert train_match_ids.isdisjoint(test_match_ids), "Train and Test must be disjoint!"
    assert val_match_ids.isdisjoint(test_match_ids), "Val and Test must be disjoint!"

    print(f"[*] Match-Disjoint Split: {len(train_match_ids)} Train, {len(val_match_ids)} Val, {len(test_match_ids)} Test (Held-out)")

    train_samples: list[WindowSample] = []
    val_samples: list[WindowSample] = []

    print("\n--- Processing Training Matches ---")
    for mid in sorted(train_match_ids, key=lambda x: int(x)):
        if mid not in all_matches:
            continue
        live, nonlive = build_samples_for_match(
            mid, all_matches[mid],
            window_size=args.window_size,
            live_stride=args.live_stride,
            nonlive_stride=args.nonlive_stride,
            margin_buffer=args.margin_buffer,
            exclusion_buffer=args.exclusion_buffer,
            max_live=args.max_live_per_match,
            max_nonlive=args.max_nonlive_per_match,
            rng=rng,
        )
        train_samples.extend(live)
        train_samples.extend(nonlive)
        print(f"  Match {mid:>2}: live={len(live):>3}, nonlive={len(nonlive):>3} (total {len(live)+len(nonlive):>3})")

    print("\n--- Processing Validation Matches ---")
    for mid in sorted(val_match_ids, key=lambda x: int(x)):
        if mid not in all_matches:
            continue
        live, nonlive = build_samples_for_match(
            mid, all_matches[mid],
            window_size=args.window_size,
            live_stride=args.live_stride,
            nonlive_stride=args.nonlive_stride,
            margin_buffer=args.margin_buffer,
            exclusion_buffer=args.exclusion_buffer,
            max_live=args.max_live_per_match,
            max_nonlive=args.max_nonlive_per_match,
            rng=rng,
        )
        val_samples.extend(live)
        val_samples.extend(nonlive)
        print(f"  Match {mid:>2}: live={len(live):>3}, nonlive={len(nonlive):>3} (total {len(live)+len(nonlive):>3})")

    # Write manifests
    train_csv = args.output_dir / "rally_state_train.csv"
    val_csv = args.output_dir / "rally_state_val.csv"
    audit_json = args.output_dir / "rally_state_audit.json"

    write_manifest(train_samples, train_csv)
    write_manifest(val_samples, val_csv)

    train_live = sum(1 for s in train_samples if s.label == 1)
    train_nonlive = sum(1 for s in train_samples if s.label == 0)
    val_live = sum(1 for s in val_samples if s.label == 1)
    val_nonlive = sum(1 for s in val_samples if s.label == 0)

    audit_data = {
        "window_size": args.window_size,
        "live_stride": args.live_stride,
        "nonlive_stride": args.nonlive_stride,
        "margin_buffer": args.margin_buffer,
        "exclusion_buffer": args.exclusion_buffer,
        "train": {
            "matches": sorted(list(train_match_ids), key=lambda x: int(x)),
            "total_samples": len(train_samples),
            "live_samples": train_live,
            "nonlive_samples": train_nonlive,
            "live_ratio": train_live / max(len(train_samples), 1),
        },
        "val": {
            "matches": sorted(list(val_match_ids), key=lambda x: int(x)),
            "total_samples": len(val_samples),
            "live_samples": val_live,
            "nonlive_samples": val_nonlive,
            "live_ratio": val_live / max(len(val_samples), 1),
        },
        "test_held_out": sorted(list(test_match_ids), key=lambda x: int(x)),
    }
    audit_json.write_text(json.dumps(audit_data, indent=2) + "\n", encoding="utf-8")

    print("\n========================================================")
    print(" [DONE] Rally-State Manifests Built Successfully")
    print(f"   Train CSV: {train_csv} -> {len(train_samples)} samples (Live: {train_live}, NonLive: {train_nonlive})")
    print(f"   Val CSV:   {val_csv}   -> {len(val_samples)} samples (Live: {val_live}, NonLive: {val_nonlive})")
    print(f"   Audit:     {audit_json}")
    print("========================================================")


if __name__ == "__main__":
    main()
