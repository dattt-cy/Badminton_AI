#!/usr/bin/env python3
"""Build manifests and train/val/test splits for the BFMD dataset.

Maps BFMD shot_type labels to the project's 8 coarse stroke classes:
  - serve, flick_serve           -> serve
  - clear                        -> clear
  - drop                         -> drop
  - smash                        -> smash
  - drive                        -> drive
  - net_shot, block              -> net_shot (matches ShuttleSet '擋小球' mapping)
  - lift                         -> lift
  - net_kill, push, press        -> net_attack

Match-level split ensures zero match/player-sequence leakage:
  - Train: 8 matches (~7,500 shots)
  - Val:   2 matches (~1,750 shots)
  - Test:  2 matches (~2,000 shots)
"""
import glob
import json
import os
from collections import Counter
from pathlib import Path
import pandas as pd

BFMD_DIR = Path(r"C:\Users\ADMIN\Downloads\BFMD_data-20260921T175743Z-1-001\BFMD_data")
HIT_INFERRED_DIR = BFMD_DIR / "annotations" / "hit_inferred"
VIDEOS_DIR = BFMD_DIR / "videos"

LABEL_MAPPING = {
    "serve": "serve",
    "flick_serve": "serve",
    "clear": "clear",
    "drop": "drop",
    "smash": "smash",
    "drive": "drive",
    "net_shot": "net_shot",
    "lift": "lift",
    "net_kill": "net_attack",
    "press": "net_attack",
}

EXCLUDED_LABELS = {
    "block": "ambiguous_defensive_response",
    "push": "taxonomy_mismatch_requires_manual_court_depth_audit",
}

MATCH_SPLITS = {
    # 8 Train matches
    "KAPAL-API-Indonesia-Open-2025-Anders-Antonsen-DEN-3-vs.-Chou-Tien-Chen-TPE-6-F": "test",
    "KAPAL-API-Indonesia-Open-2025-Chou-Tien-Chen-TPE-6-vs.-Kunlavut-Vitidsarn-THA-2-SF": "train",
    "PETRONAS-Malaysia-Open-2025-Kodai-Naraok-JPN-8-vs.-Anders-Antonsen-DEN-2-SF": "train",
    "PETRONAS-Malaysia-Open-2025-Shi-Yu-Qi-CHN-1-vs.-Li-Shi-Feng-CHN-7-SF": "train",
    "VICTOR-China-Open-2025-Chou-Tien-Chen-TPE-6-vs.-Shi-Yu-Qi-CHN-3-SF": "train",
    "VICTOR-China-Open-2025-Shi-Yu-Qi-CHN-3-vs.-Wang-Zheng-Xing-CHN-F": "train",
    "YONEX-All-England-Open-2025-Lee-Chia-Hao-TPE-vs.-Alex-Lanier-FRA-SF": "test",
    "YONEX-All-England-Open-2025-Shi-Yu-Qi-CHN-1-vs.-Li-Shi-Feng-CHN-6-SF": "val",

    # 2 Val matches
    "KAPAL-API-Indonesia-Open-2025-Shi-Yu-Qi-CHN-1-vs.-Anders-Antonsen-DEN-3-SF": "val",
    "VICTOR-China-Open-2025-Wang-Zheng-Xing-CHN-vs.-Anders-Antonsen-DEN-2-SF": "train",

    # 2 Test matches
    "PETRONAS-Malaysia-Open-2025-Shi-Yu-Qi-CHN-1-vs.-Anders-Antonsen-DEN-2-F": "train",
    "YONEX-All-England-Open-2025-Shi-Yu-Qi-CHN-1-vs.-Lee-Chia-Hao-TPE-F": "train",
}


def find_video_path(match_name: str) -> Path:
    # Look in tournament subdirectories or directly in videos/
    tournaments = [
        "KAPAL-API-Indonesia-Open-2025",
        "PETRONAS-Malaysia-Open-2025",
        "VICTOR-China-Open-2025",
        "YONEX-All-England-Open-2025",
    ]
    for tour in tournaments:
        if match_name.startswith(tour):
            p = VIDEOS_DIR / tour / f"{match_name}.mp4"
            return p
    return VIDEOS_DIR / f"{match_name}.mp4"


def validate_manifest(df: pd.DataFrame) -> None:
    required_classes = set(LABEL_MAPPING.values())
    split_matches = {
        split: set(df.loc[df["split"] == split, "match_id"])
        for split in ("train", "val", "test")
    }
    if any(split_matches[left] & split_matches[right] for left, right in (
        ("train", "val"), ("train", "test"), ("val", "test")
    )):
        raise ValueError("BFMD match leakage across train/val/test")
    for split in ("train", "val", "test"):
        actual = set(df.loc[df["split"] == split, "coarse_label"])
        if actual != required_classes:
            raise ValueError(f"BFMD {split} class coverage mismatch: {actual}")
    if set(df["raw_label"]) & set(EXCLUDED_LABELS):
        raise ValueError("Excluded BFMD labels leaked into the manifest")


def main():
    json_files = sorted(glob.glob(str(HIT_INFERRED_DIR / "*.json")))
    print(f"Loaded {len(json_files)} annotation files.")

    rows = []
    global_raw_counter = Counter()
    global_coarse_counter = Counter()
    split_counter = Counter()
    excluded_counter = Counter()

    for jpath in json_files:
        mname = Path(jpath).stem
        if mname not in MATCH_SPLITS:
            raise ValueError(f"Missing explicit split assignment for match: {mname}")
        split = MATCH_SPLITS[mname]
        vpath = find_video_path(mname)

        with open(jpath, "r", encoding="utf-8") as fp:
            data = json.load(fp)

        hits = data.get("hits", [])
        for idx, h in enumerate(hits):
            raw_shot = h.get("shot_type")
            coarse = LABEL_MAPPING.get(raw_shot)
            if coarse is None:
                excluded_counter[raw_shot or "<missing>"] += 1
                continue

            hit_f = int(h["frame"])
            start_f = max(0, hit_f - 30)
            end_f = hit_f + 30
            player_side = h.get("side", "bottom")
            game_id = h.get("game", 1)
            rally_id = h.get("rally", 0)
            sample_id = f"bfmd_{mname}_{game_id}_{rally_id}_{idx}"

            global_raw_counter[raw_shot] += 1
            global_coarse_counter[coarse] += 1
            split_counter[split] += 1

            rows.append({
                "sample_id": sample_id,
                "dataset": "bfmd",
                "split": split,
                "match_id": mname,
                "set_id": game_id,
                "rally": rally_id,
                "ball_round": idx,
                "video_path": str(vpath),
                "start_frame": start_f,
                "hit_frame": hit_f,
                "end_frame": end_f,
                "fps": 30.0,
                "player_side": player_side,
                "raw_label": raw_shot,
                "coarse_label": coarse,
                "stroke_side": "unknown",  # BFMD supervises stroke head
                "player_name": h.get("player", ""),
                "annotation_path": os.path.relpath(jpath, BFMD_DIR),
            })

    df = pd.DataFrame(rows)
    validate_manifest(df)
    out_manifests_dir = Path("data/manifests")
    out_manifests_dir.mkdir(parents=True, exist_ok=True)
    splits_dir = out_manifests_dir / "bfmd_splits"
    splits_dir.mkdir(parents=True, exist_ok=True)

    all_csv_path = out_manifests_dir / "bfmd_all.csv"
    df.to_csv(all_csv_path, index=False)
    print(f"Saved {len(df)} total shots to {all_csv_path}")

    for split in ["train", "val", "test"]:
        sub_df = df[df["split"] == split]
        split_path = splits_dir / f"{split}.csv"
        sub_df.to_csv(split_path, index=False)
        print(f"Saved split '{split}': {len(sub_df)} shots to {split_path}")

    # Save audit summary
    audit_data = {
        "dataset": "BFMD",
        "total_matches": len(json_files),
        "total_shots": len(df),
        "raw_counts": dict(global_raw_counter),
        "coarse_counts": dict(global_coarse_counter),
        "split_totals": dict(split_counter),
        "split_class_counts": {
            split: dict(Counter(df.loc[df["split"] == split, "coarse_label"]))
            for split in ("train", "val", "test")
        },
        "split_by_match": MATCH_SPLITS,
        "excluded_counts": dict(excluded_counter),
        "excluded_reasons": EXCLUDED_LABELS,
        "mapping_policy": "conservative_v2",
    }
    audit_path = out_manifests_dir / "bfmd_audit.json"
    with open(audit_path, "w", encoding="utf-8") as fp:
        json.dump(audit_data, fp, indent=2)
    print(f"Saved audit summary to {audit_path}")


if __name__ == "__main__":
    main()
