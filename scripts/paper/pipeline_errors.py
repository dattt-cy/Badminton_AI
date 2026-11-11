"""Extractor errors of the deployed pipeline (YOLOv8n-Pose + TrackNet) on the untrimmed test clips.

Reads the per-hit feature caches written by compare_kap_checkpoints_8clips.py and
measures, over the raw 46-frame window of every candidate hit (15 before, 30 after):
  * shuttle frames lost by TrackNet, gap lengths, and jumps above ``--jump``;
  * frames where the hitter or the opponent has no detection (foot position 0);
  * joints missing while the rest of that player's skeleton is present (32-step pose).

The same statistics for the published ShuttleSet extraction come from natural_errors.py.

Example:
    python scripts/paper/pipeline_errors.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
TEST_CLIPS = ("clip1_match44_10min", "clip2_match39_10min", "clip3_match43_10min",
              "clip4_match40_10min", "clip5_match42_10min")


def runs_of(mask: np.ndarray) -> list[int]:
    padded = np.concatenate([[False], mask, [False]])
    edges = np.flatnonzero(np.diff(padded.astype(int)))
    return list(edges[1::2] - edges[::2])


def describe(values: np.ndarray) -> dict:
    return {"mean": float(values.mean()), "median": float(np.median(values)),
            "p90": float(np.quantile(values, 0.9)), "p99": float(np.quantile(values, 0.99)),
            "max": float(values.max()), "share_nonzero": float((values > 0).mean())}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, default=REPO_ROOT / "work_dirs/kap_ab_8clips/feature_cache")
    parser.add_argument("--jump", type=float, default=0.15)
    parser.add_argument("--output", type=Path, default=REPO_ROOT / "work_dirs/paper_lan4/pipeline_errors.json")
    args = parser.parse_args()

    stats = {k: [] for k in ("shuttle_lost", "shuttle_jump", "player_lost", "joint_lost", "longest_gap")}
    gaps: list[int] = []
    frames = {"shuttle": [0, 0], "player": [0, 0]}
    for clip in TEST_CLIPS:
        cached = torch.load(args.cache / f"{clip}.pt", map_location="cpu", weights_only=False)
        for item in cached["features"]:
            context = item["feature"]["analytics_context"]
            shuttle = np.asarray(context["shuttle_image"])
            lost = np.abs(shuttle).sum(-1) == 0
            frames["shuttle"][0] += int(lost.sum()); frames["shuttle"][1] += len(lost)
            stats["shuttle_lost"].append(lost.mean())
            seen = np.flatnonzero(~lost)
            if len(seen) > 1:
                step = np.linalg.norm(np.diff(shuttle[seen], axis=0), axis=-1) / np.diff(seen)
                stats["shuttle_jump"].append((step > args.jump).mean())
            else:
                stats["shuttle_jump"].append(0.0)
            run = runs_of(lost)
            gaps.extend(run)
            stats["longest_gap"].append(max(run, default=0) / len(lost))
            feet = np.stack([np.asarray(context["player_court"]), np.asarray(context["opponent_court"])], 1)
            missing = np.abs(feet).sum(-1) == 0                      # (46, 2)
            frames["player"][0] += int(missing.sum()); frames["player"][1] += missing.size
            stats["player_lost"].append(missing.mean())
            pose = item["feature"]["s_j"][0].numpy().reshape(-1, 2, 17, 2)
            zero = np.abs(pose).sum(-1) == 0
            whole = zero.all(-1)
            stats["joint_lost"].append((zero & ~whole[..., None]).mean())

    result = {
        "hits": len(stats["shuttle_lost"]), "clips": list(TEST_CLIPS), "jump_threshold": args.jump,
        "frames": {k: v[0] / v[1] for k, v in frames.items()},
        "per_event": {k: describe(np.asarray(v)) for k, v in stats.items()},
        "gap_lengths": {**describe(np.asarray(gaps, dtype=float)), "share_le_6": float((np.asarray(gaps) <= 6).mean())},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
