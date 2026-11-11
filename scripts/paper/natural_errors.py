"""Natural extractor errors on the ShuttleSet test events and accuracy on events that have them.

Measures, inside each event's valid frames of the published BST extraction (MMPose
poses, TrackNetV3 shuttle):
  * shuttle frames lost by TrackNetV3, and how long the gaps are;
  * shuttle jumps larger than ``--jump`` (normalised image units per frame), a proxy
    for wrong detections since ShuttleSet has no shuttle ground truth per frame;
  * player-frames whose 17 joints are all missing, and joints missing on their own.

Then compares Robust KAP (main seeds) and BST-CG-AP (official checkpoints) on the
clean test predictions saved by degradation_eval.py, grouped by natural error level,
with a paired bootstrap over events for the accuracy difference.

Example:
    python scripts/paper/natural_errors.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts" / "paper"))
for path in (REPO_ROOT / "src", REPO_ROOT / "scripts" / "training"):
    sys.path.insert(0, str(path))

import degradation_eval  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, default=REPO_ROOT / "work_dirs/paper_final/clean_predictions.npz")
    parser.add_argument("--kap-prefix", default="kap__main_seed")
    parser.add_argument("--jump", type=float, default=0.15)
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--output", type=Path, default=REPO_ROOT / "work_dirs/paper_lan4/natural_errors.json")
    return parser.parse_args()


def runs_of(mask: np.ndarray) -> list[int]:
    """Lengths of the runs of True in a 1-D mask."""
    padded = np.concatenate([[False], mask, [False]])
    edges = np.flatnonzero(np.diff(padded.astype(int)))
    return list(edges[1::2] - edges[::2])


def event_errors(joints: np.ndarray, shuttle: np.ndarray, lengths: np.ndarray, jump: float) -> dict[str, np.ndarray]:
    count = len(lengths)
    out = {k: np.zeros(count) for k in ("shuttle_lost", "shuttle_jump", "player_lost", "joint_lost", "longest_gap")}
    gaps: list[int] = []
    for i, n in enumerate(lengths):
        j, s = joints[i, :n], shuttle[i, :n]
        lost = np.abs(s).sum(-1) == 0
        out["shuttle_lost"][i] = lost.mean()
        seen = np.flatnonzero(~lost)
        if len(seen) > 1:
            step = np.linalg.norm(np.diff(s[seen], axis=0), axis=-1) / np.diff(seen)
            out["shuttle_jump"][i] = (step > jump).mean()
        run = runs_of(lost)
        gaps.extend(run)
        out["longest_gap"][i] = max(run, default=0) / n
        zero = np.abs(j).sum(-1) == 0                     # (n, 2, 17)
        player = zero.all(-1)                             # (n, 2)
        out["player_lost"][i] = player.mean()
        out["joint_lost"][i] = (zero & ~player[..., None]).mean()
    out["gap_lengths"] = np.asarray(gaps)
    return out


def describe(values: np.ndarray) -> dict:
    return {"mean": float(values.mean()), "median": float(np.median(values)),
            "p90": float(np.quantile(values, 0.9)), "p99": float(np.quantile(values, 0.99)),
            "max": float(values.max()), "share_nonzero": float((values > 0).mean())}


def compare(truth: np.ndarray, kap: list[np.ndarray], bst: list[np.ndarray], rows: np.ndarray,
            rng: np.random.Generator, draws: int) -> dict:
    """Mean accuracy over seeds/checkpoints and a paired event bootstrap of the difference."""
    k = np.mean([(p[rows] == truth[rows]) for p in kap], axis=0)   # per-event share of KAP seeds correct
    b = np.mean([(p[rows] == truth[rows]) for p in bst], axis=0)
    diff = k - b
    picks = rng.integers(0, len(rows), size=(draws, len(rows)))
    boot = diff[picks].mean(1)
    return {"events": int(len(rows)), "kap": float(k.mean() * 100), "bst": float(b.mean() * 100),
            "diff": float(diff.mean() * 100),
            "ci95": [float(np.quantile(boot, 0.025) * 100), float(np.quantile(boot, 0.975) * 100)]}


def main() -> None:
    args = parse_args()
    sys.argv = sys.argv[:1]
    data = degradation_eval.bst_setup(degradation_eval.parse_args())
    order = np.argsort(data["sample_ids"])
    common = data["common"][order]
    errors = event_errors(data["joints"][common], data["shuttle"][common], data["lengths"][common], args.jump)

    saved = np.load(args.predictions)
    if not np.array_equal(saved["sample_id"], data["sample_ids"][order]):
        raise RuntimeError("Saved predictions and BST test events differ")
    truth = saved["truth"]
    kap = [saved[k].argmax(1) for k in saved.files if k.startswith(args.kap_prefix)]
    bst = [saved[k].argmax(1) for k in saved.files if k.startswith("bst__")]

    result: dict = {"events": int(len(truth)), "kap_runs": len(kap), "bst_checkpoints": len(bst),
                    "jump_threshold": args.jump,
                    "frames": {"shuttle_lost": float(np.concatenate([
                        np.abs(data["shuttle"][i, :n]).sum(-1) == 0
                        for i, n in zip(common, data["lengths"][common])]).mean())},
                    "per_event": {k: describe(v) for k, v in errors.items() if k != "gap_lengths"},
                    "gap_lengths": describe(errors["gap_lengths"].astype(float)),
                    "groups": {}}
    gaps = errors["gap_lengths"]
    result["gap_lengths"]["share_le_6"] = float((gaps <= 6).mean())

    rng = np.random.default_rng(20261109)
    everything = np.arange(len(truth))
    groups = {"all": everything}
    for name in ("shuttle_lost", "player_lost", "shuttle_jump"):
        v = errors[name]
        groups[f"{name}=0"] = np.flatnonzero(v == 0)
        groups[f"{name}>0"] = np.flatnonzero(v > 0)
        if (v > 0).sum() > 40:
            q = np.quantile(v[v > 0], 0.75)
            groups[f"{name}>=p75({q:.2f})"] = np.flatnonzero(v >= q)
    worst = np.maximum(errors["shuttle_lost"], errors["player_lost"])
    groups["any_lost>=0.25"] = np.flatnonzero(worst >= 0.25)
    groups["clean (no loss, no jump)"] = np.flatnonzero(
        (errors["shuttle_lost"] == 0) & (errors["player_lost"] == 0) & (errors["shuttle_jump"] == 0))
    for name, rows in groups.items():
        result["groups"][name] = compare(truth, kap, bst, rows, rng, args.bootstrap)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: result[k] for k in ("frames", "per_event", "gap_lengths")}, indent=1))
    for name, g in result["groups"].items():
        print(f"{name:32s} n={g['events']:5d} KAP={g['kap']:.2f} BST={g['bst']:.2f} "
              f"diff={g['diff']:+.2f} [{g['ci95'][0]:+.2f}; {g['ci95'][1]:+.2f}]")


if __name__ == "__main__":
    main()
