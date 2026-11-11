"""95% bootstrap confidence intervals for the end-to-end hit F1 and Joint F1 on the test clips.

Matching follows joint_f1.py (one-to-one, nearest prediction within the tolerance).
Every matched pair, missed ground-truth hit and unmatched prediction is one unit;
F1 = 2 TP / (GT + Pred) is recomputed on units resampled with replacement:
  * ``stroke``: units resampled independently (per stroke, as requested);
  * ``minute``: units grouped by clip-minute and the 50 minutes resampled, which keeps
    the dependence between neighbouring strokes.

Example:
    python scripts/paper/e2e_ci.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.evaluation.benchmark_8_matches_10min import CLIPS_CONFIG  # noqa: E402
from scripts.evaluation.benchmark_worker_pipeline_8clips import load_gt  # noqa: E402
from scripts.evaluation.compare_kap_checkpoints_8clips import TEST_MATCHES  # noqa: E402
from scripts.paper.joint_f1 import SIDE  # noqa: E402


def units(predictions: list[dict], truth: list[dict], tolerance: int, fps: float, clip: int) -> list[tuple]:
    """(gt, pred, hit_tp, joint_tp, block) per unit, matched exactly as joint_f1.joint_counts."""
    used, out = set(), []
    for row in truth:
        choices = [(abs(p["frame"] - row["frame"]), i, p) for i, p in enumerate(predictions)
                   if i not in used and abs(p["frame"] - row["frame"]) <= tolerance]
        block = (clip, int(row["frame"] // (60 * fps)))
        if not choices:
            out.append((1, 0, 0, 0, block))
            continue
        _, index, pred = min(choices)
        used.add(index)
        side_ok = SIDE.get(str(pred.get("hitter") or pred.get("player_side")).lower()) == SIDE.get(str(row["player_side"]).lower())
        out.append((1, 1, 1, int(side_ok and pred.get("stroke") == row["stroke"]), block))
    for index, pred in enumerate(predictions):
        if index not in used:
            out.append((0, 1, 0, 0, (clip, int(pred["frame"] // (60 * fps)))))
    return out


def f1(sample: np.ndarray) -> tuple[float, float]:
    gt, pred = sample[:, 0].sum(), sample[:, 1].sum()
    return 200 * sample[:, 2].sum() / (gt + pred), 200 * sample[:, 3].sum() / (gt + pred)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=REPO_ROOT / "work_dirs/worker_8clips_soup_t75")
    parser.add_argument("--manifest", type=Path, default=REPO_ROOT / "data/manifests/shuttleset_rgb.csv")
    parser.add_argument("--tolerances", type=int, nargs="+", default=[5, 12])
    parser.add_argument("--bootstrap", type=int, default=5000)
    parser.add_argument("--output", type=Path, default=REPO_ROOT / "work_dirs/paper_lan4/e2e_ci.json")
    args = parser.parse_args()

    report = {}
    rng = np.random.default_rng(20261109)
    for tolerance in args.tolerances:
        rows = []
        for clip, cfg in enumerate(CLIPS_CONFIG):
            if cfg["match_id"] not in TEST_MATCHES:
                continue
            payload = json.loads((args.run_dir / f"{cfg['video'].stem}.json").read_text(encoding="utf-8"))
            fps = float(payload.get("video", {}).get("fps", 30.0))
            rows += units(payload["predictions"], load_gt(args.manifest, cfg), tolerance, fps, clip)
        data = np.asarray([r[:4] for r in rows], dtype=float)
        blocks = np.asarray([hash(r[4]) for r in rows])
        hit, joint = f1(data)
        stroke = np.asarray([f1(data[rng.integers(0, len(data), len(data))]) for _ in range(args.bootstrap)])
        groups = [np.flatnonzero(blocks == b) for b in np.unique(blocks)]
        minute = np.asarray([f1(data[np.concatenate([groups[i] for i in rng.integers(0, len(groups), len(groups))])])
                             for _ in range(args.bootstrap)])
        report[f"test@{tolerance}"] = {
            "gt": int(data[:, 0].sum()), "pred": int(data[:, 1].sum()), "minutes": len(groups),
            "hit_f1": hit, "joint_f1": joint,
            "hit_f1_ci_stroke": list(np.quantile(stroke[:, 0], [0.025, 0.975])),
            "joint_f1_ci_stroke": list(np.quantile(stroke[:, 1], [0.025, 0.975])),
            "hit_f1_ci_minute": list(np.quantile(minute[:, 0], [0.025, 0.975])),
            "joint_f1_ci_minute": list(np.quantile(minute[:, 1], [0.025, 0.975])),
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
