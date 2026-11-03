"""Joint F1 (time + hitter + stroke type) of saved worker-pipeline outputs.

A prediction is a joint true positive when it is matched one-to-one to a
ground-truth hit within the frame tolerance, names the correct hitter
(upper/lower court) and the correct 8-class stroke type.

Example:
    python scripts/paper/joint_f1.py --run-dir work_dirs/worker_8clips_soup_t75
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.evaluation.benchmark_8_matches_10min import CLIPS_CONFIG  # noqa: E402
from scripts.evaluation.benchmark_worker_pipeline_8clips import load_gt  # noqa: E402
from scripts.evaluation.compare_kap_checkpoints_8clips import TEST_MATCHES  # noqa: E402

SIDE = {"top": "upper", "bottom": "lower", "upper": "upper", "lower": "lower"}


def joint_counts(predictions: list[dict], truth: list[dict], tolerance: int) -> dict:
    used, matched, hitter, joint = set(), 0, 0, 0
    for row in truth:
        choices = [
            (abs(pred["frame"] - row["frame"]), index, pred)
            for index, pred in enumerate(predictions)
            if index not in used and abs(pred["frame"] - row["frame"]) <= tolerance
        ]
        if not choices:
            continue
        _, index, pred = min(choices)
        used.add(index)
        matched += 1
        side = SIDE.get(str(pred.get("hitter") or pred.get("player_side")).lower())
        side_ok = side == SIDE.get(str(row["player_side"]).lower())
        hitter += side_ok
        joint += side_ok and pred.get("stroke") == row["stroke"]
    return {"gt": len(truth), "pred": len(predictions), "matched": matched, "hitter": hitter, "joint": joint}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=REPO_ROOT / "work_dirs/worker_8clips_soup_t75")
    parser.add_argument("--manifest", type=Path, default=REPO_ROOT / "data/manifests/shuttleset_rgb.csv")
    parser.add_argument("--tolerances", type=int, nargs="+", default=[5, 12])
    args = parser.parse_args()

    report = {}
    for tolerance in args.tolerances:
        rows, seconds, duration = [], 0.0, 0.0
        for cfg in CLIPS_CONFIG:
            path = args.run_dir / f"{cfg['video'].stem}.json"
            if not path.exists():
                continue
            payload = json.loads(path.read_text(encoding="utf-8"))
            counts = joint_counts(payload["predictions"], load_gt(args.manifest, cfg), tolerance)
            counts.update(match_id=cfg["match_id"], split="test" if cfg["match_id"] in TEST_MATCHES else "val")
            rows.append(counts)
            if counts["split"] == "test":
                seconds += payload.get("timing_seconds", {}).get("total", 0.0)
                duration += payload.get("video", {}).get("duration_seconds", 0.0)
        for group in ("test", "all"):
            subset = [r for r in rows if group == "all" or r["split"] == group]
            gt = sum(r["gt"] for r in subset)
            pred = sum(r["pred"] for r in subset)
            total = {k: sum(r[k] for r in subset) for k in ("matched", "hitter", "joint")}
            report[f"{group}@{tolerance}"] = {
                "clips": len(subset), "gt": gt, "pred": pred, **total,
                "hit_precision": 100 * total["matched"] / pred, "hit_recall": 100 * total["matched"] / gt,
                "hit_f1": 200 * total["matched"] / (gt + pred),
                "joint_precision": 100 * total["joint"] / pred, "joint_recall": 100 * total["joint"] / gt,
                "joint_f1": 200 * total["joint"] / (gt + pred),
                "hitter_acc_matched": 100 * total["hitter"] / total["matched"],
            }
        report[f"per_clip@{tolerance}"] = rows
        report["test_processing_seconds"] = seconds
        report["test_video_seconds"] = duration
    print(json.dumps({k: v for k, v in report.items() if not k.startswith("per_clip")}, indent=2))
    (args.run_dir / "joint_f1.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
