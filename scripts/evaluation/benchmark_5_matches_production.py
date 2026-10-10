"""Run multi-match benchmark for locked production pipeline on matches 39, 40, 42, 43, 44.

Locks production parameters:
- Hit threshold: 0.75
- NMS radius: 8
- Same-side NMS radius: 15
- Contact refine: 6
- KAP checkpoint: work_dirs/kap_online_balanced_selection_20261008/best.pth
- Pose model: models/checkpoints/pose/yolov8n-pose.pt
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path
from sklearn.metrics import f1_score

REPO_ROOT = Path(__file__).resolve().parents[2]

MATCH_CONFIGS = [
    {
        "id": "39",
        "name": "Match 39 (Kodai Naraoka vs Kunlavut Vitidsarn)",
        "video": REPO_ROOT / "work_dirs" / "match39_5min_clip.mp4",
        "scan": REPO_ROOT / "work_dirs" / "scan_match39_5min_fresh.json",
        "offset": 10000,
    },
    {
        "id": "40",
        "name": "Match 40 (Lee Zii Jia vs Loh Kean Yew)",
        "video": REPO_ROOT / "work_dirs" / "match40_5min_clip.mp4",
        "scan": REPO_ROOT / "work_dirs" / "scan_match40_5min.json",
        "offset": 12000,
    },
    {
        "id": "42",
        "name": "Match 42 (Anthony Ginting vs Jonatan Christie)",
        "video": REPO_ROOT / "work_dirs" / "match42_5min_clip.mp4",
        "scan": REPO_ROOT / "work_dirs" / "scan_match42_5min_latest_20261001.json",
        "offset": 14500,
    },
    {
        "id": "43",
        "name": "Match 43 (Kento Momota vs Chou Tien Chen)",
        "video": REPO_ROOT / "work_dirs" / "match43_5min_clip.mp4",
        "scan": REPO_ROOT / "work_dirs" / "scan_match43_5min_robust_kap.json",
        "offset": 18000,
    },
    {
        "id": "44",
        "name": "Match 44 (Viktor Axelsen vs Anders Antonsen)",
        "video": REPO_ROOT / "work_dirs" / "match44_5min_clip.mp4",
        "scan": REPO_ROOT / "work_dirs" / "scan_match44_5min.json",
        "offset": 45000,
    },
]

KAP_CKPT = REPO_ROOT / "work_dirs" / "kap_online_balanced_selection_20261008" / "best.pth"
POSE_MODEL = REPO_ROOT / "models" / "checkpoints" / "pose" / "yolov8n-pose.pt"
MANIFEST = REPO_ROOT / "data" / "manifests" / "shuttleset_rgb.csv"


def run_single_match(cfg: dict) -> Path:
    out_path = REPO_ROOT / "work_dirs" / f"production_match{cfg['id']}_eval.json"
    cmd = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "evaluation" / "run_eval_robust.py"),
        "--video", str(cfg["video"]),
        "--scan", str(cfg["scan"]),
        "--manifest", str(MANIFEST),
        "--match-id", cfg["id"],
        "--offset", str(cfg["offset"]),
        "--raw-threshold", "0.75",
        "--raw-nms-radius", "8",
        "--raw-same-side-nms-radius", "15",
        "--contact-refine-radius", "6",
        "--kap-checkpoint", str(KAP_CKPT),
        "--pose-model", str(POSE_MODEL),
        "--output", str(out_path),
    ]
    print(f"\n========================================================")
    print(f"RUNNING EVALUATION: {cfg['name']}")
    print(f"Command: {' '.join(cmd)}")
    print(f"========================================================")
    t0 = time.time()
    res = subprocess.run(cmd, check=True)
    dt = time.time() - t0
    print(f"[COMPLETED] {cfg['name']} in {dt:.1f}s")
    return out_path


def analyze_all_matches(output_paths: list[dict]) -> dict:
    all_results = []
    tot_gt = 0
    tot_cand = 0
    tot_matched = 0
    tot_top1 = 0
    tot_top2 = 0
    tot_runtime = 0.0

    all_y_true = []
    all_y_pred = []
    class_stats = defaultdict(lambda: {"gt": 0, "matched": 0, "top1": 0, "top2": 0})

    for item in output_paths:
        cfg = item["cfg"]
        path = item["path"]
        data = json.loads(path.read_text(encoding="utf-8"))
        
        gt_count = data["gt_count"]
        cand_count = data["candidate_count"]
        matched_count = data["matched_count"]
        matches = data["matches"]
        runtime = data["runtime_seconds"]

        top1_count = sum(m["top1"] for m in matches)
        top2_count = sum(m["top2"] for m in matches)

        tot_gt += gt_count
        tot_cand += cand_count
        tot_matched += matched_count
        tot_top1 += top1_count
        tot_top2 += top2_count
        tot_runtime += runtime

        y_true = [m["gt"]["stroke"] for m in matches if m["prediction"] is not None]
        y_pred = [m["prediction"]["stroke"] for m in matches if m["prediction"] is not None]
        all_y_true.extend(y_true)
        all_y_pred.extend(y_pred)

        m_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0) * 100 if y_true else 0.0

        for m in matches:
            gt_label = m["gt"]["stroke"]
            class_stats[gt_label]["gt"] += 1
            if m["prediction"] is not None:
                class_stats[gt_label]["matched"] += 1
                if m["top1"]:
                    class_stats[gt_label]["top1"] += 1
                if m["top2"]:
                    class_stats[gt_label]["top2"] += 1

        all_results.append({
            "id": cfg["id"],
            "name": cfg["name"],
            "gt_count": gt_count,
            "candidate_count": cand_count,
            "matched_count": matched_count,
            "false_positives": cand_count - matched_count,
            "missed_hits": gt_count - matched_count,
            "hit_recall": (matched_count / gt_count) * 100 if gt_count else 0.0,
            "hit_precision": (matched_count / cand_count) * 100 if cand_count else 0.0,
            "end_to_end_top1": (top1_count / gt_count) * 100 if gt_count else 0.0,
            "end_to_end_top2": (top2_count / gt_count) * 100 if gt_count else 0.0,
            "conditional_top1": (top1_count / matched_count) * 100 if matched_count else 0.0,
            "macro_f1": m_f1,
            "runtime_seconds": runtime,
            "per_class": data.get("per_class", {}),
        })

    overall_macro_f1 = f1_score(all_y_true, all_y_pred, average="macro", zero_division=0) * 100

    overall = {
        "total_clips": len(all_results),
        "total_gt": tot_gt,
        "total_candidates": tot_cand,
        "total_matched": tot_matched,
        "total_false_positives": tot_cand - tot_matched,
        "total_missed_hits": tot_gt - tot_matched,
        "hit_recall": (tot_matched / tot_gt) * 100 if tot_gt else 0.0,
        "hit_precision": (tot_matched / tot_cand) * 100 if tot_cand else 0.0,
        "end_to_end_top1": (tot_top1 / tot_gt) * 100 if tot_gt else 0.0,
        "end_to_end_top2": (tot_top2 / tot_gt) * 100 if tot_gt else 0.0,
        "conditional_top1": (tot_top1 / tot_matched) * 100 if tot_matched else 0.0,
        "macro_f1": overall_macro_f1,
        "total_runtime_seconds": tot_runtime,
        "per_class_aggregated": {
            cls: {
                "gt": stats["gt"],
                "matched": stats["matched"],
                "top1": stats["top1"],
                "top2": stats["top2"],
                "top1_acc": (stats["top1"] / stats["gt"]) * 100 if stats["gt"] else 0.0,
                "top2_acc": (stats["top2"] / stats["gt"]) * 100 if stats["gt"] else 0.0,
            }
            for cls, stats in class_stats.items()
        },
        "clips": all_results,
    }
    return overall


def main():
    paths = []
    for cfg in MATCH_CONFIGS:
        p = run_single_match(cfg)
        paths.append({"cfg": cfg, "path": p})

    summary = analyze_all_matches(paths)
    out_file = REPO_ROOT / "work_dirs" / "benchmark_5_matches_production_results.json"
    out_file.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print("\n" + "=" * 60)
    print("BENCHMARK SUMMARY SAVED TO:", out_file)
    print("=" * 60)
    print(json.dumps(summary["overall"] if "overall" in summary else {k: v for k, v in summary.items() if k != "clips"}, indent=2))


if __name__ == "__main__":
    main()

