"""Evaluate Full Production Pipeline on Completely New Clip: Match 37 (10 Minutes).

Match 37: Viktor Axelsen vs Hans-Kristian Solberg Vittinghus (TOYOTA Thailand Open 2021 Finals).
Clip 9: Raw frames 15000 to 33000 (10 minutes, 18,151 frames at 30 fps).
Ground Truth hits: 211 hits.
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path
from collections import Counter

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from scripts.evaluation.eval_compare_hit_models import evaluate_predictions


def main():
    output_json = REPO_ROOT / "work_dirs" / "eval_clips" / "clip9_match37_10min_pipeline.json"
    manifest_p = REPO_ROOT / "data" / "manifests" / "shuttleset_rgb.csv"

    # 1. Load Ground Truth hits for Match 37 in frame window [15000, 33000]
    gt = []
    with open(manifest_p, encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            if r.get("match_id") == "37":
                hf = int(float(r["hit_frame"]))
                if 15000 <= hf < 33000:
                    gt.append({
                        "frame": hf - 15000,
                        "stroke": r["coarse_label"],
                        "player_side": r.get("player_side", "unknown"),
                    })
    gt.sort(key=lambda x: x["frame"])

    # 2. Load Predictions from pipeline JSON
    with open(output_json, "r", encoding="utf-8") as f:
        data = json.load(f)
    preds = data.get("predictions", [])

    # Apply TrackNet shuttle visibility filter (>= 0.25)
    preds = [
        p for p in preds
        if p.get("analytics", {}).get("shuttle_visibility_ratio", 1.0) is None
        or p.get("analytics", {}).get("shuttle_visibility_ratio", 1.0) >= 0.25
    ]

    # 3. Evaluate Metrics (Hit Spotting & End-to-End Stroke Classification)
    res = evaluate_predictions(preds, gt, tolerance=12)

    # Format result without cp1258 encoding errors
    print("=" * 90)
    print(">>> KET QUA DANH GIA THUC TE TREN CLIP HOAN TOAN MOI: MATCH 37 (10 PHUT) <<<")
    print("=" * 90)
    print("Tran dau: Match 37 (Chung ket Toyota Thailand Open 2021)")
    print("Van dong vien: Viktor Axelsen vs Hans-Kristian Solberg Vittinghus")
    print("Thoi luong video: 10 phut (18,151 frames @ 30fps)")
    print("-" * 90)
    print(f"Tong so cu danh that (Ground Truth Hits)      : {res['ground_truth_hits']}")
    print(f"Tong so cu danh AI bat duoc (Predicted Hits) : {res['predicted_hits']}")
    print(f"So cu bat trung chinh xac (Matched Hits)      : {res['matched_hits']} / {res['ground_truth_hits']}")
    print(f"So cu bat truot (Missed Hits)                 : {res['missed_hits']}")
    print(f"So cu bat rac (False Positives)               : {res['false_positives']}")
    print("-" * 90)
    print(f"DO BAT TRUNG CU DANH (RECALL)          : {res['hit_recall']:.2f}%")
    print(f"DO CHINH XAC PHAN BIET (PRECISION)     : {res['hit_precision']:.2f}%")
    print(f"DIEM F1 TOAN DIEN (HIT F1)             : {res['hit_f1']:.2f}%")
    print(f"PHAN LOAI KY THUAT (END-TO-END ACC)    : {res['end_to_end_top1']:.2f}%")
    print("=" * 90)

    # Summary of strokes detected
    stroke_counts = Counter(p["stroke"] for p in preds)
    print("\nPhan bo ky thuat danh (AI Predictions):")
    for stroke, count in sorted(stroke_counts.items(), key=lambda x: -x[1]):
        pct = (count / len(preds)) * 100 if preds else 0
        print(f"  - {stroke:<15}: {count:>3} cu ({pct:>5.1f}%)")

    # Rally info
    rallies = data.get("scoreboard", {}).get("rallies", [])
    print(f"\nSo luong pha cau (Rallies) phat hien: {len(rallies)}")
    print(f"Ty so uoc luong (Scoreboard): {data.get('summary', {}).get('estimated_score', {}).get('text')}")


if __name__ == "__main__":
    main()
