"""Evaluate Full Production Pipeline on Match 37 (10 Minutes) with Stroke Side (Forehand/Backhand).

Match 37: Viktor Axelsen vs Hans-Kristian Solberg Vittinghus (TOYOTA Thailand Open 2021 Finals).
Clip 9: Raw frames 15000 to 33000 (10 minutes, 18,151 frames at 30 fps).
Ground Truth hits: 211 hits.
"""

from __future__ import annotations

import csv
import json
import subprocess
import sys
import time
from pathlib import Path
from collections import Counter

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from scripts.evaluation.eval_compare_hit_models import evaluate_predictions


def main():
    clip_video = REPO_ROOT / "work_dirs" / "eval_clips" / "clip9_match37_10min.mp4"
    output_json = REPO_ROOT / "work_dirs" / "eval_clips" / "clip9_match37_10min_pipeline.json"
    manifest_p = REPO_ROOT / "data" / "manifests" / "shuttleset_rgb.csv"

    # 1. Run Pipeline on GPU if not already executed or forced
    force_rerun = True
    if force_rerun or not output_json.exists():
        print("=" * 95)
        print(">>> RUNNING FULL END-TO-END PIPELINE WITH STROKE SIDE: MATCH 37 (10 MINS) <<<")
        print("=" * 95)
        t0 = time.time()
        cmd = [
            sys.executable,
            str(REPO_ROOT / "run_pipeline.py"),
            "--video", str(clip_video),
            "--output", str(output_json),
            "--player-upper", "Viktor Axelsen",
            "--player-lower", "Hans-Kristian Vittinghus",
            "--threshold", "0.72",
            "--nms-radius", "6",
            "--same-side-nms-radius", "20",
            "--serve-rescue-threshold", "0.55",
            "--bridge-rescue-threshold", "0.55",
            "--min-shuttle-visibility", "0.25",
            "--use-tactical-decoder",
            "--no-auto-ocr",
        ]
        ret = subprocess.run(cmd)
        if ret.returncode != 0:
            raise RuntimeError(f"Pipeline failed with exit code {ret.returncode}")
        print(f"\n[+] Pipeline completed in {time.time() - t0:.1f}s\n")

    # 2. Load Ground Truth hits for Match 37 in frame window [15000, 33000]
    gt = []
    with open(manifest_p, encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            if r.get("match_id") == "37":
                hf = int(float(r["hit_frame"]))
                if 15000 <= hf < 33000:
                    gt.append({
                        "frame": hf - 15000,
                        "stroke": r["coarse_label"],
                        "stroke_side": r.get("stroke_side", "unknown"),
                        "player_side": r.get("player_side", "unknown"),
                    })
    gt.sort(key=lambda x: x["frame"])

    # 3. Load Predictions from pipeline JSON
    with open(output_json, "r", encoding="utf-8") as f:
        data = json.load(f)
    preds = data.get("predictions", [])

    # Apply TrackNet shuttle visibility filter (>= 0.25)
    preds = [
        p for p in preds
        if p.get("analytics", {}).get("shuttle_visibility_ratio", 1.0) is None
        or p.get("analytics", {}).get("shuttle_visibility_ratio", 1.0) >= 0.25
    ]

    # 4. Evaluate Metrics (Hit Spotting, Stroke, Stroke Side, and Combined Fine-grained)
    res = evaluate_predictions(preds, gt, tolerance=12)

    print("=" * 95)
    print(">>> KET QUA FULL PIPELINE KET HOP STROKE + HUONG TAY (FOREHAND/BACKHAND) <<<")
    print("=" * 95)
    print("Tran dau: Match 37 (Chung ket Toyota Thailand Open 2021)")
    print("Van dong vien: Viktor Axelsen vs Hans-Kristian Solberg Vittinghus")
    print("Thoi luong video: 10 phut (18,151 frames @ 30fps)")
    print("-" * 95)
    print(f"Tong so cu danh that (Ground Truth Hits)      : {res['ground_truth_hits']}")
    print(f"Tong so cu danh AI bat duoc (Predicted Hits) : {res['predicted_hits']}")
    print(f"So cu bat trung chinh xac (Matched Hits)      : {res['matched_hits']} / {res['ground_truth_hits']}")
    print(f"So cu bat truot (Missed Hits)                 : {res['missed_hits']}")
    print(f"So cu bat rac (False Positives)               : {res['false_positives']}")
    print("-" * 95)
    print("--- 1. PHAT HIEN THOI DIEM CU DANH (HIT SPOTTING) ---")
    print(f"  * DO BAT TRUNG CU DANH (RECALL)             : {res['hit_recall']:.2f}%")
    print(f"  * DO CHINH XAC PHAN BIET (PRECISION)        : {res['hit_precision']:.2f}%")
    print(f"  * DIEM F1 TOAN DIEN (HIT F1)                : {res['hit_f1']:.2f}%")
    print("-" * 95)
    print("--- 2. PHAN LOAI HUONG TAY (STROKE SIDE: FOREHAND / BACKHAND / AROUNDHEAD) ---")
    print(f"  * DO CHINH XAC 3 MAT VOT (FOREHAND / BACKHAND / AROUNDHEAD): {res['stroke_side_accuracy']:.2f}%")
    print(f"  * DO CHINH XAC MAT VOT (GOP AROUNDHEAD VAO FOREHAND)       : {res['face_accuracy']:.2f}%")
    print("-" * 95)
    print("--- 3. END-TO-END (KET HOP THOI DIEM + KY THUAT + HUONG TAY) ---")
    print(f"  * END-TO-END COARSE (Hit + 8 Ky Thuat)                     : {res['end_to_end_top1']:.2f}%")
    print(f"  * END-TO-END FINE (Hit + 8 Ky Thuat + 3 Huong Tay)         : {res['end_to_end_fine']:.2f}%")
    print(f"  * END-TO-END BINARY (Hit + 8 Ky Thuat + Mat Thuan/Trai)    : {res['end_to_end_fine_face']:.2f}%")
    print(f"  * CONDITIONAL FINE FACE (Tren cac cu da bat trung)         : {res['conditional_fine_face']:.2f}%")
    print("=" * 95)

    # Breakdown by stroke side
    side_counts = Counter(p.get("stroke_side", "unknown") for p in preds)
    print("\nPhan bo huong mat vot (Predicted Stroke Side):")
    for side, count in sorted(side_counts.items(), key=lambda x: -x[1]):
        pct = (count / len(preds)) * 100 if preds else 0
        print(f"  - {side:<12}: {count:>3} cu ({pct:>5.1f}%)")

    # Breakdown by 8 strokes
    stroke_counts = Counter(p["stroke"] for p in preds)
    print("\nPhan bo ky thuat danh (Predicted Strokes):")
    for stroke, count in sorted(stroke_counts.items(), key=lambda x: -x[1]):
        pct = (count / len(preds)) * 100 if preds else 0
        print(f"  - {stroke:<15}: {count:>3} cu ({pct:>5.1f}%)")

    # Save detailed evaluation JSON
    eval_out = REPO_ROOT / "work_dirs" / "eval_clips" / "clip9_match37_stroke_side_eval.json"
    with open(eval_out, "w", encoding="utf-8") as f:
        json.dump(res, f, indent=2)
    print(f"\n[+] Saved detailed evaluation to: {eval_out}")


if __name__ == "__main__":
    main()
