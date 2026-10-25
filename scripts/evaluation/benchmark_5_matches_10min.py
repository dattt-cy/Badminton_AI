"""Comprehensive 5-Match Benchmark on 10-Minute Independent Clips.

Matches evaluated (All 5 Held-out BWF World Tour Finals Test Matches):
1. Clip 1: Match 44 (Men's Final) - Viktor Axelsen vs Anders Antonsen (158 hits)
2. Clip 2: Match 39 (Men's QF) - Anthony Ginting vs Lee Zii Jia (152 hits)
3. Clip 3: Match 43 (Women's SF) - Carolina Marin vs Pornpawee Chochuwong (123 hits)
4. Clip 4: Match 40 (Women's QF) - Evgeniya Kosetskaya vs Michelle Li (160 hits)
5. Clip 5: Match 42 (Women's QF) - Pusarla V. Sindhu vs Pornpawee Chochuwong (133 hits)

Total Ground Truth Hits: ~726 hits across 50 minutes of real match video.
"""

from __future__ import annotations

import csv
import json
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from scripts.evaluation.eval_compare_hit_models import evaluate_predictions

CLIPS_CONFIG = [
    {
        "clip_id": "Clip 1",
        "name": "Match 44 (Final - Men)",
        "players": "Viktor Axelsen vs Anders Antonsen",
        "match_id": 44,
        "raw_start": 45000,
        "raw_end": 63000,
        "player_upper": "Anders Antonsen",
        "player_lower": "Viktor Axelsen",
        "video": REPO_ROOT / "work_dirs" / "eval_clips" / "clip1_match44_10min.mp4",
        "pipeline_json": REPO_ROOT / "work_dirs" / "eval_clips" / "clip1_match44_10min_pipeline.json",
    },
    {
        "clip_id": "Clip 2",
        "name": "Match 39 (QF - Men)",
        "players": "Anthony Ginting vs Lee Zii Jia",
        "match_id": 39,
        "raw_start": 15000,
        "raw_end": 33000,
        "player_upper": "Lee Zii Jia",
        "player_lower": "Anthony Ginting",
        "video": REPO_ROOT / "work_dirs" / "eval_clips" / "clip2_match39_10min.mp4",
        "pipeline_json": REPO_ROOT / "work_dirs" / "eval_clips" / "clip2_match39_10min_pipeline.json",
    },
    {
        "clip_id": "Clip 3",
        "name": "Match 43 (SF - Women)",
        "players": "Carolina Marin vs Pornpawee Chochuwong",
        "match_id": 43,
        "raw_start": 15000,
        "raw_end": 33000,
        "player_upper": "Pornpawee Chochuwong",
        "player_lower": "Carolina Marin",
        "video": REPO_ROOT / "work_dirs" / "eval_clips" / "clip3_match43_10min.mp4",
        "pipeline_json": REPO_ROOT / "work_dirs" / "eval_clips" / "clip3_match43_10min_pipeline.json",
    },
    {
        "clip_id": "Clip 4",
        "name": "Match 40 (QF - Women)",
        "players": "Evgeniya Kosetskaya vs Michelle Li",
        "match_id": 40,
        "raw_start": 15000,
        "raw_end": 33000,
        "player_upper": "Michelle Li",
        "player_lower": "Evgeniya Kosetskaya",
        "video": REPO_ROOT / "work_dirs" / "eval_clips" / "clip4_match40_10min.mp4",
        "pipeline_json": REPO_ROOT / "work_dirs" / "eval_clips" / "clip4_match40_10min_pipeline.json",
    },
    {
        "clip_id": "Clip 5",
        "name": "Match 42 (QF - Women)",
        "players": "Pusarla V. Sindhu vs Pornpawee Chochuwong",
        "match_id": 42,
        "raw_start": 15000,
        "raw_end": 33000,
        "player_upper": "Pornpawee Chochuwong",
        "player_lower": "Pusarla V. Sindhu",
        "video": REPO_ROOT / "work_dirs" / "eval_clips" / "clip5_match42_10min.mp4",
        "pipeline_json": REPO_ROOT / "work_dirs" / "eval_clips" / "clip5_match42_10min_pipeline.json",
    },
]


def load_clip_gt(manifest_path: Path, match_id: int, start_frame: int, end_frame: int):
    gt = []
    with open(manifest_path, encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            if str(r.get("match_id")) == str(match_id) or f"match{match_id}" in r.get("video_path", "").lower():
                hf = int(float(r["hit_frame"]))
                if start_frame <= hf < end_frame:
                    gt.append({
                        "frame": hf - start_frame,
                        "stroke": r["coarse_label"],
                        "player_side": r.get("player_side", "unknown")
                    })
    gt.sort(key=lambda x: x["frame"])
    return gt


def run_pipeline_if_missing(cfg: dict) -> None:
    json_path = cfg["pipeline_json"]
    if json_path.exists():
        return

    print(f"\n>>> Running pipeline for {cfg['clip_id']} ({cfg['name']}) on GPU... <<<", flush=True)
    t0 = time.time()
    ckpt_hit = REPO_ROOT / "src" / "ai_classifier" / "models" / "checkpoints" / "badminton_hitnet_v2.pth"

    cmd = [
        sys.executable,
        str(REPO_ROOT / "run_pipeline.py"),
        "--video", str(cfg["video"]),
        "--output", str(json_path),
        "--player-upper", cfg["player_upper"],
        "--player-lower", cfg["player_lower"],
        "--hit-checkpoint", str(ckpt_hit),
        "--threshold", "0.72",
        "--nms-radius", "6",
        "--same-side-nms-radius", "20",
        "--serve-rescue-threshold", "0.55",
        "--bridge-rescue-threshold", "0.55",
        "--use-tactical-decoder",
        "--no-auto-ocr",
    ]
    ret = subprocess.run(cmd)
    elapsed = time.time() - t0
    if ret.returncode != 0:
        raise RuntimeError(f"Pipeline failed for {cfg['clip_id']} with code {ret.returncode}")
    print(f"[+] Completed {cfg['clip_id']} in {elapsed:.1f}s -> {json_path.name}\n", flush=True)


def main():
    manifest_p = REPO_ROOT / "data" / "manifests" / "shuttleset_rgb.csv"
    results = []

    total_gt = 0
    total_pred = 0
    total_matched = 0
    total_fp = 0
    total_missed = 0
    total_top1 = 0

    print("=" * 115)
    print(">>> 5-MATCH BWF PRODUCTION BENCHMARK (50 MINS / 5 HELD-OUT TEST MATCHES) <<<")
    print("=" * 115)

    # 1. Ensure all clips have pipeline JSON
    for c in CLIPS_CONFIG:
        run_pipeline_if_missing(c)

    # 2. Evaluate all 5 clips
    print("\n" + "=" * 115)
    print(f"{'Clip ID':<8} | {'Match / Event':<26} | {'GT Hits':<8} | {'Pred Hits':<9} | {'Matched':<12} | {'FP (Rac)':<9} | {'Recall':<8} | {'Prec':<8} | {'Hit F1':<8} | {'End-to-End'}")
    print("=" * 115)

    for c in CLIPS_CONFIG:
        gt = load_clip_gt(manifest_p, c["match_id"], c["raw_start"], c["raw_end"])
        with open(c["pipeline_json"], "r", encoding="utf-8") as f:
            data = json.load(f)
        preds = data.get("predictions", [])
        res = evaluate_predictions(preds, gt, tolerance=12)

        total_gt += res["ground_truth_hits"]
        total_pred += res["predicted_hits"]
        total_matched += res["matched_hits"]
        total_fp += res["false_positives"]
        total_missed += res["missed_hits"]
        total_top1 += int(round(res["end_to_end_top1"] * res["ground_truth_hits"] / 100.0))

        results.append({
            "clip_id": c["clip_id"],
            "name": c["name"],
            "players": c["players"],
            "gt_hits": len(gt),
            "pred_hits": res["predicted_hits"],
            "matched": res["matched_hits"],
            "false_positives": res["false_positives"],
            "recall": res["hit_recall"],
            "precision": res["hit_precision"],
            "f1": res["hit_f1"],
            "e2e_top1": res["end_to_end_top1"],
        })

        m_str = f"{res['matched_hits']}/{len(gt)}"
        print(f"{c['clip_id']:<8} | {c['name']:<26} | {len(gt):<8} | {res['predicted_hits']:<9} | {m_str:<12} | {res['false_positives']:<9} | {res['hit_recall']:.2f}%{'':<1} | {res['hit_precision']:.2f}%{'':<1} | {res['hit_f1']:.2f}%{'':<1} | {res['end_to_end_top1']:.2f}%")

    # Overall Summary
    overall_rec = (total_matched / total_gt * 100.0) if total_gt else 0.0
    overall_prec = (total_matched / total_pred * 100.0) if total_pred else 0.0
    overall_f1 = (2 * overall_rec * overall_prec / (overall_rec + overall_prec)) if (overall_rec + overall_prec) else 0.0
    overall_e2e = (total_top1 / total_gt * 100.0) if total_gt else 0.0

    print("=" * 115)
    all_str = f"{total_matched}/{total_gt}"
    print(f"{'OVERALL':<8} | {'5 Clips x 10m (50 mins)':<26} | {total_gt:<8} | {total_pred:<9} | {all_str:<12} | {total_fp:<9} | {overall_rec:.2f}%{'':<1} | {overall_prec:.2f}%{'':<1} | {overall_f1:.2f}%{'':<1} | {overall_e2e:.2f}%")
    print("=" * 115)

    summary_file = REPO_ROOT / "work_dirs" / "benchmark_5_matches_results.json"
    summary_file.parent.mkdir(parents=True, exist_ok=True)
    summary_data = {
        "benchmark_name": "5_matches_10min_test_held_out",
        "total_clips": len(results),
        "total_gt_hits": total_gt,
        "total_pred_hits": total_pred,
        "total_matched_hits": total_matched,
        "total_false_positives": total_fp,
        "overall_recall": round(overall_rec, 2),
        "overall_precision": round(overall_prec, 2),
        "overall_f1": round(overall_f1, 2),
        "overall_e2e_top1": round(overall_e2e, 2),
        "clips": results,
    }
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=2, ensure_ascii=False)
    print(f"\n[+] Detailed benchmark results saved to: {summary_file}")


if __name__ == "__main__":
    main()

