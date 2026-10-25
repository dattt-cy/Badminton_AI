import csv, json, sys, time
from pathlib import Path

REPO_ROOT = Path(r"E:\HK1-2026\PBL6\AI_Classifier")
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

from scripts.evaluation.eval_compare_hit_models import evaluate_predictions

def load_clip_gt(manifest_path: Path, match_id: int, start_frame: int, end_frame: int):
    gt = []
    with open(manifest_path, encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            if str(r.get("match_id")) == str(match_id) or f"match{match_id}" in r.get("video_path", "").lower():
                hf = int(r["hit_frame"])
                if start_frame <= hf < end_frame:
                    gt.append({
                        "frame": hf - start_frame,
                        "stroke": r["coarse_label"],
                        "player_side": r.get("player_side", "unknown")
                    })
    gt.sort(key=lambda x: x["frame"])
    return gt

CLIPS_CONFIG = [
    {
        "clip_id": "Clip 1",
        "name": "Match 44 (Final - Men)",
        "players": "Viktor Axelsen vs Anders Antonsen",
        "match_id": 44,
        "raw_start": 45000,
        "raw_end": 63000,
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
        "video": REPO_ROOT / "work_dirs" / "eval_clips" / "clip3_match43_10min.mp4",
        "pipeline_json": REPO_ROOT / "work_dirs" / "eval_clips" / "clip3_match43_10min_pipeline.json",
    },
]

def evaluate_all():
    manifest_p = REPO_ROOT / "data" / "manifests" / "shuttleset_rgb.csv"
    results = []

    total_gt = 0
    total_pred = 0
    total_matched = 0
    total_fp = 0
    total_missed = 0
    total_top1 = 0

    print("\n" + "="*115)
    print(f"{'Clip ID':<8} | {'Match / Event':<26} | {'GT Hits':<8} | {'Pred Hits':<9} | {'Matched':<12} | {'FP (Rác)':<9} | {'Recall':<8} | {'Prec':<8} | {'Hit F1':<8} | {'End-to-End'}")
    print("="*115)

    for c in CLIPS_CONFIG:
        gt = load_clip_gt(manifest_p, c["match_id"], c["raw_start"], c["raw_end"])
        json_p = c["pipeline_json"]
        if not json_p.exists():
            print(f"{c['clip_id']:<8} | {c['name']:<26} | {len(gt):<8} | [WAITING - Pipeline not yet run]")
            continue
        
        with open(json_p, "r", encoding="utf-8") as f:
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
            "config": c,
            "gt_count": len(gt),
            "res": res,
        })

        m_str = f"{res['matched_hits']}/{len(gt)}"
        print(f"{c['clip_id']:<8} | {c['name']:<26} | {len(gt):<8} | {res['predicted_hits']:<9} | {m_str:<12} | {res['false_positives']:<9} | {res['hit_recall']:.2f}%{'':<1} | {res['hit_precision']:.2f}%{'':<1} | {res['hit_f1']:.2f}%{'':<1} | {res['end_to_end_top1']:.2f}%")

    if results:
        overall_recall = (total_matched / total_gt * 100.0) if total_gt else 0.0
        overall_prec = (total_matched / total_pred * 100.0) if total_pred else 0.0
        overall_f1 = (2 * overall_prec * overall_recall / (overall_prec + overall_recall)) if (overall_prec + overall_recall) else 0.0
        overall_e2e = (total_top1 / total_gt * 100.0) if total_gt else 0.0

        print("="*115)
        m_tot_str = f"{total_matched}/{total_gt}"
        print(f"{'OVERALL':<8} | {'3 Clips x 10 mins (30 mins)':<26} | {total_gt:<8} | {total_pred:<9} | {m_tot_str:<12} | {total_fp:<9} | {overall_recall:.2f}%{'':<1} | {overall_prec:.2f}%{'':<1} | {overall_f1:.2f}%{'':<1} | {overall_e2e:.2f}%")
        print("="*115)

if __name__ == "__main__":
    evaluate_all()

