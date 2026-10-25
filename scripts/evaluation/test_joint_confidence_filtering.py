import json, csv
from pathlib import Path

# Load GT
manifest_path = Path("AI_Classifier/data/manifests/shuttleset_rgb.csv")
gt_20min = []
with manifest_path.open(encoding="utf-8-sig", newline="") as handle:
    for row in csv.DictReader(handle):
        absolute = int(row["hit_frame"])
        match_id = row.get("match_id", "")
        name = Path(row["video_path"]).name
        same_match = match_id == "44" or name.startswith("44 ") or "match44" in name.lower()
        if same_match and 45000 <= absolute < 81000:
            gt_20min.append({
                "frame": absolute - 45000,
                "stroke": row["coarse_label"],
                "player_side": row.get("player_side", "unknown")
            })
gt_20min.sort(key=lambda r: r["frame"])

new_file = Path("AI_Classifier/work_dirs/match44_20min_hitnet_cleaned.json")
with new_file.open("r", encoding="utf-8") as f:
    n_data = json.load(f)

preds = n_data.get("predictions", [])

import sys
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
from scripts.evaluation.eval_compare_hit_models import evaluate_predictions

print("=== GOC (CURRENT 257 PREDICTIONS) ===")
res_orig = evaluate_predictions(preds, gt_20min, tolerance=12)
print(f"Preds: {res_orig['predicted_hits']} | Matched: {res_orig['matched_hits']} | FP: {res_orig['false_positives']} | Prec: {res_orig['hit_precision']:.2f}% | F1: {res_orig['hit_f1']:.2f}% | E2E: {res_orig['end_to_end_top1']:.2f}%")

print("\n=== THU NGHIEM CAC BO LOC LIEN KET (HIT SCORE + KAP CONFIDENCE) ===")
for min_hscore in [0.75, 0.78, 0.80]:
    for min_kconf in [0.50, 0.55, 0.58, 0.60]:
        # Filter: reject if hit_score < min_hscore AND kap_conf < min_kconf
        filtered = [
            p for p in preds
            if not (p.get("hit_score", 0) < min_hscore and p.get("confidence", 0) < min_kconf)
        ]
        r = evaluate_predictions(filtered, gt_20min, tolerance=12)
        if r['false_positives'] < res_orig['false_positives'] and r['matched_hits'] >= 234:
            print(f"Reject if (HitScore<{min_hscore} and KAPConf<{min_kconf}): Preds={r['predicted_hits']:3d} | Matched={r['matched_hits']:3d}/247 | FP={r['false_positives']:2d} (Giam!) | Prec={r['hit_precision']:.2f}% | F1={r['hit_f1']:.2f}% | E2E={r['end_to_end_top1']:.2f}%")
