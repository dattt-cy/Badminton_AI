import json, csv
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

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

from scripts.evaluation.eval_compare_hit_models import evaluate_predictions

old_file = Path("AI_Classifier/work_dirs/match44_20min_pipeline_with_tactical_decoder.json")
with old_file.open("r", encoding="utf-8") as f:
    d = json.load(f)

preds = d.get("predictions", [])
res = evaluate_predictions(preds, gt_20min, tolerance=12)

print("\n=== BASELINE R(2+1)D KET QUA TREN CLIP 20 PHUT ===")
print(f"Total GT Hits: {len(gt_20min)}")
print(f"Predicted: {res['predicted_hits']}")
print(f"Matched: {res['matched_hits']} / {len(gt_20min)}")
print(f"False Positives (Junk): {res['false_positives']}")
print(f"Missed: {res['missed_hits']}")
print(f"Hit Recall: {res['hit_recall']:.2f}%")
print(f"Hit Precision: {res['hit_precision']:.2f}%")
print(f"Hit F1-Score: {res['hit_f1']:.2f}%")
print(f"Conditional Top-1: {res['conditional_top1']:.2f}%")
print(f"End-to-End Top-1: {res['end_to_end_top1']:.2f}%")
print(f"Hitter Accuracy: {res['hitter_accuracy']:.2f}%")

with open("AI_Classifier/work_dirs/baseline_20min_eval.json", "w", encoding="utf-8") as f:
    json.dump(res, f, indent=2)

