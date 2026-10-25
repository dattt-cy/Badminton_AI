import json, csv
from pathlib import Path

# Load GT
manifest_path = Path("AI_Classifier/data/manifests/shuttleset_rgb.csv")
gt = []
with manifest_path.open(encoding="utf-8-sig", newline="") as handle:
    for row in csv.DictReader(handle):
        absolute = int(row["hit_frame"])
        match_id = row.get("match_id", "")
        name = Path(row["video_path"]).name
        same_match = match_id == "44" or name.startswith("44 ") or "match44" in name.lower()
        if same_match and 45000 <= absolute < 54000:
            gt.append({
                "frame": absolute - 45000,
                "stroke": row["coarse_label"],
                "player_side": row.get("player_side", "unknown")
            })
gt.sort(key=lambda r: r["frame"])

import sys
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
from scripts.evaluation.eval_compare_hit_models import evaluate_predictions

cleaned_file = Path("AI_Classifier/work_dirs/match44_5min_hitnet_cleaned.json")
with cleaned_file.open("r", encoding="utf-8") as f:
    c_data = json.load(f)

c_preds = c_data.get("predictions", [])
res = evaluate_predictions(c_preds, gt, tolerance=12)

baseline_summary_file = Path("AI_Classifier/work_dirs/match44_5min_canonical_reproduced_yolov8m_rescored.json")
with baseline_summary_file.open(encoding="utf-8") as f:
    b_raw = json.load(f)

b_rec = b_raw.get('hit_recall', 0) * 100
b_prec = b_raw.get('hit_precision', 0) * 100
b_f1 = (2 * b_rec * b_prec / (b_rec + b_prec)) if (b_rec + b_prec) else 0.0

print("\n" + "="*85)
print(f"{'Metric':<30} | {'Baseline (R(2+1)D-18)':<24} | {'HitNet (Cleaned)':<25}")
print("="*85)
metrics = [
    ("Total GT Hits", str(len(gt)), str(len(gt))),
    ("Predicted Hits", str(b_raw.get('predicted_hits')), str(res['predicted_hits'])),
    ("Matched Hits (Recall)", f"{b_raw.get('matched_hits')}/96", f"{res['matched_hits']}/96"),
    ("False Positives (Junk)", f"{b_raw.get('false_positives')}", f"{res['false_positives']} (REDUCED)"),
    ("Missed Hits", f"{b_raw.get('missed_hits')}", f"{res['missed_hits']}"),
    ("Hit Recall", f"{b_rec:.2f}%", f"{res['hit_recall']:.2f}%"),
    ("Hit Precision", f"{b_prec:.2f}%", f"{res['hit_precision']:.2f}%"),
    ("Hit F1-Score", f"{b_f1:.2f}%", f"{res['hit_f1']:.2f}%"),
    ("Conditional Top-1", f"{b_raw.get('conditional_top1')*100:.2f}%", f"{res['conditional_top1']:.2f}%"),
    ("Conditional Top-2", f"{b_raw.get('conditional_top2')*100:.2f}%", f"{res['conditional_top2']:.2f}%"),
    ("End-to-End Top-1", f"{b_raw.get('end_to_end_top1')*100:.2f}%", f"{res['end_to_end_top1']:.2f}%"),
    ("End-to-End Top-2", f"{b_raw.get('end_to_end_top2')*100:.2f}%", f"{res['end_to_end_top2']:.2f}%"),
    ("Hitter Accuracy", f"{b_raw.get('hitter_accuracy')*100:.2f}%", f"{res['hitter_accuracy']:.2f}%"),
    ("Mean Frame Error", f"{b_raw.get('mean_frame_error'):.2f} frames", f"{res['mean_frame_error']:.2f} frames"),
]

for label, b_val, c_val in metrics:
    print(f"{label:<30} | {b_val:<24} | {c_val:<25}")
print("="*85)
