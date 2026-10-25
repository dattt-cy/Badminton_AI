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

# 1. Baseline R(2+1)D-18
old_file = Path("AI_Classifier/work_dirs/match44_20min_pipeline_with_tactical_decoder.json")
with old_file.open("r", encoding="utf-8") as f:
    b_data = json.load(f)
b_preds = b_data.get("predictions", [])
b_res = evaluate_predictions(b_preds, gt_20min, tolerance=12)

# 2. BadmintonHitNet Cleaned
new_file = Path("AI_Classifier/work_dirs/match44_20min_hitnet_cleaned.json")
with new_file.open("r", encoding="utf-8") as f:
    n_data = json.load(f)
n_preds = n_data.get("predictions", [])
n_res = evaluate_predictions(n_preds, gt_20min, tolerance=12)

print("\n" + "="*85)
print(f"{'Metric':<30} | {'Baseline (R(2+1)D-18)':<24} | {'BadmintonHitNet (New)':<25}")
print("="*85)
metrics = [
    ("Architecture", "R(2+1)D (Kinetics Pretrained)", "BadmintonHitNet (100% Scratch)"),
    ("Model Parameters", "31.5 Million", "0.55 Million (58x lighter)"),
    ("Total GT Hits (20 mins)", str(len(gt_20min)), str(len(gt_20min))),
    ("Predicted Hits", str(b_res['predicted_hits']), str(n_res['predicted_hits'])),
    ("Matched Hits (Recall hits)", f"{b_res['matched_hits']} / {len(gt_20min)}", f"{n_res['matched_hits']} / {len(gt_20min)}"),
    ("False Positives (Junk Hits)", f"{b_res['false_positives']} hits", f"{n_res['false_positives']} hits"),
    ("Missed Hits", f"{b_res['missed_hits']} hits", f"{n_res['missed_hits']} hits"),
    ("Hit Recall", f"{b_res['hit_recall']:.2f}%", f"{n_res['hit_recall']:.2f}%"),
    ("Hit Precision", f"{b_res['hit_precision']:.2f}%", f"{n_res['hit_precision']:.2f}%"),
    ("Hit F1-Score", f"{b_res['hit_f1']:.2f}%", f"{n_res['hit_f1']:.2f}%"),
    ("Mean Frame Error (MAE)", f"{b_res['mean_frame_error']:.2f} frames", f"{n_res['mean_frame_error']:.2f} frames"),
    ("Hitter Side Accuracy", f"{b_res['hitter_accuracy']:.2f}%", f"{n_res['hitter_accuracy']:.2f}%"),
    ("Conditional Stroke Top-1", f"{b_res['conditional_top1']:.2f}%", f"{n_res['conditional_top1']:.2f}%"),
    ("Conditional Stroke Top-2", f"{b_res['conditional_top2']:.2f}%", f"{n_res['conditional_top2']:.2f}%"),
    ("End-to-End Stroke Top-1", f"{b_res['end_to_end_top1']:.2f}%", f"{n_res['end_to_end_top1']:.2f}%"),
    ("End-to-End Stroke Top-2", f"{b_res['end_to_end_top2']:.2f}%", f"{n_res['end_to_end_top2']:.2f}%"),
]

for label, b_val, n_val in metrics:
    print(f"{label:<30} | {b_val:<24} | {n_val:<25}")
print("="*85)

# Save comparison JSON
comparison_dump = {
    "baseline_r2plus1d_20min": b_res,
    "badminton_hitnet_20min": n_res
}
with open("AI_Classifier/work_dirs/match44_20min_comparison_results.json", "w", encoding="utf-8") as f:
    json.dump(comparison_dump, f, indent=2)
print("\nSaved comparison to AI_Classifier/work_dirs/match44_20min_comparison_results.json")

