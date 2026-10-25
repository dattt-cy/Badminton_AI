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
b_file = Path("AI_Classifier/work_dirs/match44_20min_pipeline_with_tactical_decoder.json")
with b_file.open("r", encoding="utf-8") as f:
    b_data = json.load(f)
b_preds = b_data.get("predictions", [])
b_res = evaluate_predictions(b_preds, gt_20min, tolerance=12)

# 2. BadmintonHitNet V1
v1_file = Path("AI_Classifier/work_dirs/match44_20min_hitnet_cleaned.json")
with v1_file.open("r", encoding="utf-8") as f:
    v1_data = json.load(f)
v1_preds = v1_data.get("predictions", [])
v1_res = evaluate_predictions(v1_preds, gt_20min, tolerance=12)

# 3. BadmintonHitNet V2 Full (Raw)
v2_file = Path("AI_Classifier/work_dirs/match44_20min_hitnet_v2_full.json")
with v2_file.open("r", encoding="utf-8") as f:
    v2_data = json.load(f)
v2_preds = v2_data.get("predictions", [])
v2_res = evaluate_predictions(v2_preds, gt_20min, tolerance=12)

# 4. BadmintonHitNet V2 with Hit-KAP Confidence Filter
# Reject ambiguous hits with low hit score AND low action confidence
v2_cleaned_preds = [
    p for p in v2_preds
    if not (p.get("hit_score", 1.0) < 0.80 and p.get("confidence", 1.0) < 0.55)
]
v2_clean_res = evaluate_predictions(v2_cleaned_preds, gt_20min, tolerance=12)

print("\n" + "="*105)
print(f"{'Metric':<28} | {'Baseline R(2+1)D':<18} | {'HitNet V1 (0.55M)':<18} | {'HitNet V2 Raw':<18} | {'HitNet V2 Cleaned':<18}")
print("="*105)

metrics = [
    ("Model Parameters", "31.5M (Pretrained)", "0.55M (Scratch)", "0.98M (Scratch)", "0.98M (Scratch)"),
    ("Total GT Hits (20 mins)", str(len(gt_20min)), str(len(gt_20min)), str(len(gt_20min)), str(len(gt_20min))),
    ("Predicted Hits", str(b_res['predicted_hits']), str(v1_res['predicted_hits']), str(v2_res['predicted_hits']), str(v2_clean_res['predicted_hits'])),
    ("Matched Hits (Recall hits)", f"{b_res['matched_hits']} / {len(gt_20min)}", f"{v1_res['matched_hits']} / {len(gt_20min)}", f"{v2_res['matched_hits']} / {len(gt_20min)}", f"{v2_clean_res['matched_hits']} / {len(gt_20min)}"),
    ("False Positives (Junk Hits)", f"{b_res['false_positives']} hits", f"{v1_res['false_positives']} hits", f"{v2_res['false_positives']} hits", f"{v2_clean_res['false_positives']} hits"),
    ("Missed Hits", f"{b_res['missed_hits']} hits", f"{v1_res['missed_hits']} hits", f"{v2_res['missed_hits']} hits", f"{v2_clean_res['missed_hits']} hits"),
    ("Hit Recall", f"{b_res['hit_recall']:.2f}%", f"{v1_res['hit_recall']:.2f}%", f"{v2_res['hit_recall']:.2f}%", f"{v2_clean_res['hit_recall']:.2f}%"),
    ("Hit Precision", f"{b_res['hit_precision']:.2f}%", f"{v1_res['hit_precision']:.2f}%", f"{v2_res['hit_precision']:.2f}%", f"{v2_clean_res['hit_precision']:.2f}%"),
    ("Hit F1-Score", f"{b_res['hit_f1']:.2f}%", f"{v1_res['hit_f1']:.2f}%", f"{v2_res['hit_f1']:.2f}%", f"{v2_clean_res['hit_f1']:.2f}%"),
    ("Mean Frame Error (MAE)", f"{b_res['mean_frame_error']:.2f} frames", f"{v1_res['mean_frame_error']:.2f} frames", f"{v2_res['mean_frame_error']:.2f} frames", f"{v2_clean_res['mean_frame_error']:.2f} frames"),
    ("Hitter Side Accuracy", f"{b_res['hitter_accuracy']:.2f}%", f"{v1_res['hitter_accuracy']:.2f}%", f"{v2_res['hitter_accuracy']:.2f}%", f"{v2_clean_res['hitter_accuracy']:.2f}%"),
    ("Conditional Stroke Top-1", f"{b_res['conditional_top1']:.2f}%", f"{v1_res['conditional_top1']:.2f}%", f"{v2_res['conditional_top1']:.2f}%", f"{v2_clean_res['conditional_top1']:.2f}%"),
    ("End-to-End Stroke Top-1", f"{b_res['end_to_end_top1']:.2f}%", f"{v1_res['end_to_end_top1']:.2f}%", f"{v2_res['end_to_end_top1']:.2f}%", f"{v2_clean_res['end_to_end_top1']:.2f}%"),
]

for label, b_v, v1_v, v2_v, v2c_v in metrics:
    print(f"{label:<28} | {b_v:<18} | {v1_v:<18} | {v2_v:<18} | {v2c_v:<18}")
print("="*105)

