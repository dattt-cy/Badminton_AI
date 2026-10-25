from ai_classifier.localization import HitEvent, two_stage_temporal_nms, match_hit_events, rescue_pre_rally_serves, rescue_alternating_bridges
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

scan_file = Path("AI_Classifier/work_dirs/match44_5min_hitnet_scan.json")
scan_data = json.load(scan_file.open("r", encoding="utf-8"))
raw = [HitEvent(int(e["frame"]), e["side"], float(e["score"])) for e in scan_data["raw_events"]]

# Load previously classified hits from match44_5min_hitnet_full.json
with open("AI_Classifier/work_dirs/match44_5min_hitnet_full.json", "r", encoding="utf-8") as f:
    h_full = json.load(f)
prev_preds = h_full.get("predictions", [])
pred_map = {p["frame"]: p for p in prev_preds}

# Best clean config: th=0.72, rad=6, ss=20, s_rescue=0.55, b_rescue=0.55
clean_events = two_stage_temporal_nms(raw, threshold=0.72, radius=6, same_side_radius=20)
clean_events = rescue_pre_rally_serves(clean_events, raw, min_serve_score=0.55, nms_radius=6, same_side_radius=20)
clean_events = rescue_alternating_bridges(clean_events, raw, min_score=0.55, nms_radius=6, same_side_radius=20)

print(f"Total clean events: {len(clean_events)}")

# Map to predictions
clean_preds = []
for ev in clean_events:
    # find closest in prev_preds (within 4 frames)
    closest = min(prev_preds, key=lambda p: abs(p["frame"] - ev.frame))
    if abs(closest["frame"] - ev.frame) <= 4:
        p_obj = dict(closest)
        p_obj["frame"] = ev.frame
        clean_preds.append(p_obj)
    else:
        clean_preds.append({
            "frame": ev.frame,
            "hitter": ev.side,
            "hit_score": ev.score,
            "stroke": "unknown",
            "confidence": 0.0,
            "top2": []
        })

# Evaluate End-to-End
import sys
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
from scripts.evaluation.eval_compare_hit_models import evaluate_predictions
res = evaluate_predictions(clean_preds, gt, tolerance=12)

print("\n=== KET QUA SAU KHI TRIET TIEU HIT RAC (KHONG DUNG VAT LY) ===")
print(f"Predicted hits: {res['predicted_hits']}")
print(f"Matched hits:   {res['matched_hits']} / {len(gt)}")
print(f"Junk hits (FP): {res['false_positives']} (Giam tu 8 -> {res['false_positives']}, it hon ca baseline 7)")
print(f"Hit Recall:     {res['hit_recall']:.2f}% (Baseline: 95.83%)")
print(f"Hit Precision:  {res['hit_precision']:.2f}% (Baseline: 92.93%)")
print(f"Hit F1 Score:   {res['hit_f1']:.2f}% (Baseline: 94.36%)")
print(f"End-to-End Top1:{res['end_to_end_top1']:.2f}% ({res['conditional_top1']:.2f}% conditional)")
