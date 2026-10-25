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

from ai_classifier.localization import HitEvent, two_stage_temporal_nms, match_hit_events, rescue_pre_rally_serves, rescue_alternating_bridges

scan_file = Path("AI_Classifier/work_dirs/match44_5min_hitnet_scan.json")
with scan_file.open("r", encoding="utf-8") as f:
    scan_data = json.load(f)

raw_events = [
    HitEvent(int(e["frame"]), e["side"], float(e["score"]))
    for e in scan_data["raw_events"]
]

print(f"Total raw scored frames: {len(raw_events)} | Total GT: {len(gt)}")

gt_events = [
    HitEvent(int(g["frame"]), "upper" if g["player_side"] == "top" else "lower", 1.0)
    for g in gt
]

results = []
for th in [0.70, 0.72, 0.74, 0.75, 0.76, 0.78, 0.80]:
    for rad in [6, 8, 10]:
        for ss_rad in [15, 18, 20, 22, 25]:
            # Test without rescue and with rescue
            for do_rescue in [False, True]:
                events = two_stage_temporal_nms(
                    raw_events,
                    threshold=th,
                    radius=rad,
                    same_side_radius=ss_rad
                )
                if do_rescue:
                    events = rescue_pre_rally_serves(
                        events, raw_events, min_serve_score=0.60,
                        nms_radius=rad, same_side_radius=ss_rad
                    )
                    events = rescue_alternating_bridges(
                        events, raw_events, min_score=0.65,
                        nms_radius=rad, same_side_radius=ss_rad
                    )
                
                m = match_hit_events(events, gt_events, tolerance=12)
                f1 = m["f1"] * 100
                prec = m["precision"] * 100
                rec = m["recall"] * 100
                fp = m["false_positives"]
                fn = m["false_negatives"]
                tp = m["true_positives"]
                
                results.append({
                    "th": th, "rad": rad, "ss_rad": ss_rad, "rescue": do_rescue,
                    "pred": len(events), "tp": tp, "fp": fp, "fn": fn,
                    "prec": prec, "rec": rec, "f1": f1
                })

# Sort by lowest false positives while maintaining recall >= 95.8% (matched >= 92)
good_candidates = [r for r in results if r["tp"] >= 92]
good_candidates.sort(key=lambda x: (x["fp"], -x["f1"]))

print("\n--- TOP 10 CAU HINH IT HIT RAC NHAT MA RECALL VAN >= 92/96 (>=95.8%) ---")
print(f"{'Th':<5} | {'Rad':<4} | {'SS_Rad':<6} | {'Rescue':<7} | {'Pred':<5} | {'Matched':<7} | {'Junk FP':<7} | {'Prec (%)':<8} | {'Rec (%)':<8} | {'F1 (%)':<8}")
print("-" * 80)
for r in good_candidates[:10]:
    print(f"{r['th']:<5.2f} | {r['rad']:<4d} | {r['ss_rad']:<6d} | {str(r['rescue']):<7} | {r['pred']:<5d} | {r['tp']:<2d}/96   | {r['fp']:<7d} | {r['prec']:<8.2f} | {r['rec']:<8.2f} | {r['f1']:<8.2f}")

