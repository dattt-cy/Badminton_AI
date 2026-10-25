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
            gt.append(HitEvent(absolute - 45000, "upper" if row.get("player_side") == "top" else "lower", 1.0))

scan_file = Path("AI_Classifier/work_dirs/match44_5min_hitnet_scan.json")
scan_data = json.load(scan_file.open("r", encoding="utf-8"))
raw = [HitEvent(int(e["frame"]), e["side"], float(e["score"])) for e in scan_data["raw_events"]]

print(f"Loaded raw: {len(raw)} events")

best_configs = []
for th in [0.70, 0.72, 0.75]:
    for rad in [6, 8]:
        for ss in [15, 18, 20]:
            for s_rescue in [0.40, 0.50, 0.55, 0.60]:
                for b_rescue in [0.55, 0.60, 0.65]:
                    ev = two_stage_temporal_nms(raw, threshold=th, radius=rad, same_side_radius=ss)
                    ev = rescue_pre_rally_serves(ev, raw, min_serve_score=s_rescue, nms_radius=rad, same_side_radius=ss)
                    ev = rescue_alternating_bridges(ev, raw, min_score=b_rescue, nms_radius=rad, same_side_radius=ss)
                    m = match_hit_events(ev, gt, tolerance=12)
                    best_configs.append((m["true_positives"], m["false_positives"], m["precision"]*100, m["f1"]*100, th, rad, ss, s_rescue, b_rescue, len(ev)))

# Filter configs with FP <= 6 and TP >= 92
good = [c for c in best_configs if c[1] <= 6 and c[0] >= 92]
good.sort(key=lambda x: (x[1], -x[0]))

print("\n--- CAU HINH FP <= 6 VA RECALL >= 92/96 ---")
print(f"{'Matched':<8} | {'FP (Junk)':<10} | {'Prec (%)':<9} | {'F1 (%)':<9} | {'Th':<5} | {'Rad':<4} | {'SS':<4} | {'S_Res':<6} | {'B_Res':<6} | {'Total'}")
for tp, fp, prec, f1, th, rad, ss, s_res, b_res, total in good[:10]:
    print(f"{tp}/96     | {fp:<10d} | {prec:<9.2f} | {f1:<9.2f} | {th:<5.2f} | {rad:<4d} | {ss:<4d} | {s_res:<6.2f} | {b_res:<6.2f} | {total}")
