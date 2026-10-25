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

with open("AI_Classifier/work_dirs/match44_5min_hitnet_full.json", "r", encoding="utf-8") as f:
    h_full = json.load(f)

predictions = h_full.get("predictions", [])

used = set()
matched_gt = set()
for truth in gt:
    choices = [
        (abs(pred["frame"] - truth["frame"]), idx, pred)
        for idx, pred in enumerate(predictions)
        if idx not in used and abs(pred["frame"] - truth["frame"]) <= 12
    ]
    if choices:
        distance, idx, pred = min(choices)
        used.add(idx)
        matched_gt.add(truth["frame"])

false_positives = [pred for idx, pred in enumerate(predictions) if idx not in used]
missed = [t for t in gt if t["frame"] not in matched_gt]

print(f"Total Predictions: {len(predictions)}")
print(f"Total Ground Truth: {len(gt)}")
print(f"Matched Hits (True Positives): {len(used)}")
print(f"False Positives (Cu rac): {len(false_positives)}")
print(f"Missed (Bo sot): {len(missed)}")

print("\n--- CHI TIET 8 CU RAC (FALSE POSITIVES) ---")
for i, fp in enumerate(false_positives, 1):
    f_num = fp["frame"]
    t_sec = fp.get("time_seconds", f_num / 30.0)
    hitter = fp.get("hitter")
    score = fp.get("hit_score", 0.0)
    stroke = fp.get("stroke")
    rally = fp.get("rally")
    # Check if there is any GT nearby (e.g. within 30 frames)
    nearby_gt = [g for g in gt if abs(g["frame"] - f_num) <= 45]
    context = ""
    if nearby_gt:
        diffs = [f"{g['frame']-f_num:+d}f ({g['stroke']})" for g in nearby_gt]
        context = f" [Near GT: {', '.join(diffs)}]"
    else:
        context = " [Between rallies / Dead time]"
    print(f" {i}. Frame {f_num:4d} ({t_sec:5.1f}s) | Side: {hitter:<5} | Score: {score:.3f} | Stroke: {stroke:<10} | Rally: {rally}{context}")

print("\n--- CHI TIET CU BI BO SOT (MISSED HITS) ---")
for i, m in enumerate(missed, 1):
    f_num = m["frame"]
    t_sec = f_num / 30.0
    side = m.get("player_side")
    stroke = m.get("stroke")
    print(f" {i}. Frame {f_num:4d} ({t_sec:5.1f}s) | Side: {side:<6} | Label: {stroke}")
