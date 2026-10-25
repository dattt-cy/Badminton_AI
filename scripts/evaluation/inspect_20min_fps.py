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

used = set()
matched_gt = set()
for truth in gt_20min:
    choices = [
        (abs(p["frame"] - truth["frame"]), idx, p)
        for idx, p in enumerate(preds)
        if idx not in used and abs(p["frame"] - truth["frame"]) <= 12
    ]
    if choices:
        d, idx, p = min(choices)
        used.add(idx)
        matched_gt.add(truth["frame"])

fps = [p for idx, p in enumerate(preds) if idx not in used]
missed = [t for t in gt_20min if t["frame"] not in matched_gt]

print(f"Total Preds: {len(preds)} | GT: {len(gt_20min)} | Matched: {len(used)} | FP: {len(fps)} | Missed: {len(missed)}")

print("\n--- PHAN LOAI 22 CU RAC THEO SCORE & CONFIDENCE ---")
tps = [p for idx, p in enumerate(preds) if idx in used]
print(f"TP count: {len(tps)}, FP count: {len(fps)}")
print(f"TP with KAP conf < 0.55: {sum(1 for p in tps if p.get('confidence', 0) < 0.55)}")
print(f"FP with KAP conf < 0.55: {sum(1 for p in fps if p.get('confidence', 0) < 0.55)}")

for p in tps:
    if p.get('confidence', 0) < 0.55:
        print(f" Low conf TP: frame {p['frame']} stroke {p.get('stroke')} conf {p.get('confidence',0):.2f}")

print("\nChi tiet 22 cu rac:")
for i, p in enumerate(fps, 1):
    f_num = p["frame"]
    t_sec = f_num / 30.0
    hitter = p.get("hitter")
    h_score = p.get("hit_score", 0)
    stroke = p.get("stroke")
    conf = p.get("confidence", 0)
    rally = p.get("rally")
    print(f" {i:2d}. Frame {f_num:5d} ({t_sec:6.1f}s) | Side: {hitter:<5} | HitScore: {h_score:.3f} | Stroke: {stroke:<10} (Conf: {conf:.2f}) | Rally: {rally}")
