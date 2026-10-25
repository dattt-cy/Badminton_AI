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
for truth in gt:
    choices = [
        (abs(pred["frame"] - truth["frame"]), idx, pred)
        for idx, pred in enumerate(predictions)
        if idx not in used and abs(pred["frame"] - truth["frame"]) <= 12
    ]
    if choices:
        dist, idx, pred = min(choices)
        used.add(idx)

fps = [p for idx, p in enumerate(predictions) if idx not in used]
tps = [p for idx, p in enumerate(predictions) if idx in used]

print("=== SO SANH CHI SO CUA CU RAC (FP) VS CU THAT (TP) ===")
print(f"Count: TP={len(tps)}, FP={len(fps)}")

tp_scores = [p.get("hit_score", 0) for p in tps]
fp_scores = [p.get("hit_score", 0) for p in fps]
print(f"Mean Hit Score: TP={sum(tp_scores)/len(tp_scores):.3f} vs FP={sum(fp_scores)/len(fp_scores):.3f}")

tp_conf = [p.get("confidence", 0) for p in tps]
fp_conf = [p.get("confidence", 0) for p in fps]
print(f"Mean KAP Confidence: TP={sum(tp_conf)/len(tp_conf):.3f} vs FP={sum(fp_conf)/len(fp_conf):.3f}")

print("\nChi tiet 8 cu rac:")
for p in fps:
    print(f"Frame {p['frame']} | HitScore={p.get('hit_score',0):.3f} | KAPConf={p.get('confidence',0):.3f} | Stroke={p.get('stroke')} | Hitter={p.get('hitter')}")

