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
matches = []
for truth in gt:
    choices = [
        (abs(pred["frame"] - truth["frame"]), idx, pred)
        for idx, pred in enumerate(predictions)
        if idx not in used and abs(pred["frame"] - truth["frame"]) <= 12
    ]
    if choices:
        dist, idx, pred = min(choices)
        used.add(idx)
        top1 = (pred.get("stroke") == truth["stroke"])
        top2 = (truth["stroke"] in pred.get("top2", []))
        matches.append({
            "frame": truth["frame"],
            "gt_stroke": truth["stroke"],
            "pred_stroke": pred.get("stroke"),
            "top1": top1,
            "top2": top2,
            "top2_list": pred.get("top2", []),
            "conf": pred.get("confidence", 0),
            "hitter": pred.get("hitter")
        })

wrong = [m for m in matches if not m["top1"]]
print(f"Total matched: {len(matches)} | Correct Top-1: {len(matches)-len(wrong)} | Wrong: {len(wrong)}")

print("\n--- CHI TIET 15 CU DANH BI DOAN SAI DONG TAC ---")
for i, w in enumerate(wrong, 1):
    in_top2 = "[TOP2 DUNG]" if w["top2"] else "[TOP2 CUNG SAI]"
    print(f" {i:2d}. Frame {w['frame']:4d} | GT: {w['gt_stroke']:<10} -> Doan: {w['pred_stroke']:<10} (Conf: {w['conf']:.2f}) {in_top2} Top2: {w['top2_list']}")

