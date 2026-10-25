import csv
from pathlib import Path

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
                "side": row.get("player_side")
            })

gt_20min.sort(key=lambda r: r["frame"])
print(f"Verified GT for 20min clip (frames 45000 to 81000): {len(gt_20min)} hits!")
print(f"First GT hit at relative frame: {gt_20min[0]['frame']} (45000 + {gt_20min[0]['frame']} = {45000 + gt_20min[0]['frame']})")
print(f"Last GT hit at relative frame: {gt_20min[-1]['frame']}")

