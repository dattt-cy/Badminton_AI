import json, csv
from pathlib import Path

with open("AI_Classifier/work_dirs/match44_20min_pipeline_with_tactical_decoder.json", "r", encoding="utf-8") as f:
    d = json.load(f)

preds = d.get("predictions", [])
print(f"Total preds: {len(preds)}")
if preds:
    print(f"First hit frame: {preds[0]['frame']}, Last hit frame: {preds[-1]['frame']}")

# Also let's check manifest for match44
manifest_path = Path("AI_Classifier/data/manifests/shuttleset_rgb.csv")
gt = []
with manifest_path.open(encoding="utf-8-sig", newline="") as handle:
    for row in csv.DictReader(handle):
        absolute = int(row["hit_frame"])
        match_id = row.get("match_id", "")
        name = Path(row["video_path"]).name
        same_match = match_id == "44" or name.startswith("44 ") or "match44" in name.lower()
        if same_match:
            gt.append(absolute)

print(f"Total GT for match44: {len(gt)} hits across frames {min(gt)} to {max(gt)}")

# Where does match44_20min_clip.mp4 start in the original video?
# Let's test possible offsets: 0, 18000, 36000, 45000, etc.
# In match44_5min_clip, frames were 45000 to 54000.
# Check where the first hit of 20min matches:
first_pred_frame = preds[0]['frame']
print(f"First pred frame in clip: {first_pred_frame}")
candidates = []
for g in gt:
    potential_offset = g - first_pred_frame
    if potential_offset >= 0:
        candidates.append(potential_offset)

from collections import Counter
counts = Counter(candidates)
print("Most likely offsets for match44_20min_clip:", counts.most_common(5))

