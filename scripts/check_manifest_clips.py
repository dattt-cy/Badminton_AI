import csv
from pathlib import Path

with open("AI_Classifier/data/manifests/shuttleset_rgb.csv", encoding="utf-8-sig") as f:
    for row in csv.DictReader(f):
        if row["match_id"] in ["13", "14", "15", "16", "17", "1", "2", "3"]:
            print(f"Match {row['match_id']:>2}: Path={Path(row['video_path']).name} | Frame={row['hit_frame']}")
            # print first 1 per match

