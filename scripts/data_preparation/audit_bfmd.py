import glob
import json
import os
from collections import Counter

BFMD_DIR = r"C:\Users\ADMIN\Downloads\BFMD_data-20260921T175743Z-1-001\BFMD_data"
HIT_INFERRED_DIR = os.path.join(BFMD_DIR, "annotations", "hit_inferred")

files = sorted(glob.glob(os.path.join(HIT_INFERRED_DIR, "*.json")))
print(f"Found {len(files)} singles annotation files.")

global_counts = Counter()
match_counts = {}

for f in files:
    mname = os.path.splitext(os.path.basename(f))[0]
    with open(f, "r", encoding="utf-8") as fp:
        data = json.load(fp)
    hits = data.get("hits", [])
    c = Counter([it.get("shot_type") for it in hits])
    match_counts[mname] = (len(hits), c)
    global_counts.update(c)

print("\n--- GLOBAL SHOT TYPE COUNTS (Singles) ---")
total_shots = sum(global_counts.values())
print(f"Total labeled shots: {total_shots}")
for st, cnt in global_counts.most_common():
    print(f"  {st:20s}: {cnt:5d} ({cnt/total_shots*100:5.2f}%)")

print("\n--- MATCH SUMMARY ---")
for mname, (tot, c) in match_counts.items():
    print(f"{mname[:50]}... : {tot:5d} shots")

