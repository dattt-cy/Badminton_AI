import json, csv
from pathlib import Path
from collections import defaultdict

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

raw_predictions = h_full.get("predictions", [])

def evaluate(preds):
    used = set()
    matches = []
    for truth in gt:
        choices = [
            (abs(p["frame"] - truth["frame"]), idx, p)
            for idx, p in enumerate(preds)
            if idx not in used and abs(p["frame"] - truth["frame"]) <= 12
        ]
        if not choices:
            matches.append({"gt": truth, "prediction": None, "top1": False})
            continue
        dist, idx, p = min(choices)
        used.add(idx)
        top1 = (p.get("stroke") == truth["stroke"])
        matches.append({"gt": truth, "prediction": p, "top1": top1})

    matched = sum(item["prediction"] is not None for item in matches)
    top1 = sum(item["top1"] for item in matches)
    fp = len(preds) - matched
    missed = len(gt) - matched
    prec = (matched / len(preds)) * 100 if preds else 0.0
    rec = (matched / len(gt)) * 100 if gt else 0.0
    f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) else 0.0
    cond_top1 = (top1 / matched) * 100 if matched else 0.0
    e2e_top1 = (top1 / len(gt)) * 100 if gt else 0.0
    return {
        "preds": len(preds),
        "matched": matched,
        "fp": fp,
        "missed": missed,
        "prec": prec,
        "rec": rec,
        "f1": f1,
        "cond_top1": cond_top1,
        "e2e_top1": e2e_top1,
        "top1_count": top1
    }

print("=== BASELINE (CURRENT 101 HITS) ===")
base_res = evaluate(raw_predictions)
for k, v in base_res.items():
    print(f"  {k}: {v:.2f}" if isinstance(v, float) else f"  {k}: {v}")

# Test Strategy 1: Score threshold filter
print("\n=== SWEEP HIT SCORE THRESHOLD ===")
for th in [0.70, 0.75, 0.77, 0.80, 0.82, 0.85]:
    filtered = [p for p in raw_predictions if p.get("hit_score", 0.0) >= th]
    res = evaluate(filtered)
    print(f"Th={th:.2f} | Preds={res['preds']:3d} | FP={res['fp']:2d} | Missed={res['missed']:2d} | Prec={res['prec']:.1f}% | Rec={res['rec']:.1f}% | F1={res['f1']:.1f}% | E2E_Top1={res['e2e_top1']:.2f}% ({res['top1_count']}/{len(gt)})")

# Test Strategy 2: Suppress same-side duplicates within radius R
print("\n=== SWEEP SAME-SIDE DUP RADIUS ===")
for same_side_rad in [15, 20, 25, 30]:
    # sort by frame
    sorted_preds = sorted(raw_predictions, key=lambda x: x["frame"])
    kept = []
    for p in sorted_preds:
        # check if previous kept from same side is too close
        too_close = False
        for k in kept:
            if k.get("hitter") == p.get("hitter") and abs(k["frame"] - p["frame"]) < same_side_rad:
                # keep higher score
                too_close = True
                break
        if not too_close:
            kept.append(p)
    res = evaluate(kept)
    print(f"Rad={same_side_rad:2d} | Preds={res['preds']:3d} | FP={res['fp']:2d} | Missed={res['missed']:2d} | Prec={res['prec']:.1f}% | Rec={res['rec']:.1f}% | F1={res['f1']:.1f}% | E2E_Top1={res['e2e_top1']:.2f}%")

# Test Strategy 3: Combined Clean Filter (Th >= 0.76 + Same-side rad 22 + Min gap between consecutive hits of opposing players 10f)
print("\n=== COMBINED FILTERING EXPERIMENTS ===")
for th in [0.75, 0.77, 0.80]:
    for rad in [18, 22, 25]:
        filtered = [p for p in raw_predictions if p.get("hit_score", 0.0) >= th]
        kept = []
        for p in filtered:
            if not any(k.get("hitter") == p.get("hitter") and abs(k["frame"] - p["frame"]) < rad for k in kept):
                kept.append(p)
        res = evaluate(kept)
        if res["fp"] < base_res["fp"]:
            print(f"Th={th:.2f}, Rad={rad:2d} -> FP reduced: {base_res['fp']} -> {res['fp']} | Missed={res['missed']} | Prec={res['prec']:.1f}% | F1={res['f1']:.1f}% | E2E_Top1={res['e2e_top1']:.2f}% ({res['top1_count']}/{len(gt)})")

