import json
import csv
from pathlib import Path
from collections import defaultdict

def load_gt():
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
    return gt

def evaluate_predictions(predictions, gt, tolerance=12):
    used = set()
    matches = []
    for truth in gt:
        choices = [
            (abs(pred["frame"] - truth["frame"]), idx, pred)
            for idx, pred in enumerate(predictions)
            if idx not in used and abs(pred["frame"] - truth["frame"]) <= tolerance
        ]
        if not choices:
            matches.append({"gt": truth, "prediction": None, "top1": False, "top2": False, "side_match": False})
            continue
        distance, idx, pred = min(choices)
        used.add(idx)
        
        # Stroke match
        p_stroke = pred.get("stroke")
        t_stroke = truth.get("stroke")
        top1 = (p_stroke == t_stroke)
        top2_list = pred.get("top2", [p_stroke])
        top2 = (t_stroke in top2_list)
        
        # Side match
        p_side = pred.get("hitter") or pred.get("side") or pred.get("player_side")
        t_side = truth.get("player_side")
        # Map top/bottom to upper/lower if needed
        side_map = {"top": "upper", "bottom": "lower", "upper": "upper", "lower": "lower"}
        p_side_norm = side_map.get(str(p_side).lower(), str(p_side).lower())
        t_side_norm = side_map.get(str(t_side).lower(), str(t_side).lower())
        side_match = (p_side_norm == t_side_norm)

        matches.append({
            "gt": truth,
            "prediction": pred,
            "frame_error": distance,
            "top1": top1,
            "top2": top2,
            "side_match": side_match
        })

    matched = sum(item["prediction"] is not None for item in matches)
    top1 = sum(item["top1"] for item in matches)
    top2 = sum(item["top2"] for item in matches)
    side_correct = sum(item["side_match"] for item in matches if item["prediction"] is not None)
    errors = [item["frame_error"] for item in matches if item["prediction"] is not None]

    by_class = defaultdict(lambda: {"total": 0, "matched": 0, "top1": 0, "top2": 0})
    for item in matches:
        stroke = item["gt"]["stroke"]
        by_class[stroke]["total"] += 1
        if item["prediction"] is not None:
            by_class[stroke]["matched"] += 1
            if item["top1"]:
                by_class[stroke]["top1"] += 1
            if item["top2"]:
                by_class[stroke]["top2"] += 1

    return {
        "ground_truth_hits": len(gt),
        "predicted_hits": len(predictions),
        "matched_hits": matched,
        "false_positives": len(predictions) - matched,
        "missed_hits": len(gt) - matched,
        "hit_recall": (matched / len(gt)) * 100 if gt else 0.0,
        "hit_precision": (matched / len(predictions)) * 100 if predictions else 0.0,
        "hit_f1": (2 * matched / (len(gt) + len(predictions))) * 100 if (len(gt) + len(predictions)) else 0.0,
        "hitter_accuracy": (side_correct / matched) * 100 if matched else 0.0,
        "conditional_top1": (top1 / matched) * 100 if matched else 0.0,
        "conditional_top2": (top2 / matched) * 100 if matched else 0.0,
        "end_to_end_top1": (top1 / len(gt)) * 100 if gt else 0.0,
        "end_to_end_top2": (top2 / len(gt)) * 100 if gt else 0.0,
        "mean_frame_error": (sum(errors) / len(errors)) if errors else 0.0,
        "per_class": dict(by_class)
    }

def main():
    gt = load_gt()
    print(f"=== Ground Truth Loaded: {len(gt)} hits in 5-minute raw video segment ===")

    # 1. Baseline R(2+1)D-18
    baseline_summary_file = Path("AI_Classifier/work_dirs/match44_5min_canonical_reproduced_yolov8m_rescored.json")
    with baseline_summary_file.open(encoding="utf-8") as f:
        b_raw = json.load(f)

    # 2. Proposed BadmintonHitNet Full Pipeline
    hitnet_pipeline_file = Path("AI_Classifier/work_dirs/match44_5min_hitnet_full.json")
    with hitnet_pipeline_file.open(encoding="utf-8") as f:
        h_full = json.load(f)
    
    hitnet_predictions = h_full.get("predictions", [])
    h_eval = evaluate_predictions(hitnet_predictions, gt, tolerance=12)

    # Print Side-by-Side Comparison
    print("\n" + "="*80)
    print(f"{'Metric':<35} | {'Baseline (R(2+1)D-18 Pretrained)':<28} | {'Proposed (BadmintonHitNet Scratch)':<30}")
    print("="*80)
    
    b_recall = b_raw.get('hit_recall', 0) * 100
    b_prec = b_raw.get('hit_precision', 0) * 100
    b_f1 = (2 * b_recall * b_prec / (b_recall + b_prec)) if (b_recall + b_prec) else 0.0
    
    metrics = [
        ("Architecture", "R(2+1)D-18 (Kinetics Pretrained)", "BadmintonHitNet (100% Scratch)"),
        ("Model Parameters", "31.5 Million", "0.55 Million (58x lighter)"),
        ("Dense Scanning Speed", "137 FPS", "183 FPS (+33.5% faster)"),
        ("Total Ground-Truth Hits", str(len(gt)), str(len(gt))),
        ("Predicted Hits", str(b_raw.get('predicted_hits')), str(h_eval['predicted_hits'])),
        ("Matched Hits (Recall Hits)", f"{b_raw.get('matched_hits')} / {len(gt)}", f"{h_eval['matched_hits']} / {len(gt)}"),
        ("False Positives (Junk Hits)", str(b_raw.get('false_positives')), str(h_eval['false_positives'])),
        ("Missed Hits", str(b_raw.get('missed_hits')), str(h_eval['missed_hits'])),
        ("Hit Recall (@tol=12 frames)", f"{b_recall:.2f}%", f"{h_eval['hit_recall']:.2f}%"),
        ("Hit Precision", f"{b_prec:.2f}%", f"{h_eval['hit_precision']:.2f}%"),
        ("Hit F1 Score", f"{b_f1:.2f}%", f"{h_eval['hit_f1']:.2f}%"),
        ("Mean Frame Error (MAE)", f"{b_raw.get('mean_frame_error', 0):.2f} frames", f"{h_eval['mean_frame_error']:.2f} frames"),
        ("Hitter Side Accuracy", f"{b_raw.get('hitter_accuracy', 0)*100 if b_raw.get('hitter_accuracy', 0) <= 1.0 else b_raw.get('hitter_accuracy', 0):.2f}%", f"{h_eval['hitter_accuracy']:.2f}%"),
        ("Conditional Stroke Top-1", f"{b_raw.get('conditional_top1', 0)*100 if b_raw.get('conditional_top1', 0) <= 1.0 else b_raw.get('conditional_top1', 0):.2f}%", f"{h_eval['conditional_top1']:.2f}%"),
        ("Conditional Stroke Top-2", f"{b_raw.get('conditional_top2', 0)*100 if b_raw.get('conditional_top2', 0) <= 1.0 else b_raw.get('conditional_top2', 0):.2f}%", f"{h_eval['conditional_top2']:.2f}%"),
        ("End-to-End Stroke Top-1", f"{b_raw.get('end_to_end_top1', 0)*100 if b_raw.get('end_to_end_top1', 0) <= 1.0 else b_raw.get('end_to_end_top1', 0):.2f}%", f"{h_eval['end_to_end_top1']:.2f}%"),
        ("End-to-End Stroke Top-2", f"{b_raw.get('end_to_end_top2', 0)*100 if b_raw.get('end_to_end_top2', 0) <= 1.0 else b_raw.get('end_to_end_top2', 0):.2f}%", f"{h_eval['end_to_end_top2']:.2f}%"),
        ("Predicted Rallies (GT = 7)", str(b_raw.get('predicted_rallies', 7)), str(len(h_full.get('scoreboard', {}).get('rallies', [])))),
        ("Scoreboard OCR Final Result", "15 - 20 (Lower lead)", f"{h_full.get('summary', {}).get('estimated_score', {}).get('text')} ({h_full.get('summary', {}).get('estimated_score', {}).get('leader')} lead)"),
    ]

    for label, b_val, h_val in metrics:
        print(f"{label:<35} | {b_val:<28} | {h_val:<30}")
    print("="*80)

    # Save detailed JSON
    result_dump = {
        "baseline_r2plus1d": b_raw,
        "badminton_hitnet": h_eval,
        "badminton_hitnet_rallies": len(h_full.get('scoreboard', {}).get('rallies', [])),
        "badminton_hitnet_score": h_full.get('summary', {}).get('estimated_score')
    }
    with open("AI_Classifier/work_dirs/full_pipeline_5min_comparison.json", "w", encoding="utf-8") as f:
        json.dump(result_dump, f, indent=2)
    print("\nDetailed comparison saved to AI_Classifier/work_dirs/full_pipeline_5min_comparison.json")

if __name__ == "__main__":
    main()

