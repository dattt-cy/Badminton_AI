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
                    "stroke_side": row.get("stroke_side", "unknown"),
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
            matches.append({
                "gt": truth, "prediction": None, "top1": False, "top2": False,
                "side_match": False, "stroke_side_match": False, "face_match": False,
                "fine_top1": False, "fine_face_top1": False
            })
            continue
        distance, idx, pred = min(choices)
        used.add(idx)
        
        # Stroke match
        p_stroke = pred.get("stroke")
        t_stroke = truth.get("stroke")
        top1 = (p_stroke == t_stroke)
        top2_list = pred.get("top2", [p_stroke])
        top2 = (t_stroke in top2_list)
        
        # Player court side match (upper/lower)
        p_side = pred.get("hitter") or pred.get("side") or pred.get("player_side")
        t_side = truth.get("player_side")
        side_map = {"top": "upper", "bottom": "lower", "upper": "upper", "lower": "lower"}
        p_side_norm = side_map.get(str(p_side).lower(), str(p_side).lower())
        t_side_norm = side_map.get(str(t_side).lower(), str(t_side).lower())
        side_match = (p_side_norm == t_side_norm)

        # 3-class Stroke side match (forehand / backhand / aroundhead)
        p_ss = str(pred.get("stroke_side", "")).lower()
        t_ss = str(truth.get("stroke_side", "")).lower()
        stroke_side_match = (p_ss == t_ss) if (p_ss and t_ss and t_ss != "unknown") else False
        fine_top1 = top1 and stroke_side_match

        # Binary racket face match: Forehand vs Backhand (Aroundhead is played with the forehand face)
        norm_face = lambda s: "forehand" if s in ("forehand", "aroundhead") else ("backhand" if s == "backhand" else s)
        p_face = norm_face(p_ss)
        t_face = norm_face(t_ss)
        face_match = (p_face == t_face) if (p_face and t_face and t_face != "unknown") else False
        fine_face_top1 = top1 and face_match

        matches.append({
            "gt": truth,
            "prediction": pred,
            "frame_error": distance,
            "top1": top1,
            "top2": top2,
            "side_match": side_match,
            "stroke_side_match": stroke_side_match,
            "face_match": face_match,
            "fine_top1": fine_top1,
            "fine_face_top1": fine_face_top1
        })

    matched = sum(item["prediction"] is not None for item in matches)
    top1 = sum(item["top1"] for item in matches)
    top2 = sum(item["top2"] for item in matches)
    fine_top1_count = sum(item["fine_top1"] for item in matches)
    fine_face_count = sum(item["fine_face_top1"] for item in matches)
    side_correct = sum(item["side_match"] for item in matches if item["prediction"] is not None)
    stroke_side_correct = sum(item["stroke_side_match"] for item in matches if item["prediction"] is not None)
    face_correct = sum(item["face_match"] for item in matches if item["prediction"] is not None)
    errors = [item["frame_error"] for item in matches if item["prediction"] is not None]

    by_class = defaultdict(lambda: {"total": 0, "matched": 0, "top1": 0, "top2": 0, "fine_top1": 0, "fine_face": 0})
    for item in matches:
        stroke = item["gt"]["stroke"]
        by_class[stroke]["total"] += 1
        if item["prediction"] is not None:
            by_class[stroke]["matched"] += 1
            if item["top1"]:
                by_class[stroke]["top1"] += 1
            if item["top2"]:
                by_class[stroke]["top2"] += 1
            if item["fine_top1"]:
                by_class[stroke]["fine_top1"] += 1
            if item["fine_face_top1"]:
                by_class[stroke]["fine_face"] += 1

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
        "stroke_side_accuracy": (stroke_side_correct / matched) * 100 if matched else 0.0,
        "face_accuracy": (face_correct / matched) * 100 if matched else 0.0,
        "conditional_top1": (top1 / matched) * 100 if matched else 0.0,
        "conditional_top2": (top2 / matched) * 100 if matched else 0.0,
        "conditional_fine": (fine_top1_count / matched) * 100 if matched else 0.0,
        "conditional_fine_face": (fine_face_count / matched) * 100 if matched else 0.0,
        "end_to_end_top1": (top1 / len(gt)) * 100 if gt else 0.0,
        "end_to_end_fine": (fine_top1_count / len(gt)) * 100 if gt else 0.0,
        "end_to_end_fine_face": (fine_face_count / len(gt)) * 100 if gt else 0.0,
        "end_to_end_top2": (top2 / len(gt)) * 100 if gt else 0.0,
        "mean_frame_error": (sum(errors) / len(errors)) if errors else 0.0,
        "per_class": dict(by_class)
    }

def main():
    gt = load_gt()
    print(f"Loaded {len(gt)} ground truth hits.")

if __name__ == "__main__":
    main()
