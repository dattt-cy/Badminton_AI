"""Extract real production errors (hard cases) from aligned online matches.

This tool extracts:
1. All misclassifications (ground_truth != predicted) under the production KAP model.
2. Complete feature vectors (emb, pose, court, shuttle, contact, quality) with real production noise.
3. Detailed confusion diagnosis (top confused pairs, confidence distributions, temporal offsets).
4. Saves structured dataset for analysis and targeted fine-tuning.
"""

from __future__ import annotations

import csv
import glob
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from ai_classifier.models import KAPScratchModel, STROKE_CLASSES
from scripts.training.finetune_kap_online_multi_matches import OFFSETS

OFFSETS["41"] = 14500
OFFSETS["44"] = 45000

MANIFEST_PATH = ROOT / "data/manifests/shuttleset_rgb.csv"
OUTPUT_DIR = ROOT / "work_dirs/error_mining"


def extract_and_audit():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt_path = ROOT / "src/ai_classifier/models/checkpoints/kap_classifier.pth"
    model = KAPScratchModel().to(device)
    payload = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(payload["model"] if "model" in payload else payload)
    model.eval()

    with MANIFEST_PATH.open(encoding="utf-8-sig", newline="") as handle:
        manifest = list(csv.DictReader(handle))

    files = sorted(glob.glob(str(ROOT / "work_dirs/*features*.pt")))

    total_samples = 0
    total_correct = 0
    errors = []
    all_aligned = []
    confusion = np.zeros((8, 8), dtype=int)
    match_stats = defaultdict(lambda: {"total": 0, "correct": 0, "errors": 0})

    for fpath_str in files:
        fpath = Path(fpath_str)
        m = re.search(r"match(\d+)", fpath.name)
        if not m:
            continue
        mid = m.group(1)

        records = torch.load(fpath, map_location="cpu", weights_only=False)

        # Check if file is already aligned with gt_stroke_idx
        if len(records) > 0 and "gt_stroke_idx" in records[0]:
            matched_pairs = []
            for idx, rec in enumerate(records):
                gt_idx = int(rec["gt_stroke_idx"])
                gt_name = STROKE_CLASSES[gt_idx]
                matched_pairs.append((rec, gt_idx, gt_name, rec.get("hit_frame", idx), rec.get("gt_frame", idx)))
        else:
            if mid not in OFFSETS:
                continue
            offset = OFFSETS[mid]
            truth = [r for r in manifest if r["match_id"] == mid]

            available = set(range(len(records)))
            matched_pairs = []
            for row in sorted(truth, key=lambda v: int(v["hit_frame"])):
                local = int(row["hit_frame"]) - offset
                choices = [
                    (abs(int(records[i]["hit_frame"]) - local), i, row)
                    for i in available
                    if abs(int(records[i]["hit_frame"]) - local) <= 12
                ]
                if not choices:
                    continue
                choices.sort()
                dist, idx, r = choices[0]
                available.remove(idx)
                rec = records[idx]
                gt_name = r["coarse_label"]
                gt_idx = STROKE_CLASSES.index(gt_name)
                matched_pairs.append((rec, gt_idx, gt_name, int(rec["hit_frame"]), int(r["hit_frame"])))

        for rec, gt_idx, gt_name, hit_frame, gt_frame in matched_pairs:
            emb = torch.as_tensor(rec["emb"]).float().view(1, 512).to(device)
            pose = torch.as_tensor(rec["s_j"]).float().view(1, 32, 68).to(device)
            court = torch.as_tensor(rec["s_p"]).float().view(1, 32, 4).to(device)
            shuttle = torch.as_tensor(rec["s_s"]).float().view(1, 32, 2).to(device)
            contact = torch.as_tensor(rec["s_c"]).float().view(1, 32, 1).to(device)
            quality = torch.as_tensor(rec["s_q"]).float().view(1, 4).to(device)

            with torch.inference_mode():
                os, _, _ = model(emb, pose, court, shuttle, contact, quality)
                probs = os.softmax(1)[0].cpu().numpy()
                pred_idx = int(probs.argmax())
                pred_prob = float(probs[pred_idx])
                gt_prob = float(probs[gt_idx])

            confusion[gt_idx, pred_idx] += 1
            total_samples += 1
            match_stats[mid]["total"] += 1

            sample_record = {
                "match_id": mid,
                "hit_frame": int(hit_frame),
                "gt_frame": int(gt_frame),
                "frame_offset_error": int(hit_frame) - int(gt_frame),
                "gt_stroke": gt_name,
                "gt_idx": gt_idx,
                "pred_stroke": STROKE_CLASSES[pred_idx],
                "pred_idx": pred_idx,
                "pred_prob": pred_prob,
                "gt_prob": gt_prob,
                "sample_id": rec.get("sample_id", f"match{mid}_{hit_frame}"),
                "is_correct": (gt_idx == pred_idx),
                "emb": rec["emb"].cpu() if hasattr(rec["emb"], "cpu") else rec["emb"],
                "s_j": rec["s_j"].cpu() if hasattr(rec["s_j"], "cpu") else rec["s_j"],
                "s_p": rec["s_p"].cpu() if hasattr(rec["s_p"], "cpu") else rec["s_p"],
                "s_s": rec["s_s"].cpu() if hasattr(rec["s_s"], "cpu") else rec["s_s"],
                "s_c": rec["s_c"].cpu() if hasattr(rec["s_c"], "cpu") else rec["s_c"],
                "s_q": rec["s_q"].cpu() if hasattr(rec["s_q"], "cpu") else rec["s_q"],
            }
            all_aligned.append(sample_record)

            if gt_idx == pred_idx:
                total_correct += 1
                match_stats[mid]["correct"] += 1
            else:
                match_stats[mid]["errors"] += 1
                errors.append(sample_record)

    # Confusion statistics
    conf_pairs = Counter()
    for err in errors:
        conf_pairs[(err["gt_stroke"], err["pred_stroke"])] += 1

    per_class = {}
    for c, name in enumerate(STROKE_CLASSES):
        tp = int(confusion[c, c])
        supp = int(confusion[c].sum())
        p_tot = int(confusion[:, c].sum())
        rec = float(tp / supp) if supp > 0 else 0.0
        prec = float(tp / p_tot) if p_tot > 0 else 0.0
        f1 = float(2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
        per_class[name] = {
            "support": supp,
            "true_positives": tp,
            "recall": round(rec * 100, 2),
            "precision": round(prec * 100, 2),
            "f1": round(f1 * 100, 2),
        }

    # Save torch datasets
    torch.save(errors, OUTPUT_DIR / "production_hard_errors.pt")
    torch.save(all_aligned, OUTPUT_DIR / "production_all_aligned.pt")

    # Save audit report (without heavy tensors)
    audit_summary = {
        "total_samples": total_samples,
        "total_correct": total_correct,
        "overall_accuracy": round(total_correct / total_samples * 100, 2),
        "total_errors": len(errors),
        "per_class": per_class,
        "top_confusion_pairs": [
            {"gt": gt, "pred": pred, "count": count}
            for (gt, pred), count in conf_pairs.most_common(20)
        ],
        "per_match_stats": {
            k: {
                "total": v["total"],
                "correct": v["correct"],
                "errors": v["errors"],
                "accuracy": round(v["correct"] / v["total"] * 100, 2) if v["total"] > 0 else 0,
            }
            for k, v in sorted(match_stats.items(), key=lambda x: int(x[0]))
        },
    }

    report_path = OUTPUT_DIR / "error_audit_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(audit_summary, f, indent=2)

    print("=" * 60)
    print("REAL PRODUCTION ERROR AUDIT COMPLETE")
    print("=" * 60)
    print(f"Total Aligned Samples : {total_samples}")
    print(f"Total Errors Extracted: {len(errors)} ({len(errors)/total_samples*100:.2f}%)")
    print(f"Overall Accuracy      : {total_correct/total_samples*100:.2f}%")
    print(f"Saved Hard Cases To   : {OUTPUT_DIR / 'production_hard_errors.pt'}")
    print(f"Saved Audit Report To : {report_path}")
    print("=" * 60)
    print("Top 5 Confusion Pairs:")
    for (gt, pred), count in conf_pairs.most_common(5):
        print(f"  {gt:12} -> {pred:12}: {count} errors")


if __name__ == "__main__":
    extract_and_audit()
