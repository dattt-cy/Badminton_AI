"""Benchmark TacticalSequenceDecoder across at least 4 independent real 5-minute match clips.

This script enforces:
1. Strict Leave-One-Match-Out: for every tested match, that match is strictly EXCLUDED
   from training the transition probabilities.
2. Evaluates real pipeline features from real 5-minute video clips.
3. Tests across 5 diverse matches: Match 44, Match 41, Match 15, Match 16, Match 17.
4. Reports baseline vs refined Top-1 Accuracy, Macro-F1, and exact error fixes.
"""

from __future__ import annotations

import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from ai_classifier.models import KAPScratchModel, STROKE_CLASSES
from ai_classifier.analytics import TacticalSequenceDecoder
from scripts.training.train_kap_scratch import confusion_metrics

TEST_MATCHES = ["44", "41", "15", "16", "17"]
MANIFEST_PATH = ROOT / "data/manifests/shuttleset_rgb.csv"
DATA_PATH = ROOT / "work_dirs/error_mining/production_all_aligned.pt"
OUTPUT_PATH = ROOT / "work_dirs/tactical_decoder_multi_match_benchmark.json"


def evaluate_predictions(preds: np.ndarray, gts: np.ndarray) -> dict:
    confusion = torch.zeros(8, 8, dtype=torch.int64)
    for p, g in zip(preds, gts):
        confusion[int(g), int(p)] += 1
    return confusion_metrics(confusion)


def main():
    print("=" * 70)
    print("MULTI-MATCH REAL BENCHMARK: BASELINE vs TACTICAL SEQUENCE DECODER")
    print("=" * 70)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt_path = ROOT / "src/ai_classifier/models/checkpoints/kap_classifier.pth"
    model = KAPScratchModel().to(device)
    payload = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(payload["model"] if "model" in payload else payload)
    model.eval()

    all_data = torch.load(DATA_PATH, map_location="cpu", weights_only=False)

    # Group by match
    by_match = defaultdict(list)
    for d in all_data:
        mid = str(d["match_id"])
        if mid in TEST_MATCHES:
            by_match[mid].append(d)

    results = {}
    total_baseline_correct = 0
    total_refined_correct = 0
    total_hits = 0

    for mid in TEST_MATCHES:
        records = by_match[mid]
        records.sort(key=lambda x: int(x["hit_frame"]))
        n_hits = len(records)
        total_hits += n_hits

        # Group into rallies by inter-hit gap
        rallies = []
        cur_rally = [records[0]]
        for r in records[1:]:
            if int(r["hit_frame"]) - int(cur_rally[-1]["hit_frame"]) > 90:
                rallies.append(cur_rally)
                cur_rally = [r]
            else:
                cur_rally.append(r)
        rallies.append(cur_rally)

        # 1. Forward inference through KAP baseline
        emb = torch.cat([torch.as_tensor(x["emb"]).float().view(1, 512) for x in records], dim=0).to(device)
        pose = torch.cat([torch.as_tensor(x["s_j"]).float().view(1, 32, 68) for x in records], dim=0).to(device)
        court = torch.cat([torch.as_tensor(x["s_p"]).float().view(1, 32, 4) for x in records], dim=0).to(device)
        shuttle = torch.cat([torch.as_tensor(x["s_s"]).float().view(1, 32, 2) for x in records], dim=0).to(device)
        contact = torch.cat([torch.as_tensor(x["s_c"]).float().view(1, 32, 1) for x in records], dim=0).to(device)
        quality = torch.cat([torch.as_tensor(x["s_q"]).float().view(1, 4) for x in records], dim=0).to(device)
        gt_labels = np.array([x["gt_idx"] for x in records])

        with torch.inference_mode():
            os, _, _ = model(emb, pose, court, shuttle, contact, quality)
            log_probs = torch.log_softmax(os, dim=1).cpu().numpy()

        baseline_preds = log_probs.argmax(axis=1)
        base_metrics = evaluate_predictions(baseline_preds, gt_labels)
        total_baseline_correct += int(np.sum(baseline_preds == gt_labels))

        # 2. Fit TacticalSequenceDecoder STRICTLY excluding this match!
        decoder = TacticalSequenceDecoder(
            MANIFEST_PATH,
            alpha=0.5,
            default_lambda=0.20,
            exclude_matches={mid},
        )

        # 3. Decode rally by rally
        refined_preds = np.zeros(n_hits, dtype=int)
        hit_ptr = 0
        fixed_count = 0
        broken_count = 0
        changed_details = []

        for r_items in rallies:
            T = len(r_items)
            r_log_probs = log_probs[hit_ptr : hit_ptr + T]
            r_hitters = [it.get("side", "") for it in r_items]

            decoded_path = decoder.decode_rally(r_log_probs, r_hitters, lam=0.20)

            for t_idx, pred_s in enumerate(decoded_path):
                global_idx = hit_ptr + t_idx
                refined_preds[global_idx] = pred_s
                old_s = baseline_preds[global_idx]
                gt_s = gt_labels[global_idx]

                if old_s != pred_s:
                    if pred_s == gt_s:
                        fixed_count += 1
                        status = "FIXED"
                    else:
                        broken_count += 1
                        status = "BROKEN"
                    changed_details.append({
                        "hit_frame": r_items[t_idx]["hit_frame"],
                        "gt": STROKE_CLASSES[gt_s],
                        "old": STROKE_CLASSES[old_s],
                        "new": STROKE_CLASSES[pred_s],
                        "status": status,
                    })

            hit_ptr += T

        refined_metrics = evaluate_predictions(refined_preds, gt_labels)
        total_refined_correct += int(np.sum(refined_preds == gt_labels))

        acc_diff = (refined_metrics["accuracy"] - base_metrics["accuracy"]) * 100
        f1_diff = (refined_metrics["macro_f1"] - base_metrics["macro_f1"]) * 100

        results[mid] = {
            "hits": n_hits,
            "rallies": len(rallies),
            "baseline": {
                "correct": int(np.sum(baseline_preds == gt_labels)),
                "accuracy": round(base_metrics["accuracy"] * 100, 2),
                "macro_f1": round(base_metrics["macro_f1"] * 100, 2),
                "balanced_acc": round(base_metrics["balanced_accuracy"] * 100, 2),
            },
            "refined": {
                "correct": int(np.sum(refined_preds == gt_labels)),
                "accuracy": round(refined_metrics["accuracy"] * 100, 2),
                "macro_f1": round(refined_metrics["macro_f1"] * 100, 2),
                "balanced_acc": round(refined_metrics["balanced_accuracy"] * 100, 2),
            },
            "changes": {
                "total": len(changed_details),
                "fixed": fixed_count,
                "broken": broken_count,
                "net_gain": fixed_count - broken_count,
                "details": changed_details,
            },
        }

        print(f"\n--- MATCH {mid} ({n_hits} hits, {len(rallies)} rallies) ---")
        print(f"  Baseline Accuracy: {base_metrics['accuracy']*100:.2f}% | Macro-F1: {base_metrics['macro_f1']*100:.2f}%")
        print(f"  Refined Accuracy : {refined_metrics['accuracy']*100:.2f}% | Macro-F1: {refined_metrics['macro_f1']*100:.2f}%")
        sign = "+" if acc_diff >= 0 else ""
        print(f"  Accuracy Impact  : {sign}{acc_diff:.2f}% (Net Gain: {fixed_count - broken_count} shots)")
        for ch in changed_details:
            print(f"    Frame {ch['hit_frame']}: GT={ch['gt']:11} | Old={ch['old']:11} -> New={ch['new']:11} [{ch['status']}]")

    print("\n" + "=" * 70)
    print("OVERALL MULTI-MATCH SUMMARY (5 INDEPENDENT MATCHES)")
    print("=" * 70)
    base_tot_acc = total_baseline_correct / total_hits * 100
    ref_tot_acc = total_refined_correct / total_hits * 100
    print(f"Total Evaluated Hits: {total_hits}")
    print(f"Total Baseline Correct: {total_baseline_correct}/{total_hits} ({base_tot_acc:.2f}%)")
    print(f"Total Refined Correct : {total_refined_correct}/{total_hits} ({ref_tot_acc:.2f}%)")
    print(f"Total Overall Gain    : +{total_refined_correct - total_baseline_correct} shots ({ref_tot_acc - base_tot_acc:+.2f}%)")
    print("=" * 70)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"Detailed benchmark saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()

