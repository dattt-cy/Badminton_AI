"""Benchmark Evaluation Script for ShuttleSet 80 Standardized Clips.

Evaluates and compares:
1. Baseline Concat-MLP (Epoch 20)
2. CR-Gated NoGate (Temporal TCN + Contact Awareness)
3. CR-Gated Full (Contact-Aware + Reliability Gated)
4. CR-Gated Ensemble (Multi-Scale Temporal Consensus)

Uses instant disk caching from work_dirs/benchmark_80_features_cache.pt.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ai_classifier.models import FusionHead, CRGatedFusionModel, STROKE_CLASSES, SIDE_CLASSES


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate ShuttleSet 80 Benchmark Clips")
    parser.add_argument(
        "--cache-file",
        type=Path,
        default=REPO_ROOT / "work_dirs" / "benchmark_80_features_cache.pt",
        help="Path to feature cache file (.pt)",
    )
    return parser.parse_args()


def load_all_models(device: torch.device):
    # 1. Baseline Concat-MLP
    base_ckpt_path = REPO_ROOT / "work_dirs" / "shuttleset_fusion_epoch20_full_b4" / "best.pth"
    base_ckpt = torch.load(base_ckpt_path, map_location="cpu", weights_only=False)
    structured_dim = base_ckpt["model"]["structured.1.weight"].shape[1]
    baseline_model = FusionHead(512, structured_dim).to(device)
    baseline_model.load_state_dict(base_ckpt["model"])
    baseline_model.eval()

    # 2. CR-Gated NoGate
    nogate_ckpt_path = REPO_ROOT / "work_dirs" / "cr_gated_fusion" / "cr_gated_no_gate" / "best.pth"
    nogate_ckpt = torch.load(nogate_ckpt_path, map_location="cpu", weights_only=False)
    nogate_model = CRGatedFusionModel(
        hidden_dim=128,
        num_stroke_classes=len(STROKE_CLASSES),
        num_side_classes=len(SIDE_CLASSES),
        use_temporal=True,
        use_contact=True,
        use_gate=False,
    ).to(device)
    nogate_model.load_state_dict(nogate_ckpt["model"])
    nogate_model.eval()

    # 3. CR-Gated Full
    full_ckpt_path = REPO_ROOT / "work_dirs" / "cr_gated_fusion" / "cr_gated_full" / "best.pth"
    full_ckpt = torch.load(full_ckpt_path, map_location="cpu", weights_only=False)
    full_model = CRGatedFusionModel(
        hidden_dim=128,
        num_stroke_classes=len(STROKE_CLASSES),
        num_side_classes=len(SIDE_CLASSES),
        use_temporal=True,
        use_contact=True,
        use_gate=True,
    ).to(device)
    full_model.load_state_dict(full_ckpt["model"])
    full_model.eval()

    # 4. CR-Gated NoTemporal
    nt_ckpt_path = REPO_ROOT / "work_dirs" / "cr_gated_fusion" / "cr_gated_no_temporal" / "best.pth"
    nt_ckpt = torch.load(nt_ckpt_path, map_location="cpu", weights_only=False)
    nt_model = CRGatedFusionModel(
        hidden_dim=128,
        num_stroke_classes=len(STROKE_CLASSES),
        num_side_classes=len(SIDE_CLASSES),
        use_temporal=False,
        use_contact=False,
        use_gate=False,
    ).to(device)
    nt_model.load_state_dict(nt_ckpt["model"])
    nt_model.eval()

    return {
        "baseline": baseline_model,
        "nogate": nogate_model,
        "full": full_model,
        "no_temporal": nt_model,
    }


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not args.cache_file.exists():
        print(f"Error: Cache file not found: {args.cache_file}")
        return

    print("=" * 96)
    print("BENCHMARK 80 CLIPS: BASELINE vs CR-GATED NOGATE vs CR-GATED FULL vs MULTI-SCALE ENSEMBLE")
    print(f"Device : {device}")
    print(f"Cache  : {args.cache_file}")
    print("=" * 96)

    models = load_all_models(device)
    cache = torch.load(args.cache_file, map_location="cpu", weights_only=False)
    print(f"[*] Loaded {len(cache)} cached clips.")

    classes = ["clear", "smash", "drop", "net_shot", "lift", "drive", "net_attack", "serve"]
    results = []

    t_start = time.time()
    for clip_name, data in sorted(cache.items()):
        gt = None
        for c in classes:
            if f"_{c}." in clip_name.lower() or f"_{c}_" in clip_name.lower():
                gt = c
                break
        if gt is None:
            continue

        emb = data["embedding"].to(device)
        t_b_struct = data["t_b_struct"].to(device)
        t_pose = data["t_pose"].to(device)
        t_court = data["t_court"].to(device)
        t_shuttle = data["t_shuttle"].to(device)
        t_contact = data["t_contact"].to(device)
        t_qual = data["t_qual"].to(device)

        with torch.inference_mode():
            # 1. Baseline
            b_logits, _ = models["baseline"](emb, t_b_struct)
            b_prob = torch.softmax(b_logits[0], dim=-1)
            b_pred = STROKE_CLASSES[b_prob.argmax().item()]

            # 2. NoGate
            ng_logits, _, _ = models["nogate"](emb, t_pose, t_court, t_shuttle, t_contact, t_qual)
            ng_prob = torch.softmax(ng_logits[0], dim=-1)
            ng_pred = STROKE_CLASSES[ng_prob.argmax().item()]

            # 3. Full
            f_logits, _, _ = models["full"](emb, t_pose, t_court, t_shuttle, t_contact, t_qual)
            f_prob = torch.softmax(f_logits[0], dim=-1)
            f_pred = STROKE_CLASSES[f_prob.argmax().item()]

            # 4. NoTemporal
            nt_logits, _, _ = models["no_temporal"](emb, t_pose, t_court, t_shuttle, t_contact, t_qual)
            nt_prob = torch.softmax(nt_logits[0], dim=-1)

            # 5. Multi-Scale Consensus Ensemble (Full 40% + NoGate 30% + NoTemporal 30%)
            ens_prob = f_prob * 0.40 + ng_prob * 0.30 + nt_prob * 0.30
            ens_pred = STROKE_CLASSES[ens_prob.argmax().item()]

        results.append({
            "clip": clip_name,
            "gt": gt,
            "baseline": b_pred,
            "baseline_ok": b_pred == gt,
            "nogate": ng_pred,
            "nogate_ok": ng_pred == gt,
            "full": f_pred,
            "full_ok": f_pred == gt,
            "ensemble": ens_pred,
            "ensemble_ok": ens_pred == gt,
        })

    elapsed = time.time() - t_start

    # Summary
    per_class_b = defaultdict(lambda: [0, 0])
    per_class_ng = defaultdict(lambda: [0, 0])
    per_class_f = defaultdict(lambda: [0, 0])
    per_class_ens = defaultdict(lambda: [0, 0])

    for r in results:
        gt = r["gt"]
        per_class_b[gt][1] += 1
        per_class_ng[gt][1] += 1
        per_class_f[gt][1] += 1
        per_class_ens[gt][1] += 1

        if r["baseline_ok"]: per_class_b[gt][0] += 1
        if r["nogate_ok"]: per_class_ng[gt][0] += 1
        if r["full_ok"]: per_class_f[gt][0] += 1
        if r["ensemble_ok"]: per_class_ens[gt][0] += 1

    print("\n" + "=" * 96)
    print("BANG TONG HOP SO SANH BENCHMARK 80 CLIPS (SHUTTLESET STANDARDIZED)")
    print("=" * 96)
    print(f"{'Stroke Class':<13} | {'Baseline':<16} | {'CR-NoGate':<16} | {'CR-Full':<16} | {'CR-Ensemble':<16} | {'Improvement'}")
    print("-" * 96)

    for c in classes:
        b_c, b_t = per_class_b[c]
        ng_c, _ = per_class_ng[c]
        f_c, _ = per_class_f[c]
        ens_c, _ = per_class_ens[c]

        b_pct = b_c / b_t * 100
        ng_pct = ng_c / b_t * 100
        f_pct = f_c / b_t * 100
        ens_pct = ens_c / b_t * 100

        diff = ens_pct - b_pct
        diff_str = f"+{diff:.1f}%" if diff > 0 else (f"{diff:.1f}%" if diff < 0 else "0.0%")
        highlight = " <--" if diff > 0 else ""

        print(
            f"{c:<13} | {b_pct:5.1f}% ({b_c:2d}/{b_t:2d})   "
            f"| {ng_pct:5.1f}% ({ng_c:2d}/{b_t:2d})   "
            f"| {f_pct:5.1f}% ({f_c:2d}/{b_t:2d})   "
            f"| {ens_pct:5.1f}% ({ens_c:2d}/{b_t:2d})   "
            f"| {diff_str:>7}{highlight}"
        )

    b_total = sum(r["baseline_ok"] for r in results)
    ng_total = sum(r["nogate_ok"] for r in results)
    f_total = sum(r["full_ok"] for r in results)
    ens_total = sum(r["ensemble_ok"] for r in results)
    total_len = len(results)

    print("-" * 96)
    print(
        f"{'OVERALL ACC':<13} | {b_total/total_len*100:5.1f}% ({b_total:2d}/{total_len:2d})   "
        f"| {ng_total/total_len*100:5.1f}% ({ng_total:2d}/{total_len:2d})   "
        f"| {f_total/total_len*100:5.1f}% ({f_total:2d}/{total_len:2d})   "
        f"| {ens_total/total_len*100:5.1f}% ({ens_total:2d}/{total_len:2d})   "
        f"| +{(ens_total - b_total)/total_len*100:.1f}%"
    )
    print("=" * 96)
    print(f"Toc do danh gia: {elapsed:.2f}s tren toan bo 80 clips (nho feature cache)")

    out_file = REPO_ROOT / "work_dirs" / "benchmark_80_quad_results.json"
    with out_file.open("w", encoding="utf-8") as fp:
        json.dump({
            "total_clips": total_len,
            "overall_accuracy": {
                "baseline": b_total / total_len * 100,
                "cr_nogate": ng_total / total_len * 100,
                "cr_full": f_total / total_len * 100,
                "cr_ensemble": ens_total / total_len * 100,
            },
            "per_class": {
                c: {
                    "baseline": per_class_b[c],
                    "nogate": per_class_ng[c],
                    "full": per_class_f[c],
                    "ensemble": per_class_ens[c],
                } for c in classes
            },
            "results": results,
        }, fp, indent=2)
    print(f"Da luu ket qua vao: {out_file}")


if __name__ == "__main__":
    main()
