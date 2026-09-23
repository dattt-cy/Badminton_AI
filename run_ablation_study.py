"""Automated Ablation Study Runner for Contact-Aware Reliability-Gated Multimodal Fusion.

Runs 4 comparative experiments:
1. Full Model (CR-Gated Fusion: Temporal TCN + Contact-Aware + Reliability Gate + Cross-Modal Attention)
2. Ablation: No Gate (w/o Reliability Gating)
3. Ablation: No Contact (w/o Contact-Aware Alignment)
4. Ablation: No Temporal (Flatten + MLP)

Generates comparison tables, per-class breakdowns, and ablation impact analysis for academic reporting.
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
PYTHON = sys.executable

MODES = [
    ("full", "Full Proposed (CR-Gated Fusion)"),
    ("no_gate", "w/o Reliability Gate"),
    ("no_contact", "w/o Contact-Aware Alignment"),
    ("no_temporal", "w/o Temporal Encoder (Concat-MLP)"),
]


def run_experiment(mode: str, label: str, epochs: int = 25, seed: int = 20260923) -> dict:
    print("\n" + "=" * 70)
    print(f"RUNNING EXPERIMENT: {mode.upper()} -> {label}")
    print("=" * 70)
    cmd = [
        PYTHON, "-u", "train_cr_gated_fusion.py",
        "--ablation", mode,
        "--epochs", str(epochs),
        "--batch-size", "128",
        "--learning-rate", "8e-4",
        "--drive-boost", "2.0",
        "--seed", str(seed),
    ]
    t0 = time.time()
    result = subprocess.run(cmd, cwd=str(REPO_ROOT), check=True)
    elapsed = time.time() - t0
    print(f"Completed {mode} in {elapsed:.1f}s ({elapsed/60:.2f} mins)")

    metrics_file = REPO_ROOT / "work_dirs" / "cr_gated_fusion" / f"cr_gated_{mode}" / "test_metrics.json"
    with metrics_file.open("r", encoding="utf-8") as f:
        data = json.load(f)
    data["elapsed_sec"] = elapsed
    data["label"] = label
    return data


def main():
    print("=" * 70)
    print("STARTING SCIENTIFIC ABLATION STUDY: CR-GATED MULTIMODAL FUSION")
    print("=" * 70)

    results = {}
    for mode, label in MODES:
        results[mode] = run_experiment(mode, label, epochs=25)

    # Print comprehensive Markdown comparison table
    print("\n" + "=" * 70)
    print("BANG TONG HOP ABLATION STUDY (TEST SET - 871 MAU)")
    print("=" * 70)
    print(f"{'Mô hình / Biến thể':<35} | {'Acc (%)':<8} | {'Macro-F1 (%)':<13} | {'Drive F1 (%)':<13} | {'Net-Attack F1 (%)'}")
    print("-" * 88)

    for mode, label in MODES:
        res = results[mode]
        acc = res["test_stroke"]["accuracy"] * 100
        mf1 = res["test_stroke"]["macro_f1"] * 100
        drive_f1 = res["test_stroke"]["f1"][6] * 100
        net_f1 = res["test_stroke"]["f1"][7] * 100
        print(f"{label:<35} | {acc:>6.2f}%  | {mf1:>10.2f}%   | {drive_f1:>10.2f}%   | {net_f1:>12.2f}%")

    print("-" * 88)

    # Save summary json
    summary_path = REPO_ROOT / "work_dirs" / "cr_gated_fusion" / "ablation_summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nDa luu bang tong hop vao: {summary_path}")


if __name__ == "__main__":
    main()
