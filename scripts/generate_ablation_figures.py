"""Generate publication-quality figures for the Ablation Study and CR-Gated Architecture.

Outputs:
1. fig_ablation_benchmark.png: Multi-metric bar chart (Accuracy, Macro-F1, Drive F1, Net-Attack F1)
2. fig_drive_netattack_evolution.png: Focus on Drive & Net Attack Precision/Recall/F1 jump
3. fig_cr_gated_confusion_matrix.png: Confusion Matrix of the best temporal architecture
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = REPO_ROOT / "docs" / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)
ARTIFACT_DIR = Path(r"C:\Users\ADMIN\.gemini\antigravity\brain\2613cee5-90a2-4966-b44b-e952b365d6dc")

# Load ablation data
summary_path = REPO_ROOT / "work_dirs" / "cr_gated_fusion" / "ablation_summary.json"
with open(summary_path, "r", encoding="utf-8") as f:
    abl_data = json.load(f)

# Baseline Concat-MLP (Epoch 20)
baseline_metrics = {
    "acc": 85.99,
    "macro_f1": 80.26,
    "drive_f1": 48.30,
    "net_f1": 58.80,
    "drive_rec": 43.75,
    "drive_prec": 53.85,
    "net_rec": 71.43,
    "net_prec": 50.00,
}

stroke_classes = ["Serve", "Clear", "Smash", "Drop", "Net Shot", "Lift", "Drive", "Net Attack"]

plt.rcParams.update({
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "figure.titlesize": 14,
    "font.family": "sans-serif",
})

# =========================================================================
# 1. Ablation Benchmark Bar Chart
# =========================================================================
models = [
    "Concat-MLP Baseline\n(Epoch 20)",
    "w/o Temporal\n(Flatten + MLP)",
    "Full Proposed\n(CR-Gated)",
    "w/o Contact\nAlignment",
    "Temporal + Contact\n(w/o Rel Gate)",
]

accs = [
    baseline_metrics["acc"],
    abl_data["no_temporal"]["test_stroke"]["accuracy"] * 100,
    abl_data["full"]["test_stroke"]["accuracy"] * 100,
    abl_data["no_contact"]["test_stroke"]["accuracy"] * 100,
    abl_data["no_gate"]["test_stroke"]["accuracy"] * 100,
]

mf1s = [
    baseline_metrics["macro_f1"],
    abl_data["no_temporal"]["test_stroke"]["macro_f1"] * 100,
    abl_data["full"]["test_stroke"]["macro_f1"] * 100,
    abl_data["no_contact"]["test_stroke"]["macro_f1"] * 100,
    abl_data["no_gate"]["test_stroke"]["macro_f1"] * 100,
]

drive_f1s = [
    baseline_metrics["drive_f1"],
    abl_data["no_temporal"]["test_stroke"]["f1"][6] * 100,
    abl_data["full"]["test_stroke"]["f1"][6] * 100,
    abl_data["no_contact"]["test_stroke"]["f1"][6] * 100,
    abl_data["no_gate"]["test_stroke"]["f1"][6] * 100,
]

x = np.arange(len(models))
width = 0.26

fig, ax = plt.subplots(figsize=(10.5, 6.0), dpi=300)

rects1 = ax.bar(x - width, accs, width, label="Top-1 Accuracy", color="#2b5c8f", edgecolor="black", linewidth=0.8)
rects2 = ax.bar(x, mf1s, width, label="Macro-F1", color="#3b82f6", edgecolor="black", linewidth=0.8)
rects3 = ax.bar(x + width, drive_f1s, width, label="Drive F1 (Chống nhầm)", color="#f59e0b", edgecolor="black", linewidth=0.8)

ax.set_ylabel("Score (%)", fontweight="bold", labelpad=8)
ax.set_title("Ablation Study Comparison on ShuttleSet Test Set (N = 871)", fontweight="bold", pad=12)
ax.set_xticks(x)
ax.set_xticklabels(models, fontweight="semibold")
ax.set_ylim(35, 102)
ax.grid(axis="y", linestyle="--", alpha=0.5)
ax.legend(frameon=True, facecolor="white", edgecolor="#cccccc", loc="upper right")

# Value annotations
def autolabel(rects, fmt="{:.1f}%"):
    for rect in rects:
        h = rect.get_height()
        ax.annotate(fmt.format(h),
                    xy=(rect.get_x() + rect.get_width() / 2, h),
                    xytext=(0, 3), textcoords="offset points",
                    ha="center", va="bottom", fontsize=8.5, fontweight="bold")

autolabel(rects1)
autolabel(rects2)
autolabel(rects3)

# Highlight Drive leap
ax.annotate(
    "Drive F1: +8.58%\n(48.3% -> 56.9%)",
    xy=(4 + width, drive_f1s[4]),
    xytext=(3.5, 72),
    arrowprops=dict(facecolor="#d97706", shrink=0.08, width=1.5, headwidth=6),
    fontweight="bold", color="#b45309",
    bbox=dict(boxstyle="round,pad=0.3", fc="#fef3c7", ec="#f59e0b", lw=1)
)

plt.tight_layout()
p1 = OUT_DIR / "fig_ablation_benchmark.png"
fig.savefig(p1)
plt.close(fig)
print(f"Generated: {p1}")

# =========================================================================
# 2. Drive & Net-Attack Evolution
# =========================================================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 5.2), dpi=300)

metrics_names = ["Precision", "Recall", "F1-Score"]

# Drive
base_drive = [baseline_metrics["drive_prec"], baseline_metrics["drive_rec"], baseline_metrics["drive_f1"]]
best_drive = [
    abl_data["no_gate"]["test_stroke"]["precision"][6] * 100,
    abl_data["no_gate"]["test_stroke"]["recall"][6] * 100,
    abl_data["no_gate"]["test_stroke"]["f1"][6] * 100,
]

x_m = np.arange(len(metrics_names))
w = 0.35

ax1.bar(x_m - w/2, base_drive, w, label="Concat-MLP Baseline", color="#94a3b8", edgecolor="black")
ax1.bar(x_m + w/2, best_drive, w, label="Temporal + Contact (Ours)", color="#f59e0b", edgecolor="black")
ax1.set_ylabel("Score (%)", fontweight="bold")
ax1.set_title("Cú DRIVE (Kỹ thuật khó nhất)", fontweight="bold")
ax1.set_xticks(x_m)
ax1.set_xticklabels(metrics_names, fontweight="semibold")
ax1.set_ylim(0, 80)
ax1.grid(axis="y", linestyle="--", alpha=0.5)
ax1.legend(loc="upper left")

for i, (b, o) in enumerate(zip(base_drive, best_drive)):
    ax1.annotate(f"{b:.1f}%", (i - w/2, b + 1.5), ha="center", fontsize=9, fontweight="bold", color="#475569")
    ax1.annotate(f"{o:.1f}%", (i + w/2, o + 1.5), ha="center", fontsize=9, fontweight="bold", color="#b45309")

# Net-Attack
base_net = [baseline_metrics["net_prec"], baseline_metrics["net_rec"], baseline_metrics["net_f1"]]
best_net = [
    abl_data["no_gate"]["test_stroke"]["precision"][7] * 100,
    abl_data["no_gate"]["test_stroke"]["recall"][7] * 100,
    abl_data["no_gate"]["test_stroke"]["f1"][7] * 100,
]

ax2.bar(x_m - w/2, base_net, w, label="Concat-MLP Baseline", color="#94a3b8", edgecolor="black")
ax2.bar(x_m + w/2, best_net, w, label="Temporal + Contact (Ours)", color="#10b981", edgecolor="black")
ax2.set_ylabel("Score (%)", fontweight="bold")
ax2.set_title("Cú NET-ATTACK (Vồ lưới phản tạt)", fontweight="bold")
ax2.set_xticks(x_m)
ax2.set_xticklabels(metrics_names, fontweight="semibold")
ax2.set_ylim(0, 85)
ax2.grid(axis="y", linestyle="--", alpha=0.5)
ax2.legend(loc="upper left")

for i, (b, o) in enumerate(zip(base_net, best_net)):
    ax2.annotate(f"{b:.1f}%", (i - w/2, b + 1.5), ha="center", fontsize=9, fontweight="bold", color="#475569")
    ax2.annotate(f"{o:.1f}%", (i + w/2, o + 1.5), ha="center", fontsize=9, fontweight="bold", color="#047857")

plt.tight_layout()
p2 = OUT_DIR / "fig_drive_netattack_evolution.png"
fig.savefig(p2)
plt.close(fig)
print(f"Generated: {p2}")

# =========================================================================
# 3. Normalized Confusion Matrix of Best Architecture (Temporal + Contact)
# =========================================================================
cm_raw = np.array(abl_data["no_gate"]["test_stroke"]["confusion_matrix"])
cm_norm = cm_raw.astype(float) / cm_raw.sum(axis=1, keepdims=True)

fig, ax = plt.subplots(figsize=(8.8, 7.5), dpi=300)
im = ax.imshow(cm_norm, interpolation="nearest", cmap="Blues", vmin=0.0, vmax=1.0)
cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
cbar.set_label("Normalized Ratio (Recall)", rotation=270, labelpad=15)

ax.set_xticks(np.arange(len(stroke_classes)))
ax.set_yticks(np.arange(len(stroke_classes)))
ax.set_xticklabels(stroke_classes, rotation=35, ha="right", fontweight="medium")
ax.set_yticklabels(stroke_classes, fontweight="medium")
ax.set_xlabel("Predicted Stroke Label", fontweight="bold", labelpad=8)
ax.set_ylabel("True Stroke Label", fontweight="bold", labelpad=8)
ax.set_title("Temporal + Contact Model: Normalized Stroke Confusion Matrix (N = 871)", fontweight="bold", pad=12)

# Text overlay
thresh = cm_norm.max() / 2.0
for i in range(len(stroke_classes)):
    for j in range(len(stroke_classes)):
        raw_val = cm_raw[i, j]
        norm_val = cm_norm[i, j]
        color = "white" if norm_val > thresh else "black"
        ax.text(j, i, f"{norm_val*100:.1f}%\n({raw_val})",
                ha="center", va="center", color=color, fontsize=8.2, fontweight="medium")

plt.tight_layout()
p3 = OUT_DIR / "fig_cr_gated_confusion_matrix.png"
fig.savefig(p3)
plt.close(fig)
print(f"Generated: {p3}")

# Copy to ARTIFACT_DIR for instant display in markdown
for p in [p1, p2, p3]:
    if ARTIFACT_DIR.exists():
        shutil.copy(p, ARTIFACT_DIR / p.name)
        print(f"Copied to artifact dir: {p.name}")
