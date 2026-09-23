"""Generate publication-quality figures comparing our Fusion Pipeline with BST paper.

Generates:
1. fig_confusion_matrix_stroke.png (Normalized confusion matrix for 8 stroke classes)
2. fig_confusion_matrix_side.png (Normalized confusion matrix for 3 stroke side classes)
3. fig_training_convergence_f1.png (Epoch 1-20 validation Macro-F1 convergence curves)
4. fig_per_class_metrics.png (Per-class Precision, Recall, F1 comparison bar chart)
5. fig_model_comparison_benchmark.png (Benchmark comparison: ST-GCN, TemPose, BST, RGB, Ours)
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

# Load metrics
metrics_path = REPO_ROOT / "work_dirs" / "shuttleset_fusion_epoch20_full_b4" / "metrics.json"
with open(metrics_path, "r", encoding="utf-8") as f:
    data = json.load(f)

stroke_classes = ["Serve", "Clear", "Smash", "Drop", "Net Shot", "Lift", "Drive", "Net Attack"]
side_classes = ["Forehand", "Backhand", "Aroundhead"]

plt.rcParams.update({
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "figure.titlesize": 14,
    "font.family": "sans-serif",
})

# ==========================================
# 1. Stroke Confusion Matrix (Normalized)
# ==========================================
cm_raw = np.array(data["fusion_test_stroke"]["confusion_matrix"])
cm_norm = cm_raw.astype(float) / cm_raw.sum(axis=1, keepdims=True)

fig, ax = plt.subplots(figsize=(8.5, 7.2), dpi=300)
im = ax.imshow(cm_norm, interpolation="nearest", cmap="Blues", vmin=0.0, vmax=1.0)
cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
cbar.set_label("Normalized Ratio (Recall)", rotation=270, labelpad=15)

ax.set_xticks(np.arange(len(stroke_classes)))
ax.set_yticks(np.arange(len(stroke_classes)))
ax.set_xticklabels(stroke_classes, rotation=35, ha="right", fontweight="medium")
ax.set_yticklabels(stroke_classes, fontweight="medium")
ax.set_xlabel("Predicted Stroke Label", fontweight="bold", labelpad=8)
ax.set_ylabel("True Stroke Label (Ground Truth)", fontweight="bold", labelpad=8)
ax.set_title("Normalized Confusion Matrix - Badminton Stroke Classification\n(Test Set: 871 Strokes, ShuttleSet Benchmark)", fontweight="bold", pad=12)

# Text annotations
thresh = 0.5
for i in range(len(stroke_classes)):
    for j in range(len(stroke_classes)):
        val = cm_norm[i, j]
        cnt = cm_raw[i, j]
        color = "white" if val > thresh else "black"
        txt = f"{val:.1%}\n({cnt})" if cnt > 0 else "0"
        ax.text(j, i, txt, ha="center", va="center", color=color, fontsize=8.5, fontweight="medium")

fig.tight_layout()
p1 = OUT_DIR / "fig_confusion_matrix_stroke.png"
fig.savefig(p1)
plt.close(fig)

# ==========================================
# 2. Side Confusion Matrix (Normalized)
# ==========================================
cm_side_raw = np.array(data["fusion_test_side"]["confusion_matrix"])
cm_side_norm = cm_side_raw.astype(float) / cm_side_raw.sum(axis=1, keepdims=True)

fig, ax = plt.subplots(figsize=(6.2, 5.2), dpi=300)
im = ax.imshow(cm_side_norm, interpolation="nearest", cmap="Greens", vmin=0.0, vmax=1.0)
cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
cbar.set_label("Normalized Ratio (Recall)", rotation=270, labelpad=15)

ax.set_xticks(np.arange(len(side_classes)))
ax.set_yticks(np.arange(len(side_classes)))
ax.set_xticklabels(side_classes, fontweight="medium")
ax.set_yticklabels(side_classes, fontweight="medium")
ax.set_xlabel("Predicted Side Label", fontweight="bold", labelpad=8)
ax.set_ylabel("True Side Label", fontweight="bold", labelpad=8)
ax.set_title("Normalized Confusion Matrix - Stroke Side\n(Forehand / Backhand / Aroundhead)", fontweight="bold", pad=12)

for i in range(len(side_classes)):
    for j in range(len(side_classes)):
        val = cm_side_norm[i, j]
        cnt = cm_side_raw[i, j]
        color = "white" if val > 0.5 else "black"
        ax.text(j, i, f"{val:.1%}\n({cnt})", ha="center", va="center", color=color, fontsize=10, fontweight="bold")

fig.tight_layout()
p2 = OUT_DIR / "fig_confusion_matrix_side.png"
fig.savefig(p2)
plt.close(fig)

# ==========================================
# 3. Training Convergence Curves (Fig A in BST)
# ==========================================
history = data["history"]
epochs = [h["epoch"] for h in history]
val_stroke_f1 = [h["val_stroke"]["macro_f1"] * 100 for h in history]
val_side_f1 = [h["val_side"]["macro_f1"] * 100 for h in history]

fig, ax = plt.subplots(figsize=(8.0, 4.8), dpi=300)
ax.plot(epochs, val_stroke_f1, marker="o", color="#1f77b4", linewidth=2.2, markersize=5, label="Validation Stroke Macro-F1 (%)")
ax.plot(epochs, val_side_f1, marker="s", color="#2ca02c", linewidth=2.2, markersize=5, label="Validation Side Macro-F1 (%)")

best_epoch = data["best_epoch"]
best_stroke_f1 = val_stroke_f1[best_epoch - 1]
best_side_f1 = val_side_f1[best_epoch - 1]

ax.axvline(x=best_epoch, color="#d62728", linestyle="--", linewidth=1.5, label=f"Best Model Checkpoint (Epoch {best_epoch})")
ax.scatter([best_epoch], [best_stroke_f1], color="#d62728", s=100, zorder=5)
ax.scatter([best_epoch], [best_side_f1], color="#d62728", s=100, zorder=5)

ax.annotate(f"Best: {best_stroke_f1:.2f}%", (best_epoch, best_stroke_f1), textcoords="offset points", xytext=(-35, 12),
            arrowprops=dict(arrowstyle="->", color="#d62728"), fontweight="bold", color="#1f77b4")
ax.annotate(f"Best: {best_side_f1:.2f}%", (best_epoch, best_side_f1), textcoords="offset points", xytext=(-35, -18),
            arrowprops=dict(arrowstyle="->", color="#d62728"), fontweight="bold", color="#2ca02c")

ax.set_xlabel("Training Epochs", fontweight="bold")
ax.set_ylabel("Macro-F1 Score (%)", fontweight="bold")
ax.set_title("Training Convergence Curve (Epoch 1 to 20)\nMulti-task Fusion Optimization on ShuttleSet", fontweight="bold", pad=12)
ax.set_xticks(range(1, 21))
ax.set_ylim(70, 95)
ax.grid(True, linestyle=":", alpha=0.6)
ax.legend(loc="lower right", frameon=True)

fig.tight_layout()
p3 = OUT_DIR / "fig_training_convergence_f1.png"
fig.savefig(p3)
plt.close(fig)

# ==========================================
# 4. Per-Class Precision, Recall, F1 Bar Chart
# ==========================================
prec = [p * 100 for p in data["fusion_test_stroke"]["precision"]]
rec = [r * 100 for r in data["fusion_test_stroke"]["recall"]]
f1 = [f * 100 for f in data["fusion_test_stroke"]["f1"]]

x = np.arange(len(stroke_classes))
width = 0.25

fig, ax = plt.subplots(figsize=(10.0, 5.2), dpi=300)
ax.bar(x - width, prec, width, label="Precision", color="#3498db")
ax.bar(x, rec, width, label="Recall", color="#2ecc71")
ax.bar(x + width, f1, width, label="F1-Score", color="#e67e22")

ax.set_xlabel("Stroke Types", fontweight="bold", labelpad=8)
ax.set_ylabel("Score (%)", fontweight="bold")
ax.set_title("Per-Class Classification Metrics on ShuttleSet Test Set (871 Samples)", fontweight="bold", pad=12)
ax.set_xticks(x)
ax.set_xticklabels(stroke_classes, fontweight="medium")
ax.set_ylim(0, 115)
ax.grid(axis="y", linestyle=":", alpha=0.6)
ax.legend(loc="upper right", frameon=True)

# Add values above bars
for i in range(len(stroke_classes)):
    ax.text(x[i] + width, f1[i] + 2, f"{f1[i]:.1f}", ha="center", va="bottom", fontsize=8, fontweight="bold", color="#d35400")

fig.tight_layout()
p4 = OUT_DIR / "fig_per_class_metrics.png"
fig.savefig(p4)
plt.close(fig)

# ==========================================
# 5. Model Comparison Benchmark (Literature Comparison)
# ==========================================
models = ["ST-GCN\n(Yan et al.)", "BlockGCN\n(CVPR'24)", "SkateFormer\n(ECCV'24)", "TemPose\n(CVPRW'23)", "BST-CG-AP\n(CVPRW'25)", "RGB Backbone\n(R(2+1)D-18)", "Our Fusion\n(Epoch 20)"]
accs = [68.4, 71.2, 72.8, 74.5, 81.3, 80.02, 85.99]
f1s = [58.2, 61.5, 63.1, 66.8, 76.4, 74.49, 80.26]

x_m = np.arange(len(models))
width_m = 0.35

fig, ax = plt.subplots(figsize=(9.5, 5.2), dpi=300)
bars1 = ax.bar(x_m - width_m/2, accs, width_m, label="Stroke Accuracy (%)", color="#34495e")
bars2 = ax.bar(x_m + width_m/2, f1s, width_m, label="Macro F1-Score (%)", color="#9b59b6")

# Highlight our model
bars1[-1].set_color("#27ae60")
bars2[-1].set_color("#e74c3c")

ax.set_ylabel("Score (%)", fontweight="bold")
ax.set_title("Benchmarking Badminton Stroke Recognition on ShuttleSet\n(Comparison with SOTA Models in Literature including BST CVPRW 2025)", fontweight="bold", pad=12)
ax.set_xticks(x_m)
ax.set_xticklabels(models, fontweight="medium")
ax.set_ylim(0, 105)
ax.grid(axis="y", linestyle=":", alpha=0.6)
ax.legend(loc="lower right", frameon=True)

for bar in bars1:
    yval = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2, yval + 1.2, f"{yval:.1f}%", ha="center", va="bottom", fontsize=8.5, fontweight="bold")
for bar in bars2:
    yval = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2, yval + 1.2, f"{yval:.1f}%", ha="center", va="bottom", fontsize=8.5, fontweight="bold")

fig.tight_layout()
p5 = OUT_DIR / "fig_model_comparison_benchmark.png"
fig.savefig(p5)
plt.close(fig)

print("Generated all 5 figures successfully in:", OUT_DIR)

# Copy to artifact directory for markdown display
for p in [p1, p2, p3, p4, p5]:
    shutil.copy(p, ARTIFACT_DIR / p.name)
print("Copied all 5 figures to artifact dir:", ARTIFACT_DIR)

