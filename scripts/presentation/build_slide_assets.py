"""Build reproducible charts used by the current project presentation."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "docs" / "slide_assets"
COLORS = {"blue": "#2563EB", "cyan": "#06B6D4", "orange": "#F59E0B", "slate": "#334155"}


def save_training_curve() -> None:
    metrics = json.loads(
        (ROOT / "work_dirs/r2plus1d18_mixed_shuttleset_finebadminton/metrics.json").read_text()
    )
    epochs = [row["epoch"] for row in metrics["history"]]
    stroke = [row["val_stroke"]["macro_f1"] * 100 for row in metrics["history"]]
    side = [row["val_side"]["macro_f1"] * 100 for row in metrics["history"]]
    fig, axis = plt.subplots(figsize=(9.6, 5.4))
    axis.plot(epochs, stroke, marker="o", linewidth=2.8, color=COLORS["blue"], label="Stroke Macro-F1")
    axis.plot(epochs, side, marker="o", linewidth=2.8, color=COLORS["cyan"], label="Side Macro-F1")
    axis.scatter([20], [side[-1]], s=150, color=COLORS["orange"], zorder=3)
    axis.annotate("checkpoint epoch 20", (20, side[-1]), xytext=(-125, 20), textcoords="offset points")
    axis.set(xlabel="Epoch", ylabel="Validation Macro-F1 (%)", ylim=(65, 78))
    axis.grid(alpha=0.22)
    axis.legend(loc="lower right", frameon=False)
    fig.tight_layout()
    fig.savefig(OUTPUT / "epoch20_training_curve.png", dpi=180, transparent=True)
    plt.close(fig)


def save_confusion_matrix() -> None:
    checkpoint = torch.load(
        ROOT / "work_dirs/r2plus1d18_mixed_shuttleset_finebadminton/best.pth",
        map_location="cpu", weights_only=False,
    )
    classes = list(checkpoint["stroke_classes"])
    confusion = np.asarray(checkpoint["val_stroke_metrics"]["confusion_matrix"], dtype=float)
    normalized = confusion / np.maximum(confusion.sum(axis=1, keepdims=True), 1)
    fig, axis = plt.subplots(figsize=(8.5, 7.2))
    image = axis.imshow(normalized * 100, cmap="Blues", vmin=0, vmax=100)
    for row in range(len(classes)):
        for column in range(len(classes)):
            value = normalized[row, column] * 100
            if value >= 3 or row == column:
                axis.text(column, row, f"{value:.0f}", ha="center", va="center",
                          color="white" if value > 52 else COLORS["slate"], fontsize=9)
    axis.set_xticks(range(len(classes)), classes, rotation=35, ha="right")
    axis.set_yticks(range(len(classes)), classes)
    axis.set_xlabel("Predicted")
    axis.set_ylabel("Ground truth")
    axis.set_title("Epoch 20 · normalized stroke confusion (%)")
    fig.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(OUTPUT / "epoch20_confusion_matrix.png", dpi=180, transparent=True)
    plt.close(fig)


def save_dataset_scale() -> None:
    labels = ["Fine-Badminton\nintervals", "ShuttleSet\nvalid manifest", "Hit-full\ntrain samples"]
    values = [9817, 32166, 41524]
    fig, axis = plt.subplots(figsize=(9.6, 5.4))
    bars = axis.bar(labels, values, color=[COLORS["cyan"], COLORS["blue"], COLORS["orange"]], width=0.62)
    axis.bar_label(bars, labels=[f"{value:,}".replace(",", ".") for value in values], padding=6, fontsize=12)
    axis.set_ylabel("Số annotation / sample")
    axis.set_ylim(0, 47000)
    axis.spines[["top", "right"]].set_visible(False)
    axis.grid(axis="y", alpha=0.18)
    fig.tight_layout()
    fig.savefig(OUTPUT / "dataset_scale.png", dpi=180, transparent=True)
    plt.close(fig)


def save_direct_ablation() -> None:
    labels = ["Độ chính xác\nloại cú", "F1 trung bình\nloại cú", "Độ chính xác\nhướng đánh", "F1 trung bình\nhướng đánh"]
    rgb = [74.05, 68.96, 78.76, 81.51]
    fusion = [86.91, 81.72, 95.29, 94.18]
    positions = np.arange(len(labels))
    width = 0.34
    fig, axis = plt.subplots(figsize=(10.5, 5.6))
    first = axis.bar(positions - width / 2, rgb, width, label="RGB cơ sở", color=COLORS["blue"])
    second = axis.bar(positions + width / 2, fusion, width, label="Kết hợp nhiều dữ liệu", color=COLORS["orange"])
    axis.bar_label(first, fmt="%.1f", padding=3, fontsize=10)
    axis.bar_label(second, fmt="%.1f", padding=3, fontsize=10)
    axis.set_xticks(positions, labels)
    axis.set_ylabel("Kết quả (%)")
    axis.set_ylim(0, 105)
    axis.set_title("So sánh trực tiếp trên cùng 871 mẫu ShuttleSet")
    axis.legend(frameon=False, loc="lower right")
    axis.grid(axis="y", alpha=0.18)
    axis.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUTPUT / "rgb_fusion_direct_ablation.png", dpi=180, transparent=True)
    plt.close(fig)


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    save_training_curve()
    save_confusion_matrix()
    save_dataset_scale()
    save_direct_ablation()
    print(f"Wrote slide assets to {OUTPUT}")


if __name__ == "__main__":
    main()
