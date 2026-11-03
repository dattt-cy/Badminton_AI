"""Sinh số liệu, dòng bảng và hình cho bài báo LaTeX từ kết quả thực nghiệm.

Đọc work_dirs/paper_ablation/summary.json và worker_8clips_soup_t75/joint_f1.json,
ghi vào <overleaf>/generated/*.tex và <overleaf>/figures/{degradation,confusion}.pdf.

Example:
    python scripts/paper/make_latex.py --overleaf ../SuaBaoCao/overleaf
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
CLASSES = ["serve", "clear", "smash", "drop", "net_shot", "lift", "drive", "net_attack"]
CLASS_VI = {
    "serve": "Serve (giao cầu)", "clear": "Clear (phông cầu)", "smash": "Smash (đập cầu)",
    "drop": "Drop (chặt cầu)", "net_shot": "Net shot (bỏ nhỏ)", "lift": "Lift (hất cầu)",
    "drive": "Drive (tạt cầu)", "net_attack": "Net attack (đẩy/vồ)",
}
SHORT = ["Serve", "Clear", "Smash", "Drop", "Net shot", "Lift", "Drive", "Net att."]
ABLATION = [
    ("main", r"\textbf{Robust KAP}", "--"),
    ("gate", "+ cổng chất lượng", "Phiên bản trước: cổng sigmoid theo vector chỉ số mất dấu"),
    ("no_phase", "w/o 3 pha", "Gom trung bình toàn chuỗi thay cho 3 pha"),
    ("no_cross", "w/o cross-attn", "Bỏ nhánh khung xương--cầu và token $z_{\\text{cross}}$"),
    ("no_hitter", "w/o phân định", "Trộn hai người chơi với $\\alpha_1=\\alpha_2=0{,}5$"),
    ("no_noise", "w/o nhánh nhiễu", "Bỏ $\\mathcal{L}_{\\text{noisy}}$ và $\\mathcal{L}_{\\text{cons}}$"),
    ("no_rgb", "w/o RGB", "Bỏ token RGB (không cần R(2+1)D, 31,3M)"),
]
CURVES = [
    ("kap_main", "Robust KAP", "#2b7bba", "-", "o"),
    ("kap_gate", "KAP + cổng CL", "#8e6cc7", "--", "s"),
    ("kap_no_noise", "KAP w/o nhánh nhiễu", "#3a9e4a", "-.", "^"),
    ("kap_no_rgb", "KAP w/o RGB", "#7f7f7f", ":", "D"),
    ("bst_cg_ap", "BST-CG-AP", "#d0453b", "-", "v"),
]


def vn(value: float, digits: int = 2) -> str:
    """Số kiểu Việt Nam: dấu phẩy thập phân, dấu chấm phân cách nghìn."""
    text = f"{value:,.{digits}f}"
    return text.replace(",", "X").replace(".", ",").replace("X", ".")


def pm(stat: dict, digits: int = 2) -> str:
    if stat.get("n", 2) < 2:
        return vn(stat["mean"], digits)  # một seed: không có độ lệch chuẩn
    return rf"\pmstd{{{vn(stat['mean'], digits)}}}{{{vn(stat['std'], digits)}}}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--overleaf", type=Path, default=REPO_ROOT.parent / "SuaBaoCao/overleaf")
    parser.add_argument("--summary", type=Path, default=REPO_ROOT / "work_dirs/paper_final/summary.json")
    parser.add_argument("--e2e", type=Path, default=REPO_ROOT / "work_dirs/worker_8clips_soup_t75/joint_f1.json")
    args = parser.parse_args()
    s = json.loads(args.summary.read_text(encoding="utf-8"))
    e2e = json.loads(args.e2e.read_text(encoding="utf-8"))
    gen = args.overleaf / "generated"
    figs = args.overleaf / "figures"
    gen.mkdir(exist_ok=True)
    v = s["variants"]
    full = v["main"]
    deg = s["degradation"]

    # ------------------------------------------------------------ macros
    counts = s["class_counts"]
    # Paired tests per (KAP seed, BST checkpoint): each single model against each single model.
    pairs = [t for k, t in s["significance"].items() if k.startswith("kap__main_seed")]
    d_acc = [t["bootstrap"]["delta_acc"]["mean"] for t in pairs]
    d_f1 = [t["bootstrap"]["delta_macro_f1"]["mean"] for t in pairs]
    only_kap = [t["mcnemar"]["only_a_correct"] for t in pairs]
    only_bst = [t["mcnemar"]["only_b_correct"] for t in pairs]
    mcnemar_sig = sum(t["mcnemar"]["p_value"] < 0.05 for t in pairs)
    f1_sig = sum(t["bootstrap"]["delta_macro_f1"]["ci95"][1] < 0 or t["bootstrap"]["delta_macro_f1"]["ci95"][0] > 0
                 for t in pairs)
    t5, t12 = e2e["test@5"], e2e["test@12"]
    speed = e2e["test_video_seconds"] / e2e["test_processing_seconds"]
    macros = {
        "kapParams": vn(full["params"], 0), "kapSeeds": str(len(full["seeds"])), "kapParamsM": vn(full["params"] / 1e6, 2) + "M",
        "kapAcc": vn(full["accuracy"]["mean"]), "kapAccStd": vn(full["accuracy"]["std"]),
        "kapFone": vn(full["macro_f1"]["mean"]), "kapFoneStd": vn(full["macro_f1"]["std"]),
        "kapBal": vn(full["balanced_accuracy"]["mean"]), "kapBalStd": vn(full["balanced_accuracy"]["std"]),
        "kapSide": vn(full["side_accuracy"]["mean"]), "kapSideStd": vn(full["side_accuracy"]["std"]),
        "bstAcc": vn(s["bst_clean"]["accuracy"]["mean"]), "bstAccStd": vn(s["bst_clean"]["accuracy"]["std"]),
        "bstFone": vn(s["bst_clean"]["macro_f1"]["mean"]), "bstFoneStd": vn(s["bst_clean"]["macro_f1"]["std"]),
        "bstBal": vn(s["bst_clean"]["balanced_accuracy"]["mean"]),
        "bstBalStd": vn(s["bst_clean"]["balanced_accuracy"]["std"]),
        "kapBothFifty": vn(deg["kap_main"]["both@0.5"]["accuracy"]["mean"]),
        "bstBothFifty": vn(deg["bst_cg_ap"]["both@0.5"]["accuracy"]["mean"]),
        "kapBothThirty": vn(deg["kap_main"]["both@0.3"]["accuracy"]["mean"]),
        "kapBurstFifty": vn(deg["kap_main"]["burst@0.5"]["accuracy"]["mean"]),
        "bstBurstFifty": vn(deg["bst_cg_ap"]["burst@0.5"]["accuracy"]["mean"]),
        "bstBothThirty": vn(deg["bst_cg_ap"]["both@0.3"]["accuracy"]["mean"]),
        "gapAcc": vn(s["bst_clean"]["accuracy"]["mean"] - full["accuracy"]["mean"], 1),
        "sigPairs": str(len(pairs)),
        "sigDaccMin": vn(min(d_acc)), "sigDaccMax": vn(max(d_acc)),
        "sigDfoneMin": vn(min(d_f1)), "sigDfoneMax": vn(max(d_f1)),
        "sigMcSig": str(mcnemar_sig), "sigFoneSig": str(f1_sig),
        "sigOnlyKap": vn(float(np.mean(only_kap)), 0), "sigOnlyBst": vn(float(np.mean(only_bst)), 0),
        "gateShLow": vn(s["gate_analysis"]["g_sh_low_loss"]), "gateShHigh": vn(s["gate_analysis"]["g_sh_high_loss"]),
        "gateShCorr": vn(s["gate_analysis"]["g_sh_corr"], 1),
        "gatePoseMin": vn(s["gate_analysis"]["g_pose_min"]), "gatePoseMax": vn(s["gate_analysis"]["g_pose_max"]),
        "mainShuttleAll": vn(deg["kap_main"]["shuttle@1.0"]["accuracy"]["mean"], 1),
        "gateShuttleAll": vn(deg["kap_gate"]["shuttle@1.0"]["accuracy"]["mean"], 1),
        "noNoiseShuttleAll": vn(deg["kap_no_noise"]["shuttle@1.0"]["accuracy"]["mean"], 1),
        "bstShuttleAll": vn(deg["bst_cg_ap"]["shuttle@1.0"]["accuracy"]["mean"], 1),
        "noRgbAcc": vn(v["no_rgb"]["accuracy"]["mean"]), "noRgbFone": vn(v["no_rgb"]["macro_f1"]["mean"]),
        "noRgbParamsM": vn(v["no_rgb"]["params"] / 1e6, 2) + "M",
        "noRgbBothFifty": vn(deg["kap_no_rgb"]["both@0.5"]["accuracy"]["mean"]),
        "noRgbBurstFifty": vn(deg["kap_no_rgb"]["burst@0.5"]["accuracy"]["mean"]),
        "noRgbPoseFifty": vn(deg["kap_no_rgb"]["pose@0.5"]["accuracy"]["mean"]),
        "kapPoseFifty": vn(deg["kap_main"]["pose@0.5"]["accuracy"]["mean"]),
        "bstPoseFifty": vn(deg["bst_cg_ap"]["pose@0.5"]["accuracy"]["mean"]),
        "noNoiseBothFifty": vn(deg["kap_no_noise"]["both@0.5"]["accuracy"]["mean"]),
        "truncShare": vn(s["window_truncation"]["truncated_share"], 1),
        "truncFullN": vn(s["window_truncation"]["full"]["n"], 0),
        "truncFullGap": vn(s["window_truncation"]["full"]["bst"] - s["window_truncation"]["full"]["kap"], 1),
        "truncCutGap": vn(s["window_truncation"]["truncated"]["bst"] - s["window_truncation"]["truncated"]["kap"], 1),
        "valBst": vn(s["validation_probe"]["bst_acc"]),
        "valBestExplore": vn(s["validation_probe"]["best_explore_acc"]),
        "valMain": vn(float(np.mean(s["validation_probe"]["main_val_acc"]))),
        "splitTrainMatches": ", ".join(map(str, s["matches"]["train"])),
        "splitTrainN": vn(sum(counts["train"].values()), 0), "splitValN": vn(sum(counts["val"].values()), 0),
        "splitTestN": vn(sum(counts["test"].values()), 0),
        "imbalance": vn(counts["test"]["net_shot"] / counts["test"]["drive"], 1),
        "bootRounds": vn(2000, 0),
        "eteGt": vn(t5["gt"], 0), "eteHitFFive": vn(t5["hit_f1"], 1), "eteJointFive": vn(t5["joint_f1"], 1),
        "eteHitFTwelve": vn(t12["hit_f1"], 1), "eteJointTwelve": vn(t12["joint_f1"], 1),
        "eteHitter": vn(t5["hitter_acc_matched"], 1), "eteSpeed": vn(speed, 1),
        "eteMinutes": vn(e2e["test_processing_seconds"] / 60, 0),
    }
    (gen / "numbers.tex").write_text(
        "% Sinh tự động bởi AI_Classifier/scripts/paper/make_latex.py -- không sửa tay.\n"
        + "".join(rf"\newcommand{{\{k}}}{{{val}}}" + "\n" for k, val in macros.items()),
        encoding="utf-8",
    )

    # ------------------------------------------------------------ bảng
    rows = [f"{CLASS_VI[c]} & {vn(counts['train'][c], 0)} & {vn(counts['val'][c], 0)} & {vn(counts['test'][c], 0)}\\\\"
            for c in CLASSES]
    rows.append(r"\midrule")
    rows.append(r"\textbf{Tổng} & \textbf{" + " & \\textbf{".join(
        vn(sum(counts[sp].values()), 0) + "}" for sp in ("train", "val", "test")) + r"\\")
    (gen / "split_rows.tex").write_text("\n".join(rows) + "\n", encoding="utf-8")

    rows = []
    for match in sorted(full["per_match"], key=int):
        m = full["per_match"][match]
        rows.append(f"Match {match} & {vn(m['n'], 0)} & {pm(m['accuracy'])} & {pm(m['macro_f1'])} & {pm(m['side_accuracy'])}\\\\")
    rows.append(r"\midrule")
    rows.append(rf"\textbf{{Toàn bộ test}} & \textbf{{3.334}} & {pm(full['accuracy'])} & {pm(full['macro_f1'])} & {pm(full['side_accuracy'])}\\")
    ams = full["across_match_std"]
    rows.append(rf"Lệch giữa các trận & -- & $\pm${vn(ams['accuracy'])} & $\pm${vn(ams['macro_f1'])} & $\pm${vn(ams['side_accuracy'])}\\")
    (gen / "per_match_rows.tex").write_text("\n".join(rows) + "\n", encoding="utf-8")

    rows = []
    for key, name, change in ABLATION:
        if key not in v:
            continue
        r = v[key]
        d = deg.get(f"kap_{key}", {}).get("both@0.3")
        noisy = f"{pm(d['accuracy'])} & {pm(d['macro_f1'])}" if d else "-- & --"
        rows.append(f"{name} ({len(r['seeds'])}) & {change} & {vn(r['params'] / 1e6, 2)}M & {pm(r['accuracy'])} & {pm(r['macro_f1'])} & {noisy}\\\\")
        if key == "main":
            rows.append(r"\midrule")
    bst = deg["bst_cg_ap"]
    rows.append(r"\midrule")
    rows.append(rf"BST-CG-AP~\cite{{chang2025bst}} & 3 checkpoint chính thức & 1,87M & {pm(s['bst_clean']['accuracy'])} & {pm(s['bst_clean']['macro_f1'])} & {pm(bst['both@0.3']['accuracy'])} & {pm(bst['both@0.3']['macro_f1'])}\\")
    (gen / "ablation_rows.tex").write_text("\n".join(rows) + "\n", encoding="utf-8")

    rates = ["0.0", "0.1", "0.2", "0.3", "0.4", "0.5"]
    rows = []
    for condition, title in (("both", "Xóa cầu + khớp ngẫu nhiên"), ("burst", "Mất liên tục một đoạn")):
        rows.append(rf"\multicolumn{{7}}{{@{{}}l}}{{\textit{{{title}}}}}\\")
        for key, label, *_ in CURVES:
            if key not in deg:
                continue
            cells = " & ".join(vn(deg[key][f"{condition}@{r}"]["accuracy"]["mean"], 1) for r in rates)
            rows.append(f"{label} & {cells}\\\\")
        if condition == "both":
            rows.append(r"\midrule")
    (gen / "degradation_rows.tex").write_text("\n".join(rows) + "\n", encoding="utf-8")

    rows = []
    for c in CLASSES:
        pc = full["per_class"][c]
        rows.append(f"{CLASS_VI[c]} & {vn(pc['support'], 0)} & {vn(pc['precision']['mean'], 1)} & "
                    f"{vn(pc['recall']['mean'], 1)} & {vn(pc['f1']['mean'], 1)}\\\\")
    (gen / "per_class_rows.tex").write_text("\n".join(rows) + "\n", encoding="utf-8")

    rows = []
    for label, key in (("Precision phát hiện cú đánh", "hit_precision"), ("Recall phát hiện cú đánh", "hit_recall"),
                       ("F1 phát hiện cú đánh", "hit_f1"), ("Đúng người đánh (trên cú khớp)", "hitter_acc_matched"),
                       ("Joint Precision", "joint_precision"), ("Joint Recall", "joint_recall"),
                       ("\\textbf{Joint F1}", "joint_f1")):
        a, b = vn(t5[key], 1), vn(t12[key], 1)
        if key == "joint_f1":
            a, b = rf"\textbf{{{a}}}", rf"\textbf{{{b}}}"
        if key == "joint_precision":
            rows.append(r"\midrule")
        rows.append(f"{label} & {a} & {b}\\\\")
    (gen / "e2e_rows.tex").write_text("\n".join(rows) + "\n", encoding="utf-8")

    # ------------------------------------------------------------ hình
    plt.rcParams.update({"font.family": "Times New Roman", "mathtext.fontset": "stix",
                         "font.size": 8, "pdf.fonttype": 42, "axes.linewidth": 0.6})
    conditions = ("shuttle", "pose", "both", "player", "burst")
    fig, axes = plt.subplots(1, len(conditions), figsize=(7.0, 2.1), sharey=True)
    xs = [0, 10, 20, 30, 40, 50]
    titles = {"shuttle": "(a) Xóa cầu", "pose": "(b) Xóa khớp", "both": "(c) Xóa cầu + khớp",
              "player": "(d) Mất cả người", "burst": "(e) Mất liên tục"}
    for ax, condition in zip(axes, conditions):
        for key, label, color, ls, marker in CURVES:
            if key not in deg:
                continue
            mean = [deg[key][f"{condition}@{r}"]["accuracy"]["mean"] for r in rates]
            ax.plot(xs, mean, ls=ls, marker=marker, ms=3.2, lw=1.3, color=color, label=label)
        ax.set_title(titles[condition], fontsize=8)
        ax.set_xlabel("r (%)", fontsize=7.5)
        ax.set_xticks([0, 25, 50])
        ax.grid(alpha=0.3, lw=0.4)
        ax.tick_params(labelsize=7, width=0.5, length=2)
    axes[0].set_ylabel("Top-1 (%)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=len(labels), frameon=False, fontsize=7.5,
               bbox_to_anchor=(0.5, 1.02))
    fig.tight_layout(rect=(0, 0, 1, 0.88), w_pad=0.3)
    fig.savefig(figs / "degradation.pdf")
    fig.savefig(args.overleaf / "figures_src/degradation_preview.png", dpi=200)
    plt.close(fig)

    cm = np.asarray(full["confusion_sum_over_seeds"], dtype=float)
    cm = 100 * cm / cm.sum(1, keepdims=True)
    fig, ax = plt.subplots(figsize=(3.3, 2.95))
    ax.imshow(cm, cmap="Blues", vmin=0, vmax=100)
    for i in range(8):
        for j in range(8):
            if cm[i, j] >= 0.5:
                ax.text(j, i, f"{cm[i, j]:.0f}", ha="center", va="center", fontsize=6.5,
                        color="white" if cm[i, j] > 55 else "#222222")
    ax.set_xticks(range(8), SHORT, rotation=40, ha="right", fontsize=7)
    ax.set_yticks(range(8), SHORT, fontsize=7)
    ax.set_xlabel("Dự đoán", fontsize=7.5)
    ax.set_ylabel("Nhãn thật", fontsize=7.5)
    ax.tick_params(width=0.5, length=2)
    fig.tight_layout(pad=0.3)
    fig.savefig(figs / "confusion.pdf")
    fig.savefig(args.overleaf / "figures_src/confusion_preview.png", dpi=200)
    print("generated", sorted(p.name for p in gen.iterdir()))


if __name__ == "__main__":
    main()
