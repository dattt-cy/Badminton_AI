"""Sinh số liệu, dòng bảng và hình cho bài báo từ work_dirs/paper_mate/simple_final.

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
    ("main", r"\textbf{Robust KAP}", "Mô hình chính"),
    (None, r"\textit{Bỏ một nhánh}", ""),
    ("no_rgb", "w/o RGB", "Không dùng MATE"),
    ("no_pose", "w/o pose", "Bỏ khung xương"),
    ("no_shuttle", "w/o cầu", "Bỏ quỹ đạo cầu"),
    ("no_pose_shuttle", "w/o pose + cầu", "Chỉ còn RGB và vị trí sân"),
    (None, r"\textit{Mã hóa và huấn luyện}", ""),
    ("no_contact", "w/o mã hóa tiếp xúc", "Đặt $c_t=0$"),
    ("no_noise", "w/o nhánh nhiễu", r"Bỏ $\mathcal{L}_{\text{noisy}}$ và $\mathcal{L}_{\text{cons}}$"),
    ("old_recipe", "Chỉ nhiễu nhẹ", "Không làm hỏng từng nhánh"),
    (None, r"\textit{Bản đầy đủ và biến thể của nó}", ""),
    ("full", "Bản đầy đủ", "Thêm gom 3 pha, cross-attention, phân định"),
    ("full_no_phase", "Đầy đủ w/o 3 pha", "Gom trung bình cho CATE"),
    ("full_no_cross", "Đầy đủ w/o cross-attn", r"Bỏ token $z_{\text{cross}}$"),
    ("full_no_hitter", "Đầy đủ w/o phân định", r"$\alpha_1=\alpha_2=0{,}5$"),
    ("full_no_rgb", "Đầy đủ w/o RGB", "Không dùng MATE"),
    ("full_gate", "Đầy đủ + cổng", "Cổng chất lượng sigmoid"),
    ("full_one_token", "Đầy đủ, RGB 1 token", "MATE gộp một vector"),
]
CURVES = [
    ("kap_main", "Robust KAP", "#2b7bba", "-", "o"),
    ("kap_old_recipe", "KAP chỉ nhiễu nhẹ", "#8e6cc7", "--", "s"),
    ("kap_no_noise", "KAP w/o nhánh nhiễu", "#3a9e4a", "-.", "^"),
    ("kap_no_rgb", "KAP w/o RGB", "#7f7f7f", ":", "D"),
    ("bst_cg_ap", "BST-CG-AP", "#d0453b", "-", "v"),
]
RGB_ROWS = [("clean@0.00", "RGB nguyên vẹn"), ("shift@0.30", "Lệch vùng cắt 30\\%"), ("shift@0.50", "Lệch vùng cắt 50\\%"), ("blur@3.00", "Mờ nặng ($\\sigma=3$)"),
            ("occlude@0.50", "Che 50\\%"), ("freeze@0.50", "Đóng băng 50\\% frame"), ("black@1.00", "Ảnh đen"), ("shuffle@1.00", "Ảnh của sự kiện khác")]
RGB_COLS = ["no_all3", "simple_no_pose", "simple_no_shuttle", "simple_no_pose_shuttle", "simple_no_noise", "branch_drop"]


def vn(value: float, digits: int = 2) -> str:
    text = f"{value:,.{digits}f}"
    return text.replace(",", "X").replace(".", ",").replace("X", ".")


def signed(value: float, digits: int = 2) -> str:
    return ("+" if value >= 0 else "$-$") + vn(abs(value), digits)


def pm(stat: dict, digits: int = 2) -> str:
    if stat.get("n", 2) < 2:
        return vn(stat["mean"], digits)
    return rf"\pmstd{{{vn(stat['mean'], digits)}}}{{{vn(stat['std'], digits)}}}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--overleaf", type=Path, default=REPO_ROOT.parent / "SuaBaoCao/overleaf")
    parser.add_argument("--runs", type=Path, default=REPO_ROOT / "work_dirs/paper_mate/simple_final")
    parser.add_argument("--rgb", type=Path, default=REPO_ROOT / "work_dirs/paper_mate/branch_drop/rgb_degradation_simple.json")
    parser.add_argument("--e2e", type=Path, default=REPO_ROOT / "work_dirs/paper_rgb_phases/finetune/ownrgb_phases_ss22x13_base1/e2e/worker_8clips/joint_f1.json")
    args = parser.parse_args()
    s = json.loads((args.runs / "summary.json").read_text(encoding="utf-8"))
    x = json.loads((args.runs / "extra.json").read_text(encoding="utf-8"))
    rgb = json.loads(args.rgb.read_text(encoding="utf-8"))["conditions"]
    e2e = json.loads(args.e2e.read_text(encoding="utf-8"))
    gen, figs = args.overleaf / "generated", args.overleaf / "figures"
    v, deg, counts = s["variants"], s["degradation"], s["class_counts"]
    full, gaps, late = v["main"], x["gaps"], x["late_fusion"]

    pairs = [t for k, t in s["significance"].items() if k.startswith("kap__main_seed")]
    d_acc = [t["bootstrap"]["delta_acc"]["mean"] for t in pairs]
    d_f1 = [t["bootstrap"]["delta_macro_f1"]["mean"] for t in pairs]
    trunc = s["window_truncation"]
    t5, t12 = e2e["test@5"], e2e["test@12"]

    def acc(key: str, cond: str, digits: int = 2) -> str:
        return vn(deg[key][cond]["accuracy"]["mean"], digits)

    def rgb_mean(cond: str, family: str, field: str = "per_seed") -> float:
        return float(np.mean(rgb[cond]["families"][family][field]))

    def gap_text(family: str, cond: str) -> str:
        g = gaps[family][cond]
        return rf"{signed(g['gap'])} [{signed(g['low'])}; {signed(g['high'])}]"

    macros = {
        "kapParams": vn(full["params"], 0), "kapParamsM": vn(full["params"] / 1e6, 2) + "M", "kapSeeds": str(len(full["seeds"])),
        "fullParamsM": vn(v["full"]["params"] / 1e6, 2) + "M", "totalParamsM": vn((full["params"] + 965981) / 1e6, 2) + "M",
        "kapAcc": vn(full["accuracy"]["mean"]), "kapAccStd": vn(full["accuracy"]["std"]),
        "kapFone": vn(full["macro_f1"]["mean"]), "kapFoneStd": vn(full["macro_f1"]["std"]),
        "kapBal": vn(full["balanced_accuracy"]["mean"]), "kapBalStd": vn(full["balanced_accuracy"]["std"]),
        "kapSide": vn(full["side_accuracy"]["mean"]),
        "bstAcc": vn(s["bst_clean"]["accuracy"]["mean"]), "bstAccStd": vn(s["bst_clean"]["accuracy"]["std"]),
        "bstFone": vn(s["bst_clean"]["macro_f1"]["mean"]), "bstFoneStd": vn(s["bst_clean"]["macro_f1"]["std"]),
        "bstBal": vn(s["bst_clean"]["balanced_accuracy"]["mean"]), "bstBalStd": vn(s["bst_clean"]["balanced_accuracy"]["std"]),
        "gapAcc": vn(s["bst_clean"]["accuracy"]["mean"] - full["accuracy"]["mean"], 2),
        "sigPairs": str(len(pairs)), "sigDaccMin": signed(min(d_acc)), "sigDaccMax": signed(max(d_acc)),
        "sigDfoneMin": signed(min(d_f1)), "sigDfoneMax": signed(max(d_f1)),
        "sigMcSig": str(sum(t["mcnemar"]["p_value"] < 0.05 for t in pairs)),
        "sigFoneSig": str(sum(t["bootstrap"]["delta_macro_f1"]["ci95"][1] < 0 or t["bootstrap"]["delta_macro_f1"]["ci95"][0] > 0 for t in pairs)),
        "sigOnlyKap": vn(float(np.mean([t["mcnemar"]["only_a_correct"] for t in pairs])), 0),
        "sigOnlyBst": vn(float(np.mean([t["mcnemar"]["only_b_correct"] for t in pairs])), 0),
        "kapBothFifty": acc("kap_main", "both@0.5"), "bstBothFifty": acc("bst_cg_ap", "both@0.5"),
        "kapBurstFifty": acc("kap_main", "burst@0.5"), "bstBurstFifty": acc("bst_cg_ap", "burst@0.5"),
        "kapPoseFifty": acc("kap_main", "pose@0.5"), "bstPoseFifty": acc("bst_cg_ap", "pose@0.5"),
        "kapShuttleFifty": acc("kap_main", "shuttle@0.5"), "bstShuttleFifty": acc("bst_cg_ap", "shuttle@0.5"),
        "kapPlayerFifty": acc("kap_main", "player@0.5"), "bstPlayerFifty": acc("bst_cg_ap", "player@0.5"),
        "kapShuttleAll": acc("kap_main", "shuttle@1.0"), "bstShuttleAll": acc("bst_cg_ap", "shuttle@1.0"),
        "kapPoseAll": acc("kap_main", "pose@1.0"), "bstPoseAll": acc("bst_cg_ap", "pose@1.0"),
        "kapBothAll": acc("kap_main", "both@1.0"), "bstBothAll": acc("bst_cg_ap", "both@1.0"),
        "noRgbAcc": vn(v["no_rgb"]["accuracy"]["mean"]), "noRgbGap": gap_text("no_rgb", "both@0.0"),
        "noRgbBothFifty": acc("kap_no_rgb", "both@0.5"), "noRgbBothGap": gap_text("no_rgb", "both@0.5"), "noRgbBothAll": acc("kap_no_rgb", "both@1.0"),
        "noPoseAcc": vn(v["no_pose"]["accuracy"]["mean"]), "noPoseGap": gap_text("no_pose", "both@0.0"),
        "noShuttleAcc": vn(v["no_shuttle"]["accuracy"]["mean"]), "noShuttleGap": gap_text("no_shuttle", "both@0.0"),
        "noBothAcc": vn(v["no_pose_shuttle"]["accuracy"]["mean"]), "noBothGap": gap_text("no_pose_shuttle", "both@0.0"),
        "noContactAcc": vn(v["no_contact"]["accuracy"]["mean"]), "noContactGap": gap_text("no_contact", "both@0.0"),
        "noNoiseAcc": vn(v["no_noise"]["accuracy"]["mean"]), "noNoiseGap": gap_text("no_noise", "both@0.0"),
        "noNoiseBothFifty": acc("kap_no_noise", "both@0.5"), "noNoiseBothGap": gap_text("no_noise", "both@0.5"),
        "oldBothFifty": acc("kap_old_recipe", "both@0.5"), "oldBothGap": gap_text("old_recipe", "both@0.5"),
        "fullAcc": vn(v["full"]["accuracy"]["mean"]), "fullFone": vn(v["full"]["macro_f1"]["mean"]), "fullGap": gap_text("full", "both@0.0"),
        "fullBothFifty": acc("kap_full", "both@0.5"), "fullShuttleAll": acc("kap_full", "shuttle@1.0"), "fullNoRgbAcc": vn(v["full_no_rgb"]["accuracy"]["mean"]),
        "fullShuffle": vn(rgb_mean("shuffle@1.00", "branch_drop")), "mainShuffle": vn(rgb_mean("shuffle@1.00", "no_all3")),
        "lateClean": vn(late["both@0.0"]), "lateBothFifty": vn(late["both@0.5"]), "lateBurstFifty": vn(late["burst@0.5"]), "latePoseFifty": vn(late["pose@0.5"]),
        "lateBothAll": vn(late["both@1.0"]), "lateShiftFifty": vn(rgb["shift@0.50"]["late_fusion"]), "lateOccFifty": vn(rgb["occlude@0.50"]["late_fusion"]),
        "lateShuffle": vn(rgb["shuffle@1.00"]["late_fusion"]), "mateAlone": vn(rgb["clean@0.00"]["mate_alone"]),
        "kapShiftFifty": vn(rgb_mean("shift@0.50", "no_all3")), "kapOccFifty": vn(rgb_mean("occlude@0.50", "no_all3")), "kapBlack": vn(rgb_mean("black@1.00", "no_all3")),
        "noNoiseOccFifty": vn(rgb_mean("occlude@0.50", "simple_no_noise")), "noNoiseShuffle": vn(rgb_mean("shuffle@1.00", "simple_no_noise")),
        "noBothShuffle": vn(rgb_mean("shuffle@1.00", "simple_no_pose_shuttle")), "noPoseShuffle": vn(rgb_mean("shuffle@1.00", "simple_no_pose")),
        "tripleBlack": vn(rgb_mean("black@1.00", "no_all3", "with_pose_shuttle_50")), "tripleShuffle": vn(rgb_mean("shuffle@1.00", "no_all3", "with_pose_shuttle_50")),
        "tripleShuffleNoPose": vn(rgb_mean("shuffle@1.00", "simple_no_pose", "with_pose_shuttle_50")),
        "tripleBlur": vn(rgb_mean("blur@3.00", "no_all3", "with_pose_shuttle_50")),
        "msMate": vn(x["ms_per_event"]["mate"], 1), "msKap": vn(x["ms_per_event"]["kap"], 1),
        "truncShare": vn(trunc["truncated_share"], 1), "truncFullN": vn(trunc["full"]["n"], 0),
        "truncFullGap": signed(trunc["full"]["bst"] - trunc["full"]["kap"], 2), "truncCutGap": signed(trunc["truncated"]["bst"] - trunc["truncated"]["kap"], 2),
        "splitTrainMatches": ", ".join(map(str, s["matches"]["train"])), "splitTrainN": vn(sum(counts["train"].values()), 0),
        "splitValN": vn(sum(counts["val"].values()), 0), "splitTestN": vn(sum(counts["test"].values()), 0),
        "imbalance": vn(counts["test"]["net_shot"] / counts["test"]["drive"], 1), "bootRounds": vn(2000, 0),
        "eteGt": vn(t5["gt"], 0), "eteHitFFive": vn(t5["hit_f1"], 1), "eteJointFive": vn(t5["joint_f1"], 1), "eteHitFTwelve": vn(t12["hit_f1"], 1),
        "eteJointTwelve": vn(t12["joint_f1"], 1), "eteHitter": vn(t5["hitter_acc_matched"], 1),
        "eteSpeed": vn(e2e["test_video_seconds"] / e2e["test_processing_seconds"], 1), "eteMinutes": vn(e2e["test_processing_seconds"] / 60, 0),
    }
    (gen / "numbers.tex").write_text("% Sinh tự động bởi AI_Classifier/scripts/paper/make_latex.py -- không sửa tay.\n"
                                     + "".join(rf"\newcommand{{\{k}}}{{{val}}}" + "\n" for k, val in macros.items()), encoding="utf-8")

    rows = [f"{CLASS_VI[c]} & {vn(counts['train'][c], 0)} & {vn(counts['val'][c], 0)} & {vn(counts['test'][c], 0)}\\\\" for c in CLASSES]
    rows += [r"\midrule", r"\textbf{Tổng} & \textbf{" + " & \\textbf{".join(vn(sum(counts[sp].values()), 0) + "}" for sp in ("train", "val", "test")) + r"\\"]
    (gen / "split_rows.tex").write_text("\n".join(rows) + "\n", encoding="utf-8")

    rows = []
    for match in sorted(full["per_match"], key=int):
        m = full["per_match"][match]
        rows.append(f"Match {match} & {vn(m['n'], 0)} & {pm(m['accuracy'])} & {pm(m['macro_f1'])} & {pm(m['side_accuracy'])}\\\\")
    ams = full["across_match_std"]
    rows += [r"\midrule", rf"\textbf{{Toàn bộ test}} & \textbf{{3.334}} & {pm(full['accuracy'])} & {pm(full['macro_f1'])} & {pm(full['side_accuracy'])}\\",
             rf"Lệch giữa các trận & -- & $\pm${vn(ams['accuracy'])} & $\pm${vn(ams['macro_f1'])} & $\pm${vn(ams['side_accuracy'])}\\"]
    (gen / "per_match_rows.tex").write_text("\n".join(rows) + "\n", encoding="utf-8")

    rows = []
    for key, name, change in ABLATION:
        if key is None:
            rows += [r"\midrule", rf"\multicolumn{{7}}{{@{{}}l}}{{{name}}}\\"]
            continue
        r, d = v[key], deg[f"kap_{key}"]
        rows.append(f"{name} ({len(r['seeds'])}) & {change} & {vn(r['params'] / 1e6, 2)}M & {pm(r['accuracy'])} & {pm(r['macro_f1'])} & {pm(d['both@0.5']['accuracy'])} & {pm(d['shuttle@1.0']['accuracy'])}\\\\")
    bst = deg["bst_cg_ap"]
    rows += [r"\midrule", rf"BST-CG-AP~\cite{{chang2025bst}} & 3 checkpoint chính thức & 1,87M & {pm(s['bst_clean']['accuracy'])} & {pm(s['bst_clean']['macro_f1'])} & {pm(bst['both@0.5']['accuracy'])} & {pm(bst['shuttle@1.0']['accuracy'])}\\"]
    (gen / "ablation_rows.tex").write_text("\n".join(rows) + "\n", encoding="utf-8")

    rates = ["0.0", "0.1", "0.2", "0.3", "0.4", "0.5"]
    rows = []
    for condition, title in (("both", "Xóa cầu + khớp ngẫu nhiên"), ("burst", "Mất liên tục một đoạn")):
        rows.append(rf"\multicolumn{{7}}{{@{{}}l}}{{\textit{{{title}}}}}\\")
        for key, label, *_ in CURVES:
            rows.append(f"{label} & " + " & ".join(vn(deg[key][f"{condition}@{r}"]["accuracy"]["mean"], 1) for r in rates) + "\\\\")
        for key, label in (("kap_full", "Bản đầy đủ"), ("kap_full_gate", "Đầy đủ + cổng")):
            rows.append(f"{label} & " + " & ".join(vn(deg[key][f"{condition}@{r}"]["accuracy"]["mean"], 1) for r in rates) + "\\\\")
        if condition == "both":
            rows.append(r"\midrule")
    (gen / "degradation_rows.tex").write_text("\n".join(rows) + "\n", encoding="utf-8")

    rows = []
    for cond, label in RGB_ROWS:
        cells = [vn(rgb_mean(cond, f), 1) for f in RGB_COLS]
        rows.append(f"{label} & " + " & ".join(cells) + f" & {vn(rgb[cond]['late_fusion'], 1)} & {vn(rgb[cond]['mate_alone'], 1)}\\\\")
    (gen / "rgb_rows.tex").write_text("\n".join(rows) + "\n", encoding="utf-8")

    rows = [f"{CLASS_VI[c]} & {vn(full['per_class'][c]['support'], 0)} & {vn(full['per_class'][c]['precision']['mean'], 1)} & "
            f"{vn(full['per_class'][c]['recall']['mean'], 1)} & {vn(full['per_class'][c]['f1']['mean'], 1)}\\\\" for c in CLASSES]
    (gen / "per_class_rows.tex").write_text("\n".join(rows) + "\n", encoding="utf-8")

    rows = []
    for label, key in (("Precision phát hiện cú đánh", "hit_precision"), ("Recall phát hiện cú đánh", "hit_recall"), ("F1 phát hiện cú đánh", "hit_f1"),
                       ("Đúng người đánh (trên cú khớp)", "hitter_acc_matched"), ("Joint Precision", "joint_precision"), ("Joint Recall", "joint_recall"), ("\\textbf{Joint F1}", "joint_f1")):
        a, b = vn(t5[key], 1), vn(t12[key], 1)
        if key == "joint_f1":
            a, b = rf"\textbf{{{a}}}", rf"\textbf{{{b}}}"
        if key == "joint_precision":
            rows.append(r"\midrule")
        rows.append(f"{label} & {a} & {b}\\\\")
    (gen / "e2e_rows.tex").write_text("\n".join(rows) + "\n", encoding="utf-8")

    plt.rcParams.update({"font.family": "Times New Roman", "mathtext.fontset": "stix", "font.size": 8, "pdf.fonttype": 42, "axes.linewidth": 0.6})
    conditions = ("shuttle", "pose", "both", "player", "burst")
    titles = {"shuttle": "(a) Xóa cầu", "pose": "(b) Xóa khớp", "both": "(c) Xóa cầu + khớp", "player": "(d) Mất cả người", "burst": "(e) Mất liên tục"}
    fig, axes = plt.subplots(1, len(conditions), figsize=(7.0, 2.1), sharey=True)
    for ax, condition in zip(axes, conditions):
        for key, label, color, ls, marker in CURVES:
            ax.plot([0, 10, 20, 30, 40, 50], [deg[key][f"{condition}@{r}"]["accuracy"]["mean"] for r in rates], ls=ls, marker=marker, ms=3.2, lw=1.3, color=color, label=label)
        ax.set_title(titles[condition], fontsize=8)
        ax.set_xlabel("r (%)", fontsize=7.5)
        ax.set_xticks([0, 25, 50])
        ax.grid(alpha=0.3, lw=0.4)
        ax.tick_params(labelsize=7, width=0.5, length=2)
    axes[0].set_ylabel("Top-1 (%)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=len(labels), frameon=False, fontsize=7.5, bbox_to_anchor=(0.5, 1.02))
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
                ax.text(j, i, f"{cm[i, j]:.0f}", ha="center", va="center", fontsize=6.5, color="white" if cm[i, j] > 55 else "#222222")
    ax.set_xticks(range(8), SHORT, rotation=40, ha="right", fontsize=7)
    ax.set_yticks(range(8), SHORT, fontsize=7)
    ax.set_xlabel("Dự đoán", fontsize=7.5)
    ax.set_ylabel("Nhãn thật", fontsize=7.5)
    ax.tick_params(width=0.5, length=2)
    fig.tight_layout(pad=0.3)
    fig.savefig(figs / "confusion.pdf")
    fig.savefig(args.overleaf / "figures_src/confusion_preview.png", dpi=200)
    print("generated", sorted(p.name for p in gen.iterdir()))
    print(json.dumps({k: macros[k] for k in macros if k not in ("splitTrainMatches",)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
