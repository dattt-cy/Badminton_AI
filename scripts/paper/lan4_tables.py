"""Tables and numbers for the revision of the 01/11 paper (R(2+1)D Robust KAP).

Steps (each reads what the previous one wrote):
  collect   copy work_dirs/paper_final runs and the extra seeds of work_dirs/paper_lan4
            into work_dirs/paper_lan4/final with their model_kwargs.json
  (then)    python scripts/paper/degradation_eval.py --runs work_dirs/paper_lan4/final \
                --output work_dirs/paper_lan4/final/degradation.json
  latex     write generated/ablation_rows.tex, real_error_rows.tex, e2e_rows.tex and
            lan4_numbers.tex into the paper directory

Example:
    python scripts/paper/lan4_tables.py collect
    python scripts/paper/lan4_tables.py latex --paper ../SuaBaoCao/lan4
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
W = REPO_ROOT / "work_dirs"
LAN4 = W / "paper_lan4"
FINAL = LAN4 / "final"
BASE = {"gate": "none", "pooling": "three", "use_cross": True, "use_hitter": True, "use_rgb": True}
KWARGS = {
    "main": BASE, "gate": {**BASE, "gate": "sigmoid"}, "no_phase": {**BASE, "pooling": "mean"},
    "no_cross": {**BASE, "use_cross": False}, "no_hitter": {**BASE, "use_hitter": False},
    "no_noise": BASE, "no_rgb": {**BASE, "use_rgb": False},
}
ROWS = [  # family, label, change, params
    ("gate", "+ cổng chất lượng", "Phiên bản trước: cổng sigmoid theo vector chỉ số mất dấu", "1,06M"),
    ("no_phase", "w/o 3 pha", "Gom trung bình toàn chuỗi thay cho 3 pha", "0,93M"),
    ("no_cross", "w/o cross-attn", "Bỏ nhánh khung xương--cầu và token $z_{\\text{cross}}$", "0,81M"),
    ("no_hitter", "w/o phân định", "Trộn hai người chơi với $\\alpha_1=\\alpha_2=0{,}5$", "1,03M"),
    ("no_noise", "w/o nhánh nhiễu", "Bỏ $\\mathcal{L}_{\\text{noisy}}$ và $\\mathcal{L}_{\\text{cons}}$", "1,06M"),
    ("no_rgb", "w/o RGB", "Bỏ token RGB (không cần R(2+1)D, 31,3M)", "0,99M"),
]


def vn(value: float, digits: int = 2) -> str:
    return f"{value:.{digits}f}".replace(".", ",")


def pm(values: list[float]) -> str:
    values = np.asarray(values) * 100
    if len(values) == 1:
        return vn(values[0])
    return f"\\pmstd{{{vn(values.mean())}}}{{{vn(values.std(ddof=1))}}}"


def collect() -> None:
    if FINAL.exists():
        shutil.rmtree(FINAL)
    sources = [p for p in (W / "paper_final").glob("*_seed*") if p.is_dir()]
    sources += [p for p in LAN4.glob("*_seed*") if p.is_dir() and (p / "test_predictions.npz").exists()]
    for run in sorted(sources):
        family, seed = run.name.rsplit("_seed", 1)
        out = FINAL / run.name
        if out.exists():   # the original run wins over a re-run of the same seed
            print(f"skip re-run {run}")
            continue
        out.mkdir(parents=True)
        for name in ("kap_best.pth", "test_predictions.npz", "metrics.json"):
            shutil.copy2(run / name, out / name)
        (out / "model_kwargs.json").write_text(json.dumps(KWARGS[family]) + "\n", encoding="utf-8")
        print(f"{out.name:24s} <- {run}")


def latex(paper: Path) -> None:
    deg = json.loads((FINAL / "degradation.json").read_text(encoding="utf-8"))
    gen = paper / "generated"

    def family(name: str) -> list[dict]:
        return [table for run, table in deg["kap"].items() if run.rsplit("_seed", 1)[0] == name]

    def cells(tables: list[dict]) -> str:
        return " & ".join(pm([t[key][metric] for t in tables])
                          for key in ("shuttle@0.0", "both@0.3") for metric in ("accuracy", "macro_f1"))

    main = family("main")
    lines = [f"\\textbf{{Robust KAP}} ({len(main)}) & -- & 1,06M & {cells(main)}\\\\", "\\midrule"]
    seeds = {}
    for name, label, change, params in ROWS:
        tables = family(name)
        seeds[name] = len(tables)
        lines.append(f"{label} ({len(tables)}) & {change} & {params} & {cells(tables)}\\\\")
    bst = list(deg["bst"].values())
    lines += ["\\midrule", f"BST-CG-AP~\\cite{{chang2025bst}} & 3 checkpoint chính thức & 1,87M & {cells(bst)}\\\\"]
    (gen / "ablation_rows.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")

    def mean(name: str, key: str, metric: str = "accuracy") -> float:
        tables = bst if name == "bst" else family(name)
        return float(np.mean([t[key][metric] for t in tables]) * 100)

    numbers = {
        "ablMinSeeds": str(min(seeds.values())),
        "noRgbAcc": vn(mean("no_rgb", "shuttle@0.0")), "noRgbFone": vn(mean("no_rgb", "shuttle@0.0", "macro_f1")),
        "noRgbBothFifty": vn(mean("no_rgb", "both@0.5")), "noRgbPoseFifty": vn(mean("no_rgb", "pose@0.5")),
        "noRgbBurstFifty": vn(mean("no_rgb", "burst@0.5")),
        "noNoiseShuttleAll": vn(mean("no_noise", "shuttle@1.0"), 1),
        "noPhaseAcc": vn(mean("no_phase", "shuttle@0.0")), "noPhaseFone": vn(mean("no_phase", "shuttle@0.0", "macro_f1")),
        "noCrossAcc": vn(mean("no_cross", "shuttle@0.0")), "noCrossFone": vn(mean("no_cross", "shuttle@0.0", "macro_f1")),
        "noHitterAcc": vn(mean("no_hitter", "shuttle@0.0")), "noHitterFone": vn(mean("no_hitter", "shuttle@0.0", "macro_f1")),
        "noPhaseThirty": vn(mean("no_phase", "both@0.3", "macro_f1")),
        "noCrossThirty": vn(mean("no_cross", "both@0.3", "macro_f1")),
        "noHitterThirty": vn(mean("no_hitter", "both@0.3", "macro_f1")),
        "kapThirtyFone": vn(mean("main", "both@0.3", "macro_f1")),
    }

    curves = [("main", "Robust KAP", "#1f4e9c", "-", "o"), ("gate", "KAP + cổng CL", "#6a9fd8", "--", "s"),
              ("no_noise", "KAP w/o nhánh nhiễu", "#2e8b57", "-.", "^"), ("no_rgb", "KAP w/o RGB", "#d98c1f", ":", "D"),
              ("bst", "BST-CG-AP", "#c0392b", "-", "x")]
    rates = ("0.0", "0.1", "0.2", "0.3", "0.4", "0.5")
    lines = []
    for condition, title in (("both", "Xóa cầu + khớp ngẫu nhiên"), ("burst", "Mất liên tục một đoạn")):
        if lines:
            lines.append("\\midrule")
        lines.append(f"\\multicolumn{{7}}{{@{{}}l}}{{\\textit{{{title}}}}}\\\\")
        for key, label, *_ in curves:
            key_rate = [("shuttle@0.0" if r == "0.0" else f"{condition}@{r}") for r in rates]
            lines.append(label + "".join(f" & {vn(mean(key, k), 1)}" for k in key_rate) + "\\\\")
    (gen / "degradation_rows.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "Times New Roman", "mathtext.fontset": "stix", "font.size": 8,
                         "pdf.fonttype": 42, "axes.linewidth": 0.6})
    titles = {"shuttle": "(a) Xóa cầu", "pose": "(b) Xóa khớp", "both": "(c) Xóa cầu + khớp",
              "player": "(d) Mất cả người", "burst": "(e) Mất liên tục"}
    fig, axes = plt.subplots(1, len(titles), figsize=(7.0, 2.1), sharey=True)
    for ax, (condition, title) in zip(axes, titles.items()):
        for key, label, color, ls, marker in curves:
            values = [mean(key, "shuttle@0.0" if r == "0.0" else f"{condition}@{r}") for r in rates]
            ax.plot([0, 10, 20, 30, 40, 50], values, ls=ls, marker=marker, ms=3.2, lw=1.3, color=color, label=label)
        ax.set_title(title, fontsize=8)
        ax.set_xlabel("r (%)", fontsize=7.5)
        ax.set_xticks([0, 25, 50])
        ax.grid(alpha=0.3, lw=0.4)
        ax.tick_params(labelsize=7, width=0.5, length=2)
    axes[0].set_ylabel("Top-1 (%)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=len(labels), frameon=False, fontsize=7.5, bbox_to_anchor=(0.5, 1.02))
    fig.tight_layout(rect=(0, 0, 1, 0.88), w_pad=0.3)
    fig.savefig(paper / "figures/degradation.pdf")
    fig.savefig(LAN4 / "degradation_preview.png", dpi=200)
    plt.close(fig)

    natural = json.loads((LAN4 / "natural_errors.json").read_text(encoding="utf-8"))
    pipeline = json.loads((LAN4 / "pipeline_errors.json").read_text(encoding="utf-8"))
    real = json.loads((LAN4 / "realistic_degradation.json").read_text(encoding="utf-8"))
    ne, pe = natural["per_event"], pipeline["per_event"]
    face = np.asarray(real["profile"]["joint_rate"])
    rows = [
        f"Khung mất cầu & {vn(natural['frames']['shuttle_lost'] * 100, 1)} & {vn(pipeline['frames']['shuttle'] * 100, 1)}\\\\",
        f"Sự kiện có mất cầu & {vn(ne['shuttle_lost']['share_nonzero'] * 100, 0)} & {vn(pe['shuttle_lost']['share_nonzero'] * 100, 0)}\\\\",
        f"Tỷ lệ mất cầu mỗi sự kiện, P90 & {vn(ne['shuttle_lost']['p90'] * 100, 0)} & {vn(pe['shuttle_lost']['p90'] * 100, 0)}\\\\",
        f"Khoảng mất cầu dài $\\le$6 frame & {vn(natural['gap_lengths']['share_le_6'] * 100, 0)} & {vn(pipeline['gap_lengths']['share_le_6'] * 100, 0)}\\\\",
        f"Khung--người mất cả khung xương & {vn(ne['player_lost']['mean'] * 100, 1)} & {vn(pe['player_lost']['mean'] * 100, 1)}\\\\",
        f"Khớp lẻ bị mất (người vẫn có) & {vn(ne['joint_lost']['mean'] * 100, 1)} & {vn(pe['joint_lost']['mean'] * 100, 1)}\\\\",
    ]
    (gen / "real_error_rows.tex").write_text("\n".join(rows) + "\n", encoding="utf-8")

    cond = real["conditions"]
    labels = {"clean": "Sạch", "pipeline": "Lỗi theo pipeline", "pipeline_x2": "Lỗi gấp đôi pipeline"}
    lines = []
    for key, label in labels.items():
        c = cond[key]
        lo, hi = c["main-bst_ci95"]
        lines.append(f"{label} & {vn(c['main'])} & {vn(c['no_rgb'])} & {vn(c['bst'])} & "
                     f"{vn(c['main-bst'])} [{vn(lo)}; {vn(hi)}]\\\\")
    (gen / "real_degradation_rows.tex").write_text("\n".join(lines).replace("-", "$-$") + "\n", encoding="utf-8")

    groups = natural["groups"]
    lines = []
    for key, label in (("shuttle_lost>0", "Có mất cầu"), ("shuttle_lost>=p75(0.20)", "Mất cầu $\\ge$20\\%"),
                       ("player_lost>0", "Có mất cả người"), ("player_lost>=p75(0.27)", "Mất người $\\ge$27\\%")):
        g = groups[key]
        lo, hi = g["ci95"]
        lines.append(f"{label} & {g['events']} & {vn(g['kap'])} & {vn(g['bst'])} & "
                     f"{vn(g['diff'])} [{vn(lo)}; {vn(hi)}]\\\\".replace("[-", "[$-$").replace("& -", "& $-$").replace("; -", "; $-$"))
    (gen / "natural_group_rows.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")
    numbers.update({
        "realCleanKap": vn(cond["clean"]["main"]), "realPipeKap": vn(cond["pipeline"]["main"]),
        "realPipeBst": vn(cond["pipeline"]["bst"]), "realPipeNoRgb": vn(cond["pipeline"]["no_rgb"]),
        "realTwoKap": vn(cond["pipeline_x2"]["main"]), "realTwoBst": vn(cond["pipeline_x2"]["bst"]),
        "realTwoNoRgb": vn(cond["pipeline_x2"]["no_rgb"]),
        "realPipeKapDrop": vn(cond["clean"]["main"] - cond["pipeline"]["main"]),
        "realPipeBstDrop": vn(cond["clean"]["bst"] - cond["pipeline"]["bst"]),
        "faceEyeRate": vn(face[1:3].mean() * 100, 0), "faceNoseRate": vn(face[0] * 100, 0),
        "bodyJointMax": vn(face[5:].max() * 100, 1),
        "pipeHits": str(pipeline["hits"]),
        "pipeGapMedian": vn(pipeline["gap_lengths"]["median"], 0), "nsGapMedian": vn(natural["gap_lengths"]["median"], 0),
    })

    ci = json.loads((LAN4 / "e2e_ci.json").read_text(encoding="utf-8"))
    for tol, tag in ((5, "Five"), (12, "Twelve")):
        c = ci[f"test@{tol}"]
        numbers[f"eteJointCi{tag}"] = f"{vn(c['joint_f1_ci_stroke'][0], 1)}; {vn(c['joint_f1_ci_stroke'][1], 1)}"
        numbers[f"eteJointCiMin{tag}"] = f"{vn(c['joint_f1_ci_minute'][0], 1)}; {vn(c['joint_f1_ci_minute'][1], 1)}"
        numbers[f"eteHitCi{tag}"] = f"{vn(c['hit_f1_ci_stroke'][0], 1)}; {vn(c['hit_f1_ci_stroke'][1], 1)}"
    numbers["eteMinutesBlocks"] = str(ci["test@5"]["minutes"])

    cost = LAN4 / "cost.json"
    if cost.exists():
        c = json.loads(cost.read_text(encoding="utf-8"))
        numbers.update({k: vn(v, 1) for k, v in c.items() if isinstance(v, float)})

    text = "% Sinh tự động bởi AI_Classifier/scripts/paper/lan4_tables.py -- không sửa tay.\n"
    # \def so that values recomputed here replace the same names in numbers.tex.
    text += "".join(f"\\def\\{k}{{{v}}}\n" for k, v in numbers.items())
    (gen / "lan4_numbers.tex").write_text(text, encoding="utf-8")
    print(json.dumps(numbers, indent=1, ensure_ascii=False))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("step", choices=("collect", "latex"))
    parser.add_argument("--paper", type=Path, default=REPO_ROOT.parent / "SuaBaoCao/lan4")
    args = parser.parse_args()
    collect() if args.step == "collect" else latex(args.paper)


if __name__ == "__main__":
    main()
