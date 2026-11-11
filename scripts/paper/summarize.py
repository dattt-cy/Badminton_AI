"""Aggregate paper ablation runs, degradation results and paired tests.

Reads work_dirs/paper_ablation/{variant}_seed{n}/ and the outputs of
degradation_eval.py, then writes work_dirs/paper_ablation/summary.json.

Example:
    python scripts/paper/summarize.py
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from math import comb
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
CLASSES = ["serve", "clear", "smash", "drop", "net_shot", "lift", "drive", "net_attack"]


def macro_f1(truth: np.ndarray, prediction: np.ndarray, classes: int = 8) -> float:
    cm = np.zeros((classes, classes), dtype=np.int64)
    np.add.at(cm, (truth, prediction), 1)
    tp = np.diag(cm).astype(float)
    p = np.divide(tp, cm.sum(0), out=np.zeros(classes), where=cm.sum(0) != 0)
    r = np.divide(tp, cm.sum(1), out=np.zeros(classes), where=cm.sum(1) != 0)
    return float(np.divide(2 * p * r, p + r, out=np.zeros(classes), where=(p + r) != 0).mean())


def mcnemar_exact(a_correct: np.ndarray, b_correct: np.ndarray) -> dict:
    b = int((a_correct & ~b_correct).sum())
    c = int((~a_correct & b_correct).sum())
    n, k = b + c, min(b, c)
    p = min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n) if n else 1.0
    return {"only_a_correct": b, "only_b_correct": c, "p_value": p}


def paired_bootstrap(truth, pred_a, pred_b, rounds: int, rng) -> dict:
    n = len(truth)
    d_acc, d_f1 = [], []
    for _ in range(rounds):
        idx = rng.integers(0, n, n)
        t = truth[idx]
        d_acc.append((pred_a[idx] == t).mean() - (pred_b[idx] == t).mean())
        d_f1.append(macro_f1(t, pred_a[idx]) - macro_f1(t, pred_b[idx]))
    out = {}
    for name, values in (("delta_acc", d_acc), ("delta_macro_f1", d_f1)):
        values = np.asarray(values) * 100
        out[name] = {
            "mean": float(values.mean()),
            "ci95": [float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))],
            "p_two_sided": float(min(1.0, 2 * min((values <= 0).mean(), (values >= 0).mean()))),
        }
    return out


def mean_std(values) -> dict:
    values = np.asarray(values, dtype=float) * 100
    return {"mean": float(values.mean()), "std": float(values.std(ddof=1)) if len(values) > 1 else 0.0, "n": len(values)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, default=REPO_ROOT / "work_dirs/paper_final")
    parser.add_argument("--features", type=Path, default=REPO_ROOT / "work_dirs/context_window_pilot/features_w15_30.npz")
    parser.add_argument("--bootstrap", type=int, default=2000)
    args = parser.parse_args()
    summary: dict = {}

    raw = np.load(args.features)
    summary["class_counts"] = {
        split: dict(zip(CLASSES, np.bincount(raw["stroke"][raw["split"] == split], minlength=8).tolist()))
        for split in ("train", "val", "test")
    }
    summary["matches"] = {
        split: sorted({int(s.split("_")[0]) for s in raw["sample_id"][raw["split"] == split]})
        for split in ("train", "val", "test")
    }

    runs = defaultdict(list)
    for path in sorted(args.runs.glob("*_seed*/metrics.json")):
        runs[path.parent.name.rsplit("_seed", 1)[0]].append(path.parent)

    variants = {}
    for variant, dirs in runs.items():
        metrics = [json.loads((d / "metrics.json").read_text()) for d in dirs]
        variants[variant] = {
            "seeds": [m["seed"] for m in metrics], "params": metrics[0]["params"],
            "best_epochs": [m["best_epoch"] for m in metrics],
            "accuracy": mean_std([m["test_stroke"]["accuracy"] for m in metrics]),
            "macro_f1": mean_std([m["test_stroke"]["macro_f1"] for m in metrics]),
            "balanced_accuracy": mean_std([m["test_stroke"]["balanced_accuracy"] for m in metrics]),
            "side_accuracy": mean_std([m["test_side"]["accuracy"] for m in metrics]),
            "side_macro_f1": mean_std([m["test_side"]["macro_f1"] for m in metrics]),
            "per_class": {
                name: {key: mean_std([m["test_stroke"][key][i] for m in metrics]) for key in ("precision", "recall", "f1")}
                | {"support": metrics[0]["test_stroke"]["support"][i]}
                for i, name in enumerate(CLASSES)
            },
        }
        # Per-match breakdown from saved per-event predictions.
        per_match = defaultdict(lambda: defaultdict(list))
        confusion = np.zeros((8, 8), dtype=np.int64)
        for d in dirs:
            p = np.load(d / "test_predictions.npz")
            matches = np.array([int(s.split("_")[0]) for s in p["sample_id"]])
            stroke_pred, side_pred = p["stroke_prob"].argmax(1), p["side_prob"].argmax(1)
            np.add.at(confusion, (p["stroke_true"], stroke_pred), 1)
            for match in sorted(set(matches)):
                m = matches == match
                per_match[match]["n"] = [int(m.sum())]
                per_match[match]["accuracy"].append((stroke_pred[m] == p["stroke_true"][m]).mean())
                per_match[match]["macro_f1"].append(macro_f1(p["stroke_true"][m], stroke_pred[m]))
                per_match[match]["side_accuracy"].append((side_pred[m] == p["side_true"][m]).mean())
        variants[variant]["per_match"] = {
            str(match): {"n": v["n"][0], **{k: mean_std(v[k]) for k in ("accuracy", "macro_f1", "side_accuracy")}}
            for match, v in per_match.items()
        }
        variants[variant]["confusion_sum_over_seeds"] = confusion.tolist()
        # Spread across matches (seed-averaged per-match accuracy).
        variants[variant]["across_match_std"] = {
            k: float(np.std([variants[variant]["per_match"][m][k]["mean"] for m in variants[variant]["per_match"]], ddof=1))
            for k in ("accuracy", "macro_f1", "side_accuracy")
        }
    summary["variants"] = variants

    degradation_path = args.runs / "degradation.json"
    if degradation_path.exists():
        degradation = json.loads(degradation_path.read_text())
        curves = {}
        families = defaultdict(list)
        for name, table in degradation["kap"].items():
            families["kap_" + name.rsplit("_seed", 1)[0]].append(table)
        families["bst_cg_ap"] = list(degradation["bst"].values())
        for family, tables in families.items():
            curves[family] = {
                key: {metric: mean_std([t[key][metric] for t in tables]) for metric in ("accuracy", "macro_f1")}
                for key in tables[0]
            }
        summary["degradation"] = curves
        if degradation.get("gate"):
            gate = list(degradation["gate"].values())
            summary["gate_analysis"] = {k: float(np.mean([g[k] for g in gate])) for k in gate[0]}

    predictions_path = args.runs / "clean_predictions.npz"
    if predictions_path.exists():
        p = np.load(predictions_path)
        truth = p["truth"]
        kap = {k: p[k].argmax(1) for k in p.files if k.startswith("kap__main_seed")}
        bst = {k: p[k].argmax(1) for k in p.files if k.startswith("bst__")}
        rng = np.random.default_rng(0)
        tests = {}
        for kn, kp in kap.items():
            for bn, bp in bst.items():
                tests[f"{kn} vs {bn}"] = {
                    "kap_acc": float((kp == truth).mean() * 100), "bst_acc": float((bp == truth).mean() * 100),
                    "mcnemar": mcnemar_exact(kp == truth, bp == truth),
                    "bootstrap": paired_bootstrap(truth, kp, bp, args.bootstrap, rng),
                }
        # Seed-averaged probabilities as a single summary comparison.
        kap_mean = np.mean([p[k] for k in p.files if k.startswith("kap__main_seed")], axis=0).argmax(1)
        bst_mean = np.mean([p[k] for k in p.files if k.startswith("bst__")], axis=0).argmax(1)
        tests["kap_seed_avg vs bst_ckpt_avg"] = {
            "kap_acc": float((kap_mean == truth).mean() * 100), "bst_acc": float((bst_mean == truth).mean() * 100),
            "mcnemar": mcnemar_exact(kap_mean == truth, bst_mean == truth),
            "bootstrap": paired_bootstrap(truth, kap_mean, bst_mean, args.bootstrap, rng),
        }
        summary["significance"] = tests
        summary["bst_clean"] = {
            "accuracy": mean_std([(v == truth).mean() for v in bst.values()]),
            "macro_f1": mean_std([macro_f1(truth, v) for v in bst.values()]),
            "balanced_accuracy": mean_std([
                np.mean([(v[truth == c] == c).mean() for c in range(8)]) for v in bst.values()
            ]),
        }

        # Window truncation: the released BST segments end at the neighbouring hits, so
        # part of KAP's fixed [-15, +30] window is zero-padded for many events.
        test = raw["split"] == "test"
        empty = (np.abs(raw["pose"][test]).sum(-1) == 0).sum(-1)[np.argsort(raw["sample_id"][test])]
        groups = {"full": empty == 0, "truncated": empty > 0}
        summary["window_truncation"] = {
            "truncated_share": float((empty > 0).mean() * 100),
            **{name: {"n": int(mask.sum()),
                      "kap": float(np.mean([(k[mask] == truth[mask]).mean() for k in kap.values()]) * 100),
                      "bst": float(np.mean([(b[mask] == truth[mask]).mean() for b in bst.values()]) * 100)}
               for name, mask in groups.items()},
        }

    # Exploratory validation runs aimed at BST (long window, fine-label heads), Match 41 only.
    bst_val = REPO_ROOT / "work_dirs/paper_ablation/bst_val_match41.json"
    # The long-window probes used the earlier R(2+1)D stream; rgbcmp_* are own-RGB comparisons, not probes.
    explore = [m for m in sorted((REPO_ROOT / "work_dirs/paper_bstgap").glob("*_seed*/metrics.json"))
               if not m.parent.name.startswith("rgbcmp")]
    if bst_val.exists() and explore:
        bst_val_runs = json.loads(bst_val.read_text(encoding="utf-8"))["results"].values()
        summary["validation_probe"] = {
            "bst_acc": float(np.mean([r["accuracy"] for r in bst_val_runs]) * 100),
            "best_explore_acc": max(json.loads(m.read_text(encoding="utf-8"))["val_stroke"]["accuracy"] for m in explore) * 100,
            "main_val_acc": [json.loads(m.read_text(encoding="utf-8"))["val_stroke"]["accuracy"] * 100
                             for m in sorted((REPO_ROOT / "work_dirs/paper_mate/runs").glob("main_seed*/metrics.json"))],
        }

    out = args.runs / "summary.json"
    out.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    for variant, v in variants.items():
        print(f"{variant:14s} params={v['params']:,} acc={v['accuracy']['mean']:.2f}±{v['accuracy']['std']:.2f} "
              f"f1={v['macro_f1']['mean']:.2f}±{v['macro_f1']['std']:.2f} side={v['side_accuracy']['mean']:.2f} n={len(v['seeds'])}")


if __name__ == "__main__":
    main()
