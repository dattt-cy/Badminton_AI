"""Numbers for the simplified paper model beyond summarize.py: seed intervals, late fusion, inference time.

Example:
    python scripts/paper/paper_extra.py --mate-alone <mate_alone_test.npy>
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from scipy import stats

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO_ROOT / "src"), str(REPO_ROOT / "scripts/training"), str(REPO_ROOT / "scripts/paper")]

import degradation_eval as de  # noqa: E402
from summarize import mcnemar_exact  # noqa: E402
from train_kap_ablation import KAPAblationModel  # noqa: E402
from train_rgb_scratch import RGBEncoder  # noqa: E402

KEYS = [("both", 0.0), ("both", 0.5), ("pose", 0.5), ("shuttle", 0.5), ("burst", 0.5), ("player", 0.5), ("shuttle", 1.0), ("pose", 1.0), ("both", 1.0)]


def interval(a: np.ndarray, b: np.ndarray) -> dict:
    gap = float(a.mean() - b.mean())
    va, vb = a.var(ddof=1) / len(a), b.var(ddof=1) / len(b)
    if va + vb == 0:
        return {"gap": gap, "low": gap, "high": gap}
    freedom = (va + vb) ** 2 / (va ** 2 / (len(a) - 1) + vb ** 2 / (len(b) - 1))
    half = float(stats.t.ppf(0.975, freedom) * (va + vb) ** 0.5)
    return {"gap": gap, "low": gap - half, "high": gap + half}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, default=REPO_ROOT / "work_dirs/paper_mate/simple_final")
    parser.add_argument("--features", type=Path, default=REPO_ROOT / "work_dirs/paper_rgb_phases/features_rgb_phases.npz")
    parser.add_argument("--encoder", type=Path, default=REPO_ROOT / "work_dirs/paper_ownrgb_ss22/x13/rgb_encoder_best.pth")
    parser.add_argument("--mate-alone", type=Path, required=True)
    parser.add_argument("--mask-seed", type=int, default=20261101)
    args = parser.parse_args()
    device = torch.device("cuda")

    degradation = json.loads((args.runs / "degradation.json").read_text(encoding="utf-8"))
    families: dict[str, list] = {}
    for name, table in degradation["kap"].items():
        families.setdefault(name.rsplit("_seed", 1)[0], []).append(table)
    result = {"gaps": {}, "late_fusion": {}}
    for family, tables in families.items():
        result["gaps"][family] = {}
        for key, rate in KEYS:
            cell = f"{key}@{rate:.1f}"
            acc = np.array([t[cell]["accuracy"] * 100 for t in tables])
            base = np.array([t[cell]["accuracy"] * 100 for t in families["main"]])
            result["gaps"][family][cell] = {"acc": float(acc.mean()), **(interval(acc, base) if len(acc) > 1 else {})}

    tensors, _ids, truth = de.kap_inputs(args.features)
    mate = np.load(args.mate_alone)
    models = []
    for run in sorted(args.runs.glob("no_rgb_seed*/kap_best.pth")):
        model = KAPAblationModel(**json.loads((run.parent / "model_kwargs.json").read_text(encoding="utf-8")))
        model.load_state_dict(torch.load(run, map_location="cpu", weights_only=False)["model"])
        models.append(model.to(device).eval())
    for key, rate in KEYS:
        degraded = de.degrade_kap(tensors, key, rate, de.mask_rng(args.mask_seed, key, rate, "kap"))
        scores = [((0.5 * de.kap_predict(m, degraded, device) + 0.5 * mate).argmax(1) == truth).mean() * 100 for m in models]
        result["late_fusion"][f"{key}@{rate:.1f}"] = float(np.mean(scores))
    print("late fusion", {k: round(v, 2) for k, v in result["late_fusion"].items()}, flush=True)

    # Paired test on clean test events: main ensemble against each family's ensemble.
    ensembles = {}
    for family in ("main", "no_rgb", "no_pose", "no_shuttle", "no_pose_shuttle", "full", "no_contact", "no_noise"):
        probs = [np.load(p)["stroke_prob"] for p in sorted(args.runs.glob(f"{family}_seed*/test_predictions.npz"))]
        ensembles[family] = np.mean(probs, axis=0).argmax(1)
    labels = np.load(next(args.runs.glob("main_seed*/test_predictions.npz")))["stroke_true"]
    result["ensemble"] = {f: {"acc": float((e == labels).mean() * 100), **mcnemar_exact(e == labels, ensembles["main"] == labels)} for f, e in ensembles.items() if f != "main"}
    result["ensemble"]["main"] = {"acc": float((ensembles["main"] == labels).mean() * 100)}

    # Inference time per event, batch size 1, on the training GPU.
    encoder = RGBEncoder().to(device).eval()
    encoder.load_state_dict(torch.load(args.encoder, map_location=device, weights_only=False)["model"])
    clip = torch.rand(1, 3, 16, 112, 112, device=device)
    run = sorted(args.runs.glob("main_seed*/kap_best.pth"))[0]
    kap = KAPAblationModel(**json.loads((run.parent / "model_kwargs.json").read_text(encoding="utf-8")))
    kap.load_state_dict(torch.load(run, map_location="cpu", weights_only=False)["model"])
    kap = kap.to(device).eval()
    inputs = [tensors[n][:1].to(device) for n in ("rgb", "pose", "court", "shuttle", "contact_dist", "quality")]

    def timed(step) -> float:
        with torch.inference_mode():
            for _ in range(20):
                step()
            torch.cuda.synchronize()
            start = time.perf_counter()
            for _ in range(200):
                step()
            torch.cuda.synchronize()
        return (time.perf_counter() - start) / 200 * 1000

    result["ms_per_event"] = {"mate": timed(lambda: encoder(clip)), "kap": timed(lambda: kap(*inputs))}
    print(result["ms_per_event"], flush=True)
    (args.runs / "extra.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
