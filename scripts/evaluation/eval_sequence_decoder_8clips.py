"""Offline A/B of the tactical sequence decoder on the 8 benchmark clips.

Uses the cached candidate features of compare_kap_checkpoints_8clips and the
rally ids of the corresponding pipeline outputs, so every variant decodes the
same classifier outputs. Variants:
  none      - classifier argmax;
  current   - production behaviour: log-probs approximated from top-2 and
              confidence (the pipeline never passes logits), lambda 0.20,
              transitions fitted on the whole manifest;
  full@lam  - full classifier log-probs, transitions fitted on train matches only.
Lambda is selected on the validation clips and reported on the test clips.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
for path in (REPO_ROOT, REPO_ROOT / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from ai_classifier.analytics.tactical_decoder import (  # noqa: E402
    STROKE_CLASSES, TacticalSequenceDecoder, refine_predictions_by_sequence,
)
from scripts.evaluation.benchmark_8_matches_10min import CLIPS_CONFIG  # noqa: E402
from scripts.evaluation.compare_kap_checkpoints_8clips import (  # noqa: E402
    TEST_MATCHES, load_classifier, load_gt, match_gt, mcnemar_exact,
)
from scripts.inference.robust_aligned_video_pipeline import DEVICE  # noqa: E402

NON_TRAIN = {str(m) for m in (35, 36, 37, 38, 39, 40, 41, 42, 43, 44)}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path,
                        default=REPO_ROOT / "src/ai_classifier/models/checkpoints/kap_classifier.pth")
    parser.add_argument("--feature-cache", type=Path, default=REPO_ROOT / "work_dirs/kap_ab_8clips/feature_cache")
    parser.add_argument("--manifest", type=Path, default=REPO_ROOT / "data/manifests/shuttleset_rgb.csv")
    parser.add_argument("--lambdas", type=float, nargs="+", default=[0.2, 0.5, 1.0, 1.5, 2.0, 3.0])
    parser.add_argument("--output", type=Path, default=REPO_ROOT / "work_dirs/sequence_decoder_8clips.json")
    return parser.parse_args()


def load_clip(cfg: dict, model, args) -> dict:
    cached = torch.load(args.feature_cache / f"{cfg['video'].stem}.pt", map_location="cpu", weights_only=False)
    pipeline = json.loads(Path(cfg["pipeline_json"]).read_text(encoding="utf-8"))["predictions"]
    rally_by_frame = {int(p["frame"]): p.get("rally", 1) for p in pipeline}
    preds = []
    with torch.no_grad():
        for item in cached["features"]:
            f = {k: v.to(DEVICE) if torch.is_tensor(v) else v for k, v in item["feature"].items()}
            logits = model(f["emb"], f["s_j"], f["s_p"], f["s_s"], f["s_c"], f["s_q"])[0][0]
            probs = torch.softmax(logits, -1).cpu().numpy()
            order = np.argsort(-probs)
            preds.append({"frame": item["frame"], "hitter": item["hitter"],
                          "rally": rally_by_frame.get(item["frame"], 1),
                          "log_probs": np.log(np.clip(probs, 1e-9, 1.0)),
                          "stroke": STROKE_CLASSES[order[0]],
                          "top2": [STROKE_CLASSES[order[0]], STROKE_CLASSES[order[1]]],
                          "confidence": float(probs[order[0]])})
    gt = load_gt(args.manifest, cfg["match_id"], cfg["raw_start"], 18000)
    return {"cfg": cfg, "preds": preds, "gt": gt,
            "split": "test" if cfg["match_id"] in TEST_MATCHES else "val"}


def decode(preds: list[dict], decoder: TacticalSequenceDecoder | None, lam: float, full: bool) -> list[str]:
    if decoder is None:
        return [p["stroke"] for p in preds]
    if not full:
        # Reproduce production: refine_predictions_by_sequence without _logits.
        rows = [{k: p[k] for k in ("frame", "hitter", "rally", "stroke", "top2", "confidence")} for p in preds]
        refined, _ = refine_predictions_by_sequence(rows, decoder, lam=lam)
        return [r["stroke"] for r in refined]
    out = [p["stroke"] for p in preds]
    rallies: dict = {}
    for i, p in enumerate(preds):
        rallies.setdefault(p["rally"], []).append(i)
    for idx in rallies.values():
        path = decoder.decode_rally(np.stack([preds[i]["log_probs"] for i in idx]),
                                    [preds[i]["hitter"] for i in idx], lam=lam)
        for i, k in zip(idx, path):
            out[i] = STROKE_CLASSES[k]
    return out


def correctness(clip: dict, strokes: list[str]) -> list[bool]:
    rows = [{"frame": p["frame"], "stroke": s} for p, s in zip(clip["preds"], strokes)]
    return [m is not None and m["stroke"] == t["stroke"]
            for t, m in zip(clip["gt"], match_gt(clip["gt"], rows, tolerance=12))]


def main() -> None:
    args = parse_args()
    model = load_classifier(args.checkpoint)
    clips = [load_clip(cfg, model, args) for cfg in CLIPS_CONFIG]
    production = TacticalSequenceDecoder(args.manifest, alpha=0.5, default_lambda=0.20)
    train_only = TacticalSequenceDecoder(args.manifest, alpha=0.5, exclude_matches=NON_TRAIN)

    variants = {"none": (None, 0.0, False), "current": (production, 0.20, False)}
    variants.update({f"full@{lam:g}": (train_only, lam, True) for lam in args.lambdas})
    results = {}
    for name, (decoder, lam, full) in variants.items():
        results[name] = {c["cfg"]["match_id"]: correctness(c, decode(c["preds"], decoder, lam, full))
                         for c in clips}

    def pooled(name: str, split: str | None) -> list[bool]:
        return [x for c in clips if split is None or c["split"] == split
                for x in results[name][c["cfg"]["match_id"]]]

    full_names = [n for n in variants if n.startswith("full@")]
    best = max(full_names, key=lambda n: sum(pooled(n, "val")))
    print(f"lambda selected on val: {best}\n")
    summary = {"selected": best, "splits": {}}
    for split in ("val", "test", None):
        label = split or "all"
        base = pooled("none", split)
        print(f"[{label}] GT={len(base)}")
        summary["splits"][label] = {}
        for name in variants:
            hits = pooled(name, split)
            only_new = sum(h and not b for h, b in zip(hits, base))
            only_base = sum(b and not h for h, b in zip(hits, base))
            p = mcnemar_exact(only_new, only_base)
            summary["splits"][label][name] = {"top1": sum(hits) / len(hits), "gain_vs_none": [only_new, only_base], "p": p}
            mark = " <- selected" if name == best else ""
            print(f"  {name:10s} top1={100 * sum(hits) / len(hits):6.2f}%  vs none: +{only_new:3d} / -{only_base:3d}  p={p:.4f}{mark}")
    args.output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
