"""Offline A/B of re-scoring strokes with the shuttle flight time to the next hit.

The KAP window ends about one second after contact, so it cannot see when the
opponent returns the shuttle. That flight time separates pushes (~1.0 s) from
lifts (~1.3 s) and drives (~0.5 s) from pushes, which look alike in pose.

score(c) = log p_model(c) + w * log p(flight | c), where p(flight | c) is a
smoothed histogram of log flight time fitted on ShuttleSet train matches. At
inference the flight time is the gap to the next *detected* hit of the same
rally; the last hit of a rally keeps the classifier output. w is selected on
the validation clips and reported on the test clips.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
for path in (REPO_ROOT, REPO_ROOT / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from ai_classifier.analytics.flight_time import fit_flight_time_model, flight_bin  # noqa: E402
from ai_classifier.analytics.tactical_decoder import STROKE_CLASSES  # noqa: E402
from scripts.evaluation.benchmark_8_matches_10min import CLIPS_CONFIG  # noqa: E402
from scripts.evaluation.compare_kap_checkpoints_8clips import load_classifier, mcnemar_exact  # noqa: E402
from scripts.evaluation.eval_sequence_decoder_8clips import correctness, load_clip  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path,
                        default=REPO_ROOT / "src/ai_classifier/models/checkpoints/kap_classifier.pth")
    parser.add_argument("--feature-cache", type=Path, default=REPO_ROOT / "work_dirs/kap_ab_8clips/feature_cache")
    parser.add_argument("--manifest", type=Path, default=REPO_ROOT / "data/manifests/shuttleset_rgb.csv")
    parser.add_argument("--weights", type=float, nargs="+", default=[0.25, 0.5, 0.75, 1.0, 1.5, 2.0])
    parser.add_argument("--output", type=Path, default=REPO_ROOT / "work_dirs/flight_time_rerank_8clips.json")
    return parser.parse_args()


def attach_flight_times(clip: dict) -> None:
    fps = cv2.VideoCapture(str(clip["cfg"]["video"])).get(cv2.CAP_PROP_FPS) or 30.0
    by_rally = defaultdict(list)
    for p in clip["preds"]:
        by_rally[p["rally"]].append(p)
    for items in by_rally.values():
        items.sort(key=lambda p: p["frame"])
        for a, b in zip(items, items[1:]):
            a["flight"] = (b["frame"] - a["frame"]) / fps
        items[-1]["flight"] = None


def rerank(clip: dict, flight_model: np.ndarray, weight: float) -> list[str]:
    out = []
    for p in clip["preds"]:
        score = p["log_probs"].copy()
        if weight and p.get("flight") is not None:
            score = score + weight * flight_model[:, flight_bin(p["flight"])]
        out.append(STROKE_CLASSES[int(np.argmax(score))])
    return out


def main() -> None:
    args = parse_args()
    model = load_classifier(args.checkpoint)
    clips = [load_clip(cfg, model, args) for cfg in CLIPS_CONFIG]
    for clip in clips:
        attach_flight_times(clip)
    flight_model = fit_flight_time_model(args.manifest)

    variants = {f"w={w:g}": w for w in [0.0, *args.weights]}
    results = {name: {c["cfg"]["match_id"]: (correctness(c, rerank(c, flight_model, w)), rerank(c, flight_model, w))
                      for c in clips} for name, w in variants.items()}

    def pooled(name: str, split: str | None) -> list[bool]:
        return [x for c in clips if split is None or c["split"] == split
                for x in results[name][c["cfg"]["match_id"]][0]]

    best = max((n for n in variants if n != "w=0"), key=lambda n: sum(pooled(n, "val")))
    print(f"weight selected on val: {best}\n")
    summary = {"selected": best, "splits": {}}
    for split in ("val", "test", None):
        label = split or "all"
        base = pooled("w=0", split)
        print(f"[{label}] GT={len(base)}")
        summary["splits"][label] = {}
        for name in variants:
            hits = pooled(name, split)
            gained = sum(h and not b for h, b in zip(hits, base))
            lost = sum(b and not h for h, b in zip(hits, base))
            p = mcnemar_exact(gained, lost)
            summary["splits"][label][name] = {"top1": sum(hits) / len(hits), "gained": gained, "lost": lost, "p": p}
            mark = " <- selected" if name == best else ""
            print(f"  {name:7s} top1={100 * sum(hits) / len(hits):6.2f}%  vs w=0: +{gained:3d} / -{lost:3d}  p={p:.4f}{mark}")

    # Per-class recall on the test clips, before and after.
    print("\n[test] recall per GT class (w=0 -> selected)")
    for cls in STROKE_CLASSES:
        rows = [(ok0, ok1) for c in clips if c["split"] == "test"
                for t, ok0, ok1 in zip(c["gt"], results["w=0"][c["cfg"]["match_id"]][0],
                                       results[best][c["cfg"]["match_id"]][0]) if t["stroke"] == cls]
        if rows:
            print(f"  {cls:10s} n={len(rows):3d}  {100 * sum(a for a, _ in rows) / len(rows):5.1f}% -> "
                  f"{100 * sum(b for _, b in rows) / len(rows):5.1f}%")
    args.output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
