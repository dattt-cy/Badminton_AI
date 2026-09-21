"""Full-video hit spotting evaluation for the R(2+1)D hit detector.

Runs dense sliding-window inference over all matches in the requested split,
performs threshold sweep on val to pick the best threshold, then evaluates
precision/recall/F1 at ±2, ±5, ±15 frame tolerances on the chosen split.

Usage:
    # 1. Tune threshold on val
    python scripts/evaluation/evaluate_rgb_hit_spotting.py \\
        --checkpoint work_dirs/r2plus1d18_hit_full/best.pth \\
        --split val \\
        --output outputs/hit_rgb_val.json

    # 2. Report on test using best threshold found above
    python scripts/evaluation/evaluate_rgb_hit_spotting.py \\
        --checkpoint work_dirs/r2plus1d18_hit_full/best.pth \\
        --split test \\
        --threshold 0.60 \\
        --output outputs/hit_rgb_test.json
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ai_classifier.localization import HitEvent, match_hit_events, temporal_nms  # noqa: E402
from scripts.inference.scan_video_hit_rgb import load_model, scan_video  # noqa: E402

SIDE_MAP = {"top": "upper", "upper": "upper", "bottom": "lower", "lower": "lower"}


# ---------------------------------------------------------------------------
# Ground truth loading
# ---------------------------------------------------------------------------

def load_ground_truth(manifest: Path, split: str, match_ids: set[str] | None = None) -> dict[str, dict]:
    """Return {match_id: {video_path, truth: [HitEvent, ...]}}."""
    matches: dict[str, dict] = {}
    with manifest.open("r", encoding="utf-8-sig", newline="") as fh:
        for row in csv.DictReader(fh):
            mid = row["match_id"]
            if row["split"] != split:
                continue
            if match_ids is not None and mid not in match_ids:
                continue
            side = SIDE_MAP.get(row["player_side"].strip().lower())
            if side is None:
                continue
            item = matches.setdefault(mid, {"video_path": Path(row["video_path"]), "truth": []})
            item["truth"].append(HitEvent(int(float(row["hit_frame"])), side, 1.0))
    for item in matches.values():
        # Deduplicate and sort
        seen: set[tuple[int, str]] = set()
        deduped: list[HitEvent] = []
        for evt in sorted(item["truth"], key=lambda e: e.frame):
            key = (evt.frame, evt.side)
            if key not in seen:
                seen.add(key)
                deduped.append(evt)
        item["truth"] = deduped
    return matches


# ---------------------------------------------------------------------------
# Aggregate metrics
# ---------------------------------------------------------------------------

def aggregate(
    matches: dict[str, dict],
    scored: dict[str, list[HitEvent]],
    threshold: float,
    nms_radius: int,
    tolerance: int,
) -> dict:
    """Return aggregated precision/recall/F1 across all matches."""
    totals: dict[str, float] = defaultdict(float)
    all_errors: list[int] = []
    per_match: dict[str, dict] = {}

    for match_id, item in matches.items():
        predicted = temporal_nms(scored[match_id], threshold=threshold, radius=nms_radius)
        result = match_hit_events(predicted, item["truth"], tolerance=tolerance)
        per_match[match_id] = {k: v for k, v in result.items() if k != "matches"}
        for key in ("truth", "predicted", "true_positives", "false_positives", "false_negatives", "joint_true_positives"):
            totals[key] += result[key]
        all_errors.extend(abs(m["frame_error"]) for m in result["matches"])

    tp = totals["true_positives"]
    pred_n = totals["predicted"]
    truth_n = totals["truth"]
    precision = tp / pred_n if pred_n else 0.0
    recall = tp / truth_n if truth_n else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    side_correct = totals["joint_true_positives"]

    import numpy as np

    return {
        "threshold": threshold,
        "tolerance": tolerance,
        "truth": int(truth_n),
        "predicted": int(pred_n),
        "true_positives": int(tp),
        "false_positives": int(totals["false_positives"]),
        "false_negatives": int(totals["false_negatives"]),
        "precision": round(precision, 6),
        "recall": round(recall, 6),
        "f1": round(f1, 6),
        "mean_absolute_frame_error": round(float(np.mean(all_errors)), 3) if all_errors else None,
        "median_absolute_frame_error": round(float(np.median(all_errors)), 3) if all_errors else None,
        "side_accuracy": round(side_correct / tp, 4) if tp else None,
        "joint_precision": round(side_correct / pred_n, 4) if pred_n else 0.0,
        "joint_recall": round(side_correct / truth_n, 4) if truth_n else 0.0,
        "per_match": per_match,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path,
                        default=Path("data/manifests/shuttleset_rgb.csv"))
    parser.add_argument("--checkpoint", type=Path,
                        default=Path("work_dirs/r2plus1d18_hit_full/best.pth"))
    parser.add_argument("--split", choices=("val", "test"), default="val")
    parser.add_argument("--match-ids", nargs="*",
                        help="Optional subset of match IDs to evaluate.")
    parser.add_argument("--output", type=Path, required=True,
                        help="Path for the output JSON report.")
    parser.add_argument("--stride", type=int, default=4,
                        help="Dense scan stride in frames (default 4).")
    parser.add_argument("--batch-size", type=int, default=8,
                        help="GPU batch size for inference.")
    parser.add_argument("--nms-radius", type=int, default=15,
                        help="Temporal NMS suppression radius in frames.")
    parser.add_argument(
        "--threshold", type=float,
        help="Fixed threshold. If omitted, sweep over --thresholds and pick best F1@5 on this split.",
    )
    parser.add_argument(
        "--thresholds", type=float, nargs="+",
        default=[0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90],
        help="Threshold sweep values (used when --threshold is not set).",
    )
    parser.add_argument("--max-frames", type=int,
                        help="Limit frames per video (debugging only).")
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument(
        "--raw-cache-dir", type=Path,
        help="Save/load per-match raw scored frames so dense scans are reusable.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # ---- load model ----
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = not args.no_amp and device.type == "cuda"
    model, ckpt_meta = load_model(args.checkpoint)
    model.to(device)
    n_frames = int(ckpt_meta.get("frames", 16))
    print(f"[INFO] checkpoint epoch={ckpt_meta.get('epoch', '?')}  frames={n_frames}  device={device}")

    # ---- load ground truth ----
    match_ids = set(args.match_ids) if args.match_ids else None
    matches = load_ground_truth(args.manifest, args.split, match_ids)
    if not matches:
        raise ValueError(f"No matches found for split={args.split}")
    print(f"[INFO] {args.split} split: {len(matches)} matches, "
          f"{sum(len(v['truth']) for v in matches.values())} ground-truth hits")

    # ---- dense scan all matches ----
    scored: dict[str, list[HitEvent]] = {}
    for match_id in sorted(matches, key=int):
        item = matches[match_id]
        cache_path = (
            args.raw_cache_dir / f"match_{match_id}.json"
            if args.raw_cache_dir is not None else None
        )
        if cache_path is not None and cache_path.is_file():
            cached = json.loads(cache_path.read_text(encoding="utf-8"))
            scored[match_id] = [HitEvent(**event) for event in cached]
            print(f"  loaded raw cache match {match_id}: {len(scored[match_id])} frames", flush=True)
            continue
        print(f"  scanning match {match_id}  truth_hits={len(item['truth'])}", flush=True)
        scored[match_id] = scan_video(
            item["video_path"], model, n_frames,
            stride=args.stride,
            batch_size=args.batch_size,
            device=device,
            use_amp=use_amp,
            max_frames=args.max_frames,
        )
        if cache_path is not None:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            cache_path.write_text(
                json.dumps([event.__dict__ for event in scored[match_id]]) + "\n",
                encoding="utf-8",
            )
        print(f"  -> {len(scored[match_id])} raw scored frames", flush=True)

    # ---- threshold sweep (unless fixed) ----
    TUNE_TOLERANCE = 5
    sweep: list[dict] = []
    if args.threshold is not None:
        best_threshold = args.threshold
        print(f"[INFO] using fixed threshold={best_threshold:.2f}")
    else:
        sweep = []
        for thr in args.thresholds:
            m = aggregate(matches, scored, thr, args.nms_radius, TUNE_TOLERANCE)
            sweep.append({"threshold": thr, "precision": m["precision"], "recall": m["recall"], "f1": m["f1"],
                          "true_positives": m["true_positives"], "predicted": m["predicted"]})
            print(
                f"  thr={thr:.2f}  P={m['precision']:.3f}  R={m['recall']:.3f}  F1={m['f1']:.3f}"
                f"  tp={m['true_positives']} / pred={m['predicted']}",
                flush=True,
            )
        best_threshold = max(sweep, key=lambda r: r["f1"])["threshold"]
        print(f"[INFO] best threshold={best_threshold:.2f} (by F1@±{TUNE_TOLERANCE})")

    # ---- evaluate at all tolerances ----
    tolerances = [2, 5, 15]
    metrics_by_tol: dict[str, dict] = {}
    for tol in tolerances:
        m = aggregate(matches, scored, best_threshold, args.nms_radius, tol)
        metrics_by_tol[str(tol)] = m
        print(
            f"  ±{tol:2d} frame  P={m['precision']:.4f}  R={m['recall']:.4f}  F1={m['f1']:.4f}"
            f"  side_acc={m['side_accuracy']}  mean_err={m['mean_absolute_frame_error']}",
        )

    # ---- save predictions ----
    predictions = {
        mid: [e.__dict__ for e in temporal_nms(evs, threshold=best_threshold, radius=args.nms_radius)]
        for mid, evs in scored.items()
    }

    result = {
        "split": args.split,
        "match_ids": sorted(matches, key=int),
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_epoch": ckpt_meta.get("epoch"),
        "stride": args.stride,
        "nms_radius": args.nms_radius,
        "threshold_selection_tolerance": TUNE_TOLERANCE,
        "threshold_sweep": sweep if args.threshold is None else [],
        "selected_threshold": best_threshold,
        "metrics_by_tolerance": metrics_by_tol,
        "predictions": predictions,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    m5 = metrics_by_tol["5"]
    print(
        f"\n[DONE] threshold={best_threshold:.2f}  "
        f"P@5={m5['precision']:.4f}  R@5={m5['recall']:.4f}  F1@5={m5['f1']:.4f}  "
        f"output={args.output}"
    )


if __name__ == "__main__":
    main()
