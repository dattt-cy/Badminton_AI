"""A/B/C comparison of KAP checkpoints on the 8-clip end-to-end benchmark.

Every checkpoint classifies the exact same hit candidates (reused from the
existing ``eval_clips/*_pipeline.json`` runs) and the exact same extracted
features, so differences come from the classifier alone. Features are cached
per clip, which makes reruns with additional checkpoints nearly free.

Pairwise significance uses the exact McNemar test on per-GT-hit correctness.
Clips are reported separately for ShuttleSet test matches (39, 40, 42, 43, 44)
and validation matches (36, 38, 41); match 41 was used to select the
production checkpoint, so its numbers are biased in favour of that model.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from collections import defaultdict
from itertools import combinations
from pathlib import Path

import cv2
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
for path in (REPO_ROOT, REPO_ROOT / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from scripts.evaluation.benchmark_8_matches_10min import CLIPS_CONFIG
from scripts.inference.auto_court_detection import detect_court_corners
from scripts.inference.robust_aligned_video_pipeline import (
    DEVICE,
    DST_PTS,
    AlignedVideoPipeline,
)
from ai_classifier.models import KAPCrossTransModel, KAPCrossTransV2Model, KAPScratchModel
from ai_classifier.models.taxonomy import SIDE_CLASSES, STROKE_CLASSES

TEST_MATCHES = {39, 40, 42, 43, 44}
DEFAULT_MODELS = {
    "base": "work_dirs/kap_robust_seed20260926/kap_best.pth",
    "production": "src/ai_classifier/models/checkpoints/kap_classifier.pth",
    "previous_production": "src/ai_classifier/models/checkpoints/kap_classifier_prev_baseline_online_20261008.pth.bak",
    "cross_v2_liftboost": "work_dirs/kap_cross_v2_liftboost_seed20260926/kap_best.pth",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model", action="append", default=None, metavar="NAME=PATH",
        help="Checkpoint to compare; repeatable. Defaults to the five KAP variants.",
    )
    parser.add_argument("--manifest", type=Path, default=Path("data/manifests/shuttleset_rgb.csv"))
    parser.add_argument("--clips", type=str, default=None, help="Comma-separated clip numbers, e.g. 1,2,8.")
    parser.add_argument("--frames", type=int, default=18000)
    parser.add_argument("--tolerance", type=int, default=12)
    parser.add_argument("--contact-refine-radius", type=int, default=6)
    # Must match run_pipeline.py, which produced the hit candidates being reused.
    parser.add_argument("--pose-model", type=Path, default=Path("models/checkpoints/pose/yolov8n-pose.pt"))
    parser.add_argument("--output-dir", type=Path, default=Path("work_dirs/kap_ab_8clips"))
    parser.add_argument(
        "--feature-cache", type=Path, default=None,
        help="Feature cache directory; defaults to OUTPUT_DIR/feature_cache.",
    )
    return parser.parse_args()


def load_classifier(path: Path) -> torch.nn.Module:
    payload = torch.load(path, map_location=DEVICE, weights_only=False)
    model_type = payload.get("model_type", "baseline")
    if model_type in ("cross_v2", "cross_v3"):
        model_class = KAPCrossTransV2Model
    elif model_type == "cross":
        model_class = KAPCrossTransModel
    else:
        model_class = KAPScratchModel
    model = model_class().to(DEVICE)
    model.load_state_dict(payload["model"])
    model.eval()
    return model


def load_gt(manifest: Path, match_id: int, offset: int, frames: int) -> list[dict]:
    gt = []
    with manifest.open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            absolute = int(float(row["hit_frame"]))
            if row["match_id"] == str(match_id) and offset <= absolute < offset + frames:
                gt.append({
                    "frame": absolute - offset,
                    "stroke": row["coarse_label"],
                    "stroke_side": row["stroke_side"],
                })
    return sorted(gt, key=lambda row: row["frame"])


def load_events(pipeline_json: Path, frames: int) -> list[dict]:
    payload = json.loads(pipeline_json.read_text(encoding="utf-8"))
    return [
        {"frame": int(row["frame"]), "side": row.get("hitter", "unknown")}
        for row in payload["predictions"]
        if 0 <= int(row["frame"]) < frames
    ]


def extract_clip_features(pipeline: AlignedVideoPipeline, cfg: dict, events: list[dict], args) -> dict:
    cap = cv2.VideoCapture(str(cfg["video"]))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    corners = detect_court_corners(str(cfg["video"]))

    # Reuse the loaded networks; only the per-video geometry and caches change.
    pipeline.orig_w, pipeline.orig_h = width, height
    pipeline.h_mat = cv2.getPerspectiveTransform(corners, DST_PTS)
    pipeline._pose_cache.clear()
    pipeline._shuttle_cache.clear()
    pipeline.prepare_event_cache(
        cap, [event["frame"] for event in events], total_frames,
        contact_refine_radius=args.contact_refine_radius,
    )
    features = []
    for index, event in enumerate(events, 1):
        feature = pipeline.extract_features(
            cap, event["frame"], total_frames, hitter_side=event["side"],
            contact_refine_radius=args.contact_refine_radius,
        )
        if feature is not None:
            features.append({"frame": event["frame"], "hitter": event["side"], "feature": feature})
        if index % 25 == 0:
            print(f"    features {index}/{len(events)}", flush=True)
    cap.release()
    return {"corners": corners.tolist(), "features": features}


def match_gt(gt: list[dict], predictions: list[dict], tolerance: int) -> list[dict | None]:
    """Greedy nearest matching in GT order, identical to run_eval_robust."""
    used, matched = set(), []
    for truth in gt:
        choices = [
            (abs(pred["frame"] - truth["frame"]), idx)
            for idx, pred in enumerate(predictions)
            if idx not in used and abs(pred["frame"] - truth["frame"]) <= tolerance
        ]
        if not choices:
            matched.append(None)
            continue
        _, idx = min(choices)
        used.add(idx)
        matched.append(predictions[idx])
    return matched


def mcnemar_exact(b: int, c: int) -> float:
    n = b + c
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, k) for k in range(min(b, c) + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def summarize(correct: dict[str, list[bool]], top2: dict[str, list[bool]], side: dict[str, list[bool]]) -> dict:
    n = len(next(iter(correct.values())))
    return {
        name: {
            "top1": sum(values) / n if n else 0.0,
            "top1_correct": sum(values),
            "top2": sum(top2[name]) / n if n else 0.0,
            "side_and_stroke": sum(side[name]) / n if n else 0.0,
        }
        for name, values in correct.items()
    } | {"gt_count": n}


def main() -> None:
    args = parse_args()
    models = dict(DEFAULT_MODELS)
    if args.model:
        models = dict(spec.split("=", 1) for spec in args.model)
    classifiers = {name: load_classifier(REPO_ROOT / path) for name, path in models.items()}
    clips = CLIPS_CONFIG
    if args.clips:
        wanted = {f"Clip {value.strip()}" for value in args.clips.split(",")}
        clips = [cfg for cfg in CLIPS_CONFIG if cfg["clip_id"] in wanted]

    cache_dir = args.feature_cache or args.output_dir / "feature_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    pipeline = None
    started = time.time()

    # Per-GT-hit correctness for every model, grouped by split for pooled stats.
    pooled = {
        group: {key: defaultdict(list) for key in ("top1", "top2", "side")}
        for group in ("all", "test", "val")
    }
    per_class = {name: defaultdict(lambda: [0, 0]) for name in classifiers}
    clip_results = []

    for cfg in clips:
        clip_started = time.time()
        cache_path = cache_dir / f"{cfg['video'].stem}.pt"
        events = load_events(cfg["pipeline_json"], args.frames)
        print(f"\n=== {cfg['clip_id']} {cfg['name']}: {len(events)} candidate hits", flush=True)
        if cache_path.exists():
            cached = torch.load(cache_path, map_location="cpu", weights_only=False)
            print("    loaded cached features", flush=True)
        else:
            if pipeline is None:
                pipeline = AlignedVideoPipeline(
                    ckpt_path=REPO_ROOT / DEFAULT_MODELS["base"], pose_model_path=args.pose_model,
                )
            cached = extract_clip_features(pipeline, cfg, events, args)
            torch.save(cached, cache_path)

        predictions = []
        for item in cached["features"]:
            feature = {
                key: value.to(DEVICE) if torch.is_tensor(value) else value
                for key, value in item["feature"].items()
            }
            row = {"frame": item["frame"], "hitter": item["hitter"], "models": {}}
            with torch.no_grad():
                for name, model in classifiers.items():
                    logits, side_logits, _ = model(
                        feature["emb"], feature["s_j"], feature["s_p"],
                        feature["s_s"], feature["s_c"], feature["s_q"],
                    )
                    top2_idx = logits.topk(2, dim=-1).indices[0].tolist()
                    row["models"][name] = {
                        "stroke": STROKE_CLASSES[top2_idx[0]],
                        "top2": [STROKE_CLASSES[k] for k in top2_idx],
                        "stroke_side": SIDE_CLASSES[int(side_logits.argmax(dim=-1).item())],
                    }
            predictions.append(row)

        gt = load_gt(args.manifest, cfg["match_id"], cfg["raw_start"], args.frames)
        matched = match_gt(gt, predictions, args.tolerance)
        group = "test" if cfg["match_id"] in TEST_MATCHES else "val"
        clip_top1, clip_top2, clip_side = defaultdict(list), defaultdict(list), defaultdict(list)
        for truth, pred in zip(gt, matched):
            for name in classifiers:
                out = pred["models"][name] if pred is not None else None
                ok = out is not None and out["stroke"] == truth["stroke"]
                ok2 = out is not None and truth["stroke"] in out["top2"]
                ok_side = ok and out["stroke_side"] == truth["stroke_side"]
                clip_top1[name].append(ok)
                clip_top2[name].append(ok2)
                clip_side[name].append(ok_side)
                per_class[name][truth["stroke"]][0] += ok
                per_class[name][truth["stroke"]][1] += 1
                for bucket in ("all", group):
                    pooled[bucket]["top1"][name].append(ok)
                    pooled[bucket]["top2"][name].append(ok2)
                    pooled[bucket]["side"][name].append(ok_side)

        summary = summarize(clip_top1, clip_top2, clip_side)
        clip_results.append({
            "clip_id": cfg["clip_id"], "name": cfg["name"], "match_id": cfg["match_id"],
            "split": group, "matched_hits": sum(pred is not None for pred in matched),
            "runtime_seconds": time.time() - clip_started, **summary,
        })
        line = "  ".join(f"{name}={summary[name]['top1'] * 100:.1f}%" for name in classifiers)
        print(f"    GT={len(gt)} matched={clip_results[-1]['matched_hits']}  {line}", flush=True)

    pooled_summary, significance = {}, {}
    for bucket, values in pooled.items():
        if not values["top1"]:
            continue
        pooled_summary[bucket] = summarize(values["top1"], values["top2"], values["side"])
        significance[bucket] = {}
        for first, second in combinations(classifiers, 2):
            a, b = values["top1"][first], values["top1"][second]
            only_first = sum(x and not y for x, y in zip(a, b))
            only_second = sum(y and not x for x, y in zip(a, b))
            significance[bucket][f"{first}_vs_{second}"] = {
                "only_first_correct": only_first, "only_second_correct": only_second,
                "p_value": mcnemar_exact(only_first, only_second),
            }

    result = {
        "protocol": "same_hit_candidates_same_features_top1_no_tta",
        "tolerance_frames": args.tolerance,
        "models": models,
        "runtime_seconds": time.time() - started,
        "pooled": pooled_summary,
        "mcnemar": significance,
        "per_class_top1": {
            name: {label: {"correct": c, "total": t, "recall": c / t} for label, (c, t) in classes.items()}
            for name, classes in per_class.items()
        },
        "clips": clip_results,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output_dir / "results.json"
    output_path.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print("\n" + "=" * 90)
    for bucket, summary in pooled_summary.items():
        print(f"[{bucket}] GT={summary['gt_count']}")
        for name in classifiers:
            stats = summary[name]
            print(f"  {name:<20} top1={stats['top1'] * 100:6.2f}%  top2={stats['top2'] * 100:6.2f}%"
                  f"  stroke+side={stats['side_and_stroke'] * 100:6.2f}%")
        for pair, stats in significance[bucket].items():
            print(f"    {pair:<42} {stats['only_first_correct']:>3} vs {stats['only_second_correct']:<3}"
                  f" p={stats['p_value']:.4f}")
    print(f"\nSaved {output_path}  ({result['runtime_seconds'] / 60:.1f} min)")


if __name__ == "__main__":
    main()
