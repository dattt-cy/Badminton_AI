"""Run the full production pipeline, with the AI worker's exact settings, on the 8 clips.

Unlike compare_kap_checkpoints_8clips (classifier A/B on frozen hit candidates),
this reruns everything a user upload goes through: hit detection, court
detection, pose, adaptive TTA classification, tactical decoding and OCR.
Outputs are cached per clip, so an interrupted run resumes where it stopped.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
for path in (REPO_ROOT, REPO_ROOT / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from run_pipeline import analyze_badminton_video  # noqa: E402
from scripts.evaluation.benchmark_8_matches_10min import CLIPS_CONFIG  # noqa: E402
from scripts.evaluation.compare_kap_checkpoints_8clips import TEST_MATCHES  # noqa: E402
from scripts.evaluation.eval_compare_hit_models import evaluate_predictions  # noqa: E402

CKPTS = REPO_ROOT / "src" / "ai_classifier" / "models" / "checkpoints"

# Mirrors backend/integrations/ai-worker/worker.py process_job().
WORKER_SETTINGS = dict(
    threshold=0.75, stride=4, batch_size=8, nms_radius=8, same_side_nms_radius=18,
    contact_refine_radius=2, shared_cache=True, use_amp=True, use_tta=True,
    full_tta=False, hit_rescue=True, serve_rescue_threshold=0.40,
    bridge_rescue_threshold=0.55,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kap-checkpoint", type=Path, default=CKPTS / "kap_classifier.pth")
    parser.add_argument("--hit-checkpoint", type=Path, default=CKPTS / "hit_detector.pth")
    parser.add_argument("--extra-hit-checkpoint", type=Path, default=None,
                        help="Second hit detector; each scan window keeps the higher of the two scores.")
    parser.add_argument("--threshold", type=float, default=WORKER_SETTINGS["threshold"])
    parser.add_argument("--alternation-gap", type=int, default=None,
                        help="Drop the weaker of two same-side hits closer than this many frames.")
    parser.add_argument("--clips", type=str, default=None, help="Comma-separated clip numbers, e.g. 3,1,2.")
    parser.add_argument("--min-shuttle-visibility", type=float, default=0.25,
                        help="Pipeline default; 0 keeps every candidate so thresholds can be swept offline.")
    parser.add_argument("--manifest", type=Path, default=Path("data/manifests/shuttleset_rgb.csv"))
    parser.add_argument("--tolerance", type=int, default=12)
    parser.add_argument("--output-dir", type=Path, default=Path("work_dirs/worker_pipeline_8clips"))
    return parser.parse_args()


def load_gt(manifest: Path, cfg: dict) -> list[dict]:
    gt = []
    with manifest.open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            frame = int(float(row["hit_frame"]))
            if row["match_id"] == str(cfg["match_id"]) and cfg["raw_start"] <= frame < cfg["raw_end"]:
                gt.append({
                    "frame": frame - cfg["raw_start"], "stroke": row["coarse_label"],
                    "player_side": row["player_side"], "stroke_side": row["stroke_side"],
                })
    return sorted(gt, key=lambda row: row["frame"])


def main() -> None:
    args = parse_args()
    clips = CLIPS_CONFIG
    if args.clips:
        order = [f"Clip {value.strip()}" for value in args.clips.split(",")]
        clips = sorted((c for c in CLIPS_CONFIG if c["clip_id"] in order), key=lambda c: order.index(c["clip_id"]))
    args.output_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for cfg in clips:
        output = args.output_dir / f"{cfg['video'].stem}.json"
        if not output.exists():
            print(f"\n>>> {cfg['clip_id']} {cfg['name']}: running worker pipeline", flush=True)
            started = time.time()
            analyze_badminton_video(
                video_path=cfg["video"], output_path=output,
                player_upper=cfg["player_upper"], player_lower=cfg["player_lower"],
                hit_checkpoint=args.hit_checkpoint, kap_checkpoint=args.kap_checkpoint,
                pose_model=CKPTS / "yolov8n-pose.pt", rally_model=CKPTS / "rally_boundary_detector.joblib",
                min_shuttle_visibility=args.min_shuttle_visibility, extra_hit_checkpoint=args.extra_hit_checkpoint,
                alternation_gap_frames=args.alternation_gap,
                **{**WORKER_SETTINGS, "threshold": args.threshold},
            )
            print(f"    done in {time.time() - started:.0f}s", flush=True)
        payload = json.loads(output.read_text(encoding="utf-8"))
        result = evaluate_predictions(payload["predictions"], load_gt(args.manifest, cfg), args.tolerance)
        result.pop("by_class", None)
        rows.append({"clip_id": cfg["clip_id"], "match_id": cfg["match_id"],
                     "split": "test" if cfg["match_id"] in TEST_MATCHES else "val",
                     "court_corners": payload.get("video", {}).get("court_corners"),
                     **{k: v for k, v in result.items() if not isinstance(v, (list, dict))}})
        r = rows[-1]
        print(f"{cfg['clip_id']} m{cfg['match_id']}: GT={r['ground_truth_hits']} pred={r['predicted_hits']} "
              f"hitF1={r['hit_f1']:.1f} e2e_top1={r['end_to_end_top1']:.1f} "
              f"stroke_side_acc={r['stroke_side_accuracy']:.1f}", flush=True)

    totals = {}
    for group in ("all", "test", "val"):
        subset = [r for r in rows if group == "all" or r["split"] == group]
        if not subset:
            continue
        gt = sum(r["ground_truth_hits"] for r in subset)
        pred = sum(r["predicted_hits"] for r in subset)
        matched = sum(r["matched_hits"] for r in subset)
        top1 = sum(r["end_to_end_top1"] * r["ground_truth_hits"] / 100 for r in subset)
        side = sum(r["stroke_side_accuracy"] * r["matched_hits"] / 100 for r in subset)
        totals[group] = {"gt": gt, "pred": pred, "matched": matched,
                         "hit_f1": 200 * matched / (gt + pred), "e2e_top1": 100 * top1 / gt,
                         "stroke_side_acc_matched": 100 * side / matched}
        t = totals[group]
        print(f"[{group}] GT={gt} hitF1={t['hit_f1']:.2f} e2e_top1={t['e2e_top1']:.2f} "
              f"stroke_side_acc={t['stroke_side_acc_matched']:.2f}")
    (args.output_dir / "summary.json").write_text(
        json.dumps({"kap_checkpoint": str(args.kap_checkpoint), "hit_checkpoint": str(args.hit_checkpoint),
                    "settings": WORKER_SETTINGS,
                    "totals": totals, "clips": rows}, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8")


if __name__ == "__main__":
    main()
