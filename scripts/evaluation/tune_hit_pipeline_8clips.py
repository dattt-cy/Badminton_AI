"""Tune the production hit stage (scan stride, NMS, serve/bridge rescue) on the 8 clips.

The video is scanned once per clip at stride 2 and cached; stride 4 is simulated
exactly by keeping the centres the stride-4 scan would evaluate. Parameters are
selected on the validation matches (36, 38, 41) and then reported unchanged on
the test matches (39, 40, 42, 43, 44), next to the AI worker's current settings.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
for path in (REPO_ROOT, REPO_ROOT / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from ai_classifier.localization import (  # noqa: E402
    HitEvent, rescue_alternating_bridges, rescue_pre_rally_serves, two_stage_temporal_nms,
)
from scripts.evaluation.benchmark_8_matches_10min import CLIPS_CONFIG  # noqa: E402
from scripts.evaluation.compare_kap_checkpoints_8clips import TEST_MATCHES  # noqa: E402

CKPT = REPO_ROOT / "src" / "ai_classifier" / "models" / "checkpoints" / "hit_detector.pth"
# Mirrors backend/integrations/ai-worker/worker.py.
WORKER = {"stride": 4, "threshold": 0.75, "radius": 8, "same_side": 18, "serve": 0.40, "bridge": 0.55}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path("data/manifests/shuttleset_rgb.csv"))
    parser.add_argument("--cache-dir", type=Path, default=Path("work_dirs/hit_scan_cache"))
    parser.add_argument("--tolerance", type=int, default=12)
    parser.add_argument("--checkpoint", type=Path, default=CKPT)
    parser.add_argument("--scan-stride", type=int, default=2,
                        help="Scan stride; 2 allows simulating stride 4, 4 halves scan time.")
    parser.add_argument("--worker-only", action="store_true",
                        help="Only report the worker's settings, without the parameter search.")
    parser.add_argument(
        "--clips", type=str, default=None,
        help="Comma-separated clip numbers. Without validation clips, parameters are "
             "selected on the given clips themselves, which is optimistic.",
    )
    parser.add_argument("--output", type=Path, default=Path("work_dirs/hit_tuning_8clips.json"))
    return parser.parse_args()


def scan_clip(cfg: dict, cache_dir: Path, checkpoint: Path = CKPT, scan_stride: int = 2) -> tuple[list[HitEvent], int, int]:
    path = cache_dir / f"{cfg['video'].stem}_{checkpoint.parent.name}_{checkpoint.stem}_stride{scan_stride}.json"
    if not path.exists():
        import torch
        from scripts.inference.scan_video_hit_rgb import load_model, scan_video

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model, meta = load_model(checkpoint)
        model.to(device).eval()
        n_frames = int(meta.get("frames", 16))
        raw = scan_video(cfg["video"], model, n_frames, stride=scan_stride, batch_size=8,
                         device=device, use_amp=device.type == "cuda")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"n_frames": n_frames, "raw_events": [e.__dict__ for e in raw]}),
                        encoding="utf-8")
        del model
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload.setdefault("scan_stride", scan_stride)
    raw = [HitEvent(int(r["frame"]), r["side"], float(r["score"])) for r in payload["raw_events"]]
    return raw, payload["n_frames"] // 2, scan_stride


def load_gt(manifest: Path, cfg: dict) -> list[dict]:
    with manifest.open(encoding="utf-8-sig", newline="") as handle:
        rows = [r for r in csv.DictReader(handle) if r["match_id"] == str(cfg["match_id"])
                and cfg["raw_start"] <= int(float(r["hit_frame"])) < cfg["raw_end"]]
    side = {"top": "upper", "bottom": "lower"}
    return sorted(({"frame": int(float(r["hit_frame"])) - cfg["raw_start"], "side": side[r["player_side"]]}
                   for r in rows), key=lambda r: r["frame"])


def detect(raw: list[HitEvent], first_center: int, p: dict, scan_stride: int = 2) -> list[HitEvent]:
    """Exactly the hit stage of run_pipeline.analyze_badminton_video."""
    if p["stride"] % scan_stride:
        raise ValueError(f"cannot simulate stride {p['stride']} from a stride-{scan_stride} scan")
    if p["stride"] != scan_stride:
        raw = [e for e in raw if (e.frame - first_center) % p["stride"] == 0]
    events = two_stage_temporal_nms(raw, threshold=p["threshold"], radius=p["radius"],
                                    same_side_radius=p["same_side"])
    if p["serve"] is not None:
        events = rescue_pre_rally_serves(events, raw, min_serve_score=p["serve"],
                                         nms_radius=p["radius"], same_side_radius=p["same_side"])
    if p["bridge"] is not None:
        events = rescue_alternating_bridges(events, raw, min_score=p["bridge"],
                                            nms_radius=p["radius"], same_side_radius=p["same_side"])
    return events


def score(events: list[HitEvent], gt: list[dict], tolerance: int) -> dict:
    """Greedy nearest matching in GT order, as in eval_compare_hit_models."""
    used, matched, side_ok = set(), 0, 0
    for truth in gt:
        choices = [(abs(e.frame - truth["frame"]), i) for i, e in enumerate(events)
                   if i not in used and abs(e.frame - truth["frame"]) <= tolerance]
        if choices:
            _, i = min(choices)
            used.add(i); matched += 1; side_ok += events[i].side == truth["side"]
    return {"gt": len(gt), "pred": len(events), "matched": matched, "side_ok": side_ok}


def pooled(rows: list[dict]) -> dict:
    gt, pred, matched = (sum(r[k] for r in rows) for k in ("gt", "pred", "matched"))
    side_ok = sum(r["side_ok"] for r in rows)
    return {"gt": gt, "pred": pred, "matched": matched,
            "recall": matched / gt, "precision": matched / pred if pred else 0.0,
            "f1": 2 * matched / (gt + pred), "hitter_acc": side_ok / matched if matched else 0.0}


def evaluate(clips, p: dict, tolerance: int, split: str | None = None) -> dict:
    rows = [score(detect(c["raw"], c["first"], p, c["scan_stride"]), c["gt"], tolerance)
            for c in clips if split is None or c["split"] == split]
    return pooled(rows) if rows else None


def main() -> None:
    args = parse_args()
    configs = CLIPS_CONFIG
    if args.clips:
        wanted = {f"Clip {value.strip()}" for value in args.clips.split(",")}
        configs = [cfg for cfg in CLIPS_CONFIG if cfg["clip_id"] in wanted]
    clips = []
    for cfg in configs:
        print(f"[scan] {cfg['clip_id']} match {cfg['match_id']}", flush=True)
        raw, first, scan_stride = scan_clip(cfg, args.cache_dir, args.checkpoint, args.scan_stride)
        clips.append({"cfg": cfg, "raw": raw, "first": first, "scan_stride": scan_stride, "gt": load_gt(args.manifest, cfg),
                      "split": "test" if cfg["match_id"] in TEST_MATCHES else "val"})

    select = "val" if any(c["split"] == "val" for c in clips) else None
    best = dict(WORKER)
    if args.worker_only:
        report_and_save(args, clips, best)
        return

    # Stage 1: scan stride, threshold and NMS radii with the worker's rescue settings.
    stage1 = []
    for stride, thr, radius, same in itertools.product(
            (4, 2), (0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85), (6, 8, 10, 12), (12, 15, 18, 21, 24)):
        if same < radius:
            continue
        p = {**WORKER, "stride": stride, "threshold": thr, "radius": radius, "same_side": same}
        stage1.append((evaluate(clips, p, args.tolerance, select)["f1"], p))
    best = max(stage1, key=lambda item: item[0])[1]
    print(f"stage1 best on {select or 'given clips'}:", best, flush=True)

    # Stage 2: rescue thresholds around the stage-1 optimum.
    stage2 = []
    for serve, bridge in itertools.product((None, 0.30, 0.40, 0.50, 0.60), (None, 0.45, 0.55, 0.65)):
        p = {**best, "serve": serve, "bridge": bridge}
        stage2.append((evaluate(clips, p, args.tolerance, select)["f1"], p))
    best = max(stage2, key=lambda item: item[0])[1]
    print(f"stage2 best on {select or 'given clips'}:", best, flush=True)

    report_and_save(args, clips, best)


def report_and_save(args, clips, best: dict) -> None:
    report = {"checkpoint": str(args.checkpoint)}
    for name, p in (("worker", WORKER), ("tuned", best)):
        if name == "tuned" and p == WORKER:
            continue
        report[name] = {"params": p, **{split or "all": evaluate(clips, p, args.tolerance, split)
                                         for split in ("val", "test", None)}}
        report[name]["per_clip"] = {
            c["cfg"]["match_id"]: score(detect(c["raw"], c["first"], p, c["scan_stride"]), c["gt"], args.tolerance)
            for c in clips}
        for split in ("val", "test", "all"):
            r = report[name][split]
            if r is None:
                continue
            print(f"{name:7s} [{split:4s}] GT={r['gt']} pred={r['pred']} recall={r['recall']*100:.2f} "
                  f"precision={r['precision']*100:.2f} F1={r['f1']*100:.2f} hitter={r['hitter_acc']*100:.2f}")
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
