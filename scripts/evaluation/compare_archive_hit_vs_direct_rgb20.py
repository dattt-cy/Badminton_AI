"""Paired archive evaluation: final HIT detector vs direct epoch-20 inference."""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import cv2
import torch
from ultralytics import YOLO

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ai_classifier.localization import temporal_nms
from scripts.inference.classify_shuttleset_rgb_multitask import (
    auto_detect_player_side,
    constrain_side_probabilities,
    read_hitter_clip,
    read_tracked_hitter_clip,
)
from scripts.inference.scan_video_hit_rgb import load_model as load_hit_model, scan_video
from scripts.training.train_shuttleset_rgb_multitask import MultiTaskR2Plus1D
from scripts.training.train_hit_candidate_selector import candidate_features


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--per-label", type=int, default=5)
    parser.add_argument(
        "--sample-policy", choices=("even", "alternate"), default="even",
        help="alternate selects a disjoint deterministic set for confirmation testing.",
    )
    parser.add_argument("--hit-checkpoint", type=Path, required=True)
    parser.add_argument("--stroke-checkpoint", type=Path, required=True)
    parser.add_argument(
        "--pose-model", type=Path,
        default=Path("models/checkpoints/pose/yolov8n-pose.pt"),
    )
    parser.add_argument("--hit-stride", type=int, default=2)
    parser.add_argument("--hit-threshold", type=float, default=0.5)
    parser.add_argument(
        "--archive-target-player", choices=("auto", "top", "bottom"), default="auto",
        help=(
            "Known actor court side for archive evaluation only. This is not used "
            "by the production video timeline inference."
        ),
    )
    parser.add_argument(
        "--event-radius-frames", type=int, default=0,
        help=(
            "Frames before/after each predicted hit passed to the stroke classifier. "
            "Zero uses one second at the source FPS (the validated default)."
        ),
    )
    parser.add_argument(
        "--fallback-hit-threshold", type=float, default=0.35,
        help="Threshold used only when the primary threshold yields no event.",
    )
    parser.add_argument("--motion-track-crop", action="store_true")
    parser.add_argument(
        "--event-selection", choices=("midpoint", "quality", "learned"), default="midpoint"
    )
    parser.add_argument("--hit-selector", type=Path)
    return parser.parse_args()


def evenly_spaced(paths: list[Path], count: int) -> list[Path]:
    if len(paths) <= count:
        return paths
    if count <= 1:
        return [paths[len(paths) // 2]]
    indices = [round(index * (len(paths) - 1) / (count - 1)) for index in range(count)]
    return [paths[index] for index in indices]


def alternate_spaced(paths: list[Path], count: int) -> list[Path]:
    if len(paths) <= count:
        return paths
    fractions = [0.10, 0.30, 0.55, 0.80, 0.90]
    if count != 5:
        fractions = [(index + 0.5) / count for index in range(count)]
    return [paths[round(fraction * (len(paths) - 1))] for fraction in fractions]


def video_metadata(path: Path) -> tuple[int, float]:
    capture = cv2.VideoCapture(str(path))
    try:
        frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = float(capture.get(cv2.CAP_PROP_FPS))
    finally:
        capture.release()
    if frames <= 0 or fps <= 0:
        raise ValueError(f"Cannot inspect video: {path}")
    return frames, fps


def classify(
    video: Path,
    player_side: str,
    start_frame: int,
    end_frame: int,
    pose_model: YOLO,
    model: MultiTaskR2Plus1D,
    checkpoint: dict,
    device: torch.device,
    *,
    hit_frame: int | None = None,
    motion_track_crop: bool = False,
) -> dict[str, object]:
    if motion_track_crop and hit_frame is not None:
        clip, crop = read_tracked_hitter_clip(
            video, int(checkpoint.get("frames", 16)), player_side, pose_model,
            float(checkpoint.get("crop_padding", 0.55)), start_frame, end_frame, hit_frame,
            crop_size=int(checkpoint.get("crop_size", 112)),
        )
    else:
        clip, crop = read_hitter_clip(
            video, int(checkpoint.get("frames", 16)), player_side, pose_model,
            float(checkpoint.get("crop_padding", 0.55)), start_frame, end_frame,
            int(checkpoint.get("crop_size", 112)),
        )
    with torch.inference_mode(), torch.autocast(
        device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"
    ):
        stroke_logits, side_logits = model(clip.to(device))
    stroke_probs = torch.softmax(stroke_logits[0], dim=0).cpu()
    side_probs = torch.softmax(side_logits[0], dim=0).cpu()
    stroke_classes = list(checkpoint["stroke_classes"])
    side_classes = list(checkpoint["side_classes"])
    if "forehand" in side_classes and "aroundhead" in side_classes:
        forehand = side_classes.index("forehand")
        aroundhead = side_classes.index("aroundhead")
        side_probs[forehand] += side_probs[aroundhead]
        side_probs[aroundhead] = 0.0
        side_probs /= side_probs.sum()
    stroke = stroke_classes[int(stroke_probs.argmax())]
    side_probs, _ = constrain_side_probabilities(side_probs, side_classes, stroke)
    side = side_classes[int(side_probs.argmax())]
    stroke_values = torch.sort(stroke_probs, descending=True).values
    stroke_margin = float(stroke_values[0] - stroke_values[1])
    return {
        "predicted_stroke": stroke,
        "predicted_side": side,
        "stroke_confidence": float(stroke_probs.max()),
        "side_confidence": float(side_probs.max()),
        "player_side": player_side,
        "clip_range": [start_frame, end_frame],
        "crop_box": crop["crop_box"],
        "stroke_margin": stroke_margin,
        "tracking": bool(crop.get("tracking", False)),
        "track_frames": int(crop.get("track_frames", 0)),
        "track_motion": float(crop.get("track_motion", 0.0)),
    }


def candidate_quality(item: dict[str, object]) -> float:
    hit_score = float(item.get("hit_score", 0.0))
    confidence = float(item.get("stroke_confidence", 0.0))
    margin = float(item.get("stroke_margin", 0.0))
    track_frames = int(item.get("track_frames", 0))
    track_bonus = min(track_frames / 7.0, 1.0) if item.get("tracking") else 0.35
    # Confidence and top-2 margin measure classification stability; HIT and
    # trajectory coverage keep visually implausible candidates from winning.
    return hit_score * (0.55 * confidence + 0.30 * margin + 0.15 * track_bonus)


def selector_probability(features: list[float], selector: dict[str, object]) -> float:
    logit = float(selector["intercept"]) + sum(
        float(weight) * float(value)
        for weight, value in zip(selector["coefficient"], features)
    )
    return 1.0 / (1.0 + math.exp(-max(-50.0, min(50.0, logit))))


def score(rows: list[dict[str, object]], branch: str) -> dict[str, object]:
    valid = [row for row in rows if "error" not in row[branch]]
    stroke = sum(item[branch]["stroke_correct"] for item in valid)
    side = sum(item[branch]["side_correct"] for item in valid)
    joint = sum(item[branch]["joint_correct"] for item in valid)
    total = len(valid)
    return {
        "evaluated": total,
        "failures": len(rows) - total,
        "stroke_correct": stroke,
        "stroke_accuracy": stroke / total if total else 0.0,
        "side_correct": side,
        "side_accuracy": side / total if total else 0.0,
        "joint_correct": joint,
        "joint_accuracy": joint / total if total else 0.0,
    }


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    hit_model, hit_checkpoint = load_hit_model(args.hit_checkpoint)
    hit_model.to(device).eval()
    stroke_checkpoint = torch.load(args.stroke_checkpoint, map_location="cpu", weights_only=False)
    stroke_model = MultiTaskR2Plus1D()
    stroke_model.load_state_dict(stroke_checkpoint["model"])
    stroke_model.to(device).eval()
    pose_model = YOLO(str(args.pose_model))
    selector = (
        json.loads(args.hit_selector.read_text(encoding="utf-8"))
        if args.hit_selector is not None else None
    )
    if args.event_selection == "learned" and selector is None:
        raise ValueError("--event-selection learned requires --hit-selector")

    jobs = []
    for folder in sorted(path for path in args.archive.iterdir() if path.is_dir()):
        paths = sorted(folder.glob("*.mp4"))
        jobs.extend(
            alternate_spaced(paths, args.per_label)
            if args.sample_policy == "alternate"
            else evenly_spaced(paths, args.per_label)
        )

    rows: list[dict[str, object]] = []
    started = time.perf_counter()
    for index, video in enumerate(jobs, 1):
        expected_side, expected_stroke = video.parent.name.split("_", 1)
        total, fps = video_metadata(video)
        row: dict[str, object] = {
            "video": str(video.resolve()),
            "label": video.parent.name,
            "expected_side": expected_side,
            "expected_stroke": expected_stroke,
        }

        try:
            raw = scan_video(
                video,
                hit_model,
                int(hit_checkpoint.get("frames", 16)),
                stride=args.hit_stride,
                batch_size=8,
                device=device,
                use_amp=device.type == "cuda",
            )
            candidate_threshold = (
                float(selector["candidate_generation"]["hit_threshold"])
                if selector is not None else args.hit_threshold
            )
            candidate_radius = (
                int(selector["candidate_generation"]["nms_radius"])
                if selector is not None else max(4, round(fps * 0.5))
            )
            events = temporal_nms(
                raw,
                threshold=candidate_threshold,
                radius=candidate_radius,
                side_aware=True,
            )
            used_fallback = False
            if not events:
                events = temporal_nms(
                    raw,
                    threshold=args.fallback_hit_threshold,
                    radius=max(4, round(fps * 0.5)),
                    side_aware=True,
                )
                used_fallback = bool(events)
            if not events:
                raise RuntimeError(
                    "No HIT event above primary or fallback threshold"
                )
            if args.archive_target_player != "auto":
                wanted_event_side = (
                    "upper" if args.archive_target_player == "top" else "lower"
                )
                events = [event for event in events if event.side == wanted_event_side]
                if not events:
                    raise RuntimeError(
                        f"No HIT event for known archive actor {args.archive_target_player}"
                    )
            event_radius = (
                args.event_radius_frames
                if args.event_radius_frames > 0
                else max(1, round(fps * 1.0))
            )
            event_results = []
            learned_scores = {}
            if selector is not None:
                for event_index, event in enumerate(events):
                    learned_scores[(event.frame, event.side)] = selector_probability(
                        candidate_features(events, event_index), selector
                    )
            for event in events:
                event_result = classify(
                    video,
                    (
                        args.archive_target_player
                        if args.archive_target_player != "auto"
                        else ("top" if event.side == "upper" else "bottom")
                    ),
                    max(0, event.frame - event_radius),
                    min(total - 1, event.frame + event_radius),
                    pose_model,
                    stroke_model,
                    stroke_checkpoint,
                    device,
                    hit_frame=event.frame,
                    motion_track_crop=args.motion_track_crop,
                )
                event_result.update({
                    "event_frame": event.frame,
                    "event_time_seconds": event.frame / fps,
                    "hit_score": event.score,
                    "low_confidence_fallback": used_fallback,
                    "selector_probability": learned_scores.get((event.frame, event.side)),
                })
                event_results.append(event_result)
            # Archive folders provide one clip label but no target event frame.
            # Keep every event for production-style output; midpoint selection is
            # used only to compute the legacy one-label-per-clip comparison.
            if args.event_selection == "learned":
                hit_result = max(
                    event_results,
                    key=lambda item: float(item.get("selector_probability") or 0.0),
                ).copy()
            elif args.event_selection == "quality":
                for item in event_results:
                    item["selection_score"] = candidate_quality(item)
                hit_result = max(event_results, key=candidate_quality).copy()
            else:
                hit_result = min(
                    event_results,
                    key=lambda item: (abs(item["event_frame"] - total / 2), -item["hit_score"]),
                ).copy()
            hit_result.update({
                "detected_events": len(event_results),
                "selection_policy": args.event_selection,
                "event_predictions": event_results,
            })
            row["with_hit"] = hit_result
        except Exception as error:
            row["with_hit"] = {"error": f"{type(error).__name__}: {error}"}

        try:
            direct_side = (
                args.archive_target_player
                if args.archive_target_player != "auto"
                else auto_detect_player_side(video, pose_model)
            )
            row["without_hit"] = classify(
                video, direct_side, 0, total - 1,
                pose_model, stroke_model, stroke_checkpoint, device,
            )
        except Exception as error:
            row["without_hit"] = {"error": f"{type(error).__name__}: {error}"}

        for branch in ("with_hit", "without_hit"):
            result = row[branch]
            if "error" not in result:
                result["stroke_correct"] = result["predicted_stroke"] == expected_stroke
                result["side_correct"] = result["predicted_side"] == expected_side
                result["joint_correct"] = result["stroke_correct"] and result["side_correct"]
        rows.append(row)
        print(
            f"[{index:02d}/{len(jobs):02d}] {video.parent.name}/{video.name} "
            f"HIT={row['with_hit'].get('predicted_side', 'ERR')} "
            f"{row['with_hit'].get('predicted_stroke', '')} "
            f"DIRECT={row['without_hit'].get('predicted_side', 'ERR')} "
            f"{row['without_hit'].get('predicted_stroke', '')}",
            flush=True,
        )

    report = {
        "sample_policy": f"{args.per_label} {args.sample_policy} clips per label",
        "sample_count": len(rows),
        "hit_checkpoint": str(args.hit_checkpoint.resolve()),
        "hit_checkpoint_epoch": hit_checkpoint.get("epoch"),
        "stroke_checkpoint": str(args.stroke_checkpoint.resolve()),
        "stroke_checkpoint_epoch": stroke_checkpoint.get("epoch"),
        "event_radius_frames": args.event_radius_frames,
        "event_radius_policy": (
            "fixed_frames" if args.event_radius_frames > 0 else "one_second_at_source_fps"
        ),
        "primary_hit_threshold": args.hit_threshold,
        "fallback_hit_threshold": args.fallback_hit_threshold,
        "archive_target_player": args.archive_target_player,
        "motion_track_crop": args.motion_track_crop,
        "event_selection": args.event_selection,
        "hit_selector": str(args.hit_selector.resolve()) if args.hit_selector else None,
        "with_hit": score(rows, "with_hit"),
        "without_hit": score(rows, "without_hit"),
        "elapsed_seconds": round(time.perf_counter() - started, 3),
        "results": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: report[key] for key in ("with_hit", "without_hit", "elapsed_seconds")}, indent=2))


if __name__ == "__main__":
    main()
