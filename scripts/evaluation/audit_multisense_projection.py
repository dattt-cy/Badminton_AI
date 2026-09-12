"""Audit per-joint projection error and render YOLO/MultiSense overlays."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np


JOINTS = (
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
)
LINKS = (
    ("left_shoulder", "right_shoulder"),
    ("left_shoulder", "left_elbow"), ("left_elbow", "left_wrist"),
    ("right_shoulder", "right_elbow"), ("right_elbow", "right_wrist"),
    ("left_shoulder", "left_hip"), ("right_shoulder", "right_hip"),
    ("left_hip", "right_hip"),
    ("left_hip", "left_knee"), ("left_knee", "left_ankle"),
    ("right_hip", "right_knee"), ("right_knee", "right_ankle"),
)


def subject_number(subject: str) -> int:
    digits = "".join(character for character in subject if character.isdigit())
    return int(digits) if digits else -1


def _torso_scale(row: dict[str, str]) -> float:
    shoulder = (
        np.asarray(_point_float(row, "left_shoulder", "yolo"))
        + np.asarray(_point_float(row, "right_shoulder", "yolo"))
    ) / 2
    hip = (
        np.asarray(_point_float(row, "left_hip", "yolo"))
        + np.asarray(_point_float(row, "right_hip", "yolo"))
    ) / 2
    return float(np.linalg.norm(shoulder - hip))


def _point_float(row: dict[str, str], joint: str, source: str) -> tuple[float, float]:
    return float(row[f"{joint}_{source}_x"]), float(row[f"{joint}_{source}_y"])


def _mapping_and_lag_audit(items: list[dict[str, str]]) -> dict[str, object]:
    required = "left_shoulder_yolo_x"
    if not items or required not in items[0]:
        return {"status": "unavailable"}
    normal, swapped = [], []
    for row in items:
        scale = _torso_scale(row)
        if scale <= 1e-6:
            continue
        for joint in ("right_elbow", "right_wrist"):
            yolo = np.asarray(_point_float(row, joint, "yolo"))
            normal.append(np.linalg.norm(yolo - _point_float(row, joint, "projected")) / scale)
            opposite = joint.replace("right_", "left_")
            swapped.append(np.linalg.norm(yolo - _point_float(row, opposite, "projected")) / scale)

    clips: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in items:
        clips[row["clip"]].append(row)
    best_lags, current_errors, best_errors = [], [], []
    for clip_rows in clips.values():
        clip_rows.sort(key=lambda row: int(row["frame"]))
        candidates = []
        for lag in range(-5, 6):
            errors = []
            for index, row in enumerate(clip_rows):
                projected_index = index + lag
                if not 0 <= projected_index < len(clip_rows):
                    continue
                scale = _torso_scale(row)
                if scale <= 1e-6:
                    continue
                projected_row = clip_rows[projected_index]
                for joint in ("right_elbow", "right_wrist"):
                    errors.append(np.linalg.norm(
                        np.asarray(_point_float(row, joint, "yolo"))
                        - np.asarray(_point_float(projected_row, joint, "projected"))
                    ) / scale)
            if errors:
                candidates.append((float(np.median(errors)), lag))
        if candidates:
            best_error, best_lag = min(candidates)
            current_error = next(error for error, lag in candidates if lag == 0)
            best_lags.append(best_lag)
            current_errors.append(current_error)
            best_errors.append(best_error)
    return {
        "normal_mapping_median_error": float(np.median(normal)),
        "swapped_mapping_median_error": float(np.median(swapped)),
        "swapped_mapping_better": float(np.median(swapped)) < float(np.median(normal)),
        "residual_temporal_audit": {
            "clip_count": len(best_lags),
            "median_best_lag_rows": float(np.median(best_lags)),
            "median_current_error": float(np.median(current_errors)),
            "median_best_shifted_error": float(np.median(best_errors)),
        },
    }


def residual_offset_calibration(
    rows: list[dict[str, str]], *, before_subject: int = 16
) -> dict[str, object]:
    """Estimate residual time offsets on training subjects only."""
    grouped: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        if subject_number(row["subject"]) < before_subject:
            grouped[(row["technique"], row["view"])].append(row)
    output = {}
    for (technique, view), items in sorted(grouped.items()):
        audit = _mapping_and_lag_audit(items)
        timing = audit.get("residual_temporal_audit")
        if not timing:
            continue
        per_clip: dict[str, list[dict[str, str]]] = defaultdict(list)
        for row in items:
            per_clip[row["clip"]].append(row)
        steps = []
        for clip_rows in per_clip.values():
            clip_rows.sort(key=lambda row: int(row["frame"]))
            if len(clip_rows) >= 2:
                steps.append(float(clip_rows[1]["timestamp_3d"]) - float(clip_rows[0]["timestamp_3d"]))
        seconds_per_row = float(np.median(steps)) if steps else 0.0
        output.setdefault(technique, {})[view] = {
            "median_best_lag_rows": timing["median_best_lag_rows"],
            "seconds_per_row": seconds_per_row,
            "residual_offset_seconds": timing["median_best_lag_rows"] * seconds_per_row,
            "calibration_clip_count": timing["clip_count"],
        }
    return output


def summarize(rows: list[dict[str, str]], *, holdout_subject: int = 16) -> dict:
    holdout = [row for row in rows if subject_number(row["subject"]) >= holdout_subject]
    grouped: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in holdout:
        grouped[(row["technique"], row["view"])].append(row)
    output = {}
    for (technique, view), items in sorted(grouped.items()):
        joints = {}
        for joint in JOINTS:
            values = np.asarray([
                float(row[f"{joint}_error_ratio"]) for row in items
                if math.isfinite(float(row[f"{joint}_error_ratio"]))
            ])
            joints[joint] = {
                "sample_count": len(values),
                "median_error_ratio": float(np.median(values)),
                "mean_error_ratio": float(np.mean(values)),
                "p90_error_ratio": float(np.percentile(values, 90)),
            }
        output.setdefault(technique, {})[view] = {
            "frame_count": len(items),
            "subject_count": len({row["subject"] for row in items}),
            "joints": joints,
            "arm_mapping_and_timing": _mapping_and_lag_audit(items),
        }
    return {
        "holdout_rule": f"subject number >= {holdout_subject}",
        "error_unit": "yolo_torso_length_ratio",
        "techniques": output,
        "training_residual_offset_calibration": residual_offset_calibration(
            rows, before_subject=holdout_subject
        ),
    }


def _point(row: dict[str, str], joint: str, source: str) -> tuple[int, int]:
    return (
        int(round(float(row[f"{joint}_{source}_x"]))),
        int(round(float(row[f"{joint}_{source}_y"]))),
    )


def render_overlay(row: dict[str, str], video: Path, output: Path) -> bool:
    capture = cv2.VideoCapture(str(video))
    try:
        capture.set(cv2.CAP_PROP_POS_FRAMES, int(row["frame"]))
        success, image = capture.read()
    finally:
        capture.release()
    if not success:
        return False
    for first, second in LINKS:
        cv2.line(image, _point(row, first, "yolo"), _point(row, second, "yolo"),
                 (0, 220, 0), 3, cv2.LINE_AA)
        cv2.line(image, _point(row, first, "projected"), _point(row, second, "projected"),
                 (220, 0, 220), 3, cv2.LINE_AA)
    for joint in JOINTS:
        cv2.circle(image, _point(row, joint, "yolo"), 5, (0, 255, 0), -1)
        cv2.circle(image, _point(row, joint, "projected"), 5, (255, 0, 255), -1)
    lines = (
        f"YOLO=green  projected MultiSense=magenta",
        f"{row['subject']} {row['technique']} {row['view']} stroke={row['stroke']} frame={row['frame']}",
        f"motion corr={float(row['motion_alignment_correlation']):.3f}  "
        f"camera median={float(row['camera_median_error_ratio']):.3f}",
        f"right elbow err={float(row['right_elbow_error_ratio']):.3f} torso  "
        f"right wrist err={float(row['right_wrist_error_ratio']):.3f} torso",
    )
    for index, text in enumerate(lines):
        cv2.putText(image, text, (20, 35 + index * 30), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (255, 255, 255), 4, cv2.LINE_AA)
        cv2.putText(image, text, (20, 35 + index * 30), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (20, 20, 20), 1, cv2.LINE_AA)
    output.parent.mkdir(parents=True, exist_ok=True)
    return bool(cv2.imwrite(str(output), image))


def select_overlay_rows(
    rows: list[dict[str, str]], *, per_group: int = 3
) -> list[dict[str, str]]:
    by_clip: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        by_clip[row["clip"]].append(row)
    clips_by_group: dict[tuple[str, str], list[list[dict[str, str]]]] = defaultdict(list)
    for items in by_clip.values():
        clips_by_group[(items[0]["technique"], items[0]["view"])].append(items)
    selected = []
    for _, clips in sorted(clips_by_group.items()):
        clips.sort(key=lambda items: (
            -float(items[0]["motion_alignment_correlation"]),
            float(items[0]["camera_median_error_ratio"]),
        ))
        for items in clips[:per_group]:
            items.sort(key=lambda row: int(row["frame"]))
            selected.append(items[len(items) // 2])
    return selected


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_csv", type=Path)
    parser.add_argument("output_json", type=Path)
    parser.add_argument("--pose-root", type=Path,
                        default=Path("data/processed/manual_clips_2class_poses"))
    parser.add_argument("--video-root", type=Path, default=Path("data/manual_clips"))
    parser.add_argument("--overlay-dir", type=Path,
                        default=Path("outputs/multisense_projection_overlays"))
    parser.add_argument("--overlays-per-group", type=int, default=3)
    args = parser.parse_args()
    with args.input_csv.open(newline="", encoding="utf-8-sig") as source:
        rows = list(csv.DictReader(source))
    report = summarize(rows)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    rendered = 0
    for row in select_overlay_rows(rows, per_group=args.overlays_per_group):
        pose = Path(row["clip"])
        try:
            relative = pose.relative_to(args.pose_root)
        except ValueError:
            continue
        video = (args.video_root / relative).with_suffix(".mov")
        name = (
            f"{row['subject']}_{row['technique']}_{row['view']}_"
            f"stroke{row['stroke']}_frame{int(row['frame']):04d}.jpg"
        )
        if video.is_file() and render_overlay(row, video, args.overlay_dir / name):
            rendered += 1
    print(f"Wrote joint audit: {args.output_json}")
    print(f"Rendered {rendered} overlays to {args.overlay_dir}")


if __name__ == "__main__":
    main()
