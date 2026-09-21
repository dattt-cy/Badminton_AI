"""Select balanced archive clips whose stroke and stroke-side are both correct."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
from collections import defaultdict
from pathlib import Path

import torch
from ultralytics import YOLO

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.inference.classify_shuttleset_rgb_multitask import (
    CLASS_PRIOR_WEIGHTS,
    auto_detect_player_side,
    constrain_side_probabilities,
    read_hitter_clip,
    video_metadata,
)
from scripts.training.train_shuttleset_rgb_multitask import MultiTaskR2Plus1D


VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--pose-model", type=Path, default=Path("models/checkpoints/pose/yolov8n-pose.pt"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/archive_joint_correct_30"))
    parser.add_argument("--per-label", type=int, default=5)
    parser.add_argument("--candidate-limit", type=int, default=30)
    parser.add_argument("--labels", nargs="*", help="Only evaluate these label folder names.")
    return parser.parse_args()


def expected_label(folder: Path) -> tuple[str, str]:
    side, stroke = folder.name.split("_", 1)
    return side, stroke


def spaced_candidates(paths: list[Path], limit: int) -> list[Path]:
    if len(paths) <= limit:
        return paths
    indices = [round(index * (len(paths) - 1) / (limit - 1)) for index in range(limit)]
    return [paths[index] for index in indices]


def main() -> None:
    args = parse_args()
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    stroke_classes = [str(value) for value in checkpoint["stroke_classes"]]
    side_classes = [str(value) for value in checkpoint["side_classes"]]
    frame_count = int(checkpoint.get("frames", 16))
    crop_padding = float(checkpoint.get("crop_padding", 0.30))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = MultiTaskR2Plus1D()
    model.load_state_dict(checkpoint["model"])
    model.to(device).eval()
    pose_model = YOLO(str(args.pose_model))

    rows: list[dict[str, object]] = []
    selected_by_label: dict[str, list[dict[str, object]]] = defaultdict(list)
    folders = sorted(
        path for path in args.archive.iterdir()
        if path.is_dir() and (not args.labels or path.name in args.labels)
    )
    for folder in folders:
        expected_side, expected_stroke = expected_label(folder)
        paths = sorted(
            path for path in folder.rglob("*")
            if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS
        )
        candidates = spaced_candidates(paths, args.candidate_limit)
        print(f"[{folder.name}] evaluating {len(candidates)}/{len(paths)} candidates", flush=True)
        for number, path in enumerate(candidates, 1):
            row: dict[str, object] = {
                "video": str(path.resolve()), "label": folder.name,
                "expected_stroke": expected_stroke, "expected_side": expected_side,
            }
            try:
                total_frames, _fps = video_metadata(path)
                player_side = auto_detect_player_side(path, pose_model)
                clip, metadata = read_hitter_clip(
                    path, frame_count, player_side, pose_model, crop_padding, 0, total_frames - 1
                )
                with torch.inference_mode(), torch.autocast(
                    device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"
                ):
                    stroke_logits, side_logits = model(clip.to(device))
                stroke_probs = torch.softmax(stroke_logits[0], dim=0).cpu()
                side_probs = torch.softmax(side_logits[0], dim=0).cpu()

                # Match the production inference calibration.
                if "serve" in stroke_classes:
                    stroke_probs[stroke_classes.index("serve")] *= 0.05
                    stroke_probs /= stroke_probs.sum()
                for index, name in enumerate(stroke_classes):
                    stroke_probs[index] /= CLASS_PRIOR_WEIGHTS.get(name, 1.0) ** 0.30
                stroke_probs /= stroke_probs.sum()
                x1, y1, x2, y2 = (float(value) for value in metadata["crop_box"])
                height = float(metadata["height"])
                center_y = (y1 + y2) / 2.0 / max(1.0, height)
                is_backcourt = (
                    (player_side == "bottom" and center_y >= 0.68)
                    or (player_side == "top" and center_y <= 0.32)
                )
                is_net = (
                    (player_side == "bottom" and center_y <= 0.52)
                    or (player_side == "top" and center_y >= 0.46)
                )
                if is_backcourt:
                    for index, name in enumerate(stroke_classes):
                        stroke_probs[index] *= (
                            2.2 if name in {"clear", "smash", "drop"}
                            else 0.25 if name in {"net_shot", "lift", "net_attack"}
                            else 1.0
                        )
                    stroke_probs /= stroke_probs.sum()
                elif is_net:
                    for index, name in enumerate(stroke_classes):
                        stroke_probs[index] *= (
                            2.0 if name in {"net_shot", "lift", "net_attack"}
                            else 0.20 if name in {"clear", "smash"}
                            else 1.0
                        )
                    stroke_probs /= stroke_probs.sum()
                if player_side == "top" and "forehand" in side_classes:
                    side_probs[side_classes.index("forehand")] *= 1.65
                elif player_side == "bottom" and "forehand" in side_classes:
                    side_probs[side_classes.index("forehand")] *= 1.25
                if "forehand" in side_classes and "aroundhead" in side_classes:
                    fh = side_classes.index("forehand")
                    ah = side_classes.index("aroundhead")
                    side_probs[fh] += side_probs[ah]
                    side_probs[ah] = 0.0
                side_probs /= side_probs.sum()

                predicted_stroke = stroke_classes[int(stroke_probs.argmax())]
                constrained_side, _ = constrain_side_probabilities(
                    side_probs, side_classes, predicted_stroke
                )
                predicted_side = side_classes[int(constrained_side.argmax())]
                stroke_confidence = float(stroke_probs.max())
                side_confidence = float(constrained_side.max())
                joint_correct = predicted_stroke == expected_stroke and predicted_side == expected_side
                row.update({
                    "predicted_stroke": predicted_stroke,
                    "predicted_side": predicted_side,
                    "stroke_confidence": stroke_confidence,
                    "side_confidence": side_confidence,
                    "joint_score": stroke_confidence * side_confidence,
                    "joint_correct": joint_correct,
                    "player_side": player_side,
                    "crop_box": metadata["crop_box"],
                })
                if joint_correct:
                    selected_by_label[folder.name].append(row)
            except Exception as error:  # Continue the audit and record corrupt clips.
                row.update({"joint_correct": False, "error": f"{type(error).__name__}: {error}"})
            rows.append(row)
            print(f"  {number:02d}/{len(candidates):02d} {path.name}: "
                  f"{row.get('predicted_side', 'ERROR')} {row.get('predicted_stroke', '')} "
                  f"joint={row.get('joint_correct', False)}", flush=True)

    selected: list[dict[str, object]] = []
    for folder in folders:
        ranked = sorted(selected_by_label[folder.name], key=lambda row: float(row["joint_score"]), reverse=True)
        selected.extend(ranked[:args.per_label])

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for row in selected:
        destination_dir = args.output_dir / "videos" / str(row["label"])
        destination_dir.mkdir(parents=True, exist_ok=True)
        source = Path(str(row["video"]))
        destination = destination_dir / source.name
        shutil.copy2(source, destination)
        row["selected_path"] = str(destination.resolve())

    report = {
        "archive": str(args.archive.resolve()),
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_epoch": int(checkpoint["epoch"]),
        "selection_rule": "top-1 stroke AND top-1 stroke_side match folder; ranked by product of confidences",
        "target_per_label": args.per_label,
        "candidate_limit_per_label": args.candidate_limit,
        "evaluated": len(rows),
        "joint_correct": sum(bool(row["joint_correct"]) for row in rows),
        "selected_count": len(selected),
        "selected": selected,
        "all_results": rows,
    }
    (args.output_dir / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    fields = [
        "label", "video", "selected_path", "expected_side", "expected_stroke",
        "predicted_side", "predicted_stroke", "side_confidence", "stroke_confidence", "joint_score",
    ]
    with (args.output_dir / "selected.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(selected)
    print(f"Selected {len(selected)} clips; report={args.output_dir / 'report.json'}", flush=True)
    if len(selected) < args.per_label * len(folders):
        print("WARNING: not every label had enough jointly correct candidates", flush=True)


if __name__ == "__main__":
    main()
