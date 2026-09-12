"""Run the complete phone-video technique preview with one command.

The user-confirmed technique and camera view always select the geometry rules.
An optional classifier JSON is recorded as a suggestion only and can never
override that selection.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path


VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv"}


def safe_stem(path: Path) -> str:
    value = re.sub(r"[^A-Za-z0-9_-]+", "_", path.stem).strip("_")
    return value or "video"


def run_command(command: list[str], *, cwd: Path) -> None:
    """Run one project CLI without shell-dependent quoting."""
    subprocess.run(command, cwd=cwd, check=True)


def assemble_analysis(
    *,
    video: Path,
    pose: Path,
    technique: str,
    view: str,
    handedness: str,
    pose_config: Path,
    quality: dict,
    technique_report: dict | None,
    preview: Path | None,
    classifier_suggestion: dict | None,
) -> dict:
    strokes = technique_report.get("strokes", []) if technique_report else []
    feedback = [stroke.get("user_feedback") for stroke in strokes]
    feedback = [item for item in feedback if item is not None]
    if not quality.get("geometry_feasible", False):
        overall = "insufficient_video_quality"
    elif not feedback:
        overall = "insufficient_data"
    elif any(item.get("overall") == "observable_good_signals" for item in feedback):
        overall = "observable_good_signals"
    else:
        overall = "review_available"
    classifier = dict(classifier_suggestion) if classifier_suggestion else None
    if classifier is not None:
        classifier["used_for_rules"] = False
        classifier["agrees_with_user_selection"] = (
            classifier.get("label") == technique
        )
        classifier["agreement_status"] = (
            "confirmed" if classifier.get("accepted")
            and classifier["agrees_with_user_selection"]
            else "conflict" if classifier.get("accepted")
            and not classifier["agrees_with_user_selection"]
            else "uncertain"
        )
        classifier["message"] = (
            "AI nhận diện trùng với kỹ thuật người dùng đã chọn."
            if classifier["agreement_status"] == "confirmed"
            else "AI nhận diện khác lựa chọn người dùng; hệ thống giữ lựa chọn người dùng."
            if classifier["agreement_status"] == "conflict"
            else "AI chưa đủ tự tin để xác nhận kỹ thuật."
        )
    good_count = sum(len(item.get("good_signals", [])) for item in feedback)
    review_count = sum(
        len(item.get("observations_for_review", [])) for item in feedback
    )
    heuristic_items = [
        criterion for item in feedback
        for criterion in item.get("heuristic_observations", [])
    ]
    unavailable_count = sum(len(item.get("not_assessed", [])) for item in feedback)
    technique_label = {
        "forehand_clear": "Forehand clear",
        "backhand_drive": "Backhand drive",
    }[technique]
    user_summary = {
        "technique": technique_label,
        "recognition": (
            classifier["message"] if classifier is not None
            else "Chưa chạy classifier; dùng kỹ thuật người dùng đã chọn."
        ),
        "video_quality": (
            "Video đủ chất lượng để phân tích hình học."
            if quality.get("geometry_feasible", False)
            else "Video chưa đủ chất lượng để phân tích hình học."
        ),
        "assessment": (
            "Có một số dấu hiệu tốt đã được kiểm định."
            if overall == "observable_good_signals"
            else "Có số liệu và quan sát để người dùng xem lại."
            if overall == "review_available"
            else "Chưa đủ dữ liệu để đưa ra quan sát kỹ thuật."
        ),
        "counts": {
            "validated_good_signals": good_count,
            "measurements_for_review": review_count,
            "heuristic_observed": sum(
                item.get("status") == "observed" for item in heuristic_items
            ),
            "heuristic_needs_review": sum(
                item.get("status") == "needs_review" for item in heuristic_items
            ),
            "not_assessed": unavailable_count,
        },
    }
    return {
        "version": 1,
        "mode": "experimental_phone_video_geometry_preview",
        "input_video": str(video),
        "selection": {
            "technique": technique,
            "view": view,
            "handedness": handedness,
            "source": "user_confirmed",
            "classifier_used_for_rules": False,
        },
        "classifier_suggestion": classifier,
        "user_summary": user_summary,
        "overall": overall,
        "score": None,
        "quality": quality,
        "strokes": strokes,
        "user_feedback": feedback,
        "artifacts": {
            "pose": str(pose),
            "pose_preview": str(preview) if preview else None,
            "technique_report": (
                str(technique_report.get("_path")) if technique_report else None
            ),
            "pose_config": str(pose_config),
        },
        "disclaimer": (
            "Bản thử nghiệm mô tả hình học quan sát được từ video; "
            "không phải điểm số hoặc kết luận chuyên môn."
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video", type=Path)
    parser.add_argument(
        "technique", choices=("forehand_clear", "backhand_drive")
    )
    parser.add_argument("--view", choices=("front", "side"), required=True)
    parser.add_argument("--handedness", choices=("left", "right"), default="right")
    parser.add_argument("--target", choices=("any", "single", "far", "near"), default="single")
    parser.add_argument(
        "--pose-config", type=Path,
        default=Path("configs/pose/yolov8_high_accuracy.yaml"),
    )
    parser.add_argument("--pose", type=Path, help="Reuse an existing pose NPZ")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--floor-angle-degrees", type=float, default=0.0)
    parser.add_argument(
        "--segmentation-mode", choices=("auto", "single", "multi"),
        default="single",
    )
    parser.add_argument("--swing-peak-frame", type=int)
    parser.add_argument("--classifier-json", type=Path)
    parser.add_argument(
        "--run-classifier-wsl", action="store_true",
        help="Run the 2-class motion-onset classifier on the extracted pose",
    )
    parser.add_argument(
        "--classifier-checkpoint", type=Path,
        default=Path(
            "models/checkpoints/action_recognition/"
            "stgcnpp_manual_clips_2class_motion_onset_best.pth"
        ),
    )
    parser.add_argument(
        "--classifier-config", type=Path,
        default=Path(
            "configs/action_recognition/experiments/"
            "stgcnpp_manual_clips_2class_motion_onset.py"
        ),
    )
    parser.add_argument("--min-action-confidence", type=float, default=0.75)
    parser.add_argument("--force-pose", action="store_true")
    parser.add_argument("--no-preview", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[2]
    video = args.video.resolve()
    if not video.is_file():
        raise FileNotFoundError(f"Video not found: {video}")
    if video.suffix.lower() not in VIDEO_EXTENSIONS:
        raise ValueError(f"Unsupported video extension: {video.suffix}")
    pose_config = (
        args.pose_config if args.pose_config.is_absolute()
        else repo_root / args.pose_config
    ).resolve()
    if not pose_config.is_file():
        raise FileNotFoundError(f"Pose config not found: {pose_config}")
    output_dir = (
        args.output_dir if args.output_dir else Path("outputs") / f"{safe_stem(video)}_analysis"
    )
    if not output_dir.is_absolute():
        output_dir = repo_root / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    pose = args.pose.resolve() if args.pose else output_dir / "pose.npz"
    if args.pose and not pose.is_file():
        raise FileNotFoundError(f"Pose not found: {pose}")
    if not pose.exists() or args.force_pose:
        run_command([
            sys.executable, str(repo_root / "scripts/inference/extract_pose.py"),
            str(video), str(pose), "--config", str(pose_config),
            "--target", args.target,
        ], cwd=repo_root)

    quality_path = output_dir / "quality.json"
    run_command([
        sys.executable,
        str(repo_root / "scripts/evaluation/check_geometry_feasibility.py"),
        str(pose), "--handedness", args.handedness,
        "--floor-angle-degrees", str(args.floor_angle_degrees),
        "--output", str(quality_path),
    ], cwd=repo_root)
    quality = json.loads(quality_path.read_text(encoding="utf-8"))

    preview = None
    if not args.no_preview:
        preview = output_dir / "pose_preview.mp4"
        run_command([
            sys.executable, str(repo_root / "scripts/inference/render_pose.py"),
            str(video), str(pose), str(preview),
        ], cwd=repo_root)

    report_path = output_dir / "technique_report.json"
    technique_report = None
    if quality.get("geometry_feasible", False):
        command = [
            sys.executable,
            str(repo_root / "scripts/evaluation/check_technique_rules.py"),
            str(pose), args.technique,
            "--view", args.view,
            "--handedness", args.handedness,
            "--floor-angle-degrees", str(args.floor_angle_degrees),
            "--segmentation-mode", args.segmentation_mode,
            "--output", str(report_path),
        ]
        if args.swing_peak_frame is not None:
            command.extend(["--swing-peak-frame", str(args.swing_peak_frame)])
        run_command(command, cwd=repo_root)
        if report_path.is_file():
            technique_report = json.loads(report_path.read_text(encoding="utf-8"))
            technique_report["_path"] = str(report_path)

    classifier_suggestion = None
    classifier_path = output_dir / "classification.json"
    if args.run_classifier_wsl:
        checkpoint = (
            args.classifier_checkpoint if args.classifier_checkpoint.is_absolute()
            else repo_root / args.classifier_checkpoint
        ).resolve()
        classifier_config = (
            args.classifier_config if args.classifier_config.is_absolute()
            else repo_root / args.classifier_config
        ).resolve()
        run_command([
            "powershell.exe", "-NoProfile", "-ExecutionPolicy", "Bypass",
            "-File", str(repo_root / "scripts/inference/classify_action_wsl.ps1"),
            "-InputPath", str(pose),
            "-Checkpoint", str(checkpoint),
            "-ActionConfig", str(classifier_config),
            "-Target", args.target,
            "-MinActionConfidence", str(args.min_action_confidence),
            "-JsonOutputPath", str(classifier_path),
        ], cwd=repo_root)
        classifier_suggestion = json.loads(
            classifier_path.read_text(encoding="utf-8")
        )
    elif args.classifier_json:
        classifier_path = args.classifier_json.resolve()
        classifier_suggestion = json.loads(classifier_path.read_text(encoding="utf-8"))

    analysis = assemble_analysis(
        video=video, pose=pose, technique=args.technique, view=args.view,
        handedness=args.handedness, pose_config=pose_config, quality=quality,
        technique_report=technique_report, preview=preview,
        classifier_suggestion=classifier_suggestion,
    )
    analysis["artifacts"]["classification"] = (
        str(classifier_path) if classifier_suggestion is not None else None
    )
    analysis_path = output_dir / "analysis.json"
    analysis_path.write_text(
        json.dumps(analysis, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Analysis complete: {analysis_path}")
    print(f"Overall: {analysis['overall']}  score: none")


if __name__ == "__main__":
    main()
