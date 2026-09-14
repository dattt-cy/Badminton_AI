"""Batch smoke-test the end-to-end technique preview across all view groups."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path


TECHNIQUES = ("forehand_clear", "backhand_drive")
VIEWS = ("front", "side")


def discover_cases(video_root: Path, pose_root: Path) -> dict[tuple[str, str], list[dict]]:
    groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for technique in TECHNIQUES:
        for view in VIEWS:
            videos = sorted(video_root.glob(f"*/{technique}/{view}/**/*.mov"))
            for video in videos:
                relative = video.relative_to(video_root)
                pose = (pose_root / relative).with_suffix(".npz")
                if pose.is_file():
                    groups[(technique, view)].append({
                        "video": video, "pose": pose,
                        "subject": relative.parts[3],
                    })
    return groups


def select_diverse(cases: list[dict], limit: int) -> list[dict]:
    """Take one case per subject before selecting additional clips."""
    by_subject: dict[str, list[dict]] = defaultdict(list)
    for case in cases:
        by_subject[str(case["subject"])].append(case)
    selected = []
    while len(selected) < limit and any(by_subject.values()):
        for subject in sorted(by_subject):
            if by_subject[subject] and len(selected) < limit:
                # Avoid a systematic seg01 bias: first clips are often truncated
                # setup trials in the manually cut archive.
                selected.append(
                    by_subject[subject].pop(len(by_subject[subject]) // 2)
                )
    return selected


def summarize(rows: list[dict]) -> dict:
    groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in rows:
        groups[(row["technique"], row["view"])].append(row)
    output, warnings = {}, []
    for (technique, view), items in sorted(groups.items()):
        successful = [item for item in items if item["success"]]
        criteria: dict[str, Counter] = defaultdict(Counter)
        for item in successful:
            for name, status in item["observable_statuses"].items():
                criteria[name][status] += 1
        criterion_summary = {}
        for name, counts in sorted(criteria.items()):
            total = sum(counts.values())
            criterion_summary[name] = {
                "count": total, "statuses": dict(sorted(counts.items())),
            }
            dominant_status, dominant_count = counts.most_common(1)[0]
            if total >= 3 and dominant_count / total >= 0.90:
                warnings.append({
                    "technique": technique, "view": view, "criterion": name,
                    "warning": "criterion_is_nearly_constant",
                    "dominant_status": dominant_status,
                    "dominant_ratio": dominant_count / total,
                })
        classified = [
            item for item in successful if item["classifier_status"] != "not_run"
        ]
        quality_reasons = Counter(
            reason for item in successful for reason in item["quality_reasons"]
        )
        output.setdefault(technique, {})[view] = {
            "case_count": len(items), "success_count": len(successful),
            "quality_pass_ratio": sum(
                item["geometry_feasible"] for item in successful
            ) / max(len(successful), 1),
            "mean_pose_valid_percent": sum(
                item["pose_valid_percent"] for item in successful
            ) / max(len(successful), 1),
            "classifier_case_count": len(classified),
            "classifier_confirmed_ratio": sum(
                item["classifier_status"] == "confirmed" for item in classified
            ) / max(len(classified), 1),
            "classifier_uncertain_ratio": sum(
                item["classifier_status"] == "uncertain" for item in classified
            ) / max(len(classified), 1),
            "criteria": criterion_summary,
            "quality_failure_reasons": dict(quality_reasons.most_common()),
        }
    return {"groups": output, "warnings": warnings}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video-root", type=Path, default=Path("data/manual_clips"))
    parser.add_argument(
        "--pose-root", type=Path,
        default=Path("data/processed/manual_clips_2class_poses"),
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path("outputs/batch_technique_analysis"),
    )
    parser.add_argument("--per-group", type=int, default=5)
    parser.add_argument("--run-classifier-wsl", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.per_group < 1:
        raise ValueError("--per-group must be positive")
    repo_root = Path(__file__).resolve().parents[2]
    video_root = (repo_root / args.video_root).resolve()
    pose_root = (repo_root / args.pose_root).resolve()
    output_dir = (repo_root / args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    groups = discover_cases(video_root, pose_root)
    selected = [
        (technique, view, case)
        for technique in TECHNIQUES for view in VIEWS
        for case in select_diverse(groups.get((technique, view), []), args.per_group)
    ]
    if not selected:
        raise RuntimeError("No matching manual video/pose cases were found")

    rows = []
    for index, (technique, view, case) in enumerate(selected, start=1):
        video, pose = Path(case["video"]), Path(case["pose"])
        case_name = f"{technique}_{view}_{case['subject']}_{video.stem}"
        case_name = "".join(
            char if char.isalnum() or char in "-_" else "_" for char in case_name
        )
        case_dir = output_dir / "cases" / case_name
        command = [
            sys.executable, str(repo_root / "scripts/inference/analyze_technique.py"),
            str(video), technique, "--view", view, "--pose", str(pose),
            "--output-dir", str(case_dir), "--no-preview",
        ]
        if args.run_classifier_wsl:
            command.append("--run-classifier-wsl")
        print(
            f"[{index}/{len(selected)}] {technique}/{view}/{case['subject']}: "
            f"{video.name}", flush=True,
        )
        completed = subprocess.run(command, cwd=repo_root, capture_output=True, text=True)
        analysis_path = case_dir / "analysis.json"
        if completed.returncode != 0 or not analysis_path.is_file():
            rows.append({
                "technique": technique, "view": view, "subject": case["subject"],
                "video": str(video), "success": False,
                "error": (completed.stderr or completed.stdout)[-1000:],
                "geometry_feasible": False, "pose_valid_percent": 0.0,
                "classifier_status": "failed" if args.run_classifier_wsl else "not_run",
                "observable_statuses": {}, "quality_reasons": ["analysis_command_failed"],
            })
            continue
        analysis = json.loads(analysis_path.read_text(encoding="utf-8"))
        user_report = json.loads((case_dir / "user_report.json").read_text(encoding="utf-8"))
        classifier = analysis.get("classifier_suggestion") or {}
        criteria = (
            analysis["strokes"][0].get("observable_criteria", [])
            if analysis.get("strokes") else []
        )
        rows.append({
            "technique": technique, "view": view, "subject": case["subject"],
            "video": str(video), "success": True, "error": "",
            "geometry_feasible": bool(analysis["quality"]["geometry_feasible"]),
            "quality_reasons": list(analysis["quality"].get("reasons", [])),
            "pose_valid_percent": float(user_report["video_quality"]["pose_valid_percent"]),
            "classifier_label": classifier.get("label", ""),
            "classifier_confidence": classifier.get("confidence", ""),
            "classifier_status": classifier.get("agreement_status", "not_run"),
            "overall": analysis["overall"],
            "observed_count": len(user_report["observed_signals"]),
            "needs_review_count": len(user_report["needs_review"]),
            "not_assessed_count": len(user_report["not_assessed"]),
            "observable_statuses": {
                item["criterion"]: item["status"] for item in criteria
            },
        })

    summary = summarize(rows)
    report = {
        "version": 1, "per_group_requested": args.per_group,
        "classifier_enabled": args.run_classifier_wsl,
        "case_count": len(rows),
        "successful_case_count": sum(row["success"] for row in rows),
        **summary,
    }
    (output_dir / "batch_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    csv_rows = []
    for row in rows:
        flat = {
            key: value for key, value in row.items()
            if key not in {"observable_statuses", "quality_reasons"}
        }
        flat["observable_statuses"] = json.dumps(
            row["observable_statuses"], ensure_ascii=False, sort_keys=True
        )
        flat["quality_reasons"] = json.dumps(
            row["quality_reasons"], ensure_ascii=False
        )
        csv_rows.append(flat)
    with (output_dir / "cases.csv").open("w", newline="", encoding="utf-8-sig") as output:
        fieldnames = sorted({key for row in csv_rows for key in row})
        writer = csv.DictWriter(output, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(csv_rows)
    print(
        f"Batch complete: {report['successful_case_count']}/{report['case_count']} "
        f"successful; warnings={len(report['warnings'])}"
    )
    print(f"Report: {output_dir / 'batch_report.json'}")


if __name__ == "__main__":
    main()
