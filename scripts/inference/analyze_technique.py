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
TECHNIQUE_LABELS = {
    "forehand_clear": "Forehand clear (phông cầu thuận tay)",
    "backhand_drive": "Backhand drive (đánh cầu ngang trái tay)",
}
PHASE_LABELS = {
    "preparation": "Chuẩn bị",
    "backswing": "Vung vợt ra sau",
    "forward_swing": "Tăng tốc",
    "contact_estimated": "Gần thời điểm đánh cầu",
    "follow_through": "Theo đà",
}
VIEW_LABELS = {"front": "Chính diện", "side": "Góc bên"}
HAND_LABELS = {"left": "Trái", "right": "Phải"}
IMPROVEMENT_TIPS = {
    "contact_above_head": "Cố gắng tiếp xúc cầu ở vị trí cao và hơi phía trước vai.",
    "contact_ahead_of_body": "Đưa điểm tiếp xúc ra phía trước cơ thể rõ hơn.",
    "non_racket_arm_coordination": "Dùng tay không thuận để định hướng cầu và giữ thăng bằng.",
    "leg_loading": "Chùng gối thêm khi chuẩn bị để hỗ trợ phát lực.",
    "body_transfer": "Phối hợp chuyển trọng tâm và xoay thân rõ hơn.",
    "followthrough_completion": "Tiếp tục vung vợt tự nhiên sau khi tiếp xúc cầu.",
    "racket_arm_preparation": "Đưa bàn tay cầm vợt lên gần ngang vai trước khi bắt đầu tăng tốc.",
    "arm_extension_excursion": "Tạo chuỗi gập rồi duỗi tay rõ ràng, không khóa cứng khuỷu.",
    "arm_reach_excursion": "Thu tay khi chuẩn bị và vươn tay rõ hơn khi đánh cầu.",
    "recovery_balance": "Kết thúc với trọng tâm nằm giữa hai chân để phục hồi nhanh.",
}
PRACTICE_DRILLS = {
    "racket_arm_preparation": [
        "Dừng lại ở tư thế chuẩn bị trước khi vung.",
        "Đưa bàn tay cầm vợt lên gần ngang vai.",
        "Thực hiện chậm 5 lần, sau đó quay lại để so sánh.",
    ],
    "contact_above_head": [
        "Đứng dưới điểm cầu tưởng tượng và đưa tay lên cao.",
        "Vươn người thoải mái, không khóa cứng khuỷu tay.",
        "Thực hiện chậm 5 lần rồi mới tăng tốc.",
    ],
    "leg_loading": [
        "Đứng với hai chân rộng vừa phải.",
        "Chùng nhẹ hai gối trước khi bắt đầu vung.",
        "Duỗi chân cùng lúc tay vợt tăng tốc, lặp lại 5 lần.",
    ],
}
OBSERVABLE_LABELS = {
    "arm_extension_excursion": "Tay co rồi duỗi khi vung",
    "arm_reach_excursion": "Tay có thu–vươn rõ",
    "followthrough_completion": "Vung theo đà",
    "racket_arm_preparation": "Tư thế tay cầm vợt",
}
OBSERVABLE_MESSAGES = {
    "racket_arm_preparation": "Bàn tay cầm vợt được đưa vào vị trí sẵn sàng.",
    "followthrough_completion": "Tay đánh tiếp tục di chuyển rõ sau vùng tăng tốc ước tính.",
}
REVIEW_LABELS = {
    "wrist_vertical_excursion": "Đường vung từ thấp lên cao",
}
FRIENDLY_VALIDATED_SIGNALS = {
    "torso_lean": (
        "Kiểm soát thân người",
        "Thân người không nghiêng quá nhiều khi tăng tốc.",
    ),
    "stance_width": (
        "Khoảng cách hai chân",
        "Hai chân tạo được vùng đứng ổn định khi chuẩn bị.",
    ),
}
REVIEW_TIPS = {
    ("wrist_shoulder_distance", "preparation"):
        "Kiểm tra xem tay vợt đã được đưa lên sớm và thoải mái chưa.",
    ("wrist_shoulder_distance", "contact_estimated"):
        "Kiểm tra xem tay đánh có vươn thoải mái ở vùng tiếp xúc không.",
    ("wrist_shoulder_distance", "follow_through"):
        "Kiểm tra xem tay vợt có tiếp tục đi hết theo đà sau cú đánh không.",
    ("wrist_vertical_excursion", None):
        "Kiểm tra xem đường vung có đi từ thấp lên cao đủ rõ không.",
}


def safe_stem(path: Path) -> str:
    value = re.sub(r"[^A-Za-z0-9_-]+", "_", path.stem).strip("_")
    return value or "video"


def run_command(command: list[str], *, cwd: Path) -> None:
    """Run one project CLI without shell-dependent quoting."""
    subprocess.run(command, cwd=cwd, check=True)


def _display_value(value: object, unit: str | None) -> str | None:
    if value is None:
        return None
    number = float(value)
    if unit == "degree":
        return f"{number:.1f}°"
    if unit in {"torso_height_ratio", "body_scale_ratio"}:
        # Normalized distances are useful for audit/model logic but are not
        # meaningful coaching units for a recreational player.
        return None
    return f"{number:.2f}"


def _display_observable_value(item: dict) -> str | None:
    value = item.get("observed")
    if value is None:
        return None
    number = float(value)
    criterion = item["criterion"]
    if criterion == "contact_above_head":
        return "ở trên đầu" if number >= 0 else "chưa ở trên đầu"
    if criterion == "contact_ahead_of_body":
        return "ở phía trước vai" if number >= 0 else "vẫn ở phía sau vai"
    if criterion == "racket_arm_preparation":
        return "cao hơn vai" if number >= 0 else "thấp hơn vai"
    if criterion == "non_racket_arm_coordination" and number < 0:
        return "chưa phối hợp đúng nhịp"
    if criterion == "recovery_balance" and number <= 0:
        return "trọng tâm nằm trong vùng hai chân"
    return _display_value(value, item.get("unit"))


def _phase_evidence(
    stroke: dict, phase: str | None, fps: float,
) -> dict | None:
    phases = stroke.get("phases", {})
    frame_range = phases.get(phase) if phase else None
    if not frame_range or len(frame_range) != 2 or fps <= 0:
        return None
    offset = int(stroke.get("start_frame", 0))
    frame = offset + int(round((frame_range[0] + frame_range[1] - 1) / 2))
    return {
        "frame": frame,
        "seconds": round(frame / fps, 2),
        "label": f"Xem chậm đoạn {frame / fps:.2f}s",
    }


def _criterion_evidence(criterion: str, stroke: dict, fps: float) -> dict | None:
    phase = {
        "contact_above_head": "contact_estimated",
        "contact_ahead_of_body": "contact_estimated",
        "contact_drive_height": "contact_estimated",
        "non_racket_arm_coordination": "backswing",
        "leg_loading": "preparation",
        "racket_arm_preparation": "preparation",
        "arm_extension_excursion": "forward_swing",
        "arm_reach_excursion": "forward_swing",
        "body_transfer": "forward_swing",
        "followthrough_completion": "follow_through",
        "recovery_balance": "follow_through",
    }.get(criterion)
    return _phase_evidence(stroke, phase, fps)


def _measurement_evidence(
    feature: str | None, phase: str | None, stroke: dict, fps: float,
) -> dict | None:
    evidence_phase = phase or {
        "wrist_vertical_excursion": "forward_swing",
        "wrist_path_length": "forward_swing",
    }.get(feature)
    return _phase_evidence(stroke, evidence_phase, fps)


def build_user_report(analysis: dict) -> dict:
    """Reduce the technical analysis to one plain-language frontend payload."""
    classifier = analysis.get("classifier_suggestion")
    selected_by_user = (
        analysis["selection"].get("source", "user_confirmed") == "user_confirmed"
    )
    quality = analysis["quality"]
    feedback = analysis.get("user_feedback", [])
    observed, needs_review, measurements, unavailable = [], [], [], []
    fps = float(quality.get("fps") or 0)
    strokes = analysis.get("strokes", [])
    for block_index, block in enumerate(feedback):
        stroke = strokes[block_index] if block_index < len(strokes) else {}
        for item in block.get("good_signals", []):
            friendly = FRIENDLY_VALIDATED_SIGNALS.get(item.get("feature"))
            observed.append({
                "label": friendly[0] if friendly else item["label"],
                "message": friendly[1] if friendly else item["message"],
                "value": _display_value(item.get("observed"), item.get("unit")),
                "evidence": "validated",
                "moment": _phase_evidence(stroke, item.get("phase"), fps),
            })
        for item in block.get("heuristic_observations", []):
            target = observed if item["status"] == "observed" else needs_review
            target.append({
                "criterion": item["criterion"],
                "label": OBSERVABLE_LABELS.get(item["criterion"], item["label"]),
                "message": OBSERVABLE_MESSAGES.get(
                    item["criterion"], item["message"]
                ),
                "value": _display_observable_value(item),
                "evidence": "visual_heuristic",
                "moment": _criterion_evidence(item["criterion"], stroke, fps),
                **(
                    {
                        "suggestion": IMPROVEMENT_TIPS.get(item["criterion"]),
                        "drill": PRACTICE_DRILLS.get(item["criterion"], []),
                    }
                    if item["status"] == "needs_review" else {}
                ),
            })
        for item in block.get("observations_for_review", []):
            measurements.append({
                "feature": item.get("feature"),
                "label": REVIEW_LABELS.get(item.get("feature"), item["label"]),
                "phase": PHASE_LABELS.get(item.get("phase"), item.get("phase")),
                "value": _display_value(item.get("observed"), item.get("unit")),
                "note": "AI chưa đủ chắc để kết luận đây là lỗi.",
                "suggestion": REVIEW_TIPS.get(
                    (item.get("feature"), item.get("phase")),
                    "Hãy xem lại video pose hoặc quay thêm một clip rõ hơn.",
                ),
                "moment": _measurement_evidence(
                    item.get("feature"), item.get("phase"), stroke, fps
                ),
            })
        for item in block.get("not_assessed", []):
            unavailable.append({
                "label": item["label"],
                "phase": PHASE_LABELS.get(item.get("phase"), item.get("phase")),
                "value": _display_value(item.get("observed"), item.get("unit")),
                "reason": item["reason"],
            })

    # A positive/negative preparation-hand observation and an uncertain reach
    # measurement describe the same visible moment to a recreational player.
    # Keep the actionable observation and remove the apparently contradictory
    # duplicate from the uncertainty list.
    if any(
        item.get("criterion") == "racket_arm_preparation"
        for item in observed + needs_review
    ):
        measurements = [
            item for item in measurements
            if not (
                item.get("feature") == "wrist_shoulder_distance"
                and item.get("phase") == PHASE_LABELS["preparation"]
            )
        ]

    motion = quality.get("motion_proposals", [])
    strokes = analysis.get("strokes", [])
    phases = strokes[0].get("phases", {}) if strokes else {}
    contact_frame = phases.get("contact_frame")
    fps = quality.get("fps")
    peak_time = (
        float(contact_frame) / float(fps)
        if contact_frame is not None and fps
        else max(motion, key=lambda item: item.get("peak_score", 0)).get("peak_time")
        if motion else None
    )
    predicted_label = (
        TECHNIQUE_LABELS.get(classifier.get("label"), classifier.get("label"))
        if classifier else None
    )
    classifier_confidence = (
        round(float(classifier["confidence"]) * 100, 2) if classifier else None
    )
    if classifier:
        recognition_message = (
            f"AI tự nhận diện {predicted_label} ({classifier_confidence:.2f}%)."
            if not selected_by_user and classifier.get("accepted")
            else
            f"AI nhận diện {predicted_label} ({classifier_confidence:.2f}%), "
            "trùng với kỹ thuật đã chọn."
            if classifier.get("agreement_status") == "confirmed"
            else f"AI nhận diện {predicted_label} ({classifier_confidence:.2f}%) "
            "khác lựa chọn; hệ thống vẫn phân tích theo lựa chọn của bạn."
            if classifier.get("agreement_status") == "conflict"
            else f"AI chưa đủ tự tin; dự đoán cao nhất là {predicted_label} "
            f"({classifier_confidence:.2f}%)."
        )
    else:
        recognition_message = (
            "Chưa chạy AI nhận diện; phân tích theo kỹ thuật người dùng đã chọn."
        )
    recognition = {
        "status": classifier.get("agreement_status") if classifier else "not_run",
        "message": recognition_message,
        "prediction": predicted_label,
        "confidence_percent": classifier_confidence,
        "scores": classifier.get("scores") if classifier else None,
    }
    if quality.get("geometry_feasible"):
        quality_message = "Video đủ chất lượng để phân tích hình học."
    else:
        quality_message = "Video chưa đủ chất lượng; hãy quay lại theo hướng dẫn."
    if not quality.get("geometry_feasible"):
        summary = "Video này chưa đủ rõ để AI đưa ra hướng dẫn kỹ thuật đáng tin cậy."
    elif needs_review:
        summary = (
            "Trong lần tập tiếp theo, hãy tập trung vào một việc: "
            f"{needs_review[0]['suggestion']}"
        )
    elif observed:
        summary = "Không phát hiện điểm cần ưu tiên sửa từ góc quay này."
    else:
        summary = "AI chưa có đủ bằng chứng để đưa ra hướng dẫn kỹ thuật."
    primary_focus = None
    if needs_review:
        primary_focus = {
            "label": needs_review[0]["label"],
            "observation": needs_review[0]["message"],
            "action": needs_review[0].get("suggestion"),
            "moment": needs_review[0].get("moment"),
            "drill": needs_review[0].get("drill", []),
        }
    view_check = analysis.get("camera_view_check") or {}
    detected_view = view_check.get("detected_body_projection")
    selected_view = analysis["selection"]["view"]
    view_warning = (
        "Video có góc quay chéo; một số phép đo cần camera chính diện hoặc "
        "góc bên rõ ràng nên sẽ không được dùng để kết luận."
        if detected_view == "oblique"
        else "Góc quay đã chọn không khớp hướng cơ thể quan sát được trong video."
        if detected_view in {"front", "side"} and detected_view != selected_view
        else None
    )
    return {
        "version": 4,
        "title": f"Kết quả phân tích {TECHNIQUE_LABELS[analysis['selection']['technique']]}",
        "status": {
            "code": analysis["overall"],
            "label": (
                "Đã phân tích — có nội dung cần xem lại"
                if needs_review else {
                "observable_good_signals": "Có dấu hiệu tốt quan sát được",
                "review_available": (
                    "Đã phân tích — có nội dung cần xem lại"
                    if needs_review else "Đã phân tích"
                ),
                "insufficient_video_quality": "Video chưa đủ chất lượng",
                "insufficient_data": "Chưa đủ dữ liệu để phân tích",
                }.get(analysis["overall"], "Đã phân tích")
            ),
        },
        "technique": {
            "selected": analysis["selection"]["technique"],
            "label": TECHNIQUE_LABELS[analysis["selection"]["technique"]],
            "view": analysis["selection"]["view"],
            "view_label": VIEW_LABELS[analysis["selection"]["view"]],
            "handedness": analysis["selection"]["handedness"],
            "handedness_label": HAND_LABELS[analysis["selection"]["handedness"]],
            "selected_by_user": selected_by_user,
        },
        "recognition": recognition,
        "video_quality": {
            "passed": bool(quality.get("geometry_feasible")),
            "message": quality_message,
            "pose_valid_percent": round(
                float(quality.get("trajectory_quality", {}).get("essential_valid_ratio", 0)) * 100,
                2,
            ),
            "player_scale_px": round(
                float(quality.get("trajectory_quality", {}).get("median_scale_px", 0)), 1
            ),
            "swing_peak_seconds": round(float(peak_time), 2) if peak_time is not None else None,
            "fps": round(float(quality.get("fps") or 30.0), 3),
            "issues": quality.get("reasons", []),
            "view_warning": view_warning,
        },
        "summary": summary,
        "primary_focus": primary_focus,
        "measurement_note": (
            "Các số đo chuyên môn được giữ trong báo cáo kỹ thuật; phần này chỉ "
            "trình bày những điều người chơi có thể áp dụng."
        ),
        "observed_signals": observed,
        "needs_review": needs_review,
        "reference_measurements": measurements,
        "not_assessed": unavailable,
        "score": None,
        "disclaimer": analysis["disclaimer"],
    }


def render_user_report_markdown(report: dict) -> str:
    """Render the user payload as a readable Vietnamese handoff."""
    technique = report["technique"]
    quality = report["video_quality"]
    recognition = report["recognition"]
    lines = [
        f"# {report['title']}", "",
        f"**Trạng thái:** {report['status']['label']}", "",
        f"- {'Kỹ thuật đã chọn' if technique['selected_by_user'] else 'Kỹ thuật AI nhận diện'}: {technique['label']}",
        f"- Góc quay: {technique['view_label']}",
        f"- Tay thuận: {technique['handedness_label']}",
        f"- AI nhận diện: {recognition['message']}",
        f"- Chất lượng video: {quality['message']}",
        f"- Thời điểm vung nhanh nhất: {quality['swing_peak_seconds']} giây"
        if quality["swing_peak_seconds"] is not None
        else "- Thời điểm vung nhanh nhất: chưa xác định",
        "", report["summary"], "",
    ]
    sections = (
        ("Dấu hiệu quan sát được", report["observed_signals"], "message"),
        ("Nên cải thiện", report["needs_review"], "message"),
        ("Cần quan sát thêm", report["reference_measurements"][:2], "suggestion"),
        ("Giới hạn của lần phân tích", report["not_assessed"], "reason"),
    )
    for title, items, message_key in sections:
        lines.extend([f"## {title}", ""])
        if not items:
            lines.extend(["Không có.", ""])
            continue
        for item in items:
            details = []
            if item.get("phase"):
                details.append(str(item["phase"]))
            suffix = f" ({'; '.join(details)})" if details else ""
            lines.append(f"- **{item['label']}**{suffix}: {item[message_key]}")
            if item.get("suggestion") and message_key != "suggestion":
                lines.append(f"  Cách cải thiện: {item['suggestion']}")
        lines.append("")
    lines.extend(["---", "", report["disclaimer"], ""])
    return "\n".join(lines)


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
    selection_source: str = "user_confirmed",
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
            "source": selection_source,
            "classifier_used_for_rules": False,
        },
        "classifier_suggestion": classifier,
        "user_summary": user_summary,
        "overall": overall,
        "score": None,
        "quality": quality,
        "camera_view_check": (
            technique_report.get("view_selection") if technique_report else None
        ),
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
        "technique", nargs="?",
        choices=("auto", "forehand_clear", "backhand_drive"), default="auto",
        help="Technique to assess; auto uses the accepted classifier prediction.",
    )
    parser.add_argument("--view", choices=("front", "side"), required=True)
    parser.add_argument("--handedness", choices=("left", "right"), default="right")
    parser.add_argument("--target", choices=("any", "single", "far", "near"), default="single")
    parser.add_argument(
        "--pose-config", type=Path,
        default=Path("configs/pose/yolov8.yaml"),
        help=(
            "Pose profile used by the complete pipeline. The default matches "
            "the pose distribution used to train the action classifier; pass "
            "yolov8_high_accuracy.yaml explicitly for geometry-only experiments."
        ),
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
    skeleton_preview = None
    if not args.no_preview:
        preview = output_dir / "pose_preview.mp4"
        run_command([
            sys.executable, str(repo_root / "scripts/inference/render_pose.py"),
            str(video), str(pose), str(preview),
        ], cwd=repo_root)
        skeleton_preview = output_dir / "skeleton_preview.mp4"
        run_command([
            sys.executable, str(repo_root / "scripts/inference/render_pose.py"),
            str(video), str(pose), str(skeleton_preview), "--skeleton-only",
        ], cwd=repo_root)

    classifier_suggestion = None
    classifier_path = output_dir / "classification.json"
    if args.run_classifier_wsl or (args.technique == "auto" and not args.classifier_json):
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

    technique = args.technique
    selection_source = "user_confirmed"
    if technique == "auto":
        if not classifier_suggestion or not classifier_suggestion.get("accepted"):
            raise ValueError(
                "AI could not identify the technique confidently. "
                "Choose forehand_clear or backhand_drive explicitly."
            )
        technique = str(classifier_suggestion.get("label"))
        if technique not in {"forehand_clear", "backhand_drive"}:
            raise ValueError(f"Unsupported classifier label: {technique}")
        selection_source = "classifier"

    report_path = output_dir / "technique_report.json"
    technique_report = None
    if quality.get("geometry_feasible", False):
        command = [
            sys.executable,
            str(repo_root / "scripts/evaluation/check_technique_rules.py"),
            str(pose), technique,
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

    analysis = assemble_analysis(
        video=video, pose=pose, technique=technique, view=args.view,
        handedness=args.handedness, pose_config=pose_config, quality=quality,
        technique_report=technique_report, preview=preview,
        classifier_suggestion=classifier_suggestion,
        selection_source=selection_source,
    )
    analysis["artifacts"]["classification"] = (
        str(classifier_path) if classifier_suggestion is not None else None
    )
    analysis["artifacts"]["skeleton_preview"] = (
        str(skeleton_preview) if skeleton_preview else None
    )
    user_report = build_user_report(analysis)
    user_report_json_path = output_dir / "user_report.json"
    user_report_markdown_path = output_dir / "user_report.md"
    user_report_json_path.write_text(
        json.dumps(user_report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    user_report_markdown_path.write_text(
        render_user_report_markdown(user_report), encoding="utf-8"
    )
    analysis["artifacts"]["user_report_json"] = str(user_report_json_path)
    analysis["artifacts"]["user_report_markdown"] = str(
        user_report_markdown_path
    )
    analysis_path = output_dir / "analysis.json"
    analysis_path.write_text(
        json.dumps(analysis, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Analysis complete: {analysis_path}")
    print(f"User report: {user_report_markdown_path}")
    print(f"Overall: {analysis['overall']}  score: none")


if __name__ == "__main__":
    main()
