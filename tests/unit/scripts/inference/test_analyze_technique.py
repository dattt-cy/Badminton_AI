import importlib.util
import sys
from pathlib import Path


SCRIPT = Path(__file__).parents[4] / "scripts/inference/analyze_technique.py"
SPEC = importlib.util.spec_from_file_location("analyze_technique", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_safe_stem_removes_shell_sensitive_characters():
    assert MODULE.safe_stem(Path("test (1).mp4")) == "test_1"


def test_default_pose_profile_matches_action_classifier_training(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        ["analyze_technique.py", "stroke.mp4", "forehand_clear", "--view", "front"],
    )

    args = MODULE.parse_args()

    assert args.pose_config == Path("configs/pose/yolov8.yaml")


def test_technique_defaults_to_auto(monkeypatch):
    monkeypatch.setattr(
        sys, "argv", ["analyze_technique.py", "stroke.mp4", "--view", "front"]
    )

    assert MODULE.parse_args().technique == "auto"


def test_assembled_report_preserves_user_selection_over_classifier():
    technique_report = {
        "_path": "report.json",
        "strokes": [{"user_feedback": {"overall": "review_available"}}],
    }
    result = MODULE.assemble_analysis(
        video=Path("video.mp4"), pose=Path("pose.npz"),
        technique="forehand_clear", view="side", handedness="right",
        pose_config=Path("pose.yaml"), quality={"geometry_feasible": True},
        technique_report=technique_report, preview=None,
        classifier_suggestion={
            "label": "backhand_drive", "confidence": 0.9, "accepted": True
        },
    )

    assert result["selection"]["technique"] == "forehand_clear"
    assert not result["selection"]["classifier_used_for_rules"]
    assert result["classifier_suggestion"]["agreement_status"] == "conflict"
    assert not result["classifier_suggestion"]["used_for_rules"]
    assert "giữ lựa chọn người dùng" in result["user_summary"]["recognition"]
    assert result["user_summary"]["counts"]["validated_good_signals"] == 0
    assert result["overall"] == "review_available"
    assert result["score"] is None


def test_infeasible_quality_stops_at_quality_report():
    result = MODULE.assemble_analysis(
        video=Path("video.mp4"), pose=Path("pose.npz"),
        technique="backhand_drive", view="front", handedness="left",
        pose_config=Path("pose.yaml"),
        quality={"geometry_feasible": False, "reasons": ["too small"]},
        technique_report=None, preview=None, classifier_suggestion=None,
    )

    assert result["overall"] == "insufficient_video_quality"
    assert result["strokes"] == []


def test_user_report_is_plain_language_and_keeps_score_empty():
    analysis = {
        "overall": "review_available",
        "selection": {
            "technique": "forehand_clear", "view": "side", "handedness": "right"
        },
        "classifier_suggestion": {
            "label": "forehand_clear", "confidence": 0.9,
            "agreement_status": "confirmed", "message": "AI xác nhận.",
        },
        "quality": {
            "geometry_feasible": True, "reasons": [],
            "trajectory_quality": {"essential_valid_ratio": 0.95, "median_scale_px": 120},
            "motion_proposals": [{"peak_time": 1.5}],
        },
        "user_feedback": [{
            "good_signals": [],
            "heuristic_observations": [{
                "criterion": "leg_loading", "label": "Sử dụng chân",
                "observed": 170.0, "unit": "degree", "status": "needs_review",
                "message": "Hai gối khá thẳng.",
            }],
            "observations_for_review": [], "not_assessed": [],
        }],
        "disclaimer": "Bản thử nghiệm.",
    }

    report = MODULE.build_user_report(analysis)
    markdown = MODULE.render_user_report_markdown(report)

    assert report["score"] is None
    assert report["needs_review"][0]["value"] == "170.0°"
    assert report["primary_focus"]["label"] == "Sử dụng chân"
    assert report["primary_focus"]["action"] == "Chùng gối thêm khi chuẩn bị để hỗ trợ phát lực."
    assert len(report["primary_focus"]["drill"]) == 3
    assert "Kết quả phân tích Forehand clear" in markdown
    assert "Chùng gối thêm" in markdown


def test_user_report_hides_normalized_model_units():
    item = {
        "criterion": "contact_above_head",
        "observed": -0.36,
        "unit": "body_scale_ratio",
    }

    assert MODULE._display_observable_value(item) == "chưa ở trên đầu"
    assert MODULE._display_value(0.36, "body_scale_ratio") is None


def test_classifier_selected_report_does_not_claim_user_selected_technique():
    analysis = MODULE.assemble_analysis(
        video=Path("video.mp4"), pose=Path("pose.npz"),
        technique="forehand_clear", view="front", handedness="right",
        pose_config=Path("pose.yaml"), quality={"geometry_feasible": False},
        technique_report=None, preview=None,
        classifier_suggestion={
            "label": "forehand_clear", "confidence": 0.95, "accepted": True
        },
        selection_source="classifier",
    )

    report = MODULE.build_user_report(analysis)
    markdown = MODULE.render_user_report_markdown(report)

    assert report["technique"]["selected_by_user"] is False
    assert "AI tự nhận diện" in report["recognition"]["message"]
    assert "Kỹ thuật AI nhận diện" in markdown


def test_review_measurement_gets_cautious_actionable_language():
    analysis = {
        "overall": "review_available",
        "selection": {
            "technique": "forehand_clear", "view": "side",
            "handedness": "right", "source": "classifier",
        },
        "classifier_suggestion": None,
        "quality": {"geometry_feasible": True, "trajectory_quality": {}},
        "user_feedback": [{
            "good_signals": [], "heuristic_observations": [],
            "observations_for_review": [{
                "feature": "wrist_shoulder_distance", "label": "Độ vươn tay",
                "phase": "contact_estimated", "observed": 0.5,
                "unit": "body_scale_ratio",
            }],
            "not_assessed": [],
        }],
        "disclaimer": "Bản thử nghiệm.",
    }

    report = MODULE.build_user_report(analysis)

    assert report["reference_measurements"][0]["note"].startswith("AI chưa đủ chắc")
    assert "vùng tiếp xúc" in report["reference_measurements"][0]["suggestion"]


def test_user_report_uses_selected_contact_instead_of_first_motion_burst():
    analysis = {
        "overall": "review_available",
        "selection": {
            "technique": "forehand_clear", "view": "side",
            "handedness": "right", "source": "classifier",
        },
        "classifier_suggestion": None,
        "quality": {
            "geometry_feasible": True, "fps": 30.0,
            "trajectory_quality": {},
            "motion_proposals": [
                {"peak_frame": 30, "peak_time": 1.0, "peak_score": 1.0},
                {"peak_frame": 51, "peak_time": 1.7, "peak_score": 2.0},
            ],
        },
        "strokes": [{"phases": {"contact_frame": 51}}],
        "user_feedback": [],
        "disclaimer": "Bản thử nghiệm.",
    }

    report = MODULE.build_user_report(analysis)

    assert report["video_quality"]["swing_peak_seconds"] == 1.7


def test_user_report_links_feedback_to_a_video_moment():
    analysis = {
        "overall": "review_available",
        "selection": {
            "technique": "forehand_clear", "view": "side",
            "handedness": "right", "source": "classifier",
        },
        "classifier_suggestion": None,
        "quality": {"geometry_feasible": True, "fps": 30.0, "trajectory_quality": {}},
        "strokes": [{
            "start_frame": 0,
            "phases": {
                "preparation": [15, 30], "contact_frame": 50,
                "contact_estimated": [48, 53],
            },
        }],
        "user_feedback": [{
            "good_signals": [], "observations_for_review": [], "not_assessed": [],
            "heuristic_observations": [{
                "criterion": "racket_arm_preparation", "label": "Tay vợt",
                "observed": -0.3, "unit": "body_scale_ratio",
                "status": "needs_review", "message": "Tay còn thấp.",
            }],
        }],
        "disclaimer": "Bản thử nghiệm.",
    }

    report = MODULE.build_user_report(analysis)

    assert report["primary_focus"]["moment"] == {
        "frame": 22, "seconds": 0.73, "label": "Xem chậm đoạn 0.73s"
    }


def test_user_report_warns_when_video_is_oblique():
    analysis = {
        "overall": "review_available",
        "selection": {
            "technique": "forehand_clear", "view": "side",
            "handedness": "right", "source": "classifier",
        },
        "classifier_suggestion": None,
        "quality": {"geometry_feasible": True, "trajectory_quality": {}},
        "camera_view_check": {"detected_body_projection": "oblique"},
        "user_feedback": [], "strokes": [],
        "disclaimer": "Bản thử nghiệm.",
    }

    report = MODULE.build_user_report(analysis)

    assert "góc quay chéo" in report["video_quality"]["view_warning"]
