import importlib.util
from pathlib import Path


SCRIPT = Path(__file__).parents[4] / "scripts/inference/analyze_technique.py"
SPEC = importlib.util.spec_from_file_location("analyze_technique", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_safe_stem_removes_shell_sensitive_characters():
    assert MODULE.safe_stem(Path("test (1).mp4")) == "test_1"


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
    assert "Kết quả phân tích Forehand clear" in markdown
    assert "Chùng gối thêm" in markdown
