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
        classifier_suggestion={"label": "backhand_drive", "confidence": 0.9},
    )

    assert result["selection"]["technique"] == "forehand_clear"
    assert not result["selection"]["classifier_used_for_rules"]
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
