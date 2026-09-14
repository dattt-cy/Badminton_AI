import importlib.util
from pathlib import Path


SCRIPT = Path(__file__).parents[4] / "scripts/evaluation/batch_test_technique_analysis.py"
SPEC = importlib.util.spec_from_file_location("batch_test_technique_analysis", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_select_diverse_uses_different_subjects_first():
    cases = [
        {"subject": "Sub08", "video": Path("a.mov")},
        {"subject": "Sub08", "video": Path("b.mov")},
        {"subject": "Sub09", "video": Path("c.mov")},
    ]
    selected = MODULE.select_diverse(cases, 2)
    assert {case["subject"] for case in selected} == {"Sub08", "Sub09"}


def test_summary_flags_nearly_constant_criterion():
    rows = [{
        "technique": "forehand_clear", "view": "side", "success": True,
        "geometry_feasible": True, "pose_valid_percent": 95.0,
        "classifier_status": "not_run",
        "quality_reasons": [],
        "observable_statuses": {"leg_loading": "needs_review"},
    } for _ in range(4)]
    report = MODULE.summarize(rows)
    assert report["warnings"][0]["criterion"] == "leg_loading"
    assert report["groups"]["forehand_clear"]["side"]["quality_pass_ratio"] == 1.0
