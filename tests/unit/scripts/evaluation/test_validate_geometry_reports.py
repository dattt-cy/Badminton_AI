import importlib.util
import json
from pathlib import Path


SCRIPT = Path(__file__).parents[4] / "scripts/evaluation/validate_geometry_reports.py"
SPEC = importlib.util.spec_from_file_location("validate_geometry_reports", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_validation_reports_peak_error_rule_metrics_and_abstention(tmp_path: Path) -> None:
    report = {
        "strokes": [{
            "stroke_id": 1,
            "estimated_swing_peak_frame": 12,
            "phases": {"contact_frame": 12},
            "rules": [
                {"rule_name": "elbow", "status": "deviation"},
                {"rule_name": "torso", "status": "review"},
            ],
        }]
    }
    report_path = tmp_path / "report.json"
    report_path.write_text(json.dumps(report), encoding="utf-8")
    manifest = tmp_path / "labels.csv"
    manifest.write_text(
        "report,stroke_id,true_swing_peak_frame,rule.elbow,rule.torso\n"
        "report.json,1,10,deviation,deviation\n",
        encoding="utf-8",
    )

    result = MODULE.validate_manifest(manifest)

    assert result["swing_peak"]["mae_frames"] == 2.0
    assert result["swing_peak"]["within_3_frames_ratio"] == 1.0
    assert result["rules"]["coverage"] == 0.5
    assert result["rules"]["aggregate"]["tp"] == 1
    assert result["rules"]["aggregate"]["fn"] == 1
    assert result["rules"]["aggregate"]["abstain"] == 1
