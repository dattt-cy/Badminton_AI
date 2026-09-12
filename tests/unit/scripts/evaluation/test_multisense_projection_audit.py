import importlib.util
from pathlib import Path


SCRIPT = Path(__file__).parents[4] / "scripts/evaluation/audit_multisense_projection.py"
SPEC = importlib.util.spec_from_file_location("audit_multisense_projection", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_summary_excludes_training_subjects():
    rows = []
    for subject, error in (("Sub09", 99.0), ("Sub16", 0.1), ("Sub19", 0.3)):
        row = {"subject": subject, "technique": "forehand_clear", "view": "side"}
        for joint in MODULE.JOINTS:
            row[f"{joint}_error_ratio"] = str(error)
        rows.append(row)

    report = MODULE.summarize(rows)
    summary = report["techniques"]["forehand_clear"]["side"]

    assert summary["frame_count"] == 2
    assert summary["joints"]["right_elbow"]["median_error_ratio"] == 0.2
