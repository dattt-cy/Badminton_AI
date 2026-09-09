import importlib.util
from pathlib import Path


SCRIPT = Path(__file__).parents[2] / "scripts/evaluation/audit_multisense_features.py"
SPEC = importlib.util.spec_from_file_location("audit_multisense_features", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_audit_excludes_invalid_phases_from_distribution():
    rows = [
        {"subject": "Sub00", "technique": "backhand_drive",
         "skill_level": "Beginner", "phase_valid": "True",
         "phase_invalid_reason": "", "contact_estimated__elbow_angle": "100"},
        {"subject": "Sub01", "technique": "backhand_drive",
         "skill_level": "Beginner", "phase_valid": "False",
         "phase_invalid_reason": "ambiguous_contact_peaks",
         "contact_estimated__elbow_angle": "999"},
    ]

    report = MODULE.audit_rows(rows)
    summary = report["techniques"]["backhand_drive"]

    assert report["subjects"] == 2
    assert summary["valid_phase_ratio"] == 0.5
    assert summary["invalid_phase_reasons"] == {"ambiguous_contact_peaks": 1}
    assert summary["feature_distributions_valid_phases"][
        "contact_estimated__elbow_angle"
    ]["median"] == 100.0
