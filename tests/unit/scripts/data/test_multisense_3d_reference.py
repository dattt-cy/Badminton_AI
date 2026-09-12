import importlib.util
from pathlib import Path

import pytest


SCRIPT = (
    Path(__file__).parents[4]
    / "scripts/data/references/build_multisense_3d_reference.py"
)
SPEC = importlib.util.spec_from_file_location("build_multisense_3d_reference", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def _row(value, **overrides):
    row = {
        "subject": "Sub11",
        "recording": "recording-a",
        "stroke_number": str(value),
        "technique": "forehand_clear",
        "skill_level": "Expert",
        "phase_valid": "True",
        "hitting_sound": "Good",
        "contact_confidence": "0.8",
        "contact_estimated__elbow_angle": str(value),
        "__source": "part2",
    }
    row.update(overrides)
    return row


def test_reference_uses_only_quality_gated_expert_strokes():
    rows = [_row(value) for value in range(100, 120)]
    rows.extend([
        _row(999, skill_level="Beginner"),
        _row(998, phase_valid="False"),
        _row(997, hitting_sound="Bad"),
        _row(996, contact_confidence="0.2"),
    ])

    document = MODULE.build_reference_document(rows)
    summary = document["techniques"]["forehand_clear"]
    interval = summary["features"]["contact_estimated__elbow_angle"]

    assert document["selected_stroke_count"] == 20
    assert document["reference_status"] == "provisional"
    assert interval["sample_count"] == 20
    assert interval["low"] == pytest.approx(101.9)
    assert interval["median"] == pytest.approx(109.5)
    assert interval["high"] == pytest.approx(117.1)
    assert interval["unit"] == "degree"
    assert document["excluded_stroke_count_by_reason"] == {
        "invalid_phase": 1,
        "low_contact_confidence": 1,
        "non_expert": 1,
        "unaccepted_hitting_sound": 1,
    }


def test_duplicate_stroke_is_not_counted_twice():
    rows = [_row(value) for value in range(100, 110)]
    rows.append(dict(rows[0], __source="part3"))

    document = MODULE.build_reference_document(rows)

    assert document["selected_stroke_count"] == 10
    assert document["excluded_stroke_count_by_reason"]["duplicate_stroke"] == 1


def test_invalid_percentiles_are_rejected():
    with pytest.raises(ValueError, match="percentiles"):
        MODULE.build_reference_document([_row(100)], low_percentile=90, high_percentile=10)


def test_holdout_validation_reports_interval_coverage():
    training = [_row(value) for value in range(100, 120)]
    document = MODULE.build_reference_document(training)
    holdout = [
        _row(102, subject="Sub19", recording="holdout"),
        _row(110, subject="Sub19", recording="holdout"),
        _row(116, subject="Sub19", recording="holdout"),
        _row(130, subject="Sub19", recording="holdout"),
    ]

    report = MODULE.validate_holdout(document, holdout)
    elbow = report["techniques"]["forehand_clear"]["features"][
        "contact_estimated__elbow_angle"
    ]

    assert elbow["sample_count"] == 4
    assert elbow["within_count"] == 3
    assert elbow["coverage"] == pytest.approx(0.75)
    assert elbow["above_count"] == 1
    assert report["status"] == "passed"
