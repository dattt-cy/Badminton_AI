import importlib.util
from pathlib import Path


SCRIPT = Path(__file__).parents[2] / "scripts/data/cut_multisense_reference_clips.py"
SPEC = importlib.util.spec_from_file_location("cut_multisense_reference_clips", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_select_rows_filters_quality_and_spreads_repetitions():
    rows = [
        {"subject": "Sub11", "technique": "backhand_drive", "phase_valid": "True",
         "hitting_sound": "Good", "stroke_number": str(index)}
        for index in range(10)
    ]
    rows.append({"subject": "Sub11", "technique": "backhand_drive",
                 "phase_valid": "False", "hitting_sound": "Good", "stroke_number": "99"})

    selected = MODULE.select_rows(
        rows, subject="Sub11", technique="backhand_drive", limit=3
    )

    assert [row["stroke_number"] for row in selected] == ["0", "4", "9"]


def test_parse_crop_validates_normalized_box():
    assert MODULE.parse_crop("0.25,0.10,0.50,0.85") == (0.25, 0.1, 0.5, 0.85)

    try:
        MODULE.parse_crop("0.8,0.1,0.5,0.8")
    except ValueError as error:
        assert "inside normalized" in str(error)
    else:
        raise AssertionError("Expected an out-of-frame crop to fail")


def test_compact_epoch_window_uses_detected_motion_bounds():
    row = {
        "start_time": "1000", "phase_window_start_seconds": "1.2",
        "phase_window_end_seconds": "2.8",
    }

    assert MODULE.compact_epoch_window(row, 0.25) == (1000.95, 1003.05)


def test_full_epoch_window_keeps_complete_trial():
    row = {"start_time": "1000", "stop_time": "1004.5"}

    assert MODULE.full_epoch_window(row, 0.25) == (999.75, 1004.75)
