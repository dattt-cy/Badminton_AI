import importlib.util
from pathlib import Path


SCRIPT = (
    Path(__file__).parents[4]
    / "scripts/evaluation/build_multisense_feature_validation.py"
)
SPEC = importlib.util.spec_from_file_location("build_multisense_feature_validation", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_validation_uses_holdout_and_marks_accurate_feature_validated():
    rows = []
    for subject in ("Sub16", "Sub19", "Sub24"):
        for index in range(40):
            rows.append({
                "subject": subject, "technique": "forehand_clear", "view": "side",
                "elbow_2d_median": str(100 + index),
                "elbow_3d_same_time_median": str(102 + index),
            })
    rows.append({
        "subject": "Sub09", "technique": "forehand_clear", "view": "side",
        "elbow_2d_median": "0", "elbow_3d_same_time_median": "180",
    })

    document = MODULE.build_validation(rows)
    elbow = document["techniques"]["forehand_clear"]["views"]["side"]["features"][
        "elbow_angle"
    ]

    assert elbow["status"] == "validated"
    assert elbow["sample_count"] == 120
    assert elbow["subject_count"] == 3
    assert elbow["mae"] == 2.0
