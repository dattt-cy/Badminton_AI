import pandas as pd

from scripts.data_preparation.prepare_bfmd_manifest import (
    EXCLUDED_LABELS,
    LABEL_MAPPING,
    validate_manifest,
)


def test_bfmd_mapping_is_conservative_for_ambiguous_labels():
    assert "block" not in LABEL_MAPPING
    assert "push" not in LABEL_MAPPING
    assert set(EXCLUDED_LABELS) == {"block", "push"}
    assert LABEL_MAPPING["net_kill"] == "net_attack"
    assert LABEL_MAPPING["press"] == "net_attack"


def test_bfmd_manifest_validation_rejects_match_leakage():
    labels = sorted(set(LABEL_MAPPING.values()))
    rows = []
    for split, match in (("train", "a"), ("val", "b"), ("test", "c")):
        rows.extend({"split": split, "match_id": match, "coarse_label": label, "raw_label": label} for label in labels)
    validate_manifest(pd.DataFrame(rows))
    rows[-1]["match_id"] = "a"
    try:
        validate_manifest(pd.DataFrame(rows))
    except ValueError as error:
        assert "leakage" in str(error)
    else:
        raise AssertionError("expected split leakage to be rejected")
