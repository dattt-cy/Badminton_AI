from pathlib import Path

import pytest

from ai_classifier.biomechanics import load_technique_registry


def test_load_project_registry() -> None:
    registry = load_technique_registry(Path("configs/biomechanics/techniques.yaml"))

    forehand = registry.technique("forehand_lift")
    assert forehand.views["side"].reference.name == "forehand_lift_side_reference.yaml"
    assert forehand.views["front"].clip_list is not None
    rules = {rule.name: rule for rule in forehand.rules}
    assert rules["elbow_too_low_in_backswing"].feature == "elbow_height"
    assert rules["stance_too_narrow"].compatible_views == ("front",)
    assert forehand.minimum_reference_clips == 10

    clear = registry.technique("forehand_clear")
    assert clear.display_name == "Forehand clear"
    assert set(clear.views) == {"front", "side"}
    assert clear.views["front"].reference.name == "forehand_clear_front_reference.yaml"


def test_unknown_technique_lists_available_names() -> None:
    registry = load_technique_registry(Path("configs/biomechanics/techniques.yaml"))

    with pytest.raises(ValueError, match="backhand_drive, forehand_clear, forehand_lift"):
        registry.technique("smash")


def test_registry_rejects_invalid_rule_direction(tmp_path: Path) -> None:
    path = tmp_path / "techniques.yaml"
    path.write_text(
        """techniques:
  smash:
    views:
      side: {reference: smash.yaml}
    rules:
      - {name: bad, feature: elbow_angle, phase: contact_estimated, direction: sideways}
""",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Invalid direction"):
        load_technique_registry(path)
