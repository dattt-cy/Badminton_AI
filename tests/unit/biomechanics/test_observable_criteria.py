import numpy as np

from ai_classifier.biomechanics.observable_criteria import evaluate_observable_criteria
from ai_classifier.pose import PoseSequence


class Phases:
    valid = True
    preparation = (0, 4)
    backswing = (4, 7)
    forward_swing = (7, 10)
    contact_estimated = (10, 13)
    follow_through = (13, 16)
    contact_confidence = 0.9


def _sequence():
    points = np.zeros((16, 17, 3), dtype=np.float32)
    points[..., 2] = 0.9
    points[:, 0, :2] = [55, 20]
    points[:, 5, :2] = [40, 40]
    points[:, 6, :2] = [60, 40]
    points[:, 11, :2] = [42, 80]
    points[:, 12, :2] = [58, 80]
    points[:, 13, :2] = [42, 110]
    points[:, 14, :2] = [58, 110]
    points[:, 15, :2] = [32, 130]
    points[:, 16, :2] = [68, 130]
    points[:, 8, :2] = [75, 30]
    points[:, 10, :2] = [90, 25]
    points[:, 7, :2] = [35, 45]
    points[:4, 9, :2] = [35, 30]
    points[10:13, 9, :2] = [40, 55]
    points[13:, 10, :2] = [45, 75]
    return PoseSequence(points, 30, 160, 140)


def test_forehand_observations_include_contact_leg_and_followthrough():
    result = evaluate_observable_criteria(
        _sequence(), Phases(), "forehand_clear", "side"
    )
    names = {item["criterion"] for item in result}
    assert "contact_above_head" in names
    assert "leg_loading" in names
    assert "followthrough_completion" in names
    assert "racket_arm_preparation" in names
    assert "arm_extension_excursion" in names
    assert "arm_reach_excursion" in names
    assert "recovery_balance" in names
    assert all(item["evidence_level"] == "experimental_heuristic" for item in result)


def test_backhand_uses_drive_contact_height_instead_of_overhead_contact():
    result = evaluate_observable_criteria(
        _sequence(), Phases(), "backhand_drive", "front"
    )
    names = {item["criterion"] for item in result}
    assert "contact_drive_height" in names
    assert "contact_above_head" not in names


def test_non_racket_arm_uses_peak_before_contact_not_short_preparation_only():
    sequence = _sequence()
    # The arm starts low, rises during the backswing, then drops at contact.
    sequence.keypoints[:4, 9, :2] = [35, 55]
    sequence.keypoints[4:9, 9, :2] = [35, 15]
    sequence.keypoints[10:13, 9, :2] = [40, 55]

    result = evaluate_observable_criteria(
        sequence, Phases(), "forehand_clear", "front"
    )
    coordination = next(
        item for item in result
        if item["criterion"] == "non_racket_arm_coordination"
    )

    assert coordination["status"] == "observed"


def test_borderline_contact_is_unavailable_instead_of_coaching_error():
    sequence = _sequence()
    # Put the racket wrist almost level with the face: too close to a noisy
    # zero threshold to justify either praise or correction.
    sequence.keypoints[10:13, 10, :2] = [90, 5]

    result = evaluate_observable_criteria(
        sequence, Phases(), "forehand_clear", "front"
    )
    contact = next(
        item for item in result if item["criterion"] == "contact_above_head"
    )

    assert contact["status"] == "unavailable"
    assert "không xác định" in contact["message"].lower()
