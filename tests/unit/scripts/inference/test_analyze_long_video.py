import math

from ai_classifier.localization import HitEvent
from scripts.inference.analyze_long_video import (
    select_hit_events,
    selector_probability,
    stable_digest,
)


def selector():
    return {
        "coefficient": [8.0, 0.0, 0.0, 0.0, 0.0],
        "intercept": -4.0,
        "threshold": 0.5,
        "candidate_generation": {
            "hit_threshold": 0.1,
            "nms_radius": 2,
            "side_aware": True,
        },
    }


def test_selector_probability_uses_serialized_logistic_model():
    probability = selector_probability([0.5, 0.0, 0.0, 0.0, 0.0], selector())
    assert math.isclose(probability, 0.5)


def test_select_hit_events_keeps_all_candidates_above_selector_threshold():
    events = [
        HitEvent(frame=10, side="upper", score=0.9),
        HitEvent(frame=50, side="lower", score=0.2),
        HitEvent(frame=90, side="upper", score=0.8),
    ]
    selected, scored = select_hit_events(events, selector(), 0.5)
    assert [event.frame for event in selected] == [10, 90]
    assert len(scored) == 3


def test_stable_digest_is_order_independent_for_mappings():
    assert stable_digest({"a": 1, "b": 2}) == stable_digest({"b": 2, "a": 1})
