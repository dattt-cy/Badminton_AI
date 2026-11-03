from ai_classifier.localization import HitEvent, enforce_alternation


def test_enforce_alternation_drops_weaker_same_side_neighbour():
    events = [HitEvent(0, "upper", 0.9), HitEvent(20, "lower", 0.9), HitEvent(30, "lower", 0.7),
              HitEvent(60, "upper", 0.95)]
    kept = enforce_alternation(events, max_gap_frames=60)
    assert [(e.frame, e.side) for e in kept] == [(0, "upper"), (20, "lower"), (60, "upper")]


def test_enforce_alternation_keeps_same_side_hits_far_apart():
    events = [HitEvent(0, "upper", 0.9), HitEvent(200, "upper", 0.6)]
    assert enforce_alternation(events, max_gap_frames=60) == events
