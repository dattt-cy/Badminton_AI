from ai_classifier.localization import HitEvent, match_hit_events, temporal_nms


def event(frame: int, side: str = "upper", score: float = 0.8) -> HitEvent:
    return HitEvent(frame, side, score)


def test_temporal_nms_keeps_strongest_event_and_preserves_order() -> None:
    events = [event(30, score=0.7), event(10, score=0.8), event(13, score=0.9)]

    assert temporal_nms(events, threshold=0.75, radius=5) == [
        event(13, score=0.9)
    ]


def test_side_aware_nms_keeps_nearby_opposing_player_hits() -> None:
    events = [event(34, "lower", 0.92), event(38, "upper", 0.83)]

    assert temporal_nms(
        events, threshold=0.5, radius=10, side_aware=True
    ) == events


def test_matching_is_one_to_one_and_reports_side_accuracy() -> None:
    predicted = [event(98, "upper"), event(102, "lower"), event(201, "lower")]
    truth = [event(100, "upper"), event(200, "upper")]

    result = match_hit_events(predicted, truth, tolerance=3)

    assert result["true_positives"] == 2
    assert result["false_positives"] == 1
    assert result["false_negatives"] == 0
    assert result["side_accuracy"] == 0.5
    assert result["mean_absolute_frame_error"] == 1.5


def test_matching_maximizes_count_before_minimizing_error() -> None:
    predicted = [event(4), event(6)]
    truth = [event(1), event(5)]

    result = match_hit_events(predicted, truth, tolerance=4)

    assert result["true_positives"] == 2
