from ai_classifier.localization.hit_spotting import HitEvent, two_stage_temporal_nms


def test_second_stage_suppresses_same_side_but_keeps_fast_exchange() -> None:
    events = [
        HitEvent(100, "upper", 0.90),
        HitEvent(116, "lower", 0.85),
        HitEvent(132, "lower", 0.70),
    ]
    kept = two_stage_temporal_nms(
        events, threshold=0.55, radius=15, same_side_radius=18
    )
    assert [(event.frame, event.side) for event in kept] == [
        (100, "upper"), (116, "lower")
    ]


if __name__ == "__main__":
    test_second_stage_suppresses_same_side_but_keeps_fast_exchange()
    print("Hit spotting test passed!")

