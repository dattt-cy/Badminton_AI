from ai_classifier.localization.live_gate import GateSample, frame_is_live, samples_to_segments


def test_live_samples_merge_and_filter_frames() -> None:
    segments = samples_to_segments(
        [GateSample(10, True, 0.8), GateSample(20, True, 0.9), GateSample(50, False, 0.0)],
        total_frames=100,
        max_gap_frames=15,
    )

    assert len(segments) == 1
    assert frame_is_live(15, segments)
    assert not frame_is_live(50, segments)
