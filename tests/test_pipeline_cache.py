from scripts.inference.robust_aligned_video_pipeline import AlignedVideoPipeline


def test_event_windows_are_merged_without_duplicate_frames() -> None:
    ranges = AlignedVideoPipeline._merge_event_ranges(
        [100, 120, 300], total_frames=400, before=21, after=36
    )
    assert ranges == [(79, 156), (279, 336)]


def test_event_windows_are_clipped_to_video_bounds() -> None:
    ranges = AlignedVideoPipeline._merge_event_ranges(
        [5, 395], total_frames=400, before=21, after=36
    )
    assert ranges == [(0, 41), (374, 399)]
