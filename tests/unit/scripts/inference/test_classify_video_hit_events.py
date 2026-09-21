from ai_classifier.localization import HitEvent
from scripts.inference.classify_video_hit_events import group_same_player_events
from scripts.evaluation.compare_archive_hit_vs_direct_rgb20 import candidate_quality


def test_group_same_player_events_merges_nearby_centers_only_for_same_side():
    events = [
        HitEvent(10, "upper", 0.7),
        HitEvent(16, "upper", 0.8),
        HitEvent(18, "lower", 0.9),
        HitEvent(40, "upper", 0.6),
    ]
    groups = group_same_player_events(events, radius_frames=10)

    assert [[event.frame for event in group] for group in groups] == [[10, 16], [18], [40]]


def test_candidate_quality_rewards_confident_tracked_candidate():
    weak = {"hit_score": 0.8, "stroke_confidence": 0.4, "stroke_margin": 0.05,
            "tracking": False, "track_frames": 0}
    strong = {"hit_score": 0.75, "stroke_confidence": 0.8, "stroke_margin": 0.5,
              "tracking": True, "track_frames": 7}
    assert candidate_quality(strong) > candidate_quality(weak)
