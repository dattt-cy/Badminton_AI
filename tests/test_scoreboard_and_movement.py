from ai_classifier.analytics import ScoreChangeTracker, analyze_sparse_movement


def test_score_change_requires_repeated_plausible_observation():
    tracker = ScoreChangeTracker(confirmations_required=2)
    tracker.observe((3, 4), frame=0, fps=30.0)
    tracker.observe((3, 5), frame=30, fps=30.0)
    assert tracker.events == []
    tracker.observe((3, 5), frame=60, fps=30.0)
    assert tracker.initial == (3, 4)
    assert tracker.events[0]["point_winner"] == "second"
    assert tracker.events[0]["score"] == [3, 5]


def test_score_change_rejects_ocr_jump():
    tracker = ScoreChangeTracker(confirmations_required=1)
    tracker.observe((3, 4), frame=0, fps=30.0)
    tracker.observe((8, 4), frame=30, fps=30.0)
    assert tracker.events == []


def test_sparse_movement_is_explicit_lower_bound():
    predictions = [
        {
            "rally": 1,
            "hitter": "upper",
            "analytics": {
                "player_position_court": [0.0, 0.0],
                "opponent_position_court": [1.0, 1.0],
            },
        },
        {
            "rally": 1,
            "hitter": "lower",
            "analytics": {
                "player_position_court": [0.5, 1.0],
                "opponent_position_court": [0.5, 0.0],
            },
        },
    ]
    result = analyze_sparse_movement(predictions)
    assert result["is_full_tracking_distance"] is False
    assert result["total_lower_bound_distance_meters"]["upper"] > 0
    assert result["total_lower_bound_distance_meters"]["lower"] > 0
