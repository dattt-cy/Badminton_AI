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


def test_score_tracker_replaces_a_misread_first_reading():
    tracker = ScoreChangeTracker(confirmations_required=2, resync_confirmations=3)
    tracker.observe((3, 6), frame=0, fps=25.0)
    for frame in (30, 60, 90):
        tracker.observe((0, 1), frame=frame, fps=25.0)
    assert tracker.initial == (0, 1)
    assert tracker.initial_frame == 90
    assert tracker.events == []


def test_score_tracker_follows_the_reset_of_a_new_game():
    tracker = ScoreChangeTracker(confirmations_required=1, resync_confirmations=3)
    tracker.observe((20, 19), frame=0, fps=25.0)
    tracker.observe((21, 19), frame=30, fps=25.0)
    # A lone misread is ignored; the 0-0 of the next game, read three times, is not.
    tracker.observe((0, 0), frame=60, fps=25.0)
    tracker.observe((21, 19), frame=90, fps=25.0)
    for frame in (120, 150, 180):
        tracker.observe((0, 0), frame=frame, fps=25.0)
    tracker.observe((1, 0), frame=210, fps=25.0)
    tracker.observe((1, 0), frame=240, fps=25.0)
    assert [e["game"] for e in tracker.events] == [1, 2]
    assert tracker.events[1]["previous_score"] == [0, 0]
    assert tracker.initial == (20, 19)
