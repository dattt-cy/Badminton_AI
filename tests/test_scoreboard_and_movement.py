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


def test_score_tracker_ignores_a_lasting_misread_that_lowers_the_score():
    # 7-8 read as 1-8 for as long as the score shows: no new game, no resync.
    tracker = ScoreChangeTracker(confirmations_required=1, resync_confirmations=3)
    tracker.observe((6, 8), frame=0, fps=25.0)
    tracker.observe((7, 8), frame=30, fps=25.0)
    tracker.observe((7, 8), frame=45, fps=25.0)
    for frame in (60, 90, 120, 150):
        tracker.observe((1, 8), frame=frame, fps=25.0)
    tracker.observe((8, 8), frame=180, fps=25.0)
    tracker.observe((8, 8), frame=195, fps=25.0)
    assert tracker.game == 1
    assert tracker.resyncs == []
    assert [e["score"] for e in tracker.events] == [[7, 8], [8, 8]]


def test_score_tracker_ignores_a_finished_games_column_during_a_replay():
    # A faded scoreboard shows only the finished game's dark digits (23-21) mid game two.
    tracker = ScoreChangeTracker(confirmations_required=1, resync_confirmations=3)
    tracker.observe((22, 21), frame=0, fps=25.0)
    tracker.observe((23, 21), frame=30, fps=25.0)
    tracker.observe((23, 21), frame=45, fps=25.0)
    for frame in (60, 90, 120):
        tracker.observe((0, 0), frame=frame, fps=25.0)
    tracker.observe((1, 0), frame=150, fps=25.0)
    tracker.observe((1, 0), frame=165, fps=25.0)
    for frame in (180, 210, 240, 270):
        tracker.observe((23, 21), frame=frame, fps=25.0)
    tracker.observe((2, 0), frame=300, fps=25.0)
    tracker.observe((2, 0), frame=315, fps=25.0)
    assert tracker.finished_games == [(23, 21)]
    assert tracker.current == (2, 0)
    assert [e["game"] for e in tracker.events] == [1, 2, 2]


def test_score_tracker_uses_the_scoreboard_column_for_the_game():
    tracker = ScoreChangeTracker(confirmations_required=1, resync_confirmations=3)
    tracker.observe((20, 15), frame=0, fps=25.0, column=0)
    # The final point was missed, then game two's column appears.
    for frame in (30, 60, 90):
        tracker.observe((0, 0), frame=frame, fps=25.0, column=1)
    # Game one's final score, shown late in its own column, changes nothing.
    for frame in (120, 150, 180, 210):
        tracker.observe((21, 15), frame=frame, fps=25.0, column=0)
    tracker.observe((1, 0), frame=240, fps=25.0, column=1)
    tracker.observe((1, 0), frame=270, fps=25.0, column=1)
    assert tracker.game == 2
    assert tracker.current == (1, 0)
    assert tracker.events[-1]["game"] == 2


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
