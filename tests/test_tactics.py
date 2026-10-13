"""Unit tests for CoachAI-aligned tactical analytics."""

from ai_classifier.analytics import (
    analyze_coachai_plus_tactics,
    analyze_scoreboard,
    build_tactical_analytics,
    extract_tactical_patterns,
    suppress_weak_duplicate_serves,
)


def test_scoreboard_calculates_winner_and_scores():
    predictions = [
        # Rally 1 (3 hits, upper ends with smash)
        {"refined_frame": 10, "stroke": "serve", "hitter": "lower", "rally": 1, "ball_round": 1, "confidence": 0.99},
        {"refined_frame": 35, "stroke": "lift", "hitter": "lower", "rally": 1, "ball_round": 2, "confidence": 0.90},
        {"refined_frame": 60, "stroke": "smash", "hitter": "upper", "rally": 1, "ball_round": 3, "confidence": 0.95},
        # Rally 2 (2 hits, upper serve, lower ends with net_attack)
        {"refined_frame": 600, "stroke": "serve", "hitter": "upper", "rally": 2, "ball_round": 1, "confidence": 0.99},
        {"refined_frame": 625, "stroke": "net_attack", "hitter": "lower", "rally": 2, "ball_round": 2, "confidence": 0.88},
    ]

    names = {"upper": "Tai Tzu Ying", "lower": "Chen Yu Fei"}
    scoreboard = analyze_scoreboard(predictions, player_names=names)

    assert scoreboard["estimated_score"]["upper"] == 1
    assert scoreboard["estimated_score"]["lower"] == 1
    assert scoreboard["estimated_score"]["text"] == "1 - 1"
    assert scoreboard["estimated_score"]["is_estimate"] is True
    assert len(scoreboard["rallies"]) == 2

    # Rally 1 winner should be upper (smash)
    assert scoreboard["rallies"][0]["winner"] == "upper"
    assert scoreboard["rallies"][0]["winner_name"] == "Tai Tzu Ying"
    assert "winner_smash" in scoreboard["rallies"][0]["win_reason"]

    # Rally 2 winner should be lower (net_attack)
    assert scoreboard["rallies"][1]["winner"] == "lower"
    assert scoreboard["rallies"][1]["winner_name"] == "Chen Yu Fei"


def test_scoreboard_excludes_partial_edge_rallies():
    predictions = [
        {"refined_frame": 10, "stroke": "net_attack", "hitter": "upper", "rally": 1, "confidence": 0.9},
        {"refined_frame": 600, "stroke": "serve", "hitter": "lower", "rally": 2, "confidence": 0.9},
        {"refined_frame": 630, "stroke": "smash", "hitter": "upper", "rally": 2, "confidence": 0.9},
    ]
    scoreboard = analyze_scoreboard(predictions, total_frames=9000)
    assert scoreboard["rallies"][0]["partial_start"] is True
    assert scoreboard["rallies"][0]["included_in_estimated_score"] is False
    assert scoreboard["estimated_score"]["complete_rallies_counted"] == 1


def test_tactical_patterns_extracts_transitions():
    predictions = [
        {"stroke": "serve", "hitter": "lower", "rally": 1},
        {"stroke": "net_shot", "hitter": "upper", "rally": 1},
        {"stroke": "lift", "hitter": "lower", "rally": 1},
        {"stroke": "smash", "hitter": "upper", "rally": 1, "confidence": 0.95},
    ]
    patterns = extract_tactical_patterns(predictions)
    assert len(patterns["top_transitions_2gram"]) >= 1
    transitions = [t["sequence"] for t in patterns["top_transitions_2gram"]]
    assert "serve -> net_shot" in transitions
    assert "net_shot -> lift" in transitions
    assert "lift -> smash" in transitions


def test_spatial_grid_rotates_upper_side_to_player_perspective():
    from ai_classifier.analytics.match_analysis import analyze_spatial_court_grid

    predictions = [
        {"hitter": "upper", "analytics": {"hitting_area_3x3": 2}},
        {"hitter": "lower", "analytics": {"hitting_area_3x3": 8}},
    ]
    result = analyze_spatial_court_grid(predictions)
    assert result["upper"][7]["name"] == "Back-Center"
    assert result["upper"][7]["count"] == 1
    assert result["lower"][7]["count"] == 1


def test_build_tactical_analytics_full_pipeline():
    predictions = [
        {"refined_frame": 10, "stroke": "serve", "hitter": "lower", "rally": 1, "ball_round": 1, "confidence": 0.95},
        {"refined_frame": 40, "stroke": "smash", "hitter": "upper", "rally": 1, "ball_round": 2, "confidence": 0.92},
    ]
    report = build_tactical_analytics(predictions)
    assert "scoreboard" in report
    assert "tactical_patterns" in report
    assert "spatial_distribution_3x3" in report
    assert "upper" in report["spatial_distribution_3x3"]
    assert "lower" in report["spatial_distribution_3x3"]
    assert "player_profiles" in report
    assert "radar_chart" in report
    assert "response_transition_matrix" in report
    assert "coach_insights" in report
    assert "structured_tactics" in report
    assert "movement" in report
    assert report["total_records"] == 2

    # Radar chart validation
    radar = report["radar_chart"]
    assert "axes" in radar
    assert len(radar["axes"]) == 6
    assert len(radar["upper"]) == 6
    assert len(radar["lower"]) == 6
    assert all(0 <= val <= 100 for val in radar["upper"])
    assert all(0 <= val <= 100 for val in radar["lower"])

    # Player profile validation
    upper_prof = report["player_profiles"]["upper"]
    assert "playstyle" in upper_prof
    assert "label" in upper_prof["playstyle"]
    assert "radar_attributes" in upper_prof
    assert "attack" in upper_prof["radar_attributes"]

    # Coach insights validation
    assert len(report["coach_insights"]) >= 1
    for ins in report["coach_insights"]:
        assert "category" in ins
        assert "insight" in ins


def test_coach_insight_suppresses_tiny_response_samples():
    from ai_classifier.analytics.match_analysis import generate_coach_insights

    profiles = {"profiles": {"upper": {}, "lower": {}}}
    matrix = {
        "upper": {
            "smash": [
                {"response_stroke": "net_shot", "count": 2, "percentage": 100.0}
            ]
        }
    }
    scoreboard = {"estimated_score": {"text": "0 - 0", "leader": "draw"}}
    insights = generate_coach_insights(profiles, matrix, scoreboard)
    assert all(row["category"] != "Tactical Response Pattern" for row in insights)


def test_stroke_analytics_builds_landing_proxy_and_speeds() -> None:
    import numpy as np
    from ai_classifier.analytics import build_stroke_analytics

    context = {
        "frame_indices": [100, 101, 102, 103, 104],
        "contact_index": 1,
        "shuttle_image": np.array(
            [[0.1, 0.2], [0.2, 0.3], [0.3, 0.4], [0.4, 0.5], [0.5, 0.6]],
            dtype=np.float32,
        ),
        "shuttle_court": np.array(
            [[0.1, 0.1], [0.2, 0.2], [0.3, 0.3], [0.4, 0.4], [0.5, 0.5]],
            dtype=np.float32,
        ),
        "player_court": np.tile([0.25, 0.75], (5, 1)).astype(np.float32),
        "opponent_court": np.tile([0.7, 0.2], (5, 1)).astype(np.float32),
        "hitter_wrists": np.array(
            [[0.1, 0.2], [0.2, 0.2], [0.3, 0.2], [0.4, 0.2], [0.5, 0.2]],
            dtype=np.float32,
        ),
    }
    result = build_stroke_analytics(
        context, stroke="smash", fps=30.0, next_hit_frame=106
    )
    assert result["landing_position_court_proxy"] == [0.5, 0.5]
    assert result["landing_area_3x3_proxy"] == 5
    assert result["attack_state_rule"] == "attacking"
    assert result["average_shuttle_speed_image_per_second"] > 0
    assert 0.0 < result["landing_proxy_confidence"] <= 0.65


def test_coachai_plus_detects_pressure_and_counterattack():
    predictions = [
        {"rally": 1, "ball_round": 1, "hitter": "lower", "stroke": "smash", "analytics": {}},
        {"rally": 1, "ball_round": 2, "hitter": "upper", "stroke": "net_shot", "analytics": {}},
        {"rally": 1, "ball_round": 3, "hitter": "lower", "stroke": "smash", "analytics": {}},
        {"rally": 1, "ball_round": 4, "hitter": "upper", "stroke": "smash", "analytics": {}},
    ]
    result = analyze_coachai_plus_tactics(predictions)
    upper = next(row for row in result["rallies"] if row["player"] == "upper")
    types = {row["type"] for row in upper["detected_tactics"]}
    assert "full_court_pressure" in types
    assert "defensive_counterattack" in types
    assert upper["primary_tactic"] == "defensive_counterattack"


def test_coachai_plus_marks_spatial_tactics_as_proxy_based():
    predictions = []
    for ball_round, point in enumerate(
        ([0.2, 0.6], [0.8, 0.9], [0.2, 0.9], [0.8, 0.6]), start=1
    ):
        predictions.append(
            {
                "rally": 1,
                "ball_round": ball_round,
                "hitter": "upper",
                "stroke": "clear",
                "analytics": {"landing_position_court_proxy": point},
            }
        )
    result = analyze_coachai_plus_tactics(predictions)
    upper = result["rallies"][0]
    four_corner = next(
        row for row in upper["detected_tactics"] if row["type"] == "four_corner"
    )
    assert four_corner["uses_landing_proxy"] is True
    assert four_corner["confidence_calibrated"] is False
    assert four_corner["evidence_quality"] == "complete_proxy_coverage"


def test_suppresses_weak_duplicate_serve_without_shuttle_evidence():
    predictions = [
        {
            "frame": 100,
            "refined_frame": 100,
            "stroke": "serve",
            "hit_score": 0.44,
            "analytics": {"shuttle_visibility_ratio": 0.0},
        },
        {
            "frame": 145,
            "refined_frame": 145,
            "stroke": "serve",
            "hit_score": 0.86,
            "analytics": {"shuttle_visibility_ratio": 1.0},
        },
    ]
    kept, audit = suppress_weak_duplicate_serves(predictions)
    assert [row["frame"] for row in kept] == [145]
    assert audit[0]["frame"] == 100


def test_keeps_low_score_serve_when_shuttle_is_visible():
    predictions = [
        {
            "frame": 100,
            "stroke": "serve",
            "hit_score": 0.44,
            "analytics": {"shuttle_visibility_ratio": 0.8},
        },
        {
            "frame": 145,
            "stroke": "serve",
            "hit_score": 0.86,
            "analytics": {"shuttle_visibility_ratio": 1.0},
        },
    ]
    kept, audit = suppress_weak_duplicate_serves(predictions)
    assert len(kept) == 2
    assert audit == []


def test_prune_trailing_phantom_hits_removes_dead_shuttle_echo():
    from ai_classifier.analytics import prune_trailing_phantom_hits

    predictions = [
        {"frame": 100, "stroke": "serve", "hitter": "lower", "rally": 1},
        {"frame": 130, "stroke": "smash", "hitter": "upper", "rally": 1},
        {"frame": 142, "stroke": "net_shot", "hitter": "lower", "rally": 1},
    ]
    cleaned = prune_trailing_phantom_hits(predictions, fps=30.0)
    assert len(cleaned) == 2
    assert cleaned[-1]["hitter"] == "upper"
    assert cleaned[-1]["stroke"] == "smash"


def test_resolve_rally_winner_recognizes_decisive_drop():
    from ai_classifier.analytics.match_analysis import resolve_rally_winner

    hits = [
        {"frame": 100, "stroke": "serve", "hitter": "lower", "confidence": 0.99},
        {"frame": 140, "stroke": "drop", "hitter": "upper", "confidence": 0.88},
    ]
    winner, reason = resolve_rally_winner(hits)
    assert winner == "upper"
    assert reason == "winner_drop"

