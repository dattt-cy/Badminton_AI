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
        # Upper player on their own baseline, image-left = their right.
        {"hitter": "upper", "analytics": {"player_position_court": [0.1, 0.05]}},
        # Lower player close to the net, centre.
        {"hitter": "lower", "analytics": {"player_position_court": [0.5, 0.55]}},
    ]
    result = analyze_spatial_court_grid(predictions)
    assert result["upper"][8]["name"] == "Back-Right"
    assert result["upper"][8]["count"] == 1
    assert result["lower"][1]["name"] == "Front-Center"
    assert result["lower"][1]["count"] == 1


def test_scoreboard_infers_server_when_serve_was_missed():
    predictions = [
        {"refined_frame": 10, "stroke": "serve", "hitter": "lower", "rally": 1, "confidence": 0.99},
        {"refined_frame": 35, "stroke": "lift", "hitter": "upper", "rally": 1, "confidence": 0.90},
        {"refined_frame": 60, "stroke": "smash", "hitter": "lower", "rally": 1, "confidence": 0.95},
        # Rally 2's serve (by upper) was not detected; the first hit is lower's return.
        {"refined_frame": 600, "stroke": "net_attack", "hitter": "lower", "rally": 2, "confidence": 0.9},
        {"refined_frame": 625, "stroke": "lift", "hitter": "upper", "rally": 2, "confidence": 0.9},
        {"refined_frame": 650, "stroke": "smash", "hitter": "lower", "rally": 2, "confidence": 0.9},
    ]
    rallies = analyze_scoreboard(predictions)["rallies"]
    assert rallies[1]["server"] == "upper"
    # Upper serves rally 2, so upper won rally 1 despite lower's final smash.
    assert rallies[0]["winner"] == "upper"


def test_scoreboard_maps_ocr_first_row_to_lower_player():
    predictions = []
    # The lower player wins rallies 1-2 (serves 2-3), the upper player wins rally 3 (serves 4).
    for rally, server in enumerate(("lower", "lower", "lower", "upper"), start=1):
        base = rally * 900
        receiver = "upper" if server == "lower" else "lower"
        predictions += [
            {"refined_frame": base, "stroke": "serve", "hitter": server, "rally": rally},
            {"refined_frame": base + 30, "stroke": "lift", "hitter": receiver, "rally": rally},
            {"refined_frame": base + 60, "stroke": "smash", "hitter": server, "rally": rally},
        ]
    # Scoreboard rows are (lower player, upper player), starting at 3-5.
    ocr = {
        "initial_score": [3, 5],
        "events": [
            {"time_seconds": 33.0, "previous_score": [3, 5], "score": [4, 5], "point_winner": "first"},
            {"time_seconds": 63.0, "previous_score": [4, 5], "score": [5, 5], "point_winner": "first"},
            {"time_seconds": 93.0, "previous_score": [5, 5], "score": [5, 6], "point_winner": "second"},
        ],
    }
    result = analyze_scoreboard(predictions, scoreboard_ocr=ocr)
    assert result["estimated_score"]["ocr_first_row_side"] == "lower"
    assert [r["winner"] for r in result["rallies"][:3]] == ["lower", "lower", "upper"]
    assert result["rallies"][2]["score_text"] == "6 - 5"


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
    assert radar["axes"] == ["attack_share", "net_share", "defence_share", "placement_share"]
    assert len(radar["upper"]) == 4
    assert len(radar["lower"]) == 4
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


def test_scoreboard_resets_score_after_game_break():
    predictions = []
    # Two rallies in game 1, then a 2-minute break, then one rally in game 2.
    for rally, start in ((1, 0), (2, 900), (3, 900 + 120 * 30)):
        predictions += [
            {"refined_frame": start, "stroke": "serve", "hitter": "lower", "rally": rally, "confidence": 0.9},
            {"refined_frame": start + 25, "stroke": "lift", "hitter": "upper", "rally": rally, "confidence": 0.9},
            {"refined_frame": start + 50, "stroke": "smash", "hitter": "lower", "rally": rally, "confidence": 0.9},
        ]
    scoreboard = analyze_scoreboard(predictions)
    assert [r["game_number"] for r in scoreboard["rallies"]] == [1, 1, 2]
    assert [g["score_text"] for g in scoreboard["games"]] == ["0 - 2", "0 - 1"]
    assert scoreboard["estimated_score"]["text"] == "0 - 1"
    assert scoreboard["estimated_score"]["game_number"] == 2


def test_coach_insights_describe_both_players_the_same_way():
    from ai_classifier.analytics.match_analysis import generate_coach_insights

    profile = {
        "stroke_counts": {"smash": 4, "net_shot": 2, "lift": 2, "clear": 2},
        "total_shots": 10,
        "playstyle": {"label": "Attack-oriented segment"},
        "error_analysis": {"most_vulnerable_zone": {"zone_id": 8, "losses_incurred": 2}},
    }
    profiles = {"profiles": {"upper": dict(profile), "lower": dict(profile)}}
    scoreboard = {"estimated_score": {"text": "0 - 0", "leader": "draw"}}
    rows = [r for r in generate_coach_insights(profiles, {}, scoreboard, {"upper": "A", "lower": "B"})
            if r["category"] == "Lối đánh"]
    assert len(rows) == 2
    assert rows[0]["insight"].replace(" A ", " X ").replace(" A.", " X.") == rows[1]["insight"].replace(" B ", " X ").replace(" B.", " X.")
    assert "40% số cú" in rows[0]["insight"] and "cuối sân, giữa (2 pha)" in rows[0]["insight"]


def test_match_insight_says_where_the_score_comes_from():
    from ai_classifier.analytics.match_analysis import generate_coach_insights

    profiles = {"profiles": {"upper": {}, "lower": {}}}
    read = {"estimated_score": {"text": "11 - 6", "leader": "upper", "is_estimate": False, "game_number": 1}}
    counted = {"estimated_score": {"text": "11 - 6", "leader": "upper", "is_estimate": True, "game_number": 1}}
    names = {"upper": "A", "lower": "B"}
    assert "đọc từ bảng điểm" in generate_coach_insights(profiles, {}, read, names)[0]["insight"]
    assert "AI tự đếm" in generate_coach_insights(profiles, {}, counted, names)[0]["insight"]


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
    assert all(row["category"] != "Cách đỡ đập" for row in insights)


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


def test_coachai_plus_keeps_lone_counterattack_ending_mid_rally():
    predictions = [
        {"rally": 1, "ball_round": 1, "hitter": "lower", "stroke": "smash", "analytics": {}},
        {"rally": 1, "ball_round": 2, "hitter": "upper", "stroke": "drop", "analytics": {}},
        {"rally": 1, "ball_round": 3, "hitter": "lower", "stroke": "lift", "analytics": {}},
        {"rally": 1, "ball_round": 4, "hitter": "upper", "stroke": "clear", "analytics": {}},
        {"rally": 1, "ball_round": 5, "hitter": "lower", "stroke": "clear", "analytics": {}},
        {"rally": 1, "ball_round": 6, "hitter": "upper", "stroke": "lift", "analytics": {}},
    ]
    result = analyze_coachai_plus_tactics(predictions)
    upper = next(row for row in result["rallies"] if row["player"] == "upper")
    assert {row["type"] for row in upper["detected_tactics"]} == {"defensive_counterattack"}
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


def test_prune_trailing_phantom_hits_keeps_confident_net_kill():
    from ai_classifier.analytics import prune_trailing_phantom_hits

    predictions = [
        {"frame": 100, "stroke": "serve", "hitter": "lower", "rally": 1, "hit_score": 0.9},
        {"frame": 130, "stroke": "net_shot", "hitter": "upper", "rally": 1, "hit_score": 0.9},
        {"frame": 150, "stroke": "net_attack", "hitter": "lower", "rally": 1, "hit_score": 0.92},
    ]
    assert len(prune_trailing_phantom_hits(predictions, fps=30.0)) == 3
    predictions[-1]["hit_score"] = 0.80
    assert len(prune_trailing_phantom_hits(predictions, fps=30.0)) == 2


def test_resolve_rally_winner_recognizes_decisive_drop():
    from ai_classifier.analytics.match_analysis import resolve_rally_winner

    hits = [
        {"frame": 100, "stroke": "serve", "hitter": "lower", "confidence": 0.99},
        {"frame": 140, "stroke": "drop", "hitter": "upper", "confidence": 0.88},
    ]
    winner, reason = resolve_rally_winner(hits)
    assert winner == "upper"
    assert reason == "winner_drop"



def test_attach_tactic_outcomes_reports_win_rate_and_rally_length():
    from ai_classifier.analytics.tactics import attach_tactic_outcomes

    predictions = []
    # Rally 1: upper presses (smash, smash) and wins; rally 2: upper presses and loses.
    for rally in (1, 2):
        base = rally * 10.0
        predictions += [
            {"rally": rally, "ball_round": 1, "hitter": "lower", "stroke": "serve", "time_seconds": base},
            {"rally": rally, "ball_round": 2, "hitter": "upper", "stroke": "smash", "time_seconds": base + 1},
            {"rally": rally, "ball_round": 3, "hitter": "lower", "stroke": "lift", "time_seconds": base + 2},
            {"rally": rally, "ball_round": 4, "hitter": "upper", "stroke": "smash", "time_seconds": base + 3},
        ]
    scoreboard = {
        "rallies": [
            {"rally_id": 1, "winner": "upper", "included_in_estimated_score": True},
            {"rally_id": 2, "winner": "lower", "included_in_estimated_score": True},
        ]
    }
    result = attach_tactic_outcomes(analyze_coachai_plus_tactics(predictions), scoreboard)

    pressure = result["effectiveness_by_player"]["upper"]["full_court_pressure"]
    assert pressure["uses"] == 2
    assert pressure["wins"] == 1
    assert pressure["win_rate"] == 0.5
    assert pressure["avg_rally_length"] == 4

    upper_rally_1 = next(
        row for row in result["rallies"] if row["player"] == "upper" and row["rally_id"] == 1
    )
    assert upper_rally_1["won"] is True
    interval = upper_rally_1["detected_tactics"][0]["intervals"][0]
    assert (interval["start_time_seconds"], interval["end_time_seconds"]) == (11.0, 13.0)


def test_attach_tactic_outcomes_ignores_uncounted_rallies():
    from ai_classifier.analytics.tactics import attach_tactic_outcomes

    predictions = [
        {"rally": 1, "ball_round": 1, "hitter": "upper", "stroke": "smash"},
        {"rally": 1, "ball_round": 2, "hitter": "lower", "stroke": "lift"},
    ]
    scoreboard = {"rallies": [{"rally_id": 1, "winner": "upper", "included_in_estimated_score": False}]}
    result = attach_tactic_outcomes(analyze_coachai_plus_tactics(predictions), scoreboard)
    stats = result["effectiveness_by_player"]["upper"]["short_rally"]
    assert stats["decided_rallies"] == 0
    assert stats["win_rate"] is None


def test_coachai_plus_reports_dominant_four_corner_subtactic():
    predictions = [
        {
            "rally": 1,
            "ball_round": index,
            "hitter": "upper",
            "stroke": "clear",
            "analytics": {"landing_position_court_proxy": point},
        }
        # All four shots land deep: back-court lock covers the whole window.
        for index, point in enumerate(([0.2, 0.95], [0.8, 0.95], [0.3, 0.9], [0.7, 0.9]), start=1)
    ]
    upper = analyze_coachai_plus_tactics(predictions)["rallies"][0]
    assert upper["primary_tactic"] == "four_corner"
    assert upper["four_corner_subtactic"] == "back_court_lock"


def _hits(*rows):
    return [
        {"rally": 1, "ball_round": index, "hitter": hitter, "stroke": stroke, "analytics": {}}
        for index, (hitter, stroke) in enumerate(rows, start=1)
    ]


def test_coachai_plus_detects_net_control_before_pressure():
    predictions = _hits(
        ("lower", "serve"), ("upper", "net_shot"), ("lower", "net_shot"),
        ("upper", "net_shot"), ("lower", "lift"), ("upper", "net_attack"),
    )
    upper = next(row for row in analyze_coachai_plus_tactics(predictions)["rallies"] if row["player"] == "upper")
    assert upper["primary_tactic"] == "net_control"


def test_coachai_plus_no_longer_reports_flat_exchange_or_tempo_change():
    # Both were removed: the pipeline's stroke and timing errors kept them at 30-40% agreement.
    predictions = _hits(
        ("upper", "serve"), ("lower", "lift"), ("upper", "clear"), ("lower", "drive"),
        ("upper", "drive"), ("lower", "drive"), ("upper", "lift"), ("lower", "clear"),
    )
    for row in analyze_coachai_plus_tactics(predictions)["rallies"]:
        types = {t["type"] for t in row["detected_tactics"]} | {row["primary_tactic"]}
        assert not types & {"fast_drive_exchange", "tempo_change"}


def test_coachai_plus_detects_early_attack_on_serve_return():
    predictions = _hits(("upper", "serve"), ("lower", "net_attack"), ("upper", "lift"))
    lower = next(row for row in analyze_coachai_plus_tactics(predictions)["rallies"] if row["player"] == "lower")
    assert lower["primary_tactic"] == "early_attack"


def test_coachai_plus_counts_net_kill_as_pressure():
    predictions = _hits(
        ("lower", "serve"), ("upper", "lift"), ("lower", "clear"), ("upper", "net_attack"),
        ("lower", "lift"), ("upper", "drop"),
    )
    upper = next(row for row in analyze_coachai_plus_tactics(predictions)["rallies"] if row["player"] == "upper")
    assert upper["primary_tactic"] == "full_court_pressure"


def test_coachai_plus_labels_player_with_two_shots_as_short_rally():
    predictions = _hits(("upper", "serve"), ("lower", "net_shot"), ("upper", "lift"), ("lower", "clear"))
    rows = analyze_coachai_plus_tactics(predictions)["rallies"]
    assert {row["primary_tactic"] for row in rows} == {"short_rally"}


def _timed(*rows):
    """(hitter, stroke, flight seconds, player position) per hit."""
    return [
        {
            "rally": 1, "ball_round": index, "hitter": hitter, "stroke": stroke,
            "analytics": {"flight_time_to_next_hit_seconds": flight, "player_position_court": position},
        }
        for index, (hitter, stroke, flight, position) in enumerate(rows, start=1)
    ]


def _types(predictions, player):
    row = next(r for r in analyze_coachai_plus_tactics(predictions)["rallies"] if r["player"] == player)
    return {t["type"] for t in row["detected_tactics"]}


def test_coachai_plus_detects_absorbing_pressure():
    predictions = _hits(
        ("upper", "serve"), ("lower", "lift"), ("upper", "smash"), ("lower", "lift"),
        ("upper", "smash"), ("lower", "lift"), ("upper", "net_attack"), ("lower", "net_shot"),
    )
    assert "absorb_pressure" in _types(predictions, "lower")


def test_coachai_plus_detects_running_the_opponent():
    corners = ([0.1, 0.95], [0.9, 0.6], [0.1, 0.95])  # lower player runs corner to corner
    rows = []
    for k in range(3):
        rows += [("upper", "clear", 1.2, [0.5, 0.1]), ("lower", "lift", 1.2, corners[k])]
    assert "run_opponent" in _types(_timed(*rows), "upper")
    assert "run_opponent" not in _types(_timed(*rows), "lower")


def test_style_profile_shares_are_disjoint_and_skip_serves():
    from ai_classifier.analytics.match_analysis import style_profile

    hits = [{"stroke": s, "analytics": {"flight_time_to_next_hit_seconds": f}} for s, f in (
        ("serve", 0.8), ("smash", 0.5), ("net_attack", 0.4), ("net_shot", 0.9),
        ("lift", 1.2), ("clear", 1.3), ("drop", 0.9), ("drive", 0.5), ("smash", 5.0),
    )]
    profile = style_profile(hits)
    assert profile["shares"] == {"attack_share": 38, "net_share": 12, "defence_share": 25, "placement_share": 25}
    assert profile["rally_strokes"] == 8
    assert profile["median_flight_seconds"] == 0.9  # the 5 s gap to the next rally is ignored
