from ai_classifier.analytics.stroke import suppress_weak_duplicate_serves


def _prediction(frame: int, score: float, stroke: str = "serve") -> dict:
    return {
        "frame": frame,
        "refined_frame": frame,
        "hit_score": score,
        "stroke": stroke,
        "hitter": "lower",
        "analytics": {"shuttle_visibility_ratio": 1.0},
    }


def test_close_serve_cluster_keeps_strongest_detector_peak() -> None:
    rows = [
        _prediction(100, 0.50),
        _prediction(128, 0.42),
        _prediction(151, 0.82),
        _prediction(260, 0.91, "clear"),
    ]

    kept, audit = suppress_weak_duplicate_serves(rows)

    assert [row["frame"] for row in kept] == [151, 260]
    assert [row["frame"] for row in audit] == [100, 128]
    assert all(row["reason"] == "duplicate_serve_cluster" for row in audit)


def test_distant_serves_are_preserved() -> None:
    rows = [_prediction(100, 0.70), _prediction(176, 0.90)]

    kept, audit = suppress_weak_duplicate_serves(rows)

    assert kept == rows
    assert audit == []


def test_landing_comes_from_the_returning_player_position():
    from ai_classifier.analytics import apply_next_hitter_landing

    def hit(rally, hitter, position, proxy):
        return {"rally": rally, "hitter": hitter, "analytics": {
            "player_position_court": position, "landing_position_court_proxy": proxy, "landing_area_3x3_proxy": 5}}

    hits = [
        hit(1, "lower", [0.5, 0.9], [0.4, 0.4]),
        hit(1, "upper", [0.1, 0.1], [0.6, 0.6]),  # last of rally 1: keeps its trajectory proxy
        hit(2, "upper", [0.9, 0.1], [0.5, 0.5]),
        hit(2, "upper", [0.8, 0.2], None),  # same hitter twice: no return to use
    ]
    apply_next_hitter_landing(hits)
    first = hits[0]["analytics"]
    assert first["landing_position_court_proxy"] == [0.1, 0.1]
    assert first["landing_area_3x3_proxy"] == 1
    assert first["landing_source"] == "next_hitter_position"
    assert first["landing_position_court_trajectory_proxy"] == [0.4, 0.4]
    for kept in hits[1:3]:
        assert kept["analytics"]["landing_source"] == "shuttle_trajectory"
    assert hits[1]["analytics"]["landing_position_court_proxy"] == [0.6, 0.6]
    assert hits[2]["analytics"]["landing_area_3x3_proxy"] == 5
