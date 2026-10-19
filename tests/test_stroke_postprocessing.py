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
