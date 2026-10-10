"""Unit tests for Rally Boundary detection (ML model and heuristic)."""

from pathlib import Path
import joblib
import pytest
from ai_classifier.analytics import assign_rallies
from ai_classifier.analytics.stroke import predict_rally_boundary

MODEL_PATH = Path("work_dirs/rally_boundary_detector.joblib")


def test_predict_rally_boundary_rule_heuristic():
    p1 = {"refined_frame": 10, "stroke": "serve", "hitter": "upper"}
    p2 = {"refined_frame": 30, "stroke": "net_shot", "hitter": "lower"}
    p3 = {"refined_frame": 250, "stroke": "drop", "hitter": "upper"}
    p4 = {"refined_frame": 280, "stroke": "serve", "hitter": "lower"}

    # Start of match -> new rally
    assert predict_rally_boundary(None, p1, fps=30.0) is True
    # Fast return (0.67s) -> same rally
    assert predict_rally_boundary(p1, p2, fps=30.0, max_gap_seconds=3.0) is False
    # Large gap (7.33s > 3.0s) -> new rally
    assert predict_rally_boundary(p2, p3, fps=30.0, max_gap_seconds=3.0) is True
    # Serve stroke -> new rally
    assert predict_rally_boundary(p3, p4, fps=30.0, max_gap_seconds=3.0) is True


def test_predict_rally_boundary_with_ml_model():
    if not MODEL_PATH.exists():
        pytest.skip("Rally Boundary model checkpoint not found")

    payload = joblib.load(MODEL_PATH)
    model = payload["model"]

    # In-rally shot: 0.8s gap, upper to lower, net_shot after smash
    p_prev = {"refined_frame": 100, "stroke": "smash", "hitter": "upper"}
    p_curr = {"refined_frame": 124, "stroke": "net_shot", "hitter": "lower"}
    assert predict_rally_boundary(p_prev, p_curr, fps=30.0, model=model) is False

    # Inter-rally transition: 18.0s gap, new serve
    p_serve = {"refined_frame": 664, "stroke": "serve", "hitter": "upper"}
    assert predict_rally_boundary(p_curr, p_serve, fps=30.0, model=model) is True

    # High defensive lift: 2.2s gap in play -> should NOT split rally
    p_lift = {"refined_frame": 100, "stroke": "lift", "hitter": "lower"}
    p_drop = {"refined_frame": 166, "stroke": "drop", "hitter": "upper"}
    assert predict_rally_boundary(p_lift, p_drop, fps=30.0, model=model) is False


def test_assign_rallies_end_to_end():
    predictions = [
        {"refined_frame": 10, "stroke": "serve", "hitter": "lower"},
        {"refined_frame": 35, "stroke": "net_shot", "hitter": "upper"},
        {"refined_frame": 60, "stroke": "smash", "hitter": "lower"},
        {"refined_frame": 600, "stroke": "serve", "hitter": "upper"},
        {"refined_frame": 630, "stroke": "lift", "hitter": "lower"},
    ]
    assign_rallies(predictions, fps=30.0)
    assert [p["rally"] for p in predictions] == [1, 1, 1, 2, 2]
    assert [p["ball_round"] for p in predictions] == [1, 2, 3, 1, 2]


def test_ml_cannot_split_short_non_serve_exchange():
    class AlwaysBoundary:
        def predict(self, features):
            return [1]

    previous = {"refined_frame": 100, "stroke": "clear", "hitter": "upper"}
    current = {"refined_frame": 168, "stroke": "net_shot", "hitter": "lower"}
    assert predict_rally_boundary(
        previous, current, fps=30.0, model=AlwaysBoundary(), max_gap_seconds=3.0
    ) is False
