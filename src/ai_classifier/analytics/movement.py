"""Sparse player-movement metrics derived from hit-time court positions."""

from __future__ import annotations

from collections import defaultdict
from math import hypot
from typing import Any


COURT_WIDTH_METERS = 6.10
COURT_LENGTH_METERS = 13.40


def _meters(point: list[float] | None) -> tuple[float, float] | None:
    if not isinstance(point, list) or len(point) != 2:
        return None
    return float(point[0]) * COURT_WIDTH_METERS, float(point[1]) * COURT_LENGTH_METERS


def analyze_sparse_movement(predictions: list[dict[str, Any]]) -> dict[str, Any]:
    """Estimate a lower bound on movement from positions sampled at hit times.

    This intentionally does not claim full running distance: motion between hit
    windows is unavailable in the current cached pipeline.
    """
    by_rally: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for prediction in predictions:
        by_rally[int(prediction.get("rally", 1))].append(prediction)

    totals = {"upper": 0.0, "lower": 0.0}
    samples = {"upper": 0, "lower": 0}
    rally_rows: list[dict[str, Any]] = []
    for rally_id, hits in sorted(by_rally.items()):
        distances = {"upper": 0.0, "lower": 0.0}
        previous: dict[str, tuple[float, float] | None] = {"upper": None, "lower": None}
        rally_samples = {"upper": 0, "lower": 0}

        for hit in hits:
            hitter = hit.get("hitter")
            analytics = hit.get("analytics") or {}
            positions = {
                hitter: _meters(analytics.get("player_position_court")),
                "lower" if hitter == "upper" else "upper": _meters(
                    analytics.get("opponent_position_court")
                ),
            }
            for side, point in positions.items():
                if side not in distances or point is None:
                    continue
                rally_samples[side] += 1
                samples[side] += 1
                if previous[side] is not None:
                    distances[side] += hypot(
                        point[0] - previous[side][0], point[1] - previous[side][1]
                    )
                previous[side] = point

        for side in totals:
            totals[side] += distances[side]
        rally_rows.append(
            {
                "rally_id": rally_id,
                "lower_bound_distance_meters": {
                    side: round(distance, 3) for side, distance in distances.items()
                },
                "position_samples": rally_samples,
            }
        )

    return {
        "method": "hit_time_position_lower_bound",
        "is_full_tracking_distance": False,
        "court_dimensions_meters": {
            "width": COURT_WIDTH_METERS,
            "length": COURT_LENGTH_METERS,
        },
        "total_lower_bound_distance_meters": {
            side: round(distance, 3) for side, distance in totals.items()
        },
        "position_samples": samples,
        "rallies": rally_rows,
        "limitations": [
            "Positions are sampled near detected hits, not on every video frame.",
            "The values are lower bounds and must not be presented as total running distance.",
        ],
    }
