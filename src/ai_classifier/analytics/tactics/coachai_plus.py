"""Rule-based tactic detection adapted from published CoachAI+ definitions.

This is an independent implementation based on the tactic definitions in the
CoachAI+ documentation (KuangDW/CoachAI-Plus). The upstream repository did not
declare a software license when inspected, so no source code is copied here.

CoachAI+ uses manually annotated 32-zone landing locations. This pipeline has
monocular 2-D landing proxies, so spatial tactics are explicitly marked as
proxy-based and use neutral left/right labels instead of forehand/backhand.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any, Callable


TACTIC_LABELS = {
    "full_court_pressure": "Full-Court Pressure",
    "defensive_counterattack": "Defensive Counterattack",
    "four_corner": "Four-Corner Movement",
    "left_side_lock": "Left-Side Lock",
    "right_side_lock": "Right-Side Lock",
    "front_court_lock": "Front-Court Lock",
    "back_court_lock": "Back-Court Lock",
    "four_corner_clear_drop": "Four-Corner Clear/Drop",
    "no_tactic": "No Tactic",
}


def _merge_intervals(intervals: list[tuple[int, int]]) -> list[tuple[int, int]]:
    if not intervals:
        return []
    merged: list[list[int]] = []
    for start, end in sorted(intervals):
        if merged and start <= merged[-1][1] + 1:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return [(start, end) for start, end in merged]


def _windows(
    values: list[Any],
    *,
    length: int,
    predicate: Callable[[list[Any]], bool],
) -> list[tuple[int, int]]:
    return _merge_intervals(
        [
            (index, index + length - 1)
            for index in range(len(values) - length + 1)
            if predicate(values[index : index + length])
        ]
    )


def _landing(hit: dict[str, Any]) -> tuple[float, float] | None:
    analytics = hit.get("analytics") or {}
    point = analytics.get("landing_position_court_proxy")
    if not isinstance(point, list) or len(point) != 2:
        return None
    return float(point[0]), float(point[1])


def _target_zone(hit: dict[str, Any]) -> str | None:
    point = _landing(hit)
    if point is None:
        return None
    x, y = point
    side = "left" if x < 0.5 else "right"
    # Distance from the net (court midpoint) is orientation-independent.
    depth = "back" if abs(y - 0.5) >= 0.25 else "front"
    return f"{depth}_{side}"


def _interval_payload(
    tactic: str,
    intervals: list[tuple[int, int]],
    player_hits: list[dict[str, Any]],
    *,
    evidence_quality: str,
    uses_landing_proxy: bool,
) -> dict[str, Any]:
    ranges = []
    for start, end in intervals:
        ranges.append(
            {
                "start_ball_round": int(player_hits[start].get("ball_round", start + 1)),
                "end_ball_round": int(player_hits[end].get("ball_round", end + 1)),
            }
        )
    return {
        "type": tactic,
        "label": TACTIC_LABELS[tactic],
        "intervals": ranges,
        "evidence_quality": evidence_quality,
        "confidence_calibrated": False,
        "uses_landing_proxy": uses_landing_proxy,
    }


def _full_court_pressure(player_hits: list[dict[str, Any]]) -> list[tuple[int, int]]:
    scores = [
        2 if hit.get("stroke") == "smash"
        else 1 if hit.get("stroke") in {"drop", "net_shot"}
        else 0
        for hit in player_hits
    ]
    intervals: list[tuple[int, int]] = []
    for length, minimum in ((2, 3), (3, 4), (4, 5)):
        intervals.extend(
            _windows(scores, length=length, predicate=lambda rows, m=minimum: sum(rows) >= m)
        )
    return _merge_intervals(intervals)


def _defensive_counterattack(
    rally_hits: list[dict[str, Any]], player: str
) -> list[tuple[int, int]]:
    intervals: list[tuple[int, int]] = []
    start: int | None = None
    for index, hit in enumerate(rally_hits):
        if hit.get("hitter") != player and hit.get("stroke") == "smash" and start is None:
            start = index
        elif (
            start is not None
            and hit.get("hitter") == player
            and hit.get("stroke") in {"smash", "drop", "net_shot"}
        ):
            intervals.append((start, index))
            start = None
    return intervals


def _spatial_intervals(
    player_hits: list[dict[str, Any]], tactic: str
) -> list[tuple[int, int]]:
    zones = [_target_zone(hit) for hit in player_hits]
    if tactic == "left_side_lock":
        return _windows(
            zones,
            length=4,
            predicate=lambda rows: rows[0] is not None
            and rows[-1] is not None
            and rows[0].endswith("_left")
            and rows[-1].endswith("_left")
            and sum(zone is not None and zone.endswith("_left") for zone in rows) >= 2,
        )
    if tactic == "right_side_lock":
        return _windows(
            zones,
            length=4,
            predicate=lambda rows: rows[0] is not None
            and rows[-1] is not None
            and rows[0].endswith("_right")
            and rows[-1].endswith("_right")
            and sum(zone is not None and zone.endswith("_right") for zone in rows) >= 2,
        )
    if tactic == "front_court_lock":
        return _windows(
            zones,
            length=3,
            predicate=lambda rows: rows[0] is not None
            and rows[-1] is not None
            and rows[0].startswith("front_")
            and rows[-1].startswith("front_")
            and sum(zone is not None and zone.startswith("front_") for zone in rows) >= 2,
        )
    if tactic == "back_court_lock":
        eligible = [
            zone if hit.get("stroke") != "smash" else None
            for zone, hit in zip(zones, player_hits)
        ]
        return _windows(
            eligible,
            length=4,
            predicate=lambda rows: rows[0] is not None
            and rows[-1] is not None
            and rows[0].startswith("back_")
            and rows[-1].startswith("back_")
            and sum(zone is not None and zone.startswith("back_") for zone in rows) >= 2,
        )
    raise ValueError(f"Unknown spatial tactic: {tactic}")


def _four_corner_clear_drop(player_hits: list[dict[str, Any]]) -> list[tuple[int, int]]:
    zones = [_target_zone(hit) for hit in player_hits]
    changes = [False]
    for previous, current in zip(zones, zones[1:]):
        changes.append(previous is not None and current is not None and previous != current)
    return _windows(
        changes,
        length=4,
        predicate=lambda rows: sum(rows) >= 3 and rows[-1],
    )


def analyze_coachai_plus_tactics(
    predictions: list[dict[str, Any]],
    player_names: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Detect CoachAI+-style tactics for every rally and player."""
    names = player_names or {"upper": "Upper Player", "lower": "Lower Player"}
    rallies: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for prediction in predictions:
        rallies[int(prediction.get("rally", 1))].append(prediction)

    records: list[dict[str, Any]] = []
    usage: dict[str, Counter[str]] = {"upper": Counter(), "lower": Counter()}
    for rally_id, rally_hits in sorted(rallies.items()):
        for player in ("upper", "lower"):
            player_hits = [hit for hit in rally_hits if hit.get("hitter") == player]
            if not player_hits:
                continue
            found: list[dict[str, Any]] = []

            fcp = _full_court_pressure(player_hits)
            if fcp:
                found.append(
                    _interval_payload(
                        "full_court_pressure", fcp, player_hits,
                        evidence_quality="stroke_sequence_rule", uses_landing_proxy=False,
                    )
                )

            dc = _defensive_counterattack(rally_hits, player)
            if dc:
                # DC indices refer to the complete rally, not only player hits.
                found.append(
                    {
                        "type": "defensive_counterattack",
                        "label": TACTIC_LABELS["defensive_counterattack"],
                        "intervals": [
                            {
                                "start_ball_round": int(rally_hits[start].get("ball_round", start + 1)),
                                "end_ball_round": int(rally_hits[end].get("ball_round", end + 1)),
                            }
                            for start, end in dc
                        ],
                        "evidence_quality": "stroke_sequence_rule",
                        "confidence_calibrated": False,
                        "uses_landing_proxy": False,
                    }
                )

            spatial: dict[str, list[tuple[int, int]]] = {}
            for tactic in (
                "left_side_lock", "right_side_lock",
                "front_court_lock", "back_court_lock",
            ):
                intervals = _spatial_intervals(player_hits, tactic)
                spatial[tactic] = intervals
                if intervals:
                    coverage = sum(_landing(hit) is not None for hit in player_hits) / len(player_hits)
                    found.append(
                        _interval_payload(
                            tactic, intervals, player_hits,
                            evidence_quality=(
                                "complete_proxy_coverage" if coverage == 1.0
                                else "partial_proxy_coverage"
                            ),
                            uses_landing_proxy=True,
                        )
                    )

            fccd = _four_corner_clear_drop(player_hits)
            if fccd:
                coverage = sum(_landing(hit) is not None for hit in player_hits) / len(player_hits)
                found.append(
                    _interval_payload(
                        "four_corner_clear_drop", fccd, player_hits,
                        evidence_quality=(
                            "complete_proxy_coverage" if coverage == 1.0
                            else "partial_proxy_coverage"
                        ),
                        uses_landing_proxy=True,
                    )
                )

            four_corner_intervals = _merge_intervals(
                [interval for rows in spatial.values() for interval in rows] + fccd
            )
            if four_corner_intervals:
                coverage = sum(_landing(hit) is not None for hit in player_hits) / len(player_hits)
                found.append(
                    _interval_payload(
                        "four_corner", four_corner_intervals, player_hits,
                        evidence_quality=(
                            "complete_proxy_coverage" if coverage == 1.0
                            else "partial_proxy_coverage"
                        ),
                        uses_landing_proxy=True,
                    )
                )

            primary = "no_tactic"
            if dc and any(end >= len(rally_hits) - 2 for _, end in dc):
                primary = "defensive_counterattack"
            elif fcp or four_corner_intervals:
                fcp_end = max((end for _, end in fcp), default=-1)
                corner_end = max((end for _, end in four_corner_intervals), default=-1)
                primary = "full_court_pressure" if fcp_end >= corner_end else "four_corner"

            usage[player][primary] += 1
            records.append(
                {
                    "rally_id": rally_id,
                    "player": player,
                    "player_name": names.get(player, player),
                    "primary_tactic": primary,
                    "primary_tactic_label": TACTIC_LABELS[primary],
                    "detected_tactics": found,
                    "shot_count": len(player_hits),
                }
            )

    return {
        "method": "coachai_plus_rules_adapted",
        "source_attribution": "CoachAI+ published tactic definitions (KuangDW/CoachAI-Plus)",
        "limitations": [
            "Landing locations are monocular 2-D proxies.",
            "Left/right locks replace forehand/backhand locks because handedness is unknown.",
            "Detected tactics are rule-based estimates, not expert annotations.",
        ],
        "rallies": records,
        "usage_by_player": {
            player: dict(sorted(counter.items())) for player, counter in usage.items()
        },
    }
