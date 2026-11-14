"""Rule-based tactic detection adapted from published CoachAI+ definitions.

This is an independent implementation based on the tactic definitions in the
CoachAI+ documentation (KuangDW/CoachAI-Plus). The upstream repository did not
declare a software license when inspected, so no source code is copied here.

CoachAI+ uses manually annotated 32-zone landing locations. This pipeline has
monocular 2-D landing proxies, so spatial tactics are explicitly marked as
proxy-based and use neutral left/right labels instead of forehand/backhand.

Beyond the three CoachAI+ tactics, two stroke-sequence tactics common in
coaching vocabulary are added (net control, early attack), and Full-Court
Pressure also counts net kills as attacking shots, like smashes. A player with
at most two shots in a rally and nothing detected gets ``short_rally``.

Two more come from coaching material and CoachAI+'s shot "exertion" idea:
absorbing pressure (defend repeated attacks without counter-attacking) and
running the opponent (make the opponent cover long distances between shots).
Between two own hits a player moves 1.7 m at the median and 3.5 m at the 90th
percentile of the stored analyses. Fast flat exchange and tempo change were
removed: their pipeline agreement stayed at 30-40% (see docs/EXPERIMENT_NOTES.md).
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
    "net_control": "Net Control",
    "early_attack": "Early Attack",
    "absorb_pressure": "Absorbing Pressure",
    "run_opponent": "Running the Opponent",
    "short_rally": "Short Rally",
    "no_tactic": "No Tactic",
}

# A player with this few shots in a rally has no room for a tactic: label it
# as a short rally instead of "no tactic".
SHORT_RALLY_MAX_OWN_SHOTS = 2

NET_STROKES = {"net_shot", "net_attack"}
NET_CONTROL_STROKES = NET_STROKES | {"drop"}
ATTACK_STROKES = {"smash", "net_attack"}
DEFENSIVE_STROKES = {"lift", "clear", "net_shot", "drive"}

# Court size used to turn normalised court positions into metres (singles width).
COURT_LENGTH_M = 13.4
COURT_WIDTH_M = 6.1
LONG_RUN_METRES = 3.0  # ~85th percentile of a player's move between two own hits

# When two tactics end on the same hit, the earlier entry wins: specific patterns
# (net control) before the general Full-Court Pressure they overlap.
PRIMARY_PRIORITY = (
    "defensive_counterattack",
    "net_control",
    "full_court_pressure",
    "four_corner",
    "absorb_pressure",
    "run_opponent",
    "early_attack",
)


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


# Key set on a copied hit: where the opponent stood for the next shot.
NEXT_HITTER_POSITION = "_next_hitter_position"


def _landing(hit: dict[str, Any]) -> tuple[float, float] | None:
    """Where the shot went.

    The opponent's position at the next hit is used first: on ShuttleSet clips it is
    1.3 m from the annotated landing point at the median, against 3.6 m for the
    shuttle-trajectory landing proxy. The proxy remains the fallback for the last
    shot of a rally.
    """
    point = hit.get(NEXT_HITTER_POSITION)
    if not (isinstance(point, list) and len(point) == 2):
        point = (hit.get("analytics") or {}).get("landing_position_court_proxy")
    if not isinstance(point, list) or len(point) != 2:
        return None
    return float(point[0]), float(point[1])


def _with_next_hitter_positions(rally_hits: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Shallow copies of the hits carrying the next hitter's court position."""
    out = []
    for index, hit in enumerate(rally_hits):
        copy = dict(hit)
        if index + 1 < len(rally_hits):
            nxt = rally_hits[index + 1]
            position = (nxt.get("analytics") or {}).get("player_position_court")
            if nxt.get("hitter") != hit.get("hitter") and isinstance(position, list) and len(position) == 2:
                copy[NEXT_HITTER_POSITION] = position
        out.append(copy)
    return out


def _target_zone(hit: dict[str, Any]) -> str | None:
    point = _landing(hit)
    if point is None:
        return None
    x, y = point
    side = "left" if x < 0.5 else "right"
    # Distance from the net (court midpoint) is orientation-independent.
    depth = "back" if abs(y - 0.5) >= 0.25 else "front"
    return f"{depth}_{side}"


FOUR_CORNER_SUBTACTICS = (
    "left_side_lock", "right_side_lock",
    "front_court_lock", "back_court_lock",
    "four_corner_clear_drop",
)


def _time_seconds(hit: dict[str, Any]) -> float | None:
    value = hit.get("time_seconds")
    return round(float(value), 3) if value is not None else None


def _interval_ranges(
    hits: list[dict[str, Any]], intervals: list[tuple[int, int]]
) -> list[dict[str, Any]]:
    """Describe index intervals over ``hits`` as ball rounds and video times.

    Ball rounds are numbered before stroke refinement, which can swap two hits, so an
    interval spans the lowest to the highest round of its hits rather than its ends'.
    """
    ranges = []
    for start, end in intervals:
        rounds = [int(hits[i].get("ball_round", i + 1)) for i in range(start, end + 1)]
        ranges.append({
            "start_ball_round": min(rounds),
            "end_ball_round": max(rounds),
            "start_time_seconds": _time_seconds(hits[start]),
            "end_time_seconds": _time_seconds(hits[end]),
        })
    return ranges


def _dominant_subtactic(subtactics: dict[str, list[tuple[int, int]]]) -> str | None:
    """Pick the Four-Corner subtactic covering the most shots.

    Mirrors CoachAI+ levels 2-4, which compare locks by covered length and
    break ties by the latest interval end.
    """
    best: tuple[int, int] | None = None
    best_tactic = None
    for tactic in FOUR_CORNER_SUBTACTICS:
        intervals = subtactics.get(tactic) or []
        if not intervals:
            continue
        key = (
            sum(end - start + 1 for start, end in intervals),
            max(end for _, end in intervals),
        )
        if best is None or key > best:
            best, best_tactic = key, tactic
    return best_tactic


def _interval_payload(
    tactic: str,
    intervals: list[tuple[int, int]],
    player_hits: list[dict[str, Any]],
    *,
    evidence_quality: str,
    uses_landing_proxy: bool,
) -> dict[str, Any]:
    return {
        "type": tactic,
        "label": TACTIC_LABELS[tactic],
        "intervals": _interval_ranges(player_hits, intervals),
        "evidence_quality": evidence_quality,
        "confidence_calibrated": False,
        "uses_landing_proxy": uses_landing_proxy,
    }


def _full_court_pressure(player_hits: list[dict[str, Any]]) -> list[tuple[int, int]]:
    scores = [
        2 if hit.get("stroke") in ATTACK_STROKES
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




def _net_control(player_hits: list[dict[str, Any]]) -> list[tuple[int, int]]:
    """Three consecutive own shots kept short, at least two of them at the net."""
    strokes = [hit.get("stroke") for hit in player_hits]
    return _windows(
        strokes,
        length=3,
        predicate=lambda rows: all(stroke in NET_CONTROL_STROKES for stroke in rows)
        and sum(stroke in NET_STROKES for stroke in rows) >= 2,
    )


def _early_attack(
    rally_hits: list[dict[str, Any]], player: str
) -> list[tuple[int, int]]:
    """Attack on the player's first chance: the serve return, or the third shot after serving.

    Indices refer to the complete rally.
    """
    if not rally_hits or rally_hits[0].get("stroke") != "serve":
        return []
    first_chance = 2 if rally_hits[0].get("hitter") == player else 1
    if first_chance >= len(rally_hits):
        return []
    hit = rally_hits[first_chance]
    if hit.get("hitter") != player or hit.get("stroke") not in ATTACK_STROKES:
        return []
    return [(0, first_chance)]


def _absorb_pressure(
    rally_hits: list[dict[str, Any]], player: str
) -> list[tuple[int, int]]:
    """Three own defensive replies in a row, at least two of them against a smash or net kill.

    Indices refer to the complete rally (from the first attack to the last reply).
    """
    own = [index for index, hit in enumerate(rally_hits) if hit.get("hitter") == player and index > 0]
    intervals = []
    for k in range(len(own) - 2):
        window = own[k : k + 3]
        if not all(rally_hits[i].get("stroke") in DEFENSIVE_STROKES for i in window):
            continue
        attacks = sum(rally_hits[i - 1].get("stroke") in ATTACK_STROKES for i in window)
        if attacks >= 2:
            intervals.append((window[0] - 1, window[-1]))
    return _merge_intervals(intervals)




def _position(hit: dict[str, Any]) -> tuple[float, float] | None:
    point = (hit.get("analytics") or {}).get("player_position_court")
    if not isinstance(point, list) or len(point) != 2:
        return None
    return float(point[0]), float(point[1])


def _run_opponent(
    rally_hits: list[dict[str, Any]], player: str
) -> list[tuple[int, int]]:
    """The opponent covers at least 2 x LONG_RUN_METRES over two consecutive runs between own hits.

    Judging the sum, not each run, keeps one under-measured run (position error ~0.7 m)
    from breaking the pattern.

    Indices refer to the complete rally (opponent's first to last hit of the run).
    """
    opponent = [
        index for index, hit in enumerate(rally_hits)
        if hit.get("hitter") in ("upper", "lower") and hit.get("hitter") != player
    ]
    runs: list[float | None] = []
    for a, b in zip(opponent, opponent[1:]):
        start, end = _position(rally_hits[a]), _position(rally_hits[b])
        if start is None or end is None:
            runs.append(None)
            continue
        runs.append(
            (((end[0] - start[0]) * COURT_WIDTH_M) ** 2 + ((end[1] - start[1]) * COURT_LENGTH_M) ** 2) ** 0.5
        )
    return _merge_intervals(
        [
            (opponent[k], opponent[k + 2])
            for k in range(len(runs) - 1)
            if runs[k] is not None and runs[k + 1] is not None
            and runs[k] + runs[k + 1] >= 2 * LONG_RUN_METRES
        ]
    )


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
    rallies = {rally_id: _with_next_hitter_positions(hits) for rally_id, hits in rallies.items()}

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
                        "intervals": _interval_ranges(rally_hits, dc),
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

            spatial["four_corner_clear_drop"] = fccd
            four_corner_intervals = _merge_intervals(
                [interval for rows in spatial.values() for interval in rows]
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

            rally_rule_intervals: dict[str, list[tuple[int, int]]] = {}
            for tactic, rule, evidence in (
                ("absorb_pressure", _absorb_pressure, "stroke_sequence_rule"),
                ("run_opponent", _run_opponent, "player_position_rule"),
            ):
                intervals = rule(rally_hits, player)
                if intervals:
                    found.append(
                        {
                            "type": tactic,
                            "label": TACTIC_LABELS[tactic],
                            "intervals": _interval_ranges(rally_hits, intervals),
                            "evidence_quality": evidence,
                            "confidence_calibrated": False,
                            "uses_landing_proxy": False,
                        }
                    )
                    rally_rule_intervals[tactic] = intervals

            net = _net_control(player_hits)
            if net:
                found.append(
                    _interval_payload(
                        "net_control", net, player_hits,
                        evidence_quality="stroke_sequence_rule", uses_landing_proxy=False,
                    )
                )

            early = _early_attack(rally_hits, player)
            if early:
                found.append(
                    {
                        "type": "early_attack",
                        "label": TACTIC_LABELS["early_attack"],
                        "intervals": _interval_ranges(rally_hits, early),
                        "evidence_quality": "stroke_sequence_rule",
                        "confidence_calibrated": False,
                        "uses_landing_proxy": False,
                    }
                )

            # Compare tactics by where they end in the rally; player-hit indices are
            # mapped to rally indices first.
            rally_index = [index for index, hit in enumerate(rally_hits) if hit.get("hitter") == player]

            def last_end(intervals: list[tuple[int, int]], own_hits: bool) -> int:
                if not intervals:
                    return -1
                end = max(end for _, end in intervals)
                return rally_index[end] if own_hits else end

            ends = {
                "full_court_pressure": last_end(fcp, True),
                "net_control": last_end(net, True),
                "four_corner": last_end(four_corner_intervals, True),
                "early_attack": last_end(early, False),
                **{tactic: last_end(rows, False) for tactic, rows in rally_rule_intervals.items()},
            }
            primary = "no_tactic"
            if dc and any(end >= len(rally_hits) - 2 for _, end in dc):
                primary = "defensive_counterattack"
            elif any(end >= 0 for end in ends.values()):
                primary = max(
                    ends,
                    key=lambda tactic: (ends[tactic], -PRIMARY_PRIORITY.index(tactic)),
                )
            elif dc:
                # CoachAI+ keeps a lone counterattack even when it ends mid-rally.
                primary = "defensive_counterattack"
            elif len(player_hits) <= SHORT_RALLY_MAX_OWN_SHOTS:
                primary = "short_rally"

            usage[player][primary] += 1
            records.append(
                {
                    "rally_id": rally_id,
                    "player": player,
                    "player_name": names.get(player, player),
                    "primary_tactic": primary,
                    "primary_tactic_label": TACTIC_LABELS[primary],
                    "detected_tactics": found,
                    "four_corner_subtactic": (
                        _dominant_subtactic(spatial) if four_corner_intervals else None
                    ),
                    "shot_count": len(player_hits),
                    "rally_length": len(rally_hits),
                    "rally_start_time_seconds": _time_seconds(rally_hits[0]),
                    "rally_end_time_seconds": _time_seconds(rally_hits[-1]),
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


def attach_tactic_outcomes(
    structured_tactics: dict[str, Any], scoreboard: dict[str, Any]
) -> dict[str, Any]:
    """Join tactic rows with rally winners and summarise tactic effectiveness.

    A rally only counts towards a win rate when the scoreboard counted it, so
    partial edge rallies and non-scoring exchanges never skew the rates.
    """
    outcomes = {
        int(rally["rally_id"]): rally
        for rally in scoreboard.get("rallies", [])
        if "rally_id" in rally
    }
    usage: dict[str, dict[str, dict[str, Any]]] = {"upper": {}, "lower": {}}
    subtactics: dict[str, Counter[str]] = {"upper": Counter(), "lower": Counter()}
    for row in structured_tactics.get("rallies", []):
        outcome = outcomes.get(int(row["rally_id"]), {})
        winner = outcome.get("winner")
        decided = bool(outcome.get("included_in_estimated_score")) and winner in ("upper", "lower")
        row["rally_winner"] = winner if decided else None
        row["won"] = (winner == row["player"]) if decided else None

        player = row["player"]
        if player not in usage:
            continue
        stats = usage[player].setdefault(
            row["primary_tactic"],
            {"uses": 0, "decided": 0, "wins": 0, "total_rally_length": 0},
        )
        stats["uses"] += 1
        stats["total_rally_length"] += int(row.get("rally_length", 0))
        if decided:
            stats["decided"] += 1
            stats["wins"] += int(row["won"])
        if row.get("primary_tactic") == "four_corner" and row.get("four_corner_subtactic"):
            subtactics[player][row["four_corner_subtactic"]] += 1

    structured_tactics["effectiveness_by_player"] = {
        player: {
            tactic: {
                "label": TACTIC_LABELS[tactic],
                "uses": stats["uses"],
                "decided_rallies": stats["decided"],
                "wins": stats["wins"],
                "win_rate": (
                    round(stats["wins"] / stats["decided"], 4) if stats["decided"] else None
                ),
                "avg_rally_length": round(stats["total_rally_length"] / stats["uses"], 2),
            }
            for tactic, stats in sorted(tactics.items())
        }
        for player, tactics in usage.items()
    }
    structured_tactics["four_corner_subtactic_usage"] = {
        player: dict(counter.most_common()) for player, counter in subtactics.items()
    }
    return structured_tactics
