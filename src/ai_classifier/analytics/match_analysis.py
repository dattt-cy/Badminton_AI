"""CoachAI-aligned Tactical and Match Analytics for Badminton Video Pipeline.

Provides high-level match intelligence extracted from rally and stroke predictions:
1. Rally Outcome & Scoreboard Progression (winner attribution, score flow).
2. Tactical Sequences & Transition Matrix (2-gram, 3-gram, winning combos).
3. Spatial Court Grid (3x3 9-zone hitting & landing distributions).
4. CoachAI-compatible structured export for downstream frontends.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any

ATTACK_AND_WINNING_STROKES = {"smash", "net_attack", "drive", "drop"}
DEFENCE_STROKES = {"lift", "clear"}
NET_STROKES = {"net_shot", "net_attack"}

ZONE_NAMES_3X3 = {
    1: "Front-Left", 2: "Front-Center", 3: "Front-Right",
    4: "Mid-Left",   5: "Mid-Center",   6: "Mid-Right",
    7: "Back-Left",  8: "Back-Center",  9: "Back-Right",
}


def opponent_of(hitter: str) -> str:
    return "lower" if hitter == "upper" else "upper"


def infer_rally_server(hits: list[dict[str, Any]]) -> str:
    """Return the serving side of a rally.

    When the serve itself was not detected, the first detected hit is the
    receiver's return, so the server is that hitter's opponent.
    """
    if not hits:
        return "unknown"
    first_hitter = hits[0].get("hitter", "unknown")
    if hits[0].get("stroke") == "serve" or first_hitter not in ("upper", "lower"):
        return first_hitter
    return opponent_of(first_hitter)


def player_centric_zone(hitter: str, analytics: dict[str, Any] | None) -> int | None:
    """Map a player's court position to a 3x3 zone of their own half.

    Rows run Front (net) -> Back (baseline) inside the hitter's half, and
    columns are left/right from the hitter's own viewpoint, so the upper
    player's grid is rotated 180 degrees relative to the image.
    """
    position = (analytics or {}).get("player_position_court")
    if not position or len(position) != 2 or hitter not in ("upper", "lower"):
        return None
    x, y = float(position[0]), float(position[1])
    if not (-0.05 <= x <= 1.05 and -0.05 <= y <= 1.05):
        return None
    depth = (0.5 - y) / 0.5 if hitter == "upper" else (y - 0.5) / 0.5
    row = min(max(int(depth * 3), 0), 2)
    column = min(max(int(x * 3), 0), 2)
    if hitter == "upper":
        column = 2 - column
    return row * 3 + column + 1


def resolve_rally_winner(hits: list[dict[str, Any]]) -> tuple[str, str]:
    """Infer the rally winner and end reason based on final stroke dynamics.

    Returns:
        tuple (winner_side, reason):
        - winner_side: 'upper' or 'lower'
        - reason: 'winner_attack', 'winning_shot', or 'unforced_error'
    """
    if not hits:
        return "unknown", "no_hits"
    
    last_hit = hits[-1]
    last_hitter = last_hit.get("hitter", "upper")
    last_stroke = last_hit.get("stroke", "net_shot")
    last_conf = float(last_hit.get("confidence", 0.8))

    # If the final shot was a decisive attacking or placed winning shot (smash, drop, drive, net_attack)
    if last_stroke in ATTACK_AND_WINNING_STROKES and last_conf >= 0.65:
        return last_hitter, f"winner_{last_stroke}"
    
    # If the final shot was net_shot with reliable confidence near rally end
    if last_stroke == "net_shot" and last_conf >= 0.60 and len(hits) >= 2:
        return last_hitter, "winner_net_shot"
    
    # Otherwise, opponent usually forced an error or final defensive return was missed/out
    return opponent_of(last_hitter), "opponent_forced_error"


def _rally_bounds_seconds(hits: list[dict[str, Any]], fps: float) -> tuple[float, float]:
    start = int(hits[0].get("refined_frame", hits[0].get("frame", 0)))
    end = int(hits[-1].get("refined_frame", hits[-1].get("frame", 0)))
    return round(start / fps, 2), round(end / fps, 2)


GAME_BREAK_SECONDS = 100.0
# Court-normalized points (x across, y from the far baseline to the near one) and 3x3
# areas numbered row by row over the whole court; both rotate 180 degrees with the ends.
_COURT_POINT_KEYS = (
    "player_position_court",
    "opponent_position_court",
    "contact_shuttle_projection_court",
    "landing_position_court_proxy",
    "landing_position_court_trajectory_proxy",
)
_COURT_AREA_KEYS = ("hitting_area_3x3", "landing_area_3x3_proxy")


def _rally_groups(predictions: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    groups: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for p in predictions:
        groups[int(p.get("rally", 1))].append(p)
    return [groups[k] for k in sorted(groups)]


def _game_of_each_rally(rallies: list[list[dict[str, Any]]], fps: float, game_break_seconds: float) -> list[int]:
    """Game number of each rally: a pause of ``game_break_seconds`` or more starts a new game.

    In ShuttleSet every between-game break lasts 105 s or more, while fewer than 0.2%
    of within-game pauses exceed 100 s.
    """
    games: list[int] = []
    game, previous_end = 1, None
    for hits in rallies:
        start, end = _rally_bounds_seconds(hits, fps)
        if previous_end is not None and start - previous_end >= game_break_seconds:
            game += 1
        games.append(game)
        previous_end = end
    return games


def _swap_ends(hits: list[dict[str, Any]], player_names: dict[str, str]) -> None:
    for hit in hits:
        if hit.get("hitter") in ("upper", "lower"):
            hit["hitter"] = opponent_of(hit["hitter"])
            hit["hitter_name"] = player_names.get(hit["hitter"], hit["hitter"])
        analytics = hit.get("analytics")
        if not isinstance(analytics, dict):
            continue
        for key in _COURT_POINT_KEYS:
            point = analytics.get(key)
            if isinstance(point, (list, tuple)) and len(point) == 2 and all(isinstance(v, (int, float)) for v in point):
                analytics[key] = [round(1.0 - float(point[0]), 5), round(1.0 - float(point[1]), 5)]
        for key in _COURT_AREA_KEYS:
            area = analytics.get(key)
            if isinstance(area, int) and 1 <= area <= 9:
                analytics[key] = 10 - area


def normalize_player_ends(
    predictions: list[dict[str, Any]],
    player_names: dict[str, str] | None = None,
    *,
    fps: float = 30.0,
    total_frames: int | None = None,
    scoreboard_ocr: dict[str, Any] | None = None,
    game_break_seconds: float = GAME_BREAK_SECONDS,
) -> list[dict[str, Any]]:
    """Make ``hitter`` name a player rather than a camera side, in place.

    The detector reports which half of the image a hitter stands in, and the user names
    the player in each half at the start of the video. Players change ends after every
    game and at 11 points in the third, so from then on the hits, court positions and
    areas of those rallies are swapped back: "upper" stays the player who started in the
    far half, and court points are drawn as if that player never changed ends.
    Returns the swapped spans as ``{"game", "from_rally", "to_rally"}``.
    """
    names = player_names or {"upper": "upper", "lower": "lower"}
    rallies = _rally_groups(predictions)
    if not rallies:
        return []
    games = _game_of_each_rally(rallies, fps, game_break_seconds)
    swapped = [False] * len(rallies)
    for i, game in enumerate(games):
        if game % 2 == 0:
            swapped[i] = True
            _swap_ends(rallies[i], names)

    # Third game: ends change when the leader reaches 11, which needs the score.
    third = [i for i, game in enumerate(games) if game == 3]
    if third:
        board = analyze_scoreboard(
            predictions, names, fps=fps, total_frames=total_frames,
            scoreboard_ocr=scoreboard_ocr, game_break_seconds=game_break_seconds,
        )
        rows = {int(r["rally_id"]): r for r in board["rallies"]}
        for position, i in enumerate(third):
            row = rows.get(int(rallies[i][0].get("rally", 0)))
            if row and max(row["score_upper"], row["score_lower"]) >= 11:
                for j in third[position + 1:]:
                    swapped[j] = True
                    _swap_ends(rallies[j], names)
                break

    spans: list[dict[str, Any]] = []
    for i, is_swapped in enumerate(swapped):
        if not is_swapped:
            continue
        rally_id = int(rallies[i][0].get("rally", 0))
        if spans and spans[-1]["game"] == games[i] and spans[-1]["_last"] == i - 1:
            spans[-1].update(to_rally=rally_id, _last=i)
        else:
            spans.append({"game": games[i], "from_rally": rally_id, "to_rally": rally_id, "_last": i})
    for span in spans:
        del span["_last"]
    return spans


def end_change_index(ocr_event: dict[str, Any]) -> int:
    """How many times the players have changed ends before this OCR point: after each game,
    and in the third game once a player reaches 11."""
    game = int(ocr_event.get("game", 1))
    previous = ocr_event.get("previous_score") or [0, 0]
    return game - 1 + (1 if game >= 3 and max(int(previous[0]), int(previous[1])) >= 11 else 0)


def first_row_side_at(start_side: str, ocr_event: dict[str, Any]) -> str:
    """The court side of the scoreboard's first row when ``ocr_event`` was scored."""
    return start_side if end_change_index(ocr_event) % 2 == 0 else opponent_of(start_side)


def infer_ocr_first_row_side(
    rallies: list[list[dict[str, Any]]],
    ocr_events: list[dict[str, Any]],
    fps: float,
) -> str:
    """Decide which court side the scoreboard's first row belongs to in the first game.

    Broadcast scoreboards list players in a fixed order unrelated to the
    camera, so each OCR point change is compared with the next rally's server
    (the rally winner serves next) and the majority vote wins. The players change
    ends after every game, so each vote is turned back to the first game's ends
    first; :func:`first_row_side_at` gives the side at a later point.
    """
    votes: Counter[str] = Counter()
    used: set[int] = set()
    for i in range(len(rallies) - 1):
        _, end_sec = _rally_bounds_seconds(rallies[i], fps)
        next_start_sec, _ = _rally_bounds_seconds(rallies[i + 1], fps)
        next_server = infer_rally_server(rallies[i + 1])
        # Across a game break the players have changed ends, so the server's side says nothing.
        if next_server not in ("upper", "lower") or next_start_sec - end_sec >= GAME_BREAK_SECONDS:
            continue
        for ev_idx, ev in enumerate(ocr_events):
            ev_time = float(ev.get("time_seconds", 0.0))
            if ev_idx not in used and (end_sec - 3.0) <= ev_time <= (next_start_sec + 5.0):
                used.add(ev_idx)
                first_won = ev.get("point_winner") in ("first", "upper")
                side = next_server if first_won else opponent_of(next_server)
                votes[side if end_change_index(ev) % 2 == 0 else opponent_of(side)] += 1
                break
    return "lower" if votes["lower"] > votes["upper"] else "upper"


def analyze_scoreboard(
    predictions: list[dict[str, Any]],
    player_names: dict[str, str] | None = None,
    *,
    fps: float = 30.0,
    total_frames: int | None = None,
    edge_window_seconds: float = 5.0,
    scoreboard_ocr: dict[str, Any] | None = None,
    game_break_seconds: float = GAME_BREAK_SECONDS,
) -> dict[str, Any]:
    """Estimate rally outcomes and align scores with OCR scoreboard evidence & Badminton server rules.

    A pause of at least ``game_break_seconds`` between rallies starts a new game
    (see :func:`_game_of_each_rally`), which starts from 0-0. An OCR score change is
    only used for a rally of the game the scoreboard was showing.
    """
    names = player_names or {"upper": "Upper Player", "lower": "Lower Player"}
    rallies_dict: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for p in predictions:
        r_id = int(p.get("rally", 1))
        rallies_dict[r_id].append(p)

    rally_keys = sorted(rallies_dict.keys())
    ocr_events: list[dict[str, Any]] = (scoreboard_ocr or {}).get("events", []) if isinstance(scoreboard_ocr, dict) else []
    initial_score = (scoreboard_ocr or {}).get("initial_score") if isinstance(scoreboard_ocr, dict) else None
    first_row_side = (
        infer_ocr_first_row_side([rallies_dict[k] for k in rally_keys], ocr_events, fps)
        if ocr_events else "upper"
    )

    def to_court_order(score: Any, ocr_event: dict[str, Any] | None = None) -> Any:
        """Reorder an OCR (first row, second row) score as (upper, lower) at ``ocr_event``'s ends."""
        side = first_row_side_at(first_row_side, ocr_event) if ocr_event else first_row_side
        if side == "lower" and isinstance(score, (list, tuple)) and len(score) == 2:
            return [score[1], score[0]]
        return score

    initial_score = to_court_order(initial_score, ocr_events[0] if ocr_events else None)
    # The first reading is the score before the first rally only when it was read by
    # then; a scoreboard first shown minutes later says nothing about the start.
    initial_time = (scoreboard_ocr or {}).get("initial_time_seconds") if isinstance(scoreboard_ocr, dict) else None
    if initial_time is not None and rally_keys:
        _, first_end = _rally_bounds_seconds(rallies_dict[rally_keys[0]], fps)
        if float(initial_time) > first_end + 30.0:
            initial_score = None
    game_of_rally = dict(zip(
        rally_keys, _game_of_each_rally([rallies_dict[k] for k in rally_keys], fps, game_break_seconds)
    ))

    if initial_score and isinstance(initial_score, (list, tuple)) and len(initial_score) == 2:
        curr_score = [int(initial_score[0]), int(initial_score[1])]
    else:
        curr_score = [0, 0]

    used_ocr_idx: set[int] = set()
    rally_summaries: list[dict[str, Any]] = []
    games: list[dict[str, Any]] = []
    game_number = 1

    def close_game() -> None:
        games.append({
            "game_number": game_number,
            "score_upper": curr_score[0],
            "score_lower": curr_score[1],
            "score_text": f"{curr_score[0]} - {curr_score[1]}",
            "rallies": sum(1 for r in rally_summaries if r["game_number"] == game_number),
        })

    for i, r_id in enumerate(rally_keys):
        hits = rallies_dict[r_id]
        start_frame = int(hits[0].get("refined_frame", hits[0].get("frame", 0)))
        end_frame = int(hits[-1].get("refined_frame", hits[-1].get("frame", 0)))
        dur_seconds = round((end_frame - start_frame) / fps, 2) if end_frame >= start_frame else 0.0
        start_sec = round(start_frame / fps, 2)
        end_sec = round(end_frame / fps, 2)

        if game_of_rally[r_id] != game_number:
            close_game()
            game_number = game_of_rally[r_id]
            curr_score = [0, 0]

        next_start_sec = (
            round(int(rallies_dict[rally_keys[i + 1]][0].get("refined_frame", rallies_dict[rally_keys[i + 1]][0].get("frame", 0))) / fps, 2)
            if i + 1 < len(rally_keys)
            else end_sec + 30.0
        )

        starts_near_video_edge = start_frame <= edge_window_seconds * fps
        ends_near_video_edge = (
            total_frames is not None
            and total_frames - 1 - end_frame <= edge_window_seconds * fps
        )
        partial_start = r_id == min(rally_keys) and starts_near_video_edge and hits[0].get("stroke") != "serve"
        partial_end = r_id == max(rally_keys) and ends_near_video_edge

        # 1. Check for matching OCR event occurring at or immediately after this rally
        matched_ev = None
        for ev_idx, ev in enumerate(ocr_events):
            ev_time = float(ev.get("time_seconds", 0.0))
            if (
                ev_idx not in used_ocr_idx
                and int(ev.get("game", game_number)) == game_number
                and (end_sec - 3.0) <= ev_time <= (next_start_sec + 5.0)
            ):
                matched_ev = ev
                used_ocr_idx.add(ev_idx)
                break

        if matched_ev:
            point_w = matched_ev.get("point_winner", "first")
            first_won = point_w in ("first", "upper")
            row_side = first_row_side_at(first_row_side, matched_ev)
            winner = row_side if first_won else opponent_of(row_side)
            reason = f"ocr_confirmed_{winner}"
            score = to_court_order(matched_ev["score"], matched_ev)
            curr_score = [int(score[0]), int(score[1])]
            complete = True
            is_estimated = False
        elif i == 0 and ocr_events and initial_score and end_sec < float(ocr_events[0].get("time_seconds", 0.0)):
            # First rally in video ending before first detected score change
            prev_s = (
                to_court_order(ocr_events[0]["previous_score"], ocr_events[0])
                if "previous_score" in ocr_events[0]
                else initial_score or [0, 0]
            )
            curr_score = [int(prev_s[0]), int(prev_s[1])]
            stroke_winner, stroke_reason = resolve_rally_winner(hits)
            winner = "upper" if curr_score[0] > curr_score[1] else ("lower" if curr_score[1] > curr_score[0] else stroke_winner)
            reason = stroke_reason if winner == stroke_winner else f"initial_ocr_{winner}"
            complete = True
            is_estimated = False
        else:
            # Check if this is a non-scoring toss or shuttlecock retrieval between rallies
            if len(hits) < 3 and hits[0].get("stroke") != "serve":
                complete = False
                winner = "unknown"
                reason = "non_scoring_exchange"
                is_estimated = True
            else:
                stroke_winner, stroke_reason = resolve_rally_winner(hits)
                # 2. Badminton Next-Rally Server Rule: The winner of rally k serves in rally k+1
                next_server = None
                if i + 1 < len(rally_keys):
                    next_server = infer_rally_server(rallies_dict[rally_keys[i + 1]])
                    # The game's winner serves the next game first, from the other end.
                    if next_server in ("upper", "lower") and game_of_rally[rally_keys[i + 1]] != game_of_rally[r_id]:
                        next_server = opponent_of(next_server)

                if next_server in ("upper", "lower"):
                    winner = next_server
                    if winner == stroke_winner:
                        reason = stroke_reason
                    else:
                        reason = "next_rally_server"
                else:
                    winner, reason = stroke_winner, stroke_reason

                complete = not partial_start and not partial_end
                if complete:
                    if winner == "upper":
                        curr_score[0] += 1
                    elif winner == "lower":
                        curr_score[1] += 1
                is_estimated = True

        server = infer_rally_server(hits)
        rally_summaries.append({
            "rally_id": r_id,
            "game_number": game_number,
            "start_time_seconds": start_sec,
            "end_time_seconds": end_sec,
            "duration_seconds": dur_seconds,
            "hits_count": len(hits),
            "server": server,
            "server_name": names.get(server, server),
            "winner": winner,
            "winner_name": names.get(winner, winner),
            "win_reason": reason,
            "winner_is_estimated": is_estimated,
            "is_complete": complete,
            "partial_start": partial_start,
            "partial_end": partial_end,
            "included_in_estimated_score": complete,
            "score_upper": curr_score[0],
            "score_lower": curr_score[1],
            "score_text": f"{curr_score[0]} - {curr_score[1]}",
            "stroke_sequence": [h.get("stroke", "unknown") for h in hits],
        })

    if rally_summaries:
        close_game()

    return {
        "games": games,
        "estimated_score": {
            "upper": curr_score[0],
            "lower": curr_score[1],
            "text": f"{curr_score[0]} - {curr_score[1]}",
            "leader": "upper" if curr_score[0] > curr_score[1] else ("lower" if curr_score[1] > curr_score[0] else "draw"),
            "is_estimate": not bool(ocr_events),
            "method": "calibrated_ocr_with_server_rule" if ocr_events else "badminton_server_rule_and_heuristics",
            "complete_rallies_counted": sum(r["included_in_estimated_score"] for r in rally_summaries),
            "game_number": game_number,
            "scope": "current_game",
            "ocr_first_row_side": first_row_side if ocr_events else None,
        },
        "rallies": rally_summaries,
    }


def extract_tactical_patterns(
    predictions: list[dict[str, Any]],
) -> dict[str, Any]:
    """Mine CoachAI-style shot transitions and winning combinations."""
    rallies_dict: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for p in predictions:
        rallies_dict[int(p.get("rally", 1))].append(p)

    transitions_2gram: Counter[str] = Counter()
    transitions_3gram: Counter[str] = Counter()
    per_hitter_transitions: dict[str, Counter[str]] = {
        "upper": Counter(),
        "lower": Counter(),
    }
    winning_combos: dict[str, Counter[str]] = {
        "upper": Counter(),
        "lower": Counter(),
    }

    for hits in rallies_dict.values():
        strokes = [h.get("stroke", "unknown") for h in hits]
        hitters = [h.get("hitter", "upper") for h in hits]
        n = len(strokes)

        winner, _ = resolve_rally_winner(hits)

        # 2-grams
        for i in range(n - 1):
            pair = f"{strokes[i]} -> {strokes[i+1]}"
            transitions_2gram[pair] += 1
            hitter = hitters[i]
            if hitter in per_hitter_transitions:
                per_hitter_transitions[hitter][pair] += 1

        # 3-grams
        for i in range(n - 2):
            triad = f"{strokes[i]} -> {strokes[i+1]} -> {strokes[i+2]}"
            transitions_3gram[triad] += 1

        # Winning combos (last 2-3 shots of a winning rally)
        if winner in winning_combos and n >= 2:
            last_combo = f"{strokes[-2]} -> {strokes[-1]}"
            winning_combos[winner][last_combo] += 1
            if n >= 3:
                last_triad = f"{strokes[-3]} -> {strokes[-2]} -> {strokes[-1]}"
                winning_combos[winner][last_triad] += 1

    return {
        "top_transitions_2gram": [
            {"sequence": seq, "count": count}
            for seq, count in transitions_2gram.most_common(10)
        ],
        "top_sequences_3gram": [
            {"sequence": seq, "count": count}
            for seq, count in transitions_3gram.most_common(10)
        ],
        "estimated_winning_combinations": {
            player: [{"sequence": seq, "count": count} for seq, count in counter.most_common(5)]
            for player, counter in winning_combos.items()
        },
        "per_player_transitions": {
            player: [{"sequence": seq, "count": count} for seq, count in counter.most_common(5)]
            for player, counter in per_hitter_transitions.items()
        },
    }


def analyze_spatial_court_grid(
    predictions: list[dict[str, Any]],
) -> dict[str, Any]:
    """Calculate 3x3 9-zone court coverage for upper and lower players."""
    zone_counts: dict[str, Counter[int]] = {
        "upper": Counter(),
        "lower": Counter(),
    }
    missing_positions = {"upper": 0, "lower": 0}

    for p in predictions:
        hitter = p.get("hitter", "upper")
        analytics = p.get("analytics")
        area = player_centric_zone(hitter, analytics if isinstance(analytics, dict) else None)
        if area is None:
            if hitter in missing_positions:
                missing_positions[hitter] += 1
            continue
        zone_counts[hitter][area] += 1

    result: dict[str, Any] = {}
    for player in ["upper", "lower"]:
        total = sum(zone_counts[player].values()) or 1
        zones_data = []
        for zone_id in range(1, 10):
            cnt = zone_counts[player][zone_id]
            pct = round((cnt / total) * 100, 1)
            zones_data.append({
                "zone_id": zone_id,
                "name": ZONE_NAMES_3X3.get(zone_id, f"Zone-{zone_id}"),
                "count": cnt,
                "percentage": pct,
            })
        result[player] = zones_data

    result["quality"] = {
        "missing_position_hits": missing_positions,
        "uses_stroke_type_fallback": False,
        "player_centric_orientation": True,
    }

    return result


def export_coachai_records(
    predictions: list[dict[str, Any]],
    player_names: dict[str, str] | None = None,
) -> list[dict[str, Any]]:
    """Format predictions into standard CoachAI/ShuttleSet tabular records."""
    names = player_names or {"upper": "A", "lower": "B"}
    records = []
    for p in predictions:
        hitter = p.get("hitter", "upper")
        records.append({
            "rally": int(p.get("rally", 1)),
            "obs_ball_round": int(p.get("ball_round", 1)),
            "obs_player": names.get(hitter, hitter),
            "player_side": hitter,
            "time_seconds": round(float(p.get("refined_frame", p.get("frame", 0))) / 30.0, 3),
            "act_ball_type": p.get("stroke", "unknown"),
            "confidence": round(float(p.get("confidence", 0.0)), 4),
            "is_serve": p.get("stroke") == "serve",
        })
    return records


# Style radar: share of a player's strokes (serves excluded) in four disjoint groups,
# so the axes share one unit (%) and add up to ~100.
STYLE_GROUPS = {
    "attack_share": ("smash", "net_attack"),
    "net_share": ("net_shot",),
    "defence_share": ("lift", "clear"),
    "placement_share": ("drive", "drop"),
}
RADAR_AXES = list(STYLE_GROUPS)
# A stroke type counts as "used" when it makes up at least this share of a player's rally strokes.
USED_STROKE_SHARE = 0.05
MIN_FLIGHT_SECONDS, MAX_FLIGHT_SECONDS = 0.15, 3.0


def style_profile(side_hits: list[dict[str, Any]]) -> dict[str, Any]:
    """Stroke-group shares (%), stroke types used, and median time between the player's shot and the reply."""
    rally_strokes = [p.get("stroke") for p in side_hits if p.get("stroke") != "serve"]
    total = len(rally_strokes)
    counts = Counter(rally_strokes)
    shares = {
        axis: round(100 * sum(counts.get(k, 0) for k in keys) / total) if total else 0
        for axis, keys in STYLE_GROUPS.items()
    }
    flights = sorted(
        float(f) for p in side_hits
        if isinstance(p.get("analytics"), dict)
        and (f := p["analytics"].get("flight_time_to_next_hit_seconds")) is not None
        and MIN_FLIGHT_SECONDS <= float(f) <= MAX_FLIGHT_SECONDS
    )
    tempo = round(flights[len(flights) // 2], 2) if flights else None
    used = sum(1 for k, n in counts.items() if total and n / total >= USED_STROKE_SHARE)
    return {"shares": shares, "rally_strokes": total, "stroke_types_used": used, "median_flight_seconds": tempo}


def analyze_player_profiles(
    predictions: list[dict[str, Any]],
    scoreboard: dict[str, Any] | None = None,
    player_names: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Extract CoachAI player style profiling, 6-axis radar metrics, and error mining."""
    import math

    names = player_names or {"upper": "Player Upper", "lower": "Player Lower"}
    rallies = scoreboard.get("rallies", []) if scoreboard else []

    profiles: dict[str, Any] = {}
    radar_values: dict[str, list[int]] = {"upper": [], "lower": []}
    radar_extra: dict[str, dict[str, Any]] = {}

    for side in ["upper", "lower"]:
        player_name = names.get(side, side)
        side_hits = [p for p in predictions if p.get("hitter") == side]
        total_shots = len(side_hits)
        stroke_counts = Counter(p.get("stroke", "unknown") for p in side_hits)

        # These are observed shot-profile indices, not athlete ability scores.
        # They intentionally avoid player-specific bonuses or unsupported elite
        # baselines. Winner-derived components are excluded because rally
        # winners are currently heuristic estimates.
        n_atk = stroke_counts.get("smash", 0) + stroke_counts.get("net_attack", 0) + 0.5 * stroke_counts.get("drive", 0) + 0.4 * stroke_counts.get("drop", 0)
        r_atk = n_atk / max(1, total_shots)
        won_rallies = [r for r in rallies if r.get("winner") == side]
        atk_score = round(min(100.0, r_atk * 200))

        # 2. Defense score (Ability to survive attacks & rally resilience)
        lost_rallies = [r for r in rallies if r.get("winner") == opponent_of(side)]
        errors_committed = len([r for r in lost_rallies if r.get("win_reason") == "opponent_forced_error"])
        err_rate = errors_committed / max(1, total_shots)
        long_rallies = [r for r in rallies if r.get("hits_count", 0) >= 6]
        long_won = [r for r in long_rallies if r.get("winner") == side]
        r_long_win = len(long_won) / max(1, len(long_rallies))
        n_def = stroke_counts.get("lift", 0) + stroke_counts.get("clear", 0)
        def_score = round(min(100.0, n_def / max(1, total_shots) * 250))

        # 3. Net Control score (Net shots & tumbling net attacks)
        n_net = stroke_counts.get("net_shot", 0) + stroke_counts.get("net_attack", 0)
        r_net = n_net / max(1, total_shots)
        net_score = round(min(100.0, r_net * 200))

        # 4. Shot Variety (Shannon entropy of 8 stroke types)
        stroke_probs = [cnt / max(1, total_shots) for cnt in stroke_counts.values()]
        entropy = -sum(p * math.log2(p) for p in stroke_probs if p > 0)
        variety_score = round(min(100.0, entropy / 3.0 * 100))

        # 5. Consistency (Error resistance and ball control stability)
        confidences = [float(p.get("confidence", 0.0)) for p in side_hits]
        cons_score = round(sum(confidences) / max(1, len(confidences)) * 100)

        # 6. Pace / Speed index (High-speed flat exchanges, footwork tempo)
        n_fast = stroke_counts.get("drive", 0) + stroke_counts.get("net_attack", 0) + stroke_counts.get("smash", 0)
        r_fast = n_fast / max(1, total_shots)
        pace_score = round(min(100.0, r_fast * 200))

        style = style_profile(side_hits)
        shares = style["shares"]
        radar_values[side] = [shares[axis] for axis in RADAR_AXES]
        radar_extra[side] = {k: style[k] for k in ("rally_strokes", "stroke_types_used", "median_flight_seconds")}

        # Descriptive label derived only from the observed segment.
        lead = max(("attack_share", "net_share", "defence_share"), key=lambda axis: shares[axis])
        if lead == "attack_share":
            style_label = "Attack-oriented segment"
            style_desc = f"{player_name} used a comparatively high share of attacking strokes in this analyzed segment."
        elif lead == "net_share":
            style_label = "Net-oriented segment"
            style_desc = f"{player_name} used a comparatively high share of net strokes in this analyzed segment."
        else:
            style_label = "Defence-oriented segment"
            style_desc = f"{player_name} used a comparatively high share of lifts and clears in this analyzed segment."

        # Error & Vulnerability Mining
        lost_rallies_count = len(lost_rallies)
        vulnerable_zones: Counter[int] = Counter()
        for r in lost_rallies:
            r_hits = [p for p in predictions if int(p.get("rally", 1)) == r.get("rally_id")]
            if r_hits:
                # Zone of the losing player's own final hit, when they made it.
                side_hits = [p for p in r_hits if p.get("hitter") == side]
                if side_hits:
                    analytics = side_hits[-1].get("analytics")
                    area = player_centric_zone(side, analytics if isinstance(analytics, dict) else None)
                    if area is not None:
                        vulnerable_zones[area] += 1

        top_zone = vulnerable_zones.most_common(1)
        vuln_zone_info = None
        if top_zone:
            z_id, z_cnt = top_zone[0]
            vuln_zone_info = {
                "zone_id": z_id,
                "name": ZONE_NAMES_3X3.get(z_id, f"Zone-{z_id}"),
                "losses_incurred": z_cnt,
            }

        profiles[side] = {
            "player_name": player_name,
            "side": side,
            "total_shots": total_shots,
            "stroke_counts": dict(stroke_counts),
            "radar_attributes": {
                "attack": atk_score,
                "defense": def_score,
                "net_control": net_score,
                "shot_variety": variety_score,
                "consistency": cons_score,
                "pace": pace_score,
            },
            "playstyle": {
                "label": style_label,
                "description": style_desc,
                "scope": "observed_segment_only",
            },
            "radar_is_ability_score": False,
            "error_analysis": {
                "is_estimated_from_heuristic_winners": True,
                "total_rallies_lost": lost_rallies_count,
                "forced_error_losses": errors_committed,
                "opponent_clean_winners_suffered": lost_rallies_count - errors_committed,
                "most_vulnerable_zone": vuln_zone_info,
            },
        }

    # Authentic CoachAI Ball Type Distribution Radar (Wang et al. 2020)
    coachai_keys = ["smash", "drive", "lift", "clear", "net_shot", "net_attack", "drop"]
    coachai_labels = ["Smash (Đập)", "Drive (Tạt)", "Lift (Bung)", "Clear (Phông)", "Net Shot (Gài)", "Net Attack (Vồ)", "Drop (Chém)"]
    coachai_radar = {
        "axes": coachai_labels,
        "keys": coachai_keys,
        "upper_counts": [profiles["upper"]["stroke_counts"].get(k, 0) for k in coachai_keys],
        "lower_counts": [profiles["lower"]["stroke_counts"].get(k, 0) for k in coachai_keys],
        "upper_pct": [round((profiles["upper"]["stroke_counts"].get(k, 0) / max(1, profiles["upper"]["total_shots"])) * 100, 1) for k in coachai_keys],
        "lower_pct": [round((profiles["lower"]["stroke_counts"].get(k, 0) / max(1, profiles["lower"]["total_shots"])) * 100, 1) for k in coachai_keys],
    }

    return {
        "profiles": profiles,
        "radar_chart": {
            "axes": RADAR_AXES,
            "unit": "percent_of_rally_strokes",
            "upper": radar_values["upper"],
            "lower": radar_values["lower"],
            "extra": radar_extra,
        },
        "coachai_balltype_radar": coachai_radar,
    }


def build_response_transition_matrix(
    predictions: list[dict[str, Any]],
) -> dict[str, dict[str, list[dict[str, Any]]]]:
    """Mine next-shot response probabilities (CoachAI ShuttleNet-style response modeling).
    
    Reveals how each player responds when facing specific incoming shots from the opponent.
    """
    rallies_dict: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for p in predictions:
        rallies_dict[int(p.get("rally", 1))].append(p)

    # response_counts[player][incoming_stroke][response_stroke] = count
    response_counts: dict[str, dict[str, Counter[str]]] = {
        "upper": defaultdict(Counter),
        "lower": defaultdict(Counter),
    }

    for hits in rallies_dict.values():
        for i in range(len(hits) - 1):
            incoming = hits[i]
            response = hits[i + 1]
            inc_hitter = incoming.get("hitter", "upper")
            res_hitter = response.get("hitter", "lower")

            # Only when hitter switched (response to opponent)
            if inc_hitter != res_hitter:
                inc_stroke = incoming.get("stroke", "unknown")
                res_stroke = response.get("stroke", "unknown")
                response_counts[res_hitter][inc_stroke][res_stroke] += 1

    matrix_result: dict[str, dict[str, list[dict[str, Any]]]] = {
        "upper": {},
        "lower": {},
    }

    for side in ["upper", "lower"]:
        for inc_stroke, counter in response_counts[side].items():
            total = sum(counter.values()) or 1
            matrix_result[side][inc_stroke] = [
                {
                    "response_stroke": resp,
                    "count": cnt,
                    "probability": round(cnt / total, 3),
                    "percentage": round((cnt / total) * 100, 1),
                }
                for resp, cnt in counter.most_common(3)
            ]

    return matrix_result


def generate_coach_insights(
    player_profiles: dict[str, Any],
    response_matrix: dict[str, Any],
    scoreboard: dict[str, Any],
    player_names: dict[str, str] | None = None,
) -> list[dict[str, str]]:
    """Synthesize actionable tactical coaching advice for players and coaches."""
    names = player_names or {"upper": "Player Upper", "lower": "Player Lower"}
    insights: list[dict[str, str]] = []

    profiles = player_profiles.get("profiles", {})
    up_prof = profiles.get("upper", {})
    low_prof = profiles.get("lower", {})

    up_name = names.get("upper", "Player Upper")
    low_name = names.get("lower", "Player Lower")

    # 1. Match overview. The score is the real one when the broadcast scoreboard was read.
    final_score = scoreboard.get("estimated_score", {})
    leader = final_score.get("leader", "draw")
    score_txt = final_score.get("text", "0 - 0").replace(" - ", "–")
    from_scoreboard = final_score.get("is_estimate") is False
    lead_txt = (
        f"{up_name} đang dẫn" if leader == "upper"
        else f"{low_name} đang dẫn" if leader == "lower" else "hai bên đang hòa"
    )
    insights.append({
        "category": "Diễn biến trận",
        "target": "General",
        "insight": (
            f"Cuối video, ván {final_score.get('game_number', 1)}: {up_name} {score_txt} {low_name}, {lead_txt} "
            f"({'đọc từ bảng điểm trên video' if from_scoreboard else 'AI tự đếm, không phải tỉ số chính thức'}). "
            f"{up_name}: {STYLE_VI.get(up_prof.get('playstyle', {}).get('label'), 'lối đánh cân bằng')}; "
            f"{low_name}: {STYLE_VI.get(low_prof.get('playstyle', {}).get('label'), 'lối đánh cân bằng')}."
        ),
    })

    # 2-3. The same observational summary for both players.
    for name, prof in ((up_name, up_prof), (low_name, low_prof)):
        shares = _stroke_group_shares(prof)
        vuln = prof.get("error_analysis", {}).get("most_vulnerable_zone")
        tip = (
            f"Trong đoạn video này (không tính giao cầu), {shares['attack']}% số cú của {name} là cú tấn công (đập, vồ lưới), "
            f"{shares['net']}% là cú kéo lưới, {shares['defence']}% là cú phòng thủ (phông, hất cầu). "
            f"Đây là tỉ lệ loại cú AI nhận ra, không phải điểm năng lực."
        )
        if vuln:
            tip += (
                f" Ở các pha thua, cú cuối của {name} thường đánh từ "
                f"{ZONE_NAMES_VI.get(vuln.get('zone_id'), vuln.get('name'))} ({vuln.get('losses_incurred')} pha)."
            )
        insights.append({
            "category": "Lối đánh",
            "target": name,
            "insight": tip,
        })

    # 4. Key response pattern from the response matrix
    up_responses = response_matrix.get("upper", {})
    if "smash" in up_responses and up_responses["smash"]:
        top_res = up_responses["smash"][0]
        if top_res["count"] >= 5:
            insights.append({
                "category": "Cách đỡ đập",
                "target": up_name,
                "insight": (
                    f"Khi bị {low_name} đập, {up_name} đỡ bằng cú "
                    f"{STROKE_VI.get(top_res['response_stroke'], top_res['response_stroke'])} "
                    f"{top_res['percentage']:.0f}% số lần ({top_res['count']} lần quan sát trong đoạn video này)."
                ),
            })

    return insights


STYLE_VI = {
    "Attack-oriented segment": "thiên về tấn công",
    "Net-oriented segment": "thiên về đánh lưới",
    "Defence-oriented segment": "thiên về phòng thủ",
}
ZONE_NAMES_VI = {
    1: "lưới, cánh trái", 2: "lưới, giữa", 3: "lưới, cánh phải",
    4: "giữa sân, cánh trái", 5: "giữa sân", 6: "giữa sân, cánh phải",
    7: "cuối sân, cánh trái", 8: "cuối sân, giữa", 9: "cuối sân, cánh phải",
}
STROKE_VI = {
    "smash": "đập cầu", "drop": "bỏ nhỏ", "clear": "phông cầu", "serve": "giao cầu", "lift": "hất cầu",
    "drive": "tạt cầu", "net_shot": "kéo lưới", "net_attack": "vồ lưới",
}


def _stroke_group_shares(profile: dict[str, Any]) -> dict[str, int]:
    """Percent of a player's detected strokes that attack, play net shots, or defend (disjoint groups)."""
    counts = profile.get("stroke_counts", {}) or {}
    # Same base as the style radar: rally strokes, serves excluded.
    total = max(1, sum(n for stroke, n in counts.items() if stroke != "serve"))
    share = lambda keys: round(100 * sum(counts.get(k, 0) for k in keys) / total)  # noqa: E731
    return {
        "attack": share(("smash", "net_attack")),
        "net": share(("net_shot",)),
        "defence": share(("lift", "clear")),
    }


def build_match_analysis(
    predictions: list[dict[str, Any]],
    player_names: dict[str, str] | None = None,
    *,
    fps: float = 30.0,
    total_frames: int | None = None,
    scoreboard_ocr: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Single-call API to generate full CoachAI-aligned tactical intelligence.

    Swaps ``predictions`` to fixed player ends first (:func:`normalize_player_ends`).
    """
    ends_swapped = normalize_player_ends(
        predictions,
        player_names,
        fps=fps,
        total_frames=total_frames,
        scoreboard_ocr=scoreboard_ocr,
    )
    scoreboard = analyze_scoreboard(
        predictions,
        player_names,
        fps=fps,
        total_frames=total_frames,
        scoreboard_ocr=scoreboard_ocr,
    )
    patterns = extract_tactical_patterns(predictions)
    spatial = analyze_spatial_court_grid(predictions)
    coachai_rows = export_coachai_records(predictions, player_names)
    
    # Advanced CoachAI intelligence modules
    player_profiles = analyze_player_profiles(predictions, scoreboard, player_names)
    response_matrix = build_response_transition_matrix(predictions)
    coach_insights = generate_coach_insights(player_profiles, response_matrix, scoreboard, player_names)
    from .tactics import analyze_coachai_plus_tactics, attach_tactic_outcomes
    structured_tactics = attach_tactic_outcomes(
        analyze_coachai_plus_tactics(predictions, player_names), scoreboard
    )
    from .movement import analyze_sparse_movement
    movement = analyze_sparse_movement(predictions)

    return {
        "scoreboard": {**scoreboard, "ends_swapped": ends_swapped},
        "tactical_patterns": patterns,
        "spatial_distribution_3x3": spatial,
        "player_profiles": player_profiles.get("profiles", {}),
        "radar_chart": player_profiles.get("radar_chart", {}),
        "coachai_balltype_radar": player_profiles.get("coachai_balltype_radar", {}),
        "response_transition_matrix": response_matrix,
        "coach_insights": coach_insights,
        "structured_tactics": structured_tactics,
        "movement": movement,
        "coachai_records_sample": coachai_rows[:10],
        "total_records": len(coachai_rows),
    }


# Backward-compatible public name used by earlier pipeline outputs and callers.
build_tactical_analytics = build_match_analysis


