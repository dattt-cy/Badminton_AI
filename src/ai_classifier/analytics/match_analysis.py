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

ATTACK_STROKES = {"smash", "net_attack", "drive"}
DEFENCE_STROKES = {"lift", "clear"}
NET_STROKES = {"net_shot", "net_attack"}

ZONE_NAMES_3X3 = {
    1: "Front-Left", 2: "Front-Center", 3: "Front-Right",
    4: "Mid-Left",   5: "Mid-Center",   6: "Mid-Right",
    7: "Back-Left",  8: "Back-Center",  9: "Back-Right",
}


def opponent_of(hitter: str) -> str:
    return "lower" if hitter == "upper" else "upper"


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

    # If the final shot was a decisive attacking shot
    if last_stroke in ATTACK_STROKES and last_conf >= 0.70:
        return last_hitter, f"winner_{last_stroke}"
    
    # If the final shot was net_shot with high confidence near rally end
    if last_stroke == "net_shot" and last_conf >= 0.85 and len(hits) > 2:
        return last_hitter, "winner_net_shot"
    
    # Otherwise, opponent usually forced an error or final defensive return was missed/out
    return opponent_of(last_hitter), "opponent_forced_error"


def analyze_scoreboard(
    predictions: list[dict[str, Any]],
    player_names: dict[str, str] | None = None,
    *,
    fps: float = 30.0,
    total_frames: int | None = None,
    edge_window_seconds: float = 5.0,
) -> dict[str, Any]:
    """Estimate rally outcomes without presenting them as official scores.

    Rally winners cannot be observed reliably from strokes alone because the
    final contact may be an error. Edge rallies are also commonly truncated by
    video clipping, so they are marked incomplete and excluded from the score.
    """
    names = player_names or {"upper": "Upper Player", "lower": "Lower Player"}
    rallies_dict: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for p in predictions:
        r_id = int(p.get("rally", 1))
        rallies_dict[r_id].append(p)

    score_upper = 0
    score_lower = 0
    rally_summaries = []

    for r_id in sorted(rallies_dict.keys()):
        hits = rallies_dict[r_id]
        start_frame = int(hits[0].get("refined_frame", hits[0].get("frame", 0)))
        end_frame = int(hits[-1].get("refined_frame", hits[-1].get("frame", 0)))
        dur_seconds = round((end_frame - start_frame) / fps, 2) if end_frame >= start_frame else 0.0

        winner, reason = resolve_rally_winner(hits)
        starts_near_video_edge = start_frame <= edge_window_seconds * fps
        ends_near_video_edge = (
            total_frames is not None
            and total_frames - 1 - end_frame <= edge_window_seconds * fps
        )
        partial_start = r_id == min(rallies_dict) and starts_near_video_edge and hits[0].get("stroke") != "serve"
        partial_end = r_id == max(rallies_dict) and ends_near_video_edge
        complete = not partial_start and not partial_end

        if complete:
            if winner == "upper":
                score_upper += 1
            elif winner == "lower":
                score_lower += 1

        server = hits[0].get("hitter", "unknown")
        rally_summaries.append({
            "rally_id": r_id,
            "start_time_seconds": round(start_frame / fps, 2),
            "end_time_seconds": round(end_frame / fps, 2),
            "duration_seconds": dur_seconds,
            "hits_count": len(hits),
            "server": server,
            "server_name": names.get(server, server),
            "winner": winner,
            "winner_name": names.get(winner, winner),
            "win_reason": reason,
            "winner_is_estimated": True,
            "is_complete": complete,
            "partial_start": partial_start,
            "partial_end": partial_end,
            "included_in_estimated_score": complete,
            "score_upper": score_upper,
            "score_lower": score_lower,
            "score_text": f"{score_upper} - {score_lower}",
            "stroke_sequence": [h.get("stroke", "unknown") for h in hits],
        })

    return {
        "estimated_score": {
            "upper": score_upper,
            "lower": score_lower,
            "text": f"{score_upper} - {score_lower}",
            "leader": "upper" if score_upper > score_lower else ("lower" if score_lower > score_upper else "draw"),
            "is_estimate": True,
            "method": "heuristic_rally_outcome",
            "complete_rallies_counted": sum(r["included_in_estimated_score"] for r in rally_summaries),
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
        # Try getting analytics area or calculate from proxy
        area = None
        analytics = p.get("analytics")
        if analytics and isinstance(analytics, dict):
            area = analytics.get("hitting_area_3x3") or analytics.get("landing_area_3x3_proxy")
        
        if area is None:
            missing_positions[hitter] += 1
            continue

        # Court coordinates are global (top-to-bottom). Rotate the upper
        # player's grid so Front/Mid/Back have the same player-centric meaning
        # for both athletes.
        if area is not None and hitter == "upper":
            zero_based = int(area) - 1
            row, column = divmod(zero_based, 3)
            area = (2 - row) * 3 + column + 1

        if area and 1 <= int(area) <= 9:
            zone_counts[hitter][int(area)] += 1

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


RADAR_AXES = [
    "Attack Stroke Share",
    "Defensive Stroke Share",
    "Net Stroke Share",
    "Shot Variety",
    "Mean Model Confidence",
    "Fast Stroke Share",
]


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

        axes_values = [atk_score, def_score, net_score, variety_score, cons_score, pace_score]
        radar_values[side] = axes_values

        # Descriptive label derived only from the observed segment.
        if atk_score == max(atk_score, def_score, net_score):
            style_label = "Attack-oriented segment"
            style_desc = f"{player_name} used a comparatively high share of attacking strokes in this analyzed segment."
        elif net_score == max(atk_score, def_score, net_score):
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
                last_hit = r_hits[-1]
                area = None
                analytics = last_hit.get("analytics")
                if analytics and isinstance(analytics, dict):
                    area = analytics.get("hitting_area_3x3")
                if area and 1 <= int(area) <= 9:
                    vulnerable_zones[int(area)] += 1

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
            "upper": radar_values["upper"],
            "lower": radar_values["lower"],
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

    # 1. Match Dynamics Overview
    final_score = scoreboard.get("estimated_score", {})
    leader = final_score.get("leader", "draw")
    leader_name = up_name if leader == "upper" else (low_name if leader == "lower" else "Tie")
    score_txt = final_score.get("text", "0 - 0")

    insights.append({
        "category": "Match Dynamic",
        "target": "General",
        "insight": f"Estimated rally outcome for this segment is {score_txt}, favoring {leader_name}; this is not an official score. "
                   f"{up_name} is styled as '{up_prof.get('playstyle', {}).get('label', 'All-Rounder')}', "
                   f"while {low_name} operates as '{low_prof.get('playstyle', {}).get('label', 'All-Rounder')}'.",
    })

    # 2. Upper player tactical advice
    up_vuln = up_prof.get("error_analysis", {}).get("most_vulnerable_zone")
    up_vuln_str = f"in {up_vuln.get('name')}" if up_vuln else "on deep defensive returns"
    up_atk = up_prof.get("radar_attributes", {}).get("attack", 50)
    if up_atk >= 70:
        up_tip = f"{up_name}'s attack rating is strong ({up_atk}/100). Keep applying downward pressure, but protect coverage {up_vuln_str} to minimize counter-attacks."
    else:
        up_tip = f"In this segment, {up_name}'s observed stroke mix was less attack-oriented; review the video before drawing a coaching conclusion. The most frequent end position among estimated losses was {up_vuln_str}."
    insights.append({
        "category": "Player Strategy",
        "target": up_name,
        "insight": up_tip,
    })

    # 3. Lower player tactical advice
    low_vuln = low_prof.get("error_analysis", {}).get("most_vulnerable_zone")
    low_vuln_str = f"in {low_vuln.get('name')}" if low_vuln else "on net exchanges"
    low_net = low_prof.get("radar_attributes", {}).get("net_control", 50)
    if low_net >= 70:
        low_tip = f"In this segment, {low_name}'s net-stroke share index was {low_net}/100. This describes detected stroke usage, not net ability; the most frequent end position among estimated losses was {low_vuln_str}."
    else:
        low_tip = f"{low_name} should reinforce defense and tighten court transitions {low_vuln_str}."
    insights.append({
        "category": "Player Strategy",
        "target": low_name,
        "insight": low_tip,
    })

    # 4. Key Counter-Tactics from Response Matrix
    up_responses = response_matrix.get("upper", {})
    if "smash" in up_responses and up_responses["smash"]:
        top_res = up_responses["smash"][0]
        if top_res["count"] >= 5:
            insights.append({
                "category": "Tactical Response Pattern",
                "target": f"{low_name} vs {up_name}",
                "insight": f"In {top_res['count']} observed cases, when receiving a smash from {low_name}, {up_name} responded with '{top_res['response_stroke']}' {top_res['percentage']}% of the time. Treat this as a segment-level pattern, not a general player tendency.",
            })

    return insights


def build_match_analysis(
    predictions: list[dict[str, Any]],
    player_names: dict[str, str] | None = None,
    *,
    fps: float = 30.0,
    total_frames: int | None = None,
) -> dict[str, Any]:
    """Single-call API to generate full CoachAI-aligned tactical intelligence."""
    scoreboard = analyze_scoreboard(
        predictions,
        player_names,
        fps=fps,
        total_frames=total_frames,
    )
    patterns = extract_tactical_patterns(predictions)
    spatial = analyze_spatial_court_grid(predictions)
    coachai_rows = export_coachai_records(predictions, player_names)
    
    # Advanced CoachAI intelligence modules
    player_profiles = analyze_player_profiles(predictions, scoreboard, player_names)
    response_matrix = build_response_transition_matrix(predictions)
    coach_insights = generate_coach_insights(player_profiles, response_matrix, scoreboard, player_names)
    from .tactics import analyze_coachai_plus_tactics
    structured_tactics = analyze_coachai_plus_tactics(predictions, player_names)
    from .movement import analyze_sparse_movement
    movement = analyze_sparse_movement(predictions)

    return {
        "scoreboard": scoreboard,
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


