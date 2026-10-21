"""Derive CoachAI-style stroke metadata from existing pipeline features.

Landing coordinates are explicitly reported as trajectory proxies. A monocular
broadcast view cannot recover the true 3-D shuttle landing point without a
dedicated estimator, especially while the shuttle is airborne.
"""

from __future__ import annotations

from typing import Any

import numpy as np


ATTACK_STROKES = {"smash", "net_attack", "drive"}
DEFENCE_STROKES = {"lift", "clear"}


def suppress_weak_duplicate_serves(
    predictions: list[dict[str, Any]],
    *,
    max_gap_frames: int = 75,
    weak_hit_score: float = 0.60,
    minimum_score_advantage: float = 0.15,
    maximum_shuttle_visibility: float = 0.10,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Collapse implausibly close serve predictions to one strongest event.

    A genuine singles player cannot serve twice within 2.5 seconds. Detector echoes may have high classification
    confidence, so only checking for a weak first event misses many duplicates.
    Consecutive serve predictions are clustered by refined contact time and the
    event with the strongest hit-detector score is retained.  The legacy
    weak/no-shuttle rule remains as a fallback when a non-serve prediction lies
    between two serve echoes.
    """
    if not predictions:
        return [], []

    serve_indices = [
        index for index, row in enumerate(predictions)
        if row.get("stroke") == "serve"
    ]
    clusters: list[list[int]] = []
    for index in serve_indices:
        frame = int(predictions[index].get("refined_frame", predictions[index].get("frame", 0)))
        if not clusters:
            clusters.append([index])
            continue
        previous = predictions[clusters[-1][-1]]
        previous_frame = int(previous.get("refined_frame", previous.get("frame", 0)))
        hitter = predictions[index].get("hitter")
        previous_hitter = previous.get("hitter")
        if (
            hitter is not None
            and hitter == previous_hitter
            and frame - previous_frame <= max_gap_frames
        ):
            clusters[-1].append(index)
        else:
            clusters.append([index])

    clustered_suppressed: dict[int, dict[str, Any]] = {}
    for cluster in clusters:
        if len(cluster) < 2:
            continue
        keep_index = max(
            cluster,
            key=lambda item: float(predictions[item].get("hit_score", 0.0)),
        )
        keeper = predictions[keep_index]
        keeper_frame = int(keeper.get("refined_frame", keeper.get("frame", 0)))
        for index in cluster:
            if index == keep_index:
                continue
            row = predictions[index]
            refined_frame = int(row.get("refined_frame", row.get("frame", 0)))
            clustered_suppressed[index] = {
                "frame": int(row.get("frame", refined_frame)),
                "refined_frame": refined_frame,
                "hit_score": round(float(row.get("hit_score", 0.0)), 4),
                "replacement_frame": int(keeper.get("frame", keeper_frame)),
                "replacement_refined_frame": keeper_frame,
                "replacement_hit_score": round(float(keeper.get("hit_score", 0.0)), 4),
                "reason": "duplicate_serve_cluster",
            }

    kept: list[dict[str, Any]] = []
    suppressed: list[dict[str, Any]] = []
    index = 0
    while index < len(predictions):
        if index in clustered_suppressed:
            suppressed.append(clustered_suppressed[index])
            index += 1
            continue
        current = predictions[index]
        following = predictions[index + 1] if index + 1 < len(predictions) else None
        if following is not None:
            current_frame = int(current.get("refined_frame", current.get("frame", 0)))
            following_frame = int(
                following.get("refined_frame", following.get("frame", 0))
            )
            current_score = float(current.get("hit_score", 1.0))
            following_score = float(following.get("hit_score", 0.0))
            visibility = float(
                (current.get("analytics") or {}).get("shuttle_visibility_ratio", 1.0)
            )
            should_suppress = (
                current.get("stroke") == "serve"
                and following.get("stroke") == "serve"
                and 0 < following_frame - current_frame <= max_gap_frames
                and current_score < weak_hit_score
                and following_score - current_score >= minimum_score_advantage
                and visibility <= maximum_shuttle_visibility
            )
            if should_suppress:
                suppressed.append(
                    {
                        "frame": int(current.get("frame", current_frame)),
                        "refined_frame": current_frame,
                        "hit_score": round(current_score, 4),
                        "replacement_frame": int(
                            following.get("frame", following_frame)
                        ),
                        "replacement_hit_score": round(following_score, 4),
                        "reason": "weak_duplicate_serve_without_shuttle_evidence",
                    }
                )
                index += 1
                continue
        kept.append(current)
        index += 1
    return kept, suppressed


def court_area(point: np.ndarray | None, rows: int = 3, columns: int = 3) -> int | None:
    """Return a one-based court grid area, or None for invalid projections."""
    if point is None or len(point) != 2 or not np.isfinite(point).all():
        return None
    x, y = float(point[0]), float(point[1])
    if not (-0.05 <= x <= 1.05 and -0.05 <= y <= 1.05):
        return None
    x, y = np.clip([x, y], 0.0, 1.0)
    column = min(int(x * columns), columns - 1)
    row = min(int(y * rows), rows - 1)
    return row * columns + column + 1


def _point(value: np.ndarray | None) -> list[float] | None:
    if value is None or not np.isfinite(value).all():
        return None
    return [round(float(value[0]), 5), round(float(value[1]), 5)]


def _court_point(value: np.ndarray | None) -> list[float] | None:
    if value is None or court_area(value) is None:
        return None
    return _point(np.clip(value, 0.0, 1.0))


def _mean_speed(points: np.ndarray, valid: np.ndarray, fps: float) -> float | None:
    pair_valid = valid[:-1] & valid[1:]
    if not np.any(pair_valid) or fps <= 0:
        return None
    distances = np.linalg.norm(np.diff(points, axis=0), axis=1)[pair_valid]
    return round(float(distances.mean() * fps), 5)


def build_stroke_analytics(
    context: dict[str, Any],
    *,
    stroke: str,
    fps: float,
    next_hit_frame: int | None = None,
) -> dict[str, Any]:
    """Build serializable analytics without changing classifier inputs."""
    frames = np.asarray(context["frame_indices"], dtype=np.int64)
    shuttle_image = np.asarray(context["shuttle_image"], dtype=np.float32)
    shuttle_court = np.asarray(context["shuttle_court"], dtype=np.float32)
    player_court = np.asarray(context["player_court"], dtype=np.float32)
    opponent_court = np.asarray(context["opponent_court"], dtype=np.float32)
    wrists = np.asarray(context["hitter_wrists"], dtype=np.float32)
    contact = int(context["contact_index"])
    shuttle_valid = np.any(shuttle_image != 0.0, axis=1)
    wrist_valid = np.any(wrists != 0.0, axis=1)

    contact_shuttle = shuttle_court[contact] if shuttle_valid[contact] else None
    contact_image = shuttle_image[contact] if shuttle_valid[contact] else None
    hit_player = player_court[contact] if np.any(player_court[contact] != 0.0) else None
    hit_opponent = (
        opponent_court[contact] if np.any(opponent_court[contact] != 0.0) else None
    )

    post = np.arange(contact + 2, len(frames))
    if next_hit_frame is not None:
        before_next = post[frames[post] <= next_hit_frame - 2]
        if len(before_next):
            post = before_next
    projected_inside = (
        np.isfinite(shuttle_court).all(axis=1)
        & (shuttle_court[:, 0] >= 0.0) & (shuttle_court[:, 0] <= 1.0)
        & (shuttle_court[:, 1] >= 0.0) & (shuttle_court[:, 1] <= 1.0)
    )
    candidates = (
        post[shuttle_valid[post] & projected_inside[post]]
        if len(post) else np.asarray([], dtype=int)
    )
    landing_index = int(candidates[-1]) if len(candidates) else None
    landing = shuttle_court[landing_index] if landing_index is not None else None
    landing_frame = int(frames[landing_index]) if landing_index is not None else None

    post_slice = slice(contact, (landing_index + 1) if landing_index is not None else len(frames))
    visibility = shuttle_valid[post_slice]
    visibility_ratio = float(visibility.mean()) if len(visibility) else 0.0
    if next_hit_frame is not None and landing_frame is not None:
        proximity = max(0.0, 1.0 - abs((next_hit_frame - 2) - landing_frame) / 30.0)
    else:
        proximity = 0.35 if landing_frame is not None else 0.0
    # This remains a monocular 2-D projection, so never report ground-truth-like
    # confidence even when tracking visibility is perfect.
    landing_confidence = round(0.65 * visibility_ratio * proximity, 4)

    flight_time = None
    if next_hit_frame is not None and fps > 0:
        flight_time = round(max(0.0, (next_hit_frame - int(frames[contact])) / fps), 4)

    attack_state = (
        "attacking" if stroke in ATTACK_STROKES
        else "defensive" if stroke in DEFENCE_STROKES
        else "neutral"
    )
    return {
        "player_position_court": _court_point(hit_player),
        "opponent_position_court": _court_point(hit_opponent),
        "contact_shuttle_projection_court": _point(contact_shuttle),
        "contact_height_image_ratio": (
            round(1.0 - float(contact_image[1]), 5) if contact_image is not None else None
        ),
        "hitting_area_3x3": court_area(hit_player),
        "landing_position_court_proxy": _court_point(landing),
        "landing_area_3x3_proxy": court_area(landing),
        "landing_proxy_frame": landing_frame,
        "landing_proxy_confidence": landing_confidence,
        "flight_time_to_next_hit_seconds": flight_time,
        "average_shuttle_speed_image_per_second": _mean_speed(
            shuttle_image[post_slice], shuttle_valid[post_slice], fps
        ),
        "average_wrist_speed_image_per_second": _mean_speed(
            wrists, wrist_valid, fps
        ),
        "shuttle_visibility_ratio": round(visibility_ratio, 4),
        "attack_state_rule": attack_state,
        "landing_is_proxy": True,
    }


def predict_rally_boundary(
    prev_prediction: dict[str, Any] | None,
    curr_prediction: dict[str, Any],
    fps: float,
    model: Any = None,
    max_gap_seconds: float = 5.0,
) -> bool:
    """Predict whether curr_prediction starts a new rally (True) or continues same rally (False)."""
    if prev_prediction is None:
        return True
    
    prev_frame = int(prev_prediction.get("refined_frame", prev_prediction.get("frame", 0)))
    curr_frame = int(curr_prediction.get("refined_frame", curr_prediction.get("frame", 0)))
    gap = max((curr_frame - prev_frame) / fps, 0.0) if fps > 0 else 0.0
    
    curr_stroke = curr_prediction.get("stroke", "")

    # A detected serve is the strongest observable boundary cue. Conversely, a
    # non-serve after a short gap is much more likely to be a slow/high return
    # than a new rally. Keep this guard in front of the learned model because
    # noisy stroke labels made the model split long rallies in production.
    if curr_stroke == "serve":
        return True
    if gap <= max_gap_seconds:
        return False

    # Fast path: ML model evaluation if available
    if model is not None and hasattr(model, "predict"):
        stroke2idx = {
            "clear": 0, "drive": 1, "drop": 2, "lift": 3,
            "net_attack": 4, "net_shot": 5, "serve": 6, "smash": 7,
        }
        prev_stroke = prev_prediction.get("stroke", "")
        same_side = 1.0 if curr_prediction.get("hitter") == prev_prediction.get("hitter") else 0.0
        curr_is_serve = 1.0 if curr_stroke == "serve" else 0.0
        prev_is_serve = 1.0 if prev_stroke == "serve" else 0.0
        curr_s_idx = float(stroke2idx.get(curr_stroke, -1))
        prev_s_idx = float(stroke2idx.get(prev_stroke, -1))
        
        feat = np.array([[
            gap,
            float(np.log(max(gap, 0.01))),
            same_side,
            curr_is_serve,
            prev_is_serve,
            curr_s_idx,
            prev_s_idx,
        ]], dtype=np.float32)
        return bool(model.predict(feat)[0] == 1)

    # Robust rule fallback
    return gap > max_gap_seconds


def assign_rallies(
    predictions: list[dict[str, Any]],
    fps: float,
    max_gap_seconds: float = 3.0,
    model: Any = None,
    prune_echoes: bool = True,
) -> list[dict[str, Any]]:
    """Add rally and ball-round identifiers using ML boundary detector or temporal gaps."""
    rally = 0
    ball_round = 0
    previous_prediction: dict[str, Any] | None = None
    for prediction in predictions:
        frame = int(prediction.get("refined_frame", prediction.get("frame", 0)))
        prev_frame = (
            int(previous_prediction.get("refined_frame", previous_prediction.get("frame", 0)))
            if previous_prediction is not None
            else None
        )
        gap = (frame - prev_frame) / fps if prev_frame is not None and fps > 0 else None
        
        new_rally = predict_rally_boundary(
            previous_prediction,
            prediction,
            fps=fps,
            model=model,
            max_gap_seconds=max_gap_seconds,
        )
        if new_rally:
            rally += 1
            ball_round = 1
        else:
            ball_round += 1
        prediction["rally"] = rally
        prediction["ball_round"] = ball_round
        prediction["gap_from_previous_hit_seconds"] = (
            round(gap, 4) if gap is not None else None
        )
        previous_prediction = prediction

    if prune_echoes and predictions:
        prune_pseudo_rallies(predictions, fps=fps)
        predictions[:] = prune_trailing_phantom_hits(predictions, fps=fps)

    return predictions


def prune_pseudo_rallies(
    predictions: list[dict[str, Any]],
    fps: float = 30.0,
    max_echo_gap_seconds: float = 2.5,
) -> int:
    """Merge echo single-hit rallies occurring immediately after a rally."""
    if not predictions:
        return 0
    rally_groups: dict[int, list[dict[str, Any]]] = {}
    for p in predictions:
        rally_groups.setdefault(p["rally"], []).append(p)

    sorted_rallies = sorted(rally_groups.keys())
    for idx, r_idx in enumerate(sorted_rallies):
        if idx == 0:
            continue
        shots = rally_groups[r_idx]
        if len(shots) == 1:
            prev_shots = rally_groups[sorted_rallies[idx - 1]]
            gap = (shots[0].get("frame", 0) - prev_shots[-1].get("frame", 0)) / fps if fps > 0 else 999.0
            if gap <= max_echo_gap_seconds:
                shots[0]["rally"] = prev_shots[-1]["rally"]
                prev_shots.append(shots[0])

    unique_rallies = sorted(set(p["rally"] for p in predictions))
    mapping = {old_r: new_r for new_r, old_r in enumerate(unique_rallies, 1)}
    for p in predictions:
        p["rally"] = mapping[p["rally"]]
    return len(mapping)


def prune_trailing_phantom_hits(
    predictions: list[dict[str, Any]],
    *,
    fps: float = 30.0,
    max_terminal_gap_seconds: float = 0.75,
) -> list[dict[str, Any]]:
    """Prune false-positive trailing swings at rally end (e.g. failing to retrieve a dead shuttle)."""
    if not predictions:
        return predictions

    rally_groups: dict[int, list[dict[str, Any]]] = {}
    for p in predictions:
        rally_groups.setdefault(p.get("rally", 1), []).append(p)

    kept: list[dict[str, Any]] = []
    for r_id in sorted(rally_groups.keys()):
        hits = rally_groups[r_id]
        if len(hits) >= 2:
            last = hits[-1]
            penult = hits[-2]
            f_last = int(last.get("refined_frame", last.get("frame", 0)))
            f_penult = int(penult.get("refined_frame", penult.get("frame", 0)))
            gap_sec = (f_last - f_penult) / fps if fps > 0 else 999.0

            is_trailing_echo = (
                last.get("stroke") in ("net_shot", "net_attack")
                and penult.get("stroke") in ("net_shot", "drop", "smash", "net_attack")
                and last.get("hitter") != penult.get("hitter")
                and gap_sec <= max_terminal_gap_seconds
            )
            if is_trailing_echo:
                hits = hits[:-1]
        kept.extend(hits)

    return kept


def refine_strokes_by_physics(
    predictions: list[dict[str, Any]],
    *,
    drive_net_threshold: float = 0.235,
    net_attack_backcourt_threshold: float = 0.40,
    net_attack_max_confidence: float = 0.70,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Refine ambiguous stroke predictions using physical court geometry and opponent reception context.

    Disambiguation rules:
    1. Drive vs Net Shot:
       If the model predicts 'drive' with 'net_shot' in top2, but the opponent receives
       the shuttle right next to the net (|y - 0.5| <= drive_net_threshold), the shot
       is geometrically a net_shot (drop/net play), not a flat drive.
    2. Net Attack vs Lift:
       If the model predicts 'net_attack' with low confidence (< net_attack_max_confidence)
       and 'lift' in top2, but the opponent is forced all the way back to the deep baseline
       (|y - 0.5| >= net_attack_backcourt_threshold), the shot is physically a lift/clear,
       not a net push.

    Returns:
        (refined_predictions, audit_log_of_changes)
    """
    changes: list[dict[str, Any]] = []
    n = len(predictions)

    for i in range(n):
        curr = predictions[i]
        nxt = predictions[i + 1] if i + 1 < n else None

        # Only apply physical disambiguation when next hit exists in the same rally
        if nxt is None or curr.get("rally") != nxt.get("rally"):
            continue

        nxt_analytics = nxt.get("analytics") or {}
        nxt_pos = nxt_analytics.get("player_position_court")
        if not nxt_pos or len(nxt_pos) < 2:
            continue

        dist_net = abs(float(nxt_pos[1]) - 0.5)
        st = curr.get("stroke")
        t2 = curr.get("top2") or []
        cf = float(curr.get("confidence", 0.0))

        # Rule 1: Drive -> Net Shot
        if st == "drive" and "net_shot" in t2 and dist_net <= drive_net_threshold:
            old_st = st
            curr["stroke"] = "net_shot"
            curr["physics_refined"] = True
            changes.append({
                "hit_index": curr.get("hit_index"),
                "frame": curr.get("frame"),
                "original_stroke": old_st,
                "refined_stroke": "net_shot",
                "reason": f"Opponent received at net (dist_to_net={dist_net:.3f} <= {drive_net_threshold})",
            })

        # Rule 2: Net Attack -> Lift
        elif (
            st == "net_attack"
            and "lift" in t2
            and cf < net_attack_max_confidence
            and dist_net >= net_attack_backcourt_threshold
        ):
            old_st = st
            curr["stroke"] = "lift"
            curr["physics_refined"] = True
            changes.append({
                "hit_index": curr.get("hit_index"),
                "frame": curr.get("frame"),
                "original_stroke": old_st,
                "refined_stroke": "lift",
                "reason": (
                    f"Opponent received at deep baseline (dist_to_net={dist_net:.3f} >= {net_attack_backcourt_threshold}, "
                    f"model_conf={cf:.2f} < {net_attack_max_confidence})"
                ),
            })

    return predictions, changes

