"""Post-processing and evaluation for frame-level badminton hit scores."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence


@dataclass(frozen=True)
class HitEvent:
    frame: int
    side: str
    score: float


def temporal_nms(
    events: Sequence[HitEvent], *, threshold: float, radius: int,
    side_aware: bool = False,
) -> list[HitEvent]:
    """Keep the strongest event in each temporal neighborhood.

    With ``side_aware=True``, nearby hits from opposing court players do not
    suppress each other. This is required for fast exchanges where upper and
    lower players can legitimately hit within one NMS radius.
    """
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must be between 0 and 1")
    if radius < 0:
        raise ValueError("radius must be non-negative")
    candidates = [event for event in events if event.score >= threshold]
    candidates.sort(key=lambda event: (-event.score, event.frame))
    kept: list[HitEvent] = []
    for event in candidates:
        if all(
            (side_aware and event.side != selected.side)
            or abs(event.frame - selected.frame) > radius
            for selected in kept
        ):
            kept.append(event)
    return sorted(kept, key=lambda event: event.frame)


def two_stage_temporal_nms(
    events: Sequence[HitEvent], *, threshold: float, radius: int,
    same_side_radius: int,
) -> list[HitEvent]:
    """Apply ordinary NMS, then suppress slower duplicate peaks per hitter.

    The second pass never suppresses nearby events from opposite court sides,
    preserving legitimate fast exchanges while removing same-player echoes.
    """
    first = temporal_nms(events, threshold=threshold, radius=radius)
    return temporal_nms(
        first, threshold=0.0, radius=same_side_radius, side_aware=True
    )


def temporal_grouping(
    events: Sequence[HitEvent], *, threshold: float, max_gap: int, max_span: int
) -> list[HitEvent]:
    """Collapse nearby candidates without merging separate long rallies."""
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must be between 0 and 1")
    if max_gap < 0 or max_span < 0:
        raise ValueError("max_gap and max_span must be non-negative")
    candidates = sorted(
        (event for event in events if event.score >= threshold),
        key=lambda event: event.frame,
    )
    groups: list[list[HitEvent]] = []
    for event in candidates:
        if (
            not groups
            or event.frame - groups[-1][-1].frame > max_gap
            or event.frame - groups[-1][0].frame > max_span
        ):
            groups.append([event])
        else:
            groups[-1].append(event)
    output: list[HitEvent] = []
    for group in groups:
        side_scores: dict[str, float] = {}
        for event in group:
            side_scores[event.side] = side_scores.get(event.side, 0.0) + event.score
        side = max(side_scores, key=side_scores.get)
        representative = max(
            (event for event in group if event.side == side),
            key=lambda event: event.score,
        )
        output.append(HitEvent(representative.frame, side, representative.score))
    return output


def match_hit_events(
    predicted: Sequence[HitEvent], truth: Sequence[HitEvent], *, tolerance: int
) -> dict:
    """One-to-one ordered matching that maximizes matches, then minimizes error."""
    if tolerance < 0:
        raise ValueError("tolerance must be non-negative")
    predictions = sorted(predicted, key=lambda event: event.frame)
    targets = sorted(truth, key=lambda event: event.frame)
    rows, cols = len(predictions) + 1, len(targets) + 1
    # Each state stores (match count, negative total absolute error).
    scores = [[(0, 0) for _ in range(cols)] for _ in range(rows)]
    choices = [["" for _ in range(cols)] for _ in range(rows)]
    for i in range(1, rows):
        choices[i][0] = "prediction"
    for j in range(1, cols):
        choices[0][j] = "truth"
    for i in range(1, rows):
        for j in range(1, cols):
            options = [
                (scores[i - 1][j], "prediction"),
                (scores[i][j - 1], "truth"),
            ]
            error = abs(predictions[i - 1].frame - targets[j - 1].frame)
            if error <= tolerance:
                previous = scores[i - 1][j - 1]
                options.append(((previous[0] + 1, previous[1] - error), "match"))
            scores[i][j], choices[i][j] = max(options, key=lambda item: item[0])

    pairs: list[tuple[int, int]] = []
    i, j = len(predictions), len(targets)
    while i or j:
        choice = choices[i][j]
        if choice == "match":
            pairs.append((i - 1, j - 1))
            i -= 1
            j -= 1
        elif choice == "prediction":
            i -= 1
        else:
            j -= 1
    pairs.reverse()
    errors = [abs(predictions[i].frame - targets[j].frame) for i, j in pairs]
    side_correct = sum(predictions[i].side == targets[j].side for i, j in pairs)
    true_positives = len(pairs)
    precision = true_positives / len(predictions) if predictions else 0.0
    recall = true_positives / len(targets) if targets else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "truth": len(targets),
        "predicted": len(predictions),
        "true_positives": true_positives,
        "false_positives": len(predictions) - true_positives,
        "false_negatives": len(targets) - true_positives,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "mean_absolute_frame_error": sum(errors) / len(errors) if errors else None,
        "median_absolute_frame_error": _median(errors),
        "side_accuracy": side_correct / true_positives if true_positives else None,
        "joint_true_positives": side_correct,
        "joint_precision": side_correct / len(predictions) if predictions else 0.0,
        "joint_recall": side_correct / len(targets) if targets else 0.0,
        "matches": [
            {
                "predicted_frame": predictions[pi].frame,
                "truth_frame": targets[ti].frame,
                "frame_error": predictions[pi].frame - targets[ti].frame,
                "predicted_side": predictions[pi].side,
                "truth_side": targets[ti].side,
                "side_correct": predictions[pi].side == targets[ti].side,
            }
            for pi, ti in pairs
        ],
    }


def _median(values: Sequence[int]) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return float(ordered[middle])
    return (ordered[middle - 1] + ordered[middle]) / 2


def rescue_pre_rally_serves(
    base_events: Sequence[HitEvent],
    raw_events: Sequence[HitEvent],
    *,
    min_serve_score: float = 0.40,
    min_dead_time_frames: int = 75,
    max_return_gap_frames: int = 85,
    nms_radius: int = 8,
    same_side_radius: int = 15,
) -> list[HitEvent]:
    """Recover low-velocity serves that occurred after dead time before a rally.

    Serves have lower kinetic energy / wrist acceleration and are frequently suppressed
    at strict production thresholds (e.g. 0.75). If a candidate scored >= min_serve_score
    follows a long dead-time interval (>= min_dead_time_frames) and is followed by a
    confirmed opponent return shot within max_return_gap_frames, it is safely restored.
    """
    if not raw_events or not base_events:
        return list(base_events)

    low_candidates = two_stage_temporal_nms(
        raw_events,
        threshold=min_serve_score,
        radius=nms_radius,
        same_side_radius=same_side_radius,
    )
    base_frames = [e.frame for e in base_events]
    existing_frames = set(base_frames)

    rescued: list[HitEvent] = []
    for cand in low_candidates:
        if cand.frame in existing_frames:
            continue
        prev_hits = [f for f in base_frames if f < cand.frame]
        prev_gap = (cand.frame - max(prev_hits)) if prev_hits else 9999
        next_hits = [e for e in base_events if e.frame > cand.frame]
        if not next_hits:
            continue
        next_event = min(next_hits, key=lambda e: e.frame)
        next_gap = next_event.frame - cand.frame

        # Guard: dead-time before, prompt opponent return after, opposing player sides
        if (
            prev_gap >= min_dead_time_frames
            and 15 <= next_gap <= max_return_gap_frames
            and cand.side != next_event.side
        ):
            rescued.append(cand)

    merged = list(base_events) + rescued
    merged.sort(key=lambda e: e.frame)
    return merged


def rescue_alternating_bridges(
    base_events: Sequence[HitEvent],
    raw_events: Sequence[HitEvent],
    *,
    min_score: float = 0.55,
    min_leg_frames: int = 12,
    max_leg_frames: int = 85,
    nms_radius: int = 8,
    same_side_radius: int = 15,
) -> list[HitEvent]:
    """Restore one plausible opposite-side hit between consecutive same-side hits.

    Singles rallies normally alternate hitters. A same-side pair can therefore
    expose a missed low-score opponent hit. This guard is deliberately local:
    it only accepts a candidate strictly between the pair, sufficiently far
    from both kept events, and chooses at most one peak per gap.
    """
    base = sorted(base_events, key=lambda event: event.frame)
    if len(base) < 2 or not raw_events:
        return base
    candidates = two_stage_temporal_nms(
        raw_events,
        threshold=min_score,
        radius=nms_radius,
        same_side_radius=same_side_radius,
    )
    rescued: list[HitEvent] = []
    for previous, following in zip(base, base[1:]):
        if previous.side != following.side:
            continue
        options = [
            candidate
            for candidate in candidates
            if candidate.side != previous.side
            and min_leg_frames <= candidate.frame - previous.frame <= max_leg_frames
            and min_leg_frames <= following.frame - candidate.frame <= max_leg_frames
        ]
        if options:
            rescued.append(max(options, key=lambda event: event.score))

    merged = base + rescued
    # A candidate can satisfy only one adjacent gap in normal ordering, but
    # de-duplicate defensively for synthetic inputs and future callers.
    unique = {(event.frame, event.side): event for event in merged}
    return sorted(unique.values(), key=lambda event: event.frame)

