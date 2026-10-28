"""Stroke re-scoring with the shuttle flight time to the next hit.

The stroke classifier sees roughly one second around contact, so it cannot
observe when the opponent returns the shuttle. That flight time separates
strokes that look alike in pose: pushes (~1.0 s) from lifts (~1.3 s), and
drives (~0.5 s) from pushes. Each hit's classifier log-probabilities are
combined with log p(flight time | stroke), a smoothed histogram fitted on
ShuttleSet train matches:

    score(c) = log p_model(c) + weight * log p(flight | c)

The last hit of a rally has no flight time and keeps the classifier output.
On the 8-clip benchmark (weight 0.75, selected on validation clips) this left
top-1 unchanged while raising drive recall by ~12 and net_attack by ~9 points.
"""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from .tactical_decoder import STROKE_CLASSES

LOG_FLIGHT_BINS = np.linspace(np.log(0.15), np.log(3.0), 41)
DEFAULT_WEIGHT = 0.75


def fit_flight_time_model(manifest_path: Path | str, split: str = "train") -> np.ndarray:
    """Return log p(log-flight-time bin | stroke) with Laplace smoothing."""
    rallies: dict[tuple, list[dict]] = defaultdict(list)
    with Path(manifest_path).open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            if row.get("split") == split:
                rallies[(row["match_id"], row["set_id"], row["rally"])].append(row)
    counts = np.ones((len(STROKE_CLASSES), len(LOG_FLIGHT_BINS) + 1))
    for strokes in rallies.values():
        strokes.sort(key=lambda r: int(r["ball_round"]))
        for current, following in zip(strokes, strokes[1:]):
            if current["coarse_label"] not in STROKE_CLASSES:
                continue
            fps = float(current.get("fps") or 30.0)
            gap = (int(float(following["hit_frame"])) - int(float(current["hit_frame"]))) / fps
            counts[STROKE_CLASSES.index(current["coarse_label"]), flight_bin(gap)] += 1
    return np.log(counts / counts.sum(axis=1, keepdims=True))


def flight_bin(seconds: float) -> int:
    return int(np.digitize(np.log(max(seconds, 1e-3)), LOG_FLIGHT_BINS))


def rescore_strokes_by_flight_time(
    predictions: list[dict[str, Any]],
    flight_model: np.ndarray,
    *,
    fps: float,
    weight: float = DEFAULT_WEIGHT,
    log_probs_key: str = "_stroke_log_probs",
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Re-score ``stroke``/``top2``/``confidence`` in place using rally-local flight times.

    Predictions need ``frame``, ``rally`` and the classifier log-probabilities
    under ``log_probs_key``; hits without log-probabilities are left untouched.
    """
    by_rally: dict[Any, list[dict[str, Any]]] = defaultdict(list)
    for prediction in predictions:
        by_rally[prediction.get("rally", 1)].append(prediction)

    changes: list[dict[str, Any]] = []
    for items in by_rally.values():
        items.sort(key=lambda p: p["frame"])
        for current, following in zip(items, items[1:]):
            log_probs = current.get(log_probs_key)
            if log_probs is None:
                continue
            flight = (following["frame"] - current["frame"]) / fps
            score = np.asarray(log_probs, dtype=float) + weight * flight_model[:, flight_bin(flight)]
            probs = np.exp(score - score.max())
            probs /= probs.sum()
            order = np.argsort(-probs)
            stroke = STROKE_CLASSES[int(order[0])]
            if stroke != current["stroke"]:
                changes.append({
                    "hit_index": current.get("hit_index"),
                    "frame": current["frame"],
                    "flight_seconds": round(flight, 3),
                    "original_stroke": current["stroke"],
                    "refined_stroke": stroke,
                })
            current["stroke"] = stroke
            current["top2"] = [STROKE_CLASSES[int(k)] for k in order[:2]]
            current["confidence"] = round(float(probs[order[0]]), 4)
    return predictions, changes
