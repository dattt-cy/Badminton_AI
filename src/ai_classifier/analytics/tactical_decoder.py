"""Tactical Sequence Decoder (Linear-Chain CRF / Viterbi).

Features:
1. Strict turn-taking: only considers cross-player transitions when players alternate sides.
2. Laplace smoothing: ensures rare or deceptive strokes never have zero probability.
3. Confidence-adaptive scaling: if the visual classifier is already confident (high margin),
   the sequence prior does not override the observation.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import numpy as np

STROKE_CLASSES = [
    "serve", "clear", "smash", "drop",
    "net_shot", "lift", "drive", "net_attack",
]


class TacticalSequenceDecoder:
    def __init__(
        self,
        manifest_path: Path | str | None = None,
        *,
        alpha: float = 0.5,
        default_lambda: float = 0.20,
        exclude_matches: set[str] | None = None,
    ):
        self.alpha = float(alpha)
        self.default_lambda = float(default_lambda)
        self.exclude_matches = exclude_matches or set()
        self.num_classes = len(STROKE_CLASSES)

        self.trans_prob = np.ones((self.num_classes, self.num_classes)) / self.num_classes
        self.start_prob = np.ones(self.num_classes) / self.num_classes
        self.log_trans = np.log(self.trans_prob)
        self.log_start = np.log(self.start_prob)

        if manifest_path is not None:
            self.fit_from_manifest(Path(manifest_path))

    def fit_from_manifest(self, manifest_path: Path) -> None:
        """Estimate transition probabilities from alternating player turns."""
        with manifest_path.open(encoding="utf-8-sig", newline="") as handle:
            rows = list(csv.DictReader(handle))

        from collections import defaultdict
        rallies = defaultdict(list)
        for r in rows:
            mid = str(r.get("match_id", ""))
            if mid in self.exclude_matches:
                continue
            key = (mid, r.get("set_id", "1"), r.get("rally", "1"))
            rallies[key].append(r)

        counts = np.zeros((self.num_classes, self.num_classes), dtype=float)
        start_counts = np.zeros(self.num_classes, dtype=float)

        for k, r_list in rallies.items():
            sorted_r = sorted(
                r_list,
                key=lambda x: int(float(x.get("ball_round", x.get("hit_frame", 0)))),
            )
            if not sorted_r:
                continue

            first_stroke = sorted_r[0].get("coarse_label")
            if first_stroke in STROKE_CLASSES:
                start_counts[STROKE_CLASSES.index(first_stroke)] += 1

            for i in range(len(sorted_r) - 1):
                s1_side = sorted_r[i].get("player_side")
                s2_side = sorted_r[i + 1].get("player_side")
                # Rule 1: Strictly alternate player turns
                if s1_side and s2_side and s1_side == s2_side:
                    continue

                c1_label = sorted_r[i].get("coarse_label")
                c2_label = sorted_r[i + 1].get("coarse_label")
                if c1_label in STROKE_CLASSES and c2_label in STROKE_CLASSES:
                    idx1 = STROKE_CLASSES.index(c1_label)
                    idx2 = STROKE_CLASSES.index(c2_label)
                    counts[idx1, idx2] += 1

        # Rule 2: Laplace smoothing
        self.start_prob = (start_counts + self.alpha) / (
            start_counts.sum() + self.num_classes * self.alpha
        )
        self.trans_prob = (counts + self.alpha) / (
            counts.sum(axis=1, keepdims=True) + self.num_classes * self.alpha
        )
        self.log_start = np.log(self.start_prob)
        self.log_trans = np.log(self.trans_prob)

    def decode_rally(
        self,
        log_probs: np.ndarray,
        hitters: list[str] | None = None,
        *,
        lam: float | None = None,
        use_confidence_gate: bool = True,
    ) -> list[int]:
        """Decode optimal stroke sequence for one rally using Viterbi."""
        T = len(log_probs)
        if T == 0:
            return []
        if T == 1:
            return [int(np.argmax(log_probs[0]))]

        effective_lam = self.default_lambda if lam is None else float(lam)

        # Convert log_probs to probabilities for margin calculation
        probs = np.exp(log_probs - np.max(log_probs, axis=-1, keepdims=True))
        probs = probs / probs.sum(axis=-1, keepdims=True)

        dp = np.zeros((T, self.num_classes))
        backpointer = np.zeros((T, self.num_classes), dtype=int)

        # t = 0
        dp[0] = log_probs[0] + effective_lam * self.log_start

        for t in range(1, T):
            is_alternating = True
            if hitters is not None and t < len(hitters):
                h_prev = hitters[t - 1]
                h_cur = hitters[t]
                if h_prev and h_cur and h_prev == h_cur:
                    is_alternating = False

            # Rule 3: Confidence gating
            sort_p = np.sort(probs[t])[::-1]
            margin = sort_p[0] - sort_p[1]
            step_lam = (
                effective_lam * (1.0 - margin * 0.6)
                if use_confidence_gate
                else effective_lam
            )

            for s in range(self.num_classes):
                if is_alternating and step_lam > 1e-4:
                    trans_score = step_lam * self.log_trans[:, s]
                else:
                    trans_score = 0.0

                scores = dp[t - 1] + trans_score
                best_prev = np.argmax(scores)
                dp[t, s] = scores[best_prev] + log_probs[t, s]
                backpointer[t, s] = best_prev

        best_end = int(np.argmax(dp[T - 1]))
        path = [best_end]
        for t in range(T - 1, 0, -1):
            path.append(backpointer[t, path[-1]])
        path.reverse()
        return path


def refine_predictions_by_sequence(
    predictions: list[dict[str, Any]],
    decoder: TacticalSequenceDecoder,
    *,
    lam: float | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Refine stroke predictions in-place across detected rallies."""
    from collections import defaultdict
    rally_groups = defaultdict(list)
    for idx, p in enumerate(predictions):
        rally_groups[p.get("rally", 1)].append((idx, p))

    changes: list[dict[str, Any]] = []

    for r_id, items in rally_groups.items():
        T = len(items)
        if T == 0:
            continue

        indices = [it[0] for it in items]
        hitters = [it[1].get("hitter", "") for it in items]

        # Extract log_probs if available, or approximate from confidence & top2
        log_probs = []
        for it in items:
            p = it[1]
            if "_logits" in p and p["_logits"] is not None:
                l = np.array(p["_logits"], dtype=float)
                l = l - np.max(l)
                lp = l - np.log(np.sum(np.exp(l)))
            else:
                lp = np.full(8, -4.0)
                st = p.get("stroke")
                t2 = p.get("top2", [])
                cf = float(p.get("confidence", 0.5))
                if st in STROKE_CLASSES:
                    lp[STROKE_CLASSES.index(st)] = np.log(max(0.1, cf))
                if len(t2) > 1 and t2[1] in STROKE_CLASSES:
                    lp[STROKE_CLASSES.index(t2[1])] = np.log(max(0.05, 1.0 - cf))
                # Normalize
                lp = lp - np.log(np.sum(np.exp(lp)))
            log_probs.append(lp)

        log_probs_arr = np.stack(log_probs)
        best_path = decoder.decode_rally(log_probs_arr, hitters, lam=lam)

        for local_idx, (global_idx, p) in enumerate(items):
            old_stroke = p.get("stroke")
            new_idx = best_path[local_idx]
            new_stroke = STROKE_CLASSES[new_idx]
            if old_stroke != new_stroke:
                changes.append({
                    "hit_index": p.get("hit_index", global_idx),
                    "frame": p.get("frame", 0),
                    "rally": r_id,
                    "original_stroke": old_stroke,
                    "refined_stroke": new_stroke,
                })
                p["stroke"] = new_stroke
                p["sequence_refined"] = True

    return predictions, changes

