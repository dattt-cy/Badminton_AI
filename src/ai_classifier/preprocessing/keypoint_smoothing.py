"""Confidence-aware temporal smoothing for pose keypoints."""

from __future__ import annotations

import numpy as np

from ai_classifier.pose.estimator import PoseSequence


class KeypointSmoother:
    """Interpolate short gaps and apply zero-phase exponential smoothing."""

    def __init__(
        self,
        *,
        alpha: float = 0.35,
        min_confidence: float = 0.3,
        max_gap: int = 4,
    ) -> None:
        if not 0.0 < alpha <= 1.0:
            raise ValueError("alpha must be in (0, 1]")
        if not 0.0 <= min_confidence <= 1.0:
            raise ValueError("min_confidence must be between 0 and 1")
        if max_gap < 0:
            raise ValueError("max_gap cannot be negative")
        self.alpha = alpha
        self.min_confidence = min_confidence
        self.max_gap = max_gap

    def smooth(self, sequence: PoseSequence) -> PoseSequence:
        keypoints = np.asarray(sequence.keypoints, dtype=np.float32)
        if keypoints.ndim != 3 or keypoints.shape[2] != 3:
            raise ValueError("keypoints must have shape (frames, joints, 3)")

        output = np.zeros_like(keypoints)
        for joint_index in range(keypoints.shape[1]):
            joint = keypoints[:, joint_index].copy()
            valid = joint[:, 2] >= self.min_confidence
            self._interpolate_short_gaps(joint, valid)
            output[:, joint_index, 2] = np.where(valid, joint[:, 2], 0.0)

            for start, end in self._valid_segments(valid):
                coordinates = joint[start:end, :2]
                forward = self._ema(coordinates)
                backward = self._ema(coordinates[::-1])[::-1]
                output[start:end, joint_index, :2] = (forward + backward) / 2.0

        return PoseSequence(
            output,
            sequence.fps,
            sequence.frame_width,
            sequence.frame_height,
        )

    def _interpolate_short_gaps(
        self, joint: np.ndarray, valid: np.ndarray
    ) -> None:
        frame_count = len(valid)
        index = 0
        while index < frame_count:
            if valid[index]:
                index += 1
                continue
            start = index
            while index < frame_count and not valid[index]:
                index += 1
            end = index
            gap = end - start
            if start == 0 or end == frame_count or gap > self.max_gap:
                continue

            left = joint[start - 1]
            right = joint[end]
            for offset, frame_index in enumerate(range(start, end), start=1):
                ratio = offset / (gap + 1)
                joint[frame_index, :2] = (
                    left[:2] * (1.0 - ratio) + right[:2] * ratio
                )
                joint[frame_index, 2] = min(left[2], right[2])
                valid[frame_index] = True

    def _ema(self, values: np.ndarray) -> np.ndarray:
        filtered = values.astype(np.float32, copy=True)
        for index in range(1, len(filtered)):
            filtered[index] = (
                self.alpha * filtered[index]
                + (1.0 - self.alpha) * filtered[index - 1]
            )
        return filtered

    @staticmethod
    def _valid_segments(valid: np.ndarray):
        index = 0
        while index < len(valid):
            while index < len(valid) and not valid[index]:
                index += 1
            start = index
            while index < len(valid) and valid[index]:
                index += 1
            if start < index:
                yield start, index

