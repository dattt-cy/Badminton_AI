"""Confidence-aware temporal smoothing for pose keypoints."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
from scipy.signal import butter, sosfiltfilt

from ai_classifier.pose.estimator import PoseSequence


class KeypointSmoother:
    """Reject isolated pose spikes, interpolate short gaps, and low-pass XY.

    ``ema`` preserves the original project behaviour. ``butterworth`` uses a
    cutoff expressed in Hz, so the effective smoothing stays comparable when
    videos have different frame rates. Both modes are run forward and backward
    to avoid shifting fast events such as the estimated racket-contact frame.
    """

    def __init__(
        self,
        *,
        alpha: float = 0.35,
        min_confidence: float = 0.3,
        max_gap: int = 4,
        filter_type: str = "ema",
        cutoff_hz: float = 6.0,
        filter_order: int = 2,
        reject_outliers: bool = False,
        hampel_window_seconds: float = 0.12,
        hampel_n_sigma: float = 3.0,
    ) -> None:
        if not 0.0 < alpha <= 1.0:
            raise ValueError("alpha must be in (0, 1]")
        if not 0.0 <= min_confidence <= 1.0:
            raise ValueError("min_confidence must be between 0 and 1")
        if max_gap < 0:
            raise ValueError("max_gap cannot be negative")
        if filter_type not in {"ema", "butterworth", "none"}:
            raise ValueError("filter_type must be ema, butterworth, or none")
        if cutoff_hz <= 0:
            raise ValueError("cutoff_hz must be positive")
        if filter_order < 1:
            raise ValueError("filter_order must be positive")
        if hampel_window_seconds <= 0:
            raise ValueError("hampel_window_seconds must be positive")
        if hampel_n_sigma <= 0:
            raise ValueError("hampel_n_sigma must be positive")
        self.alpha = alpha
        self.min_confidence = min_confidence
        self.max_gap = max_gap
        self.filter_type = filter_type
        self.cutoff_hz = cutoff_hz
        self.filter_order = filter_order
        self.reject_outliers = reject_outliers
        self.hampel_window_seconds = hampel_window_seconds
        self.hampel_n_sigma = hampel_n_sigma

    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> "KeypointSmoother":
        """Build a smoother from a YAML ``smoothing`` mapping."""
        return cls(
            alpha=float(config.get("alpha", 0.35)),
            min_confidence=float(config.get("min_confidence", 0.3)),
            max_gap=int(config.get("max_gap", 4)),
            filter_type=str(config.get("filter_type", "ema")),
            cutoff_hz=float(config.get("cutoff_hz", 6.0)),
            filter_order=int(config.get("filter_order", 2)),
            reject_outliers=bool(config.get("reject_outliers", False)),
            hampel_window_seconds=float(config.get("hampel_window_seconds", 0.12)),
            hampel_n_sigma=float(config.get("hampel_n_sigma", 3.0)),
        )

    def smooth(self, sequence: PoseSequence) -> PoseSequence:
        keypoints = np.asarray(sequence.keypoints, dtype=np.float32)
        if keypoints.ndim != 3 or keypoints.shape[2] != 3:
            raise ValueError("keypoints must have shape (frames, joints, 3)")
        if sequence.fps <= 0:
            raise ValueError("fps must be positive")

        output = np.zeros_like(keypoints)
        for joint_index in range(keypoints.shape[1]):
            joint = keypoints[:, joint_index].copy()
            valid = joint[:, 2] >= self.min_confidence
            if self.reject_outliers:
                self._reject_hampel_outliers(joint, valid, sequence.fps)
            self._interpolate_short_gaps(joint, valid)
            output[:, joint_index, 2] = np.where(valid, joint[:, 2], 0.0)

            for start, end in self._valid_segments(valid):
                coordinates = joint[start:end, :2]
                output[start:end, joint_index, :2] = self._filter_coordinates(
                    coordinates, sequence.fps
                )

        return PoseSequence(
            output,
            sequence.fps,
            sequence.frame_width,
            sequence.frame_height,
        )

    def _reject_hampel_outliers(
        self, joint: np.ndarray, valid: np.ndarray, fps: float
    ) -> None:
        """Turn isolated 2D localization spikes into gaps before interpolation."""
        radius = max(1, int(round(self.hampel_window_seconds * fps / 2.0)))
        original_valid = valid.copy()
        for index in np.flatnonzero(original_valid):
            start = max(0, index - radius)
            end = min(len(joint), index + radius + 1)
            neighbours = joint[start:end, :2][original_valid[start:end]]
            if len(neighbours) < 3:
                continue
            center = np.median(neighbours, axis=0)
            distances = np.linalg.norm(neighbours - center, axis=1)
            median_distance = float(np.median(distances))
            mad = float(np.median(np.abs(distances - median_distance)))
            robust_sigma = 1.4826 * mad
            residual = float(np.linalg.norm(joint[index, :2] - center))
            threshold = median_distance + self.hampel_n_sigma * max(robust_sigma, 1e-6)
            if residual > threshold:
                valid[index] = False
                joint[index, 2] = 0.0

    def _filter_coordinates(self, values: np.ndarray, fps: float) -> np.ndarray:
        if self.filter_type == "none" or len(values) < 2:
            return values.astype(np.float32, copy=True)
        if self.filter_type == "ema":
            return self._zero_phase_ema(values, self.alpha)

        # Keep the normalized cutoff strictly below Nyquist for low-FPS input.
        effective_cutoff = min(self.cutoff_hz, fps * 0.45)
        if fps <= 0 or effective_cutoff <= 0:
            raise ValueError("fps must be positive")
        sos = butter(self.filter_order, effective_cutoff, btype="low", fs=fps, output="sos")
        try:
            return sosfiltfilt(sos, values, axis=0).astype(np.float32)
        except ValueError:
            # filtfilt needs padding around a segment. Use the closest
            # frequency-derived zero-phase EMA for short valid runs.
            alpha = 1.0 - np.exp(-2.0 * np.pi * effective_cutoff / fps)
            return self._zero_phase_ema(values, float(alpha))

    def _zero_phase_ema(self, values: np.ndarray, alpha: float) -> np.ndarray:
        forward = self._ema(values, alpha)
        backward = self._ema(values[::-1], alpha)[::-1]
        return ((forward + backward) / 2.0).astype(np.float32)

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

    @staticmethod
    def _ema(values: np.ndarray, alpha: float) -> np.ndarray:
        filtered = values.astype(np.float32, copy=True)
        for index in range(1, len(filtered)):
            filtered[index] = (
                alpha * filtered[index]
                + (1.0 - alpha) * filtered[index - 1]
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
