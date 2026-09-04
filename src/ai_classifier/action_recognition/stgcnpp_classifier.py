"""ST-GCN++ inference for badminton pose sequences."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np

from ai_classifier.pose import PoseSequence

DEFAULT_LABELS = ("backhand_drive", "forehand_lift")


@dataclass(frozen=True)
class ActionPrediction:
    """A classified action and pose-quality diagnostics."""

    label: str
    confidence: float
    scores: dict[str, float]
    frame_count: int
    detected_ratio: float
    mean_confidence: float


class STGCNPPClassifier:
    """Load a PySKL ST-GCN++ checkpoint and classify one pose sequence."""

    def __init__(
        self,
        config_path: str | Path,
        checkpoint_path: str | Path,
        *,
        device: str | None = None,
        labels: Sequence[str] = DEFAULT_LABELS,
        min_detected_ratio: float = 0.5,
        min_mean_confidence: float = 0.3,
        model: Any | None = None,
        inference_fn: Callable[[Any, dict], list[tuple[int, float]]] | None = None,
    ) -> None:
        self.config_path = Path(config_path)
        self.checkpoint_path = Path(checkpoint_path)
        self.labels = tuple(labels)
        self.min_detected_ratio = min_detected_ratio
        self.min_mean_confidence = min_mean_confidence
        if not self.labels:
            raise ValueError("labels cannot be empty")
        if not 0 <= min_detected_ratio <= 1:
            raise ValueError("min_detected_ratio must be between 0 and 1")
        if not 0 <= min_mean_confidence <= 1:
            raise ValueError("min_mean_confidence must be between 0 and 1")

        if model is None:
            model, inference_fn = self._load_model(device)
        if inference_fn is None:
            raise ValueError("inference_fn is required when injecting a model")
        self.model = model
        self._inference = inference_fn

    def predict(self, sequence: PoseSequence) -> ActionPrediction:
        """Classify a smoothed COCO-17 pose sequence."""
        annotation, detected_ratio, mean_confidence = self.prepare_input(sequence)
        if detected_ratio < self.min_detected_ratio:
            raise ValueError(
                f"Pose quality too low: detected frames {detected_ratio:.1%} "
                f"< {self.min_detected_ratio:.1%}"
            )
        if mean_confidence < self.min_mean_confidence:
            raise ValueError(
                f"Pose quality too low: mean confidence {mean_confidence:.3f} "
                f"< {self.min_mean_confidence:.3f}"
            )

        ranked_scores = self._inference(self.model, annotation)
        scores = {
            self.labels[int(index)]: float(score)
            for index, score in ranked_scores
            if 0 <= int(index) < len(self.labels)
        }
        if len(scores) != len(self.labels):
            raise ValueError(
                f"Model returned {len(scores)} known classes, expected {len(self.labels)}"
            )
        label = max(scores, key=scores.get)
        return ActionPrediction(
            label=label,
            confidence=scores[label],
            scores=scores,
            frame_count=int(sequence.keypoints.shape[0]),
            detected_ratio=detected_ratio,
            mean_confidence=mean_confidence,
        )

    @staticmethod
    def prepare_input(sequence: PoseSequence) -> tuple[dict, float, float]:
        """Convert a pose sequence to the dictionary consumed by PySKL."""
        keypoints = np.asarray(sequence.keypoints, dtype=np.float32)
        if keypoints.ndim != 3 or keypoints.shape[1:] != (17, 3):
            raise ValueError(
                f"Expected keypoints with shape (T, 17, 3), got {keypoints.shape}"
            )
        if keypoints.shape[0] == 0:
            raise ValueError("Pose sequence is empty")
        if sequence.frame_width <= 0 or sequence.frame_height <= 0:
            raise ValueError("Frame width and height must be positive")

        confidence = np.clip(keypoints[..., 2], 0.0, 1.0)
        detected_ratio = float((confidence.max(axis=1) > 0).mean())
        mean_confidence = float(confidence.mean())
        annotation = {
            "frame_dir": "inference",
            "label": -1,
            "total_frames": int(keypoints.shape[0]),
            "img_shape": (sequence.frame_height, sequence.frame_width),
            "original_shape": (sequence.frame_height, sequence.frame_width),
            "start_index": 0,
            "modality": "Pose",
            "keypoint": keypoints[None, ..., :2],
            "keypoint_score": confidence[None, ...],
        }
        return annotation, detected_ratio, mean_confidence

    def _load_model(self, device: str | None) -> tuple[Any, Callable]:
        if not self.config_path.is_file():
            raise FileNotFoundError(f"Action config not found: {self.config_path}")
        if not self.checkpoint_path.is_file():
            raise FileNotFoundError(f"Checkpoint not found: {self.checkpoint_path}")
        try:
            import torch
            from pyskl.apis import inference_recognizer, init_recognizer
        except ImportError as exc:
            raise RuntimeError(
                "PySKL/MMCV is not installed in this Python environment. "
                "Run inference in the same pyskl_310 environment used for training."
            ) from exc

        # Registers the project-specific transform referenced by the config.
        from ai_classifier.action_recognition import pyskl_transforms  # noqa: F401

        selected_device = device or ("cuda:0" if torch.cuda.is_available() else "cpu")
        model = init_recognizer(
            str(self.config_path), str(self.checkpoint_path), device=selected_device
        )
        return model, inference_recognizer
