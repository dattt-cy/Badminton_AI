"""ST-GCN++ inference for badminton pose sequences."""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np

from ai_classifier.biomechanics import extract_geometry_features
from ai_classifier.pose import PoseSequence
from ai_classifier.segmentation import motion_score

TWO_CLASS_LABELS = ("backhand_drive", "forehand_lift")
THREE_CLASS_LABELS = ("background", "backhand_drive", "forehand_clear")


@dataclass(frozen=True)
class ActionPrediction:
    """A classified action and pose-quality diagnostics."""

    label: str
    confidence: float
    scores: dict[str, float]
    frame_count: int
    detected_ratio: float
    mean_confidence: float
    source_frame_count: int
    alignment_applied: bool
    window_start_frame: int
    window_end_frame: int
    motion_peak_frame: int | None
    padding_start_frames: int
    padding_end_frames: int


class STGCNPPClassifier:
    """Load a PySKL ST-GCN++ checkpoint and classify one pose sequence."""

    def __init__(
        self,
        config_path: str | Path,
        checkpoint_path: str | Path,
        *,
        device: str | None = None,
        labels: Sequence[str] | None = None,
        min_detected_ratio: float = 0.5,
        min_mean_confidence: float = 0.3,
        model_clip_frames: int = 64,
        model: Any | None = None,
        inference_fn: Callable[[Any, dict], list[tuple[int, float]]] | None = None,
    ) -> None:
        self.config_path = Path(config_path)
        self.checkpoint_path = Path(checkpoint_path)
        self.min_detected_ratio = min_detected_ratio
        self.min_mean_confidence = min_mean_confidence
        self.model_clip_frames = model_clip_frames
        if not 0 <= min_detected_ratio <= 1:
            raise ValueError("min_detected_ratio must be between 0 and 1")
        if not 0 <= min_mean_confidence <= 1:
            raise ValueError("min_mean_confidence must be between 0 and 1")
        if model_clip_frames <= 0:
            raise ValueError("model_clip_frames must be positive")

        if model is None:
            model, inference_fn = self._load_model(device)
        if labels is None:
            class_count = getattr(getattr(model, "cls_head", None), "num_classes", 2)
            configured_labels = getattr(getattr(model, "cfg", None), "class_names", None)
            if configured_labels is not None:
                labels = tuple(str(label) for label in configured_labels)
                if len(labels) != class_count:
                    raise ValueError(
                        f"Config defines {len(labels)} class names for a "
                        f"{class_count}-class model"
                    )
            elif class_count == len(TWO_CLASS_LABELS):
                labels = TWO_CLASS_LABELS
            elif class_count == len(THREE_CLASS_LABELS):
                labels = THREE_CLASS_LABELS
            else:
                raise ValueError(f"No label mapping for a {class_count}-class model")
        self.labels = tuple(labels)
        if not self.labels:
            raise ValueError("labels cannot be empty")
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

        annotation, padding_start, padding_end = self._pad_short_annotation(annotation)
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
            source_frame_count=int(sequence.keypoints.shape[0]),
            alignment_applied=False,
            window_start_frame=0,
            window_end_frame=int(sequence.keypoints.shape[0]),
            motion_peak_frame=None,
            padding_start_frames=padding_start,
            padding_end_frames=padding_end,
        )

    def predict_aligned(
        self,
        sequence: PoseSequence,
        *,
        window_seconds: float = 3.5,
        window_frames: int | None = None,
        peak_position: float = 0.55,
        handedness: str = "right",
    ) -> ActionPrediction:
        """Find the strongest swing and classify a time-normalized window.

        Missing context at either video boundary is filled by repeating the
        boundary pose. This prevents PySKL's short-clip sampler from wrapping
        the end of an action back to its beginning.
        """
        if window_seconds <= 0:
            raise ValueError("window_seconds must be positive")
        if window_frames is not None and window_frames <= 0:
            raise ValueError("window_frames must be positive")
        if not 0.0 <= peak_position <= 1.0:
            raise ValueError("peak_position must be between 0 and 1")
        if handedness not in {"left", "right"}:
            raise ValueError("handedness must be left or right")

        source_frames = int(sequence.keypoints.shape[0])
        if source_frames == 0:
            return self.predict(sequence)
        if sequence.fps <= 0 and window_frames is None:
            raise ValueError("fps must be positive when using window_seconds")

        target_frames = (
            int(window_frames)
            if window_frames is not None
            else max(self.model_clip_frames, int(round(window_seconds * sequence.fps)))
        )

        features = extract_geometry_features(sequence, handedness=handedness)
        scores = motion_score(features)
        peak_frame = int(np.argmax(scores)) if len(scores) and scores.max() > 0 else None

        if peak_frame is None:
            requested_start = (source_frames - target_frames) // 2
        else:
            peak_offset = int(round(peak_position * (target_frames - 1)))
            requested_start = peak_frame - peak_offset
        requested_end = requested_start + target_frames
        start_frame = max(0, requested_start)
        end_frame = min(source_frames, requested_end)
        padding_start = max(0, -requested_start)
        padding_end = max(0, requested_end - source_frames)
        keypoints = np.asarray(sequence.keypoints)[start_frame:end_frame]
        if padding_start or padding_end:
            keypoints = np.pad(
                keypoints,
                ((padding_start, padding_end), (0, 0), (0, 0)),
                mode="edge",
            )
        if len(keypoints) != target_frames:
            raise RuntimeError(
                f"Aligned window has {len(keypoints)} frames, expected {target_frames}"
            )
        aligned = PoseSequence(
            np.asarray(keypoints, dtype=np.float32),
            sequence.fps,
            sequence.frame_width,
            sequence.frame_height,
        )
        prediction = self.predict(aligned)
        return replace(
            prediction,
            source_frame_count=source_frames,
            alignment_applied=True,
            window_start_frame=start_frame,
            window_end_frame=end_frame,
            motion_peak_frame=peak_frame,
            padding_start_frames=padding_start + prediction.padding_start_frames,
            padding_end_frames=padding_end + prediction.padding_end_frames,
        )

    def _pad_short_annotation(self, annotation: dict) -> tuple[dict, int, int]:
        """Edge-pad inputs so UniformSample never circularly wraps a clip."""
        frame_count = int(annotation["total_frames"])
        if frame_count >= self.model_clip_frames:
            return annotation, 0, 0
        padding = self.model_clip_frames - frame_count
        padding_start = padding // 2
        padding_end = padding - padding_start
        padded = dict(annotation)
        padded["keypoint"] = np.pad(
            annotation["keypoint"],
            ((0, 0), (padding_start, padding_end), (0, 0), (0, 0)),
            mode="edge",
        )
        padded["keypoint_score"] = np.pad(
            annotation["keypoint_score"],
            ((0, 0), (padding_start, padding_end), (0, 0)),
            mode="edge",
        )
        padded["total_frames"] = self.model_clip_frames
        return padded, padding_start, padding_end

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
            "test_mode": True,
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
