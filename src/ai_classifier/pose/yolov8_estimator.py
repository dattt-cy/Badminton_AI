'''YOLOv8 Pose adapter backed by Ultralytics.'''

from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .estimator import PoseSequence


class YOLOv8PoseEstimator:
    '''Extract COCO-17 keypoints with an Ultralytics YOLOv8 pose model.'''

    def __init__(
        self,
        model_path: str | Path = 'yolov8n-pose.pt',
        *,
        confidence: float = 0.25,
        image_size: int = 640,
        device: str | None = None,
        tracking_distance_weight: float = 1.0,
        max_tracking_distance: float = 0.15,
        max_tracking_gap: int = 5,
        target_region: str = 'any',
        model: Any | None = None,
    ) -> None:
        if not 0.0 <= confidence <= 1.0:
            raise ValueError('confidence must be between 0 and 1')
        if image_size <= 0:
            raise ValueError('image_size must be positive')
        if tracking_distance_weight < 0:
            raise ValueError('tracking_distance_weight cannot be negative')
        if max_tracking_distance <= 0:
            raise ValueError('max_tracking_distance must be positive')
        if max_tracking_gap < 0:
            raise ValueError('max_tracking_gap cannot be negative')
        if target_region not in {'any', 'single', 'far', 'near'}:
            raise ValueError('target_region must be any, single, far, or near')

        if model is None:
            try:
                from ultralytics import YOLO
            except ImportError as exc:
                raise RuntimeError(
                    'Ultralytics is not installed. Run: pip install -e .'
                ) from exc
            model = YOLO(str(model_path))

        self.model = model
        self.confidence = confidence
        self.image_size = image_size
        self.device = device
        self.tracking_distance_weight = tracking_distance_weight
        self.max_tracking_distance = max_tracking_distance
        self.max_tracking_gap = max_tracking_gap
        self.target_region = target_region

    def extract(self, video_path: str | Path) -> PoseSequence:
        source = Path(video_path)
        if not source.is_file():
            raise FileNotFoundError(f'Video not found: {source}')

        fps, width, height = self._read_video_metadata(source)
        options: dict[str, Any] = {
            'source': str(source),
            'stream': True,
            'verbose': False,
            'conf': self.confidence,
            'imgsz': self.image_size,
        }
        if self.device:
            options['device'] = self.device

        frames: list[np.ndarray] = []
        joint_count = 17
        previous_pose: np.ndarray | None = None
        missed_frames = 0
        frame_diagonal = float(np.hypot(width, height))
        for result in self.model.predict(**options):
            data = self._keypoint_data(result)
            if data is None or data.shape[0] == 0:
                frames.append(np.zeros((joint_count, 3), dtype=np.float32))
                missed_frames += 1
                if missed_frames > self.max_tracking_gap:
                    previous_pose = None
                continue

            joint_count = data.shape[1]
            data = self._filter_target_region(data, width, height)
            if data.shape[0] == 0:
                frames.append(np.zeros((joint_count, 3), dtype=np.float32))
                missed_frames += 1
                if missed_frames > self.max_tracking_gap:
                    previous_pose = None
                continue
            selected = self._select_tracked_pose(
                data, previous_pose, frame_diagonal
            )
            if selected is None:
                frames.append(np.zeros((joint_count, 3), dtype=np.float32))
                missed_frames += 1
                if missed_frames > self.max_tracking_gap:
                    previous_pose = None
                continue
            selected = selected.astype(np.float32, copy=False)
            frames.append(selected)
            previous_pose = selected
            missed_frames = 0

        keypoints = (
            np.stack(frames).astype(np.float32, copy=False)
            if frames
            else np.empty((0, joint_count, 3), dtype=np.float32)
        )
        return PoseSequence(keypoints, fps, width, height)

    def _filter_target_region(
        self, candidates: np.ndarray, frame_width: int, frame_height: int
    ) -> np.ndarray:
        if self.target_region in {'any', 'single'}:
            return candidates
        centers = np.stack([self._pose_center(candidate) for candidate in candidates])
        centers_x = centers[:, 0]
        centers_y = centers[:, 1]
        if self.target_region == 'far':
            # Exclude spectators, coaches, and line judges outside the court.
            in_far_court = (
                (centers_y >= frame_height * 0.16)
                & (centers_y < frame_height * 0.58)
                & (centers_x >= frame_width * 0.18)
                & (centers_x <= frame_width * 0.82)
            )
            return candidates[in_far_court]
        return candidates[centers_y >= frame_height * 0.42]

    def _select_tracked_pose(
        self,
        candidates: np.ndarray,
        previous_pose: np.ndarray | None,
        frame_diagonal: float,
    ) -> np.ndarray | None:
        confidence_scores = candidates[:, :, 2].mean(axis=1)
        if previous_pose is None or frame_diagonal <= 0:
            return candidates[int(np.argmax(confidence_scores))]

        previous_center = self._pose_center(previous_pose)
        candidate_centers = np.stack(
            [self._pose_center(candidate) for candidate in candidates]
        )
        distances = np.linalg.norm(candidate_centers - previous_center, axis=1)
        nearby = distances <= self.max_tracking_distance * frame_diagonal
        if not nearby.any():
            return None
        tracking_scores = confidence_scores - (
            self.tracking_distance_weight * distances / frame_diagonal
        )
        tracking_scores[~nearby] = -np.inf
        return candidates[int(np.argmax(tracking_scores))]

    @staticmethod
    def _pose_center(pose: np.ndarray) -> np.ndarray:
        visible = pose[:, 2] >= 0.3
        if visible.any():
            return pose[visible, :2].mean(axis=0)
        return pose[:, :2].mean(axis=0)

    @staticmethod
    def _keypoint_data(result: Any) -> np.ndarray | None:
        keypoints = getattr(result, 'keypoints', None)
        if keypoints is None:
            return None
        tensor = getattr(keypoints, 'data', None)
        if tensor is None:
            return None
        array = tensor.detach().cpu().numpy()
        if array.ndim != 3 or array.shape[2] < 2:
            raise ValueError(f'Unexpected keypoint shape: {array.shape}')
        if array.shape[2] == 2:
            confidence = np.ones((*array.shape[:2], 1), dtype=array.dtype)
            array = np.concatenate((array, confidence), axis=2)
        return np.asarray(array[:, :, :3], dtype=np.float32)

    @staticmethod
    def _read_video_metadata(video_path: Path) -> tuple[float, int, int]:
        capture = cv2.VideoCapture(str(video_path))
        try:
            if not capture.isOpened():
                raise ValueError(f'Cannot open video: {video_path}')
            fps = float(capture.get(cv2.CAP_PROP_FPS))
            width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        finally:
            capture.release()
        return fps, width, height
