'''Shared types for pose-estimation backends.'''

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class PoseSequence:
    '''Pose keypoints and video metadata.

    Keypoints has shape (frames, joints, 3). The last dimension is
    (x, y, confidence) in pixel coordinates. Missing poses are all zeros.
    '''

    keypoints: NDArray[np.float32]
    fps: float
    frame_width: int
    frame_height: int

    def save(self, output_path: str | Path) -> Path:
        '''Save keypoints and metadata in a compressed NumPy archive.'''
        destination = Path(output_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            destination,
            keypoints=self.keypoints,
            fps=np.float32(self.fps),
            frame_width=np.int32(self.frame_width),
            frame_height=np.int32(self.frame_height),
        )
        return destination


class PoseEstimator(Protocol):
    '''Interface implemented by pose-estimation backends.'''

    def extract(self, video_path: str | Path) -> PoseSequence:
        '''Extract one person pose sequence from a video.'''
