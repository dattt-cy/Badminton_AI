"""Project-specific PySKL transforms."""

from __future__ import annotations

import numpy as np

try:
    from pyskl.datasets.builder import PIPELINES
except ImportError:  # Allows unit tests before the optional PySKL install.
    PIPELINES = None


def _register(cls):
    return PIPELINES.register_module()(cls) if PIPELINES is not None else cls


@_register
class BadmintonRandomRot2D:
    """Rotate 2D skeletons by a uniformly sampled symmetric angle."""

    def __init__(self, theta: float = 0.12) -> None:
        if theta < 0:
            raise ValueError("theta cannot be negative")
        self.theta = theta

    def __call__(self, results: dict) -> dict:
        skeleton = results["keypoint"]
        if skeleton.shape[-1] != 2:
            raise ValueError("BadmintonRandomRot2D requires 2D keypoints")
        angle = np.random.uniform(-self.theta, self.theta)
        cosine, sine = np.cos(angle), np.sin(angle)
        rotation = np.asarray(
            [[cosine, -sine], [sine, cosine]], dtype=skeleton.dtype
        )
        results["keypoint"] = np.einsum("ab,mtvb->mtva", rotation, skeleton)
        return results

