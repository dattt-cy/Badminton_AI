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
        if skeleton.shape[-1] < 2:
            raise ValueError("BadmintonRandomRot2D requires x/y keypoints")
        angle = np.random.uniform(-self.theta, self.theta)
        cosine, sine = np.cos(angle), np.sin(angle)
        rotation = np.asarray(
            [[cosine, -sine], [sine, cosine]], dtype=skeleton.dtype
        )
        rotated_xy = np.einsum("ab,mtvb->mtva", rotation, skeleton[..., :2])
        results["keypoint"] = (
            rotated_xy
            if skeleton.shape[-1] == 2
            else np.concatenate([rotated_xy, skeleton[..., 2:]], axis=-1)
        )
        return results


@_register
class BadmintonEdgePad:
    """Repeat the final pose so short clips do not wrap in UniformSample."""

    def __init__(self, min_frames: int = 64) -> None:
        if min_frames <= 0:
            raise ValueError("min_frames must be positive")
        self.min_frames = min_frames

    def __call__(self, results: dict) -> dict:
        frame_count = int(results["total_frames"])
        if frame_count <= 0:
            raise ValueError("Cannot pad an empty pose sequence")
        if frame_count >= self.min_frames:
            return results

        padding = self.min_frames - frame_count
        output = dict(results)
        output["keypoint"] = np.pad(
            results["keypoint"],
            ((0, 0), (0, padding), (0, 0), (0, 0)),
            mode="edge",
        )
        if "keypoint_score" in results:
            output["keypoint_score"] = np.pad(
                results["keypoint_score"],
                ((0, 0), (0, padding), (0, 0)),
                mode="edge",
            )
        output["total_frames"] = self.min_frames
        return output
