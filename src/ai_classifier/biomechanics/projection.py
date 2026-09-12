"""Robust weak-perspective projection for synchronized 3D and 2D skeletons."""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np


@dataclass(frozen=True)
class WeakPerspectiveCamera:
    rotation_3x2: np.ndarray
    scale: float
    translation_2d: np.ndarray
    median_error: float
    p90_error: float
    point_count: int

    def project(self, points_3d: np.ndarray) -> np.ndarray:
        points = np.asarray(points_3d, dtype=np.float64)
        return self.scale * points @ self.rotation_3x2 + self.translation_2d


@dataclass(frozen=True)
class PerspectiveCamera:
    matrix_3x4: np.ndarray
    median_error: float
    p90_error: float
    point_count: int

    def project(self, points_3d: np.ndarray) -> np.ndarray:
        points = np.asarray(points_3d, dtype=np.float64)
        homogeneous = np.column_stack([
            points.reshape(-1, 3), np.ones(points.size // 3)
        ])
        projected = homogeneous @ self.matrix_3x4.T
        denominator = projected[:, 2:3]
        safe = np.where(
            np.abs(denominator) > 1e-9,
            denominator,
            np.where(denominator < 0, -1e-9, 1e-9),
        )
        return (projected[:, :2] / safe).reshape(points.shape[:-1] + (2,))


def fit_weak_perspective_camera(
    points_3d: np.ndarray,
    points_2d: np.ndarray,
    weights: np.ndarray | None = None,
    *,
    trim_iterations: int = 2,
) -> WeakPerspectiveCamera:
    """Fit scale, orthographic rotation, and translation with robust trimming."""
    xyz = np.asarray(points_3d, dtype=np.float64).reshape(-1, 3)
    xy = np.asarray(points_2d, dtype=np.float64).reshape(-1, 2)
    weight = (
        np.ones(len(xyz), dtype=np.float64)
        if weights is None else np.asarray(weights, dtype=np.float64).reshape(-1)
    )
    valid = (
        np.isfinite(xyz).all(axis=1) & np.isfinite(xy).all(axis=1)
        & np.isfinite(weight) & (weight > 0)
    )
    if valid.sum() < 6:
        raise ValueError("At least six valid 3D/2D correspondences are required")

    keep = valid.copy()
    rotation = np.zeros((3, 2), dtype=np.float64)
    scale = 1.0
    translation = np.zeros(2, dtype=np.float64)
    for _ in range(trim_iterations + 1):
        current_weight = weight[keep]
        current_weight = current_weight / np.sum(current_weight)
        center_3d = np.sum(xyz[keep] * current_weight[:, None], axis=0)
        center_2d = np.sum(xy[keep] * current_weight[:, None], axis=0)
        centered_3d = xyz[keep] - center_3d
        centered_2d = xy[keep] - center_2d
        weighted_3d = centered_3d * np.sqrt(current_weight[:, None])
        weighted_2d = centered_2d * np.sqrt(current_weight[:, None])
        left, singular, right_t = np.linalg.svd(
            weighted_3d.T @ weighted_2d, full_matrices=False
        )
        rotation = left @ right_t
        projected = centered_3d @ rotation
        denominator = np.sum(current_weight * np.sum(projected ** 2, axis=1))
        if denominator <= 1e-12:
            raise ValueError("Degenerate 3D correspondences")
        scale = float(
            np.sum(current_weight * np.sum(centered_2d * projected, axis=1))
            / denominator
        )
        translation = center_2d - scale * center_3d @ rotation
        residual = np.linalg.norm(scale * xyz @ rotation + translation - xy, axis=1)
        finite_residual = residual[keep]
        median = float(np.median(finite_residual))
        mad = float(np.median(np.abs(finite_residual - median)))
        threshold = median + max(2.5 * 1.4826 * mad, 1.0)
        refined = valid & (residual <= threshold)
        if refined.sum() < 6 or np.array_equal(refined, keep):
            break
        keep = refined

    residual = np.linalg.norm(scale * xyz @ rotation + translation - xy, axis=1)
    used_error = residual[keep]
    return WeakPerspectiveCamera(
        rotation, scale, translation,
        float(np.median(used_error)), float(np.percentile(used_error, 90)),
        int(keep.sum()),
    )


def fit_perspective_camera(
    points_3d: np.ndarray,
    points_2d: np.ndarray,
    weights: np.ndarray | None = None,
    *,
    trim_iterations: int = 2,
) -> PerspectiveCamera:
    """Fit a projective 3x4 camera matrix with normalized robust DLT."""
    xyz = np.asarray(points_3d, dtype=np.float64).reshape(-1, 3)
    xy = np.asarray(points_2d, dtype=np.float64).reshape(-1, 2)
    weight = (
        np.ones(len(xyz), dtype=np.float64)
        if weights is None else np.asarray(weights, dtype=np.float64).reshape(-1)
    )
    valid = (
        np.isfinite(xyz).all(axis=1) & np.isfinite(xy).all(axis=1)
        & np.isfinite(weight) & (weight > 0)
    )
    if valid.sum() < 12:
        raise ValueError("At least twelve valid correspondences are required")
    keep = valid.copy()
    matrix = np.zeros((3, 4), dtype=np.float64)
    for _ in range(trim_iterations + 1):
        source, target = xyz[keep], xy[keep]
        source_center, target_center = np.mean(source, axis=0), np.mean(target, axis=0)
        source_rms = np.sqrt(np.mean(np.sum((source - source_center) ** 2, axis=1)))
        target_rms = np.sqrt(np.mean(np.sum((target - target_center) ** 2, axis=1)))
        if source_rms <= 1e-9 or target_rms <= 1e-9:
            raise ValueError("Degenerate camera correspondences")
        source_scale, target_scale = np.sqrt(3) / source_rms, np.sqrt(2) / target_rms
        transform_3d = np.eye(4)
        transform_3d[:3, :3] *= source_scale
        transform_3d[:3, 3] = -source_scale * source_center
        transform_2d = np.array([
            [target_scale, 0, -target_scale * target_center[0]],
            [0, target_scale, -target_scale * target_center[1]],
            [0, 0, 1],
        ])
        normalized_3d = np.column_stack([source, np.ones(len(source))]) @ transform_3d.T
        normalized_2d = (
            np.column_stack([target, np.ones(len(target))]) @ transform_2d.T
        )[:, :2]
        design = []
        for point, (u, v), confidence in zip(
            normalized_3d, normalized_2d, weight[keep]
        ):
            root_weight = math.sqrt(confidence)
            design.append(root_weight * np.r_[point, np.zeros(4), -u * point])
            design.append(root_weight * np.r_[np.zeros(4), point, -v * point])
        _, _, right_t = np.linalg.svd(np.asarray(design), full_matrices=False)
        normalized_matrix = right_t[-1].reshape(3, 4)
        matrix = np.linalg.inv(transform_2d) @ normalized_matrix @ transform_3d
        camera = PerspectiveCamera(matrix, 0.0, 0.0, int(keep.sum()))
        residual = np.linalg.norm(camera.project(xyz) - xy, axis=1)
        current = residual[keep]
        median = float(np.median(current))
        mad = float(np.median(np.abs(current - median)))
        threshold = median + max(2.5 * 1.4826 * mad, 1.0)
        refined = valid & (residual <= threshold)
        if refined.sum() < 12 or np.array_equal(refined, keep):
            break
        keep = refined
    residual = np.linalg.norm(PerspectiveCamera(matrix, 0, 0, 0).project(xyz) - xy, axis=1)
    used = residual[keep]
    return PerspectiveCamera(
        matrix, float(np.median(used)), float(np.percentile(used, 90)), int(keep.sum())
    )
