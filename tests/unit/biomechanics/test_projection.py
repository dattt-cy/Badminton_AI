import numpy as np
import pytest

from ai_classifier.biomechanics.projection import (
    fit_perspective_camera,
    fit_weak_perspective_camera,
)


def test_fits_known_weak_perspective_camera():
    points = np.array([
        [0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0],
        [0, 0, 1], [1, 0, 1], [0, 1, 1], [1, 1, 1],
    ], dtype=float)
    expected = 3.0 * points[:, :2] + [20.0, 30.0]

    camera = fit_weak_perspective_camera(points, expected)

    np.testing.assert_allclose(camera.project(points), expected, atol=1e-8)
    assert camera.scale == pytest.approx(3.0)
    assert camera.median_error < 1e-8


def test_requires_enough_correspondences():
    with pytest.raises(ValueError, match="six"):
        fit_weak_perspective_camera(np.zeros((5, 3)), np.zeros((5, 2)))


def test_fits_known_perspective_camera():
    rng = np.random.default_rng(42)
    points = rng.uniform([-1, -1, 3], [1, 1, 7], size=(30, 3))
    matrix = np.array([
        [800, 0, 320, 0], [0, 800, 240, 0], [0, 0, 1, 0]
    ], dtype=float)
    homogeneous = np.column_stack([points, np.ones(len(points))]) @ matrix.T
    expected = homogeneous[:, :2] / homogeneous[:, 2:3]

    camera = fit_perspective_camera(points, expected)

    np.testing.assert_allclose(camera.project(points), expected, atol=1e-6)
    assert camera.median_error < 1e-6
