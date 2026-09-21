import cv2
import numpy as np

from scripts.inference.classify_shuttleset_fusion_video import court_homography


def test_court_homography_preserves_bl_br_tr_tl_order():
    corners = np.asarray([
        [100.0, 900.0], [900.0, 900.0], [700.0, 100.0], [300.0, 100.0]
    ], dtype=np.float32)
    projected = cv2.perspectiveTransform(corners[None], court_homography(corners))[0]
    expected = np.asarray([[0, 1], [1, 1], [1, 0], [0, 0]], dtype=np.float32)
    np.testing.assert_allclose(projected, expected, atol=1e-5)
