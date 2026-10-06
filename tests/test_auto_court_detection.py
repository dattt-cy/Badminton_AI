import numpy as np

from scripts.inference.auto_court_detection import _validate_corners


FRAME_SHAPE = (720, 1280, 3)


def test_accepts_full_centered_broadcast_court() -> None:
    corners = np.float32([[222, 694], [1061, 709], [873, 262], [412, 260]])
    assert _validate_corners(corners, FRAME_SHAPE)


def test_rejects_one_sided_or_shallow_green_regions() -> None:
    right_half = np.float32([[858, 719], [1279, 719], [1279, 364], [794, 364]])
    shallow = np.float32([[0, 719], [877, 719], [877, 584], [0, 584]])
    non_perspective = np.float32([[0, 719], [1279, 719], [1177, 360], [0, 388]])
    assert not _validate_corners(right_half, FRAME_SHAPE)
    assert not _validate_corners(shallow, FRAME_SHAPE)
    assert not _validate_corners(non_perspective, FRAME_SHAPE)
