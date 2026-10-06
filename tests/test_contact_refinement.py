import numpy as np
import pytest

from ai_classifier.localization.contact_refinement import refine_contact_index


def test_refines_to_plausible_nearby_wrist_contact() -> None:
    shuttle = np.full((9, 2), [0.8, 0.8], dtype=np.float32)
    wrists = np.full((9, 2), [0.1, 0.1], dtype=np.float32)
    shuttle[6] = [0.42, 0.40]
    wrists[6] = [0.40, 0.40]

    assert refine_contact_index(shuttle, wrists, center=4, radius=3) == 6


def test_keeps_center_when_tracks_are_missing_or_implausible() -> None:
    missing = np.zeros((7, 2), dtype=np.float32)
    assert refine_contact_index(missing, missing, center=3, radius=3) == 3

    shuttle = np.full((7, 2), [0.9, 0.9], dtype=np.float32)
    wrists = np.full((7, 2), [0.1, 0.1], dtype=np.float32)
    assert refine_contact_index(shuttle, wrists, center=3, radius=3) == 3


def test_zero_radius_is_disabled_and_invalid_center_is_rejected() -> None:
    track = np.ones((5, 2), dtype=np.float32)
    assert refine_contact_index(track, track, center=2, radius=0) == 2
    with pytest.raises(ValueError):
        refine_contact_index(track, track, center=5, radius=2)


if __name__ == "__main__":
    test_refines_to_plausible_nearby_wrist_contact()
    test_keeps_center_when_tracks_are_missing_or_implausible()
    test_zero_radius_is_disabled_and_invalid_center_is_rejected()
    print("Contact refinement tests passed!")

