import numpy as np
import pytest

from ai_classifier.biomechanics.features_3d import (
    MULTISENSE_JOINT_INDEX,
    extract_multisense_geometry_features,
)


def _skeleton(frames=8):
    xyz = np.zeros((frames, 21, 3), dtype=np.float32)
    j = MULTISENSE_JOINT_INDEX
    xyz[:, j["Hips"]] = (0, 0, 0)
    xyz[:, j["Spine3"]] = (0, 2, 0)
    xyz[:, j["RightShoulder"]] = (-0.5, 2, 0)
    xyz[:, j["RightArm"]] = (-1, 2, 0)
    xyz[:, j["RightForeArm"]] = (-1, 1, 0)
    xyz[:, j["RightHand"]] = (0, 1, 0)
    xyz[:, j["LeftArm"]] = (1, 2, 0)
    xyz[:, j["RightUpLeg"]] = (-0.4, 0, 0)
    xyz[:, j["RightLeg"]] = (-0.4, -1, 0)
    xyz[:, j["RightFoot"]] = (-0.4, -2, 0)
    xyz[:, j["LeftFoot"]] = (0.4, -2, 0)
    return xyz


def test_extracts_anatomical_3d_geometry():
    xyz = _skeleton()
    features = extract_multisense_geometry_features(xyz, np.arange(len(xyz))/60)
    np.testing.assert_allclose(features.column("elbow_angle"), 90, atol=1e-5)
    np.testing.assert_allclose(features.column("torso_lean"), 0, atol=1e-5)
    np.testing.assert_allclose(features.column("stance_width"), 0.4, atol=1e-5)
    np.testing.assert_allclose(
        features.column("elbow_torso_distance"), 0.5, atol=1e-5
    )
    assert features.values.shape == (len(xyz), 15)


def test_accepts_flat_hdf5_shape():
    xyz = _skeleton(2).reshape(2, 63)
    features = extract_multisense_geometry_features(xyz, np.array([10.0, 10.02]))
    np.testing.assert_allclose(features.timestamps, [0.0, 0.02])


def test_rejects_non_monotonic_timestamps():
    with pytest.raises(ValueError, match="strictly increasing"):
        extract_multisense_geometry_features(_skeleton(2), np.array([10.0, 10.0]))
