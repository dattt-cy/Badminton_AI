import numpy as np

from ai_classifier.features import modality_quality_vector


def test_modality_quality_vector_preserves_training_scale() -> None:
    pose = np.array(
        [
            [[0.0, 0.0], [0.2, 0.4]],
            [[0.1, 0.2], [0.4, 0.8]],
            [[0.2, 0.4], [0.6, 1.2]],
        ],
        dtype=np.float32,
    )
    shuttle = np.array(
        [[0.0, 0.0], [0.2, 0.1], [0.7, 0.5]],
        dtype=np.float32,
    )

    quality = modality_quality_vector(pose, shuttle)
    expected_speed_std = np.linalg.norm(np.diff(shuttle, axis=0), axis=-1).std()
    expected_pose_motion = pose.reshape(len(pose), -1).std(axis=0).mean()

    assert quality.shape == (4,)
    assert quality.dtype == np.float32
    assert quality[0] == np.float32(1 / 3)
    assert quality[1] == np.float32(expected_speed_std)
    assert quality[2] == np.float32(1 / 6)
    assert quality[3] == np.float32(expected_pose_motion)
