import numpy as np

from ai_classifier.action_recognition.pyskl_transforms import BadmintonRandomRot2D


def test_random_rotation_stays_inside_symmetric_limit(monkeypatch) -> None:
    captured = {}

    def fake_uniform(low: float, high: float) -> float:
        captured.update(low=low, high=high)
        return 0.0

    monkeypatch.setattr(np.random, "uniform", fake_uniform)
    skeleton = np.ones((1, 4, 17, 2), dtype=np.float32)

    result = BadmintonRandomRot2D(theta=0.12)({"keypoint": skeleton})

    assert captured == {"low": -0.12, "high": 0.12}
    np.testing.assert_array_equal(result["keypoint"], skeleton)
