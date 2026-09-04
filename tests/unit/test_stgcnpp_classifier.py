import numpy as np
import pytest

from ai_classifier.action_recognition import STGCNPPClassifier
from ai_classifier.pose import PoseSequence


def make_sequence(confidence: float = 0.8) -> PoseSequence:
    keypoints = np.ones((20, 17, 3), dtype=np.float32)
    keypoints[..., 2] = confidence
    return PoseSequence(keypoints, 30.0, 1280, 720)


def test_prepare_input_matches_pyskl_pose_shape() -> None:
    annotation, detected_ratio, mean_confidence = STGCNPPClassifier.prepare_input(
        make_sequence()
    )

    assert annotation["keypoint"].shape == (1, 20, 17, 2)
    assert annotation["keypoint_score"].shape == (1, 20, 17)
    assert annotation["img_shape"] == (720, 1280)
    assert detected_ratio == 1.0
    assert mean_confidence == pytest.approx(0.8)


def test_prediction_maps_model_indices_to_labels() -> None:
    classifier = STGCNPPClassifier(
        "unused.py",
        "unused.pth",
        model=object(),
        inference_fn=lambda model, data: [(1, 0.75), (0, 0.25)],
    )

    prediction = classifier.predict(make_sequence())

    assert prediction.label == "forehand_lift"
    assert prediction.confidence == pytest.approx(0.75)
    assert prediction.scores == {
        "forehand_lift": pytest.approx(0.75),
        "backhand_drive": pytest.approx(0.25),
    }


def test_prediction_rejects_low_quality_pose() -> None:
    classifier = STGCNPPClassifier(
        "unused.py",
        "unused.pth",
        model=object(),
        inference_fn=lambda model, data: [(0, 1.0), (1, 0.0)],
    )

    with pytest.raises(ValueError, match="mean confidence"):
        classifier.predict(make_sequence(confidence=0.1))


def test_three_class_model_selects_other_label() -> None:
    class Head:
        num_classes = 3

    class Model:
        cls_head = Head()

    classifier = STGCNPPClassifier(
        "unused.py",
        "unused.pth",
        model=Model(),
        inference_fn=lambda model, data: [(2, 0.8), (0, 0.15), (1, 0.05)],
    )

    prediction = classifier.predict(make_sequence())

    assert prediction.label == "other"
    assert prediction.confidence == pytest.approx(0.8)
