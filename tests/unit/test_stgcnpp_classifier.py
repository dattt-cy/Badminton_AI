import numpy as np
import pytest

from ai_classifier.action_recognition import STGCNPPClassifier
from ai_classifier.biomechanics import GeometryFeatures
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
    assert annotation["test_mode"] is True
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


def test_prediction_edge_pads_short_input_before_inference() -> None:
    sequence = make_sequence()
    sequence.keypoints[:, :, 0] = np.arange(20, dtype=np.float32)[:, None]
    captured = {}

    def inference(model, annotation):
        captured.update(annotation)
        return [(0, 0.75), (1, 0.25)]

    classifier = STGCNPPClassifier(
        "unused.py", "unused.pth", model=object(), inference_fn=inference
    )

    prediction = classifier.predict(sequence)

    assert prediction.frame_count == 20
    assert prediction.padding_start_frames == 22
    assert prediction.padding_end_frames == 22
    assert captured["total_frames"] == 64
    assert captured["keypoint"].shape == (1, 64, 17, 2)
    assert np.all(captured["keypoint"][0, :22, :, 0] == 0)
    assert np.all(captured["keypoint"][0, 22:42, 0, 0] == np.arange(20))
    assert np.all(captured["keypoint"][0, 42:, :, 0] == 19)


def test_prediction_rejects_low_quality_pose() -> None:
    classifier = STGCNPPClassifier(
        "unused.py",
        "unused.pth",
        model=object(),
        inference_fn=lambda model, data: [(0, 1.0), (1, 0.0)],
    )

    with pytest.raises(ValueError, match="mean confidence"):
        classifier.predict(make_sequence(confidence=0.1))


def test_three_class_model_uses_multisense_label_mapping() -> None:
    class Head:
        num_classes = 3

    class Model:
        cls_head = Head()

    classifier = STGCNPPClassifier(
        "unused.py", "unused.pth", model=Model(),
        inference_fn=lambda model, data: [(2, 0.8), (0, 0.15), (1, 0.05)],
    )

    prediction = classifier.predict(make_sequence())

    assert prediction.label == "forehand_clear"
    assert prediction.scores["background"] == pytest.approx(0.15)


def test_configured_class_names_override_legacy_two_class_mapping() -> None:
    class Head:
        num_classes = 2

    class Config:
        class_names = ("backhand_drive", "forehand_clear")

    class Model:
        cls_head = Head()
        cfg = Config()

    classifier = STGCNPPClassifier(
        "unused.py",
        "unused.pth",
        model=Model(),
        inference_fn=lambda model, data: [(1, 0.9), (0, 0.1)],
    )

    assert classifier.predict(make_sequence()).label == "forehand_clear"


def test_predict_aligned_selects_window_around_motion_peak(monkeypatch) -> None:
    sequence = PoseSequence(
        np.ones((100, 17, 3), dtype=np.float32), 30.0, 1280, 720
    )
    scores = np.zeros(100, dtype=np.float32)
    scores[70] = 5.0
    features = GeometryFeatures(
        ("wrist_speed", "elbow_angular_speed"),
        np.zeros((100, 2), dtype=np.float32),
        np.ones((100, 2), dtype=np.float32),
        np.arange(100, dtype=np.float32) / 30.0,
    )
    monkeypatch.setattr(
        "ai_classifier.action_recognition.stgcnpp_classifier.extract_geometry_features",
        lambda sequence, handedness: features,
    )
    monkeypatch.setattr(
        "ai_classifier.action_recognition.stgcnpp_classifier.motion_score",
        lambda value: scores,
    )
    captured = {}

    def inference(model, annotation):
        captured["frames"] = annotation["total_frames"]
        return [(1, 0.8), (0, 0.2)]

    classifier = STGCNPPClassifier(
        "unused.py", "unused.pth", model=object(), inference_fn=inference
    )

    prediction = classifier.predict_aligned(sequence, window_frames=64)

    assert captured["frames"] == 64
    assert prediction.frame_count == 64
    assert prediction.source_frame_count == 100
    assert prediction.alignment_applied is True
    assert prediction.window_start_frame == 35
    assert prediction.window_end_frame == 99
    assert prediction.motion_peak_frame == 70
    assert prediction.padding_start_frames == 0
    assert prediction.padding_end_frames == 0


def test_predict_aligned_edge_pads_short_clip(monkeypatch) -> None:
    scores = np.zeros(20, dtype=np.float32)
    scores[12] = 1.0
    features = GeometryFeatures(
        ("wrist_speed", "elbow_angular_speed"),
        np.zeros((20, 2), dtype=np.float32),
        np.ones((20, 2), dtype=np.float32),
        np.arange(20, dtype=np.float32) / 30.0,
    )
    monkeypatch.setattr(
        "ai_classifier.action_recognition.stgcnpp_classifier.extract_geometry_features",
        lambda sequence, handedness: features,
    )
    monkeypatch.setattr(
        "ai_classifier.action_recognition.stgcnpp_classifier.motion_score",
        lambda value: scores,
    )
    classifier = STGCNPPClassifier(
        "unused.py",
        "unused.pth",
        model=object(),
        inference_fn=lambda model, annotation: [(0, 0.7), (1, 0.3)],
    )

    prediction = classifier.predict_aligned(make_sequence(), window_frames=64)

    assert prediction.frame_count == 64
    assert prediction.source_frame_count == 20
    assert prediction.alignment_applied is True
    assert prediction.window_start_frame == 0
    assert prediction.window_end_frame == 20
    assert prediction.motion_peak_frame == 12
    assert prediction.padding_start_frames == 23
    assert prediction.padding_end_frames == 21


def test_predict_aligned_window_duration_is_fps_aware(monkeypatch) -> None:
    captured_frames = []

    def fake_features(sequence, handedness):
        frame_count = len(sequence.keypoints)
        return GeometryFeatures(
            ("wrist_speed", "elbow_angular_speed"),
            np.zeros((frame_count, 2), dtype=np.float32),
            np.ones((frame_count, 2), dtype=np.float32),
            np.arange(frame_count, dtype=np.float32) / sequence.fps,
        )

    def fake_score(features):
        scores = np.zeros(len(features.values), dtype=np.float32)
        scores[len(scores) // 2] = 1.0
        return scores

    def inference(model, annotation):
        captured_frames.append(annotation["total_frames"])
        return [(0, 0.7), (1, 0.3)]

    monkeypatch.setattr(
        "ai_classifier.action_recognition.stgcnpp_classifier.extract_geometry_features",
        fake_features,
    )
    monkeypatch.setattr(
        "ai_classifier.action_recognition.stgcnpp_classifier.motion_score", fake_score
    )
    classifier = STGCNPPClassifier(
        "unused.py", "unused.pth", model=object(), inference_fn=inference
    )
    sequence_30fps = PoseSequence(
        np.ones((120, 17, 3), dtype=np.float32), 30.0, 1280, 720
    )
    sequence_60fps = PoseSequence(
        np.ones((240, 17, 3), dtype=np.float32), 60.0, 1280, 720
    )

    prediction_30fps = classifier.predict_aligned(sequence_30fps)
    prediction_60fps = classifier.predict_aligned(sequence_60fps)

    assert prediction_30fps.frame_count == 105
    assert prediction_60fps.frame_count == 210
    assert captured_frames == [105, 210]
