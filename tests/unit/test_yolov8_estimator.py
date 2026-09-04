from pathlib import Path
from types import SimpleNamespace

import numpy as np

from ai_classifier.pose import PoseSequence, YOLOv8PoseEstimator


class FakeTensor:
    def __init__(self, array: np.ndarray) -> None:
        self.array = array

    def detach(self) -> 'FakeTensor':
        return self

    def cpu(self) -> 'FakeTensor':
        return self

    def numpy(self) -> np.ndarray:
        return self.array


class FakeModel:
    def __init__(self, results: list[object]) -> None:
        self.results = results
        self.options = None

    def predict(self, **kwargs: object) -> list[object]:
        self.options = kwargs
        return self.results


def test_extract_selects_person_with_highest_mean_confidence(
    tmp_path: Path, monkeypatch
) -> None:
    video = tmp_path / 'sample.mp4'
    video.touch()
    people = np.zeros((2, 17, 3), dtype=np.float32)
    people[0, :, 2] = 0.2
    people[1, :, 0] = 42.0
    people[1, :, 2] = 0.9
    model = FakeModel([SimpleNamespace(keypoints=SimpleNamespace(data=FakeTensor(people)))])
    monkeypatch.setattr(
        YOLOv8PoseEstimator,
        '_read_video_metadata',
        staticmethod(lambda _: (30.0, 1920, 1080)),
    )

    sequence = YOLOv8PoseEstimator(model=model).extract(video)

    assert sequence.keypoints.shape == (1, 17, 3)
    assert np.all(sequence.keypoints[0, :, 0] == 42.0)
    assert sequence.fps == 30.0
    assert model.options['stream'] is True


def test_extract_inserts_zero_pose_when_detection_is_missing(
    tmp_path: Path, monkeypatch
) -> None:
    video = tmp_path / 'sample.mp4'
    video.touch()
    model = FakeModel([SimpleNamespace(keypoints=None)])
    monkeypatch.setattr(
        YOLOv8PoseEstimator,
        '_read_video_metadata',
        staticmethod(lambda _: (25.0, 640, 480)),
    )

    sequence = YOLOv8PoseEstimator(model=model).extract(video)

    assert sequence.keypoints.shape == (1, 17, 3)
    assert not sequence.keypoints.any()


def test_tracking_prefers_nearby_person_over_higher_confidence(
    tmp_path: Path, monkeypatch
) -> None:
    video = tmp_path / 'sample.mp4'
    video.touch()
    first_frame = np.zeros((1, 17, 3), dtype=np.float32)
    first_frame[0, :, :2] = (100.0, 100.0)
    first_frame[0, :, 2] = 0.8
    second_frame = np.zeros((2, 17, 3), dtype=np.float32)
    second_frame[0, :, :2] = (110.0, 100.0)
    second_frame[0, :, 2] = 0.75
    second_frame[1, :, :2] = (900.0, 900.0)
    second_frame[1, :, 2] = 0.95
    model = FakeModel([
        SimpleNamespace(keypoints=SimpleNamespace(data=FakeTensor(first_frame))),
        SimpleNamespace(keypoints=SimpleNamespace(data=FakeTensor(second_frame))),
    ])
    monkeypatch.setattr(
        YOLOv8PoseEstimator,
        '_read_video_metadata',
        staticmethod(lambda _: (30.0, 1000, 1000)),
    )

    sequence = YOLOv8PoseEstimator(model=model).extract(video)

    assert np.all(sequence.keypoints[1, :, 0] == 110.0)


def test_pose_sequence_saves_keypoints_and_metadata(tmp_path: Path) -> None:
    sequence = PoseSequence(np.zeros((2, 17, 3), np.float32), 60.0, 1280, 720)
    output = sequence.save(tmp_path / 'nested' / 'pose.npz')

    saved = np.load(output)
    assert saved['keypoints'].shape == (2, 17, 3)
    assert saved['fps'] == 60.0
