import pickle
from pathlib import Path

import numpy as np

from scripts.data.export.export_manual_clips_2class_pyskl import export_dataset
from scripts.data.pose.extract_manual_clips_poses import discover_jobs


def _write_pose(path: Path, confidence: float = 0.9, frames: int = 40) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keypoints = np.zeros((frames, 17, 3), dtype=np.float32)
    keypoints[..., :2] = 10.0
    keypoints[..., 2] = confidence
    np.savez_compressed(
        path,
        keypoints=keypoints,
        fps=np.float32(30),
        frame_width=np.int32(640),
        frame_height=np.int32(480),
    )


def test_discover_jobs_mirrors_nested_manual_paths(tmp_path):
    source = tmp_path / "videos"
    output = tmp_path / "poses"
    video = source / "train" / "backhand_drive" / "front" / "Sub08" / "a.mov"
    video.parent.mkdir(parents=True)
    video.touch()
    (video.parent / "a-proj.llc").touch()
    for split in ("train", "val"):
        for action in ("backhand_drive", "forehand_clear"):
            for view in ("front", "side"):
                (source / split / action / view).mkdir(parents=True, exist_ok=True)

    jobs = discover_jobs(
        source,
        output,
        splits=["train", "val"],
        classes=["backhand_drive", "forehand_clear"],
        views=["front", "side"],
        overwrite=False,
    )

    assert jobs == [
        (
            video,
            output / "train" / "backhand_drive" / "front" / "Sub08" / "a.npz",
            "front",
        )
    ]


def test_export_keeps_subject_splits_and_filters_bad_poses(tmp_path):
    pose_root = tmp_path / "poses"
    for split, subject in (("train", "Sub08"), ("val", "Sub14")):
        for action in ("backhand_drive", "forehand_clear"):
            _write_pose(pose_root / split / action / "front" / subject / "good.npz")
    _write_pose(
        pose_root / "train" / "backhand_drive" / "side" / "Sub08" / "bad.npz",
        confidence=0.0,
    )
    output = tmp_path / "manual.pkl"
    config = {
        "pose_output_root": str(pose_root),
        "annotation_output": str(output),
        "classes": {"backhand_drive": 0, "forehand_clear": 1},
        "splits": ["train", "val"],
        "quality": {
            "min_frames": 30,
            "min_detected_ratio": 0.8,
            "min_mean_confidence": 0.3,
        },
    }

    summary = export_dataset(config)

    assert summary["exported"] == 4
    assert summary["rejected_low_detection"] == 1
    with output.open("rb") as source:
        dataset = pickle.load(source)
    assert len(dataset["split"]["train"]) == 2
    assert len(dataset["split"]["val"]) == 2
    assert dataset["split"]["test"] == []
    assert {item["label"] for item in dataset["annotations"]} == {0, 1}
    assert all(
        identifier.startswith("train/")
        for identifier in dataset["split"]["train"]
    )


def test_export_combines_original_and_additional_pose_roots(tmp_path):
    original = tmp_path / "original"
    augmented = tmp_path / "augmented"
    for root, stem in ((original, "full"), (augmented, "start_at_swing")):
        for split, subject in (("train", "Sub08"), ("val", "Sub14")):
            for action in ("backhand_drive", "forehand_clear"):
                _write_pose(
                    root / split / action / "front" / subject / f"{stem}.npz"
                )
    output = tmp_path / "combined.pkl"
    config = {
        "pose_output_root": str(original),
        "additional_pose_output_roots": [str(augmented)],
        "annotation_output": str(output),
        "classes": {"backhand_drive": 0, "forehand_clear": 1},
        "splits": ["train", "val"],
    }

    summary = export_dataset(config)

    assert summary["exported"] == 8
    assert summary["split_train"] == 4
    assert summary["split_val"] == 4


def test_additional_pose_roots_can_be_limited_to_train(tmp_path):
    original = tmp_path / "original"
    augmented = tmp_path / "augmented"
    for split, subject in (("train", "Sub08"), ("val", "Sub14")):
        for action in ("backhand_drive", "forehand_clear"):
            _write_pose(original / split / action / "front" / subject / "full.npz")
            _write_pose(
                augmented / split / action / "front" / subject / "partial.npz"
            )
    output = tmp_path / "combined.pkl"
    config = {
        "pose_output_root": str(original),
        "additional_pose_output_roots": [str(augmented)],
        "additional_pose_splits": ["train"],
        "annotation_output": str(output),
        "classes": {"backhand_drive": 0, "forehand_clear": 1},
        "splits": ["train", "val"],
    }

    summary = export_dataset(config)

    assert summary["split_train"] == 4
    assert summary["split_val"] == 2
