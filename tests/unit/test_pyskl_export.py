import pickle
from pathlib import Path

import numpy as np
import pytest

from ai_classifier.action_recognition.pyskl_export import (
    _single_player_source_group,
    build_pyskl_dataset,
)


def save_pose(path: Path, frame_count: int = 8) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keypoints = np.zeros((frame_count, 17, 3), dtype=np.float32)
    keypoints[..., 2] = 0.9
    np.savez_compressed(
        path,
        keypoints=keypoints,
        fps=np.float32(30),
        frame_width=np.int32(1920),
        frame_height=np.int32(1080),
    )


def test_build_pyskl_dataset_shapes_and_splits(tmp_path: Path) -> None:
    pose_root = tmp_path / "poses"
    for action in ("backhand_drive", "forehand_lift"):
        for index in range(10):
            save_pose(pose_root / action / "match" / f"{index:03}.npz")
    output = tmp_path / "annotations.pkl"

    dataset = build_pyskl_dataset(
        pose_root,
        output,
        {"backhand_drive": 0, "forehand_lift": 1},
    )

    assert len(dataset["annotations"]) == 20
    assert len(dataset["split"]["train"]) == 14
    assert len(dataset["split"]["val"]) == 4
    assert len(dataset["split"]["test"]) == 2
    annotation = dataset["annotations"][0]
    assert annotation["keypoint"].shape == (1, 8, 17, 2)
    assert annotation["keypoint_score"].shape == (1, 8, 17)
    with output.open("rb") as input_file:
        assert len(pickle.load(input_file)["annotations"]) == 20


def test_numbered_clips_share_single_player_source_group() -> None:
    assert _single_player_source_group(
        "backhand_drive/single_player/001_videoplayback_03"
    ) == "backhand_drive/single_player/001_videoplayback"
    assert _single_player_source_group(
        "backhand_drive/single_player/011"
    ) == "backhand_drive/single_player/011"


def test_export_rejects_a_class_without_valid_samples(tmp_path: Path) -> None:
    pose_root = tmp_path / "poses"
    save_pose(pose_root / "backhand_drive" / "match" / "001.npz")

    with pytest.raises(ValueError, match="other"):
        build_pyskl_dataset(
            pose_root,
            tmp_path / "annotations.pkl",
            {"backhand_drive": 0, "other": 1},
        )


def test_single_player_source_group_does_not_span_splits(tmp_path: Path) -> None:
    pose_root = tmp_path / "poses"
    for index in range(6):
        save_pose(
            pose_root
            / "backhand_drive"
            / "single_player"
            / f"source_a_{index:02}.npz"
        )
    for index in range(4):
        save_pose(
            pose_root
            / "backhand_drive"
            / "single_player"
            / f"source_b_{index:02}.npz"
        )

    dataset = build_pyskl_dataset(
        pose_root,
        tmp_path / "annotations.pkl",
        {"backhand_drive": 0},
    )
    split_by_identifier = {
        identifier: split_name
        for split_name, identifiers in dataset["split"].items()
        for identifier in identifiers
    }
    source_splits: dict[str, set[str]] = {}
    for annotation in dataset["annotations"]:
        source_splits.setdefault(annotation["source_group"], set()).add(
            split_by_identifier[annotation["frame_dir"]]
        )

    assert all(len(splits) == 1 for splits in source_splits.values())
