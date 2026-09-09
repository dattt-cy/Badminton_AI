import pickle
from pathlib import Path

import numpy as np

from scripts.data.export_multisense_3class_pyskl import (
    load_subject_split,
    selected_pose_paths,
)


def _assert_subject_safe(path: Path, expected_train: set[str]):
    if not path.exists():
        return
    with path.open("rb") as source:
        dataset = pickle.load(source)
    annotations = {item["frame_dir"]: item for item in dataset["annotations"]}
    subjects_by_split = {}
    for split, identifiers in dataset["split"].items():
        subjects_by_split[split] = {
            Path(identifier).parts[2] for identifier in identifiers
        }
        assert all(identifier in annotations for identifier in identifiers)

    assert subjects_by_split["train"] == expected_train
    assert subjects_by_split["val"] == {"Sub20"}
    assert subjects_by_split["test"] == {"Sub24"}
    assert {item["label"] for item in dataset["annotations"]} == {0, 1, 2}


def test_multisense_three_class_dataset_has_subject_safe_splits():
    _assert_subject_safe(
        Path("data/annotations/multisense_actions_3class.pkl"), {"Sub11", "Sub19"}
    )


def test_legacy_augmented_dataset_keeps_legacy_poses_in_train_only():
    _assert_subject_safe(
        Path("data/annotations/multisense_actions_3class_legacy_augmented.pkl"),
        {"Sub11", "Sub19", "LegacyTrain"},
    )


def test_legacy_backhand_and_clear_augmented_dataset_keeps_legacy_in_train():
    _assert_subject_safe(
        Path("data/annotations/multisense_actions_3class_legacy_both_augmented.pkl"),
        {"Sub11", "Sub19", "LegacyTrain", "LegacyClearTrain"},
    )


def _write_pose(path: Path, confidence: float = 0.9):
    path.parent.mkdir(parents=True, exist_ok=True)
    keypoints = np.zeros((12, 17, 3), dtype=np.float32)
    keypoints[..., 2] = confidence
    np.savez_compressed(
        path,
        keypoints=keypoints,
        fps=np.float32(30),
        frame_width=np.int32(640),
        frame_height=np.int32(480),
    )


def test_expanded_split_config_is_subject_exclusive():
    mapping = load_subject_split(
        Path("configs/action_recognition/dataset_multisense_3class_expanded.yaml")
    )
    assert mapping["Sub13"] == "train"
    assert mapping["Sub20"] == "val"
    assert mapping["Sub24"] == "test"
    assert len(mapping) == 7


def test_balanced_selection_uses_equal_classes_per_subject(tmp_path):
    for label, count in {
        "background": 4,
        "backhand_drive": 2,
        "forehand_clear": 3,
    }.items():
        for index in range(count):
            _write_pose(tmp_path / label / "front" / "Sub08" / f"{index}.npz")

    selected = selected_pose_paths(
        tmp_path,
        {"Sub08": "train"},
        balance_per_subject=True,
        max_per_subject_class=20,
        min_detected_ratio=0.8,
        min_mean_confidence=0.3,
    )
    counts = {
        label: sum(selected_label == label for selected_label, _ in selected)
        for label in ("background", "backhand_drive", "forehand_clear")
    }
    assert counts == {"background": 2, "backhand_drive": 2, "forehand_clear": 2}


def test_quality_filter_runs_before_balancing(tmp_path):
    for label in ("background", "backhand_drive", "forehand_clear"):
        _write_pose(tmp_path / label / "front" / "Sub08" / "good.npz")
        _write_pose(
            tmp_path / label / "front" / "Sub08" / "bad.npz", confidence=0.0
        )

    selected = selected_pose_paths(
        tmp_path,
        {"Sub08": "train"},
        balance_per_subject=True,
        max_per_subject_class=None,
        min_detected_ratio=0.8,
        min_mean_confidence=0.3,
    )
    assert len(selected) == 3
    assert all(path.name == "good.npz" for _, path in selected)
