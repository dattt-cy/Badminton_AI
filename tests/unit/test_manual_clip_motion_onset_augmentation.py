from pathlib import Path

import numpy as np

from scripts.data.augment_manual_clips_motion_onset import (
    MotionOnsetJob,
    resample_pose,
    select_paired_jobs,
)


def _job(action: str, subject: str, index: int) -> MotionOnsetJob:
    relative = Path("train") / action / "front" / subject / f"clip{index}.mov"
    return MotionOnsetJob(
        Path("videos") / relative,
        Path("poses") / relative.with_suffix(".npz"),
        relative,
        "train",
        action,
        "front",
        subject,
        10,
        58,
        22,
        30.0,
    )


def test_selection_is_class_paired_per_subject_and_view() -> None:
    candidates = [
        _job(action, subject, index)
        for action in ("backhand_drive", "forehand_clear")
        for subject in ("Sub01", "Sub02")
        for index in range(10)
    ]

    selected = select_paired_jobs(
        candidates, split_ratios={"train": 0.5}, seed=9
    )

    for subject in ("Sub01", "Sub02"):
        assert sum(
            job.subject == subject and job.action == "backhand_drive"
            for job in selected
        ) == 5
        assert sum(
            job.subject == subject and job.action == "forehand_clear"
            for job in selected
        ) == 5


def test_resample_pose_interpolates_to_exact_length() -> None:
    keypoints = np.zeros((4, 17, 3), dtype=np.float32)
    keypoints[:, :, 0] = np.arange(4)[:, None]

    sampled = resample_pose(keypoints, target_frames=7)

    assert sampled.shape == (7, 17, 3)
    np.testing.assert_allclose(sampled[:, 0, 0], np.linspace(0, 3, 7))
