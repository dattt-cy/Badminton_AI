from pathlib import Path

import pytest

from scripts.data.augmentation.augment_manual_clips_temporal import (
    AugmentationJob,
    output_relative_path,
    select_balanced_jobs,
    validate_output_roots,
)


def _job(action: str, subject: str, index: int) -> AugmentationJob:
    relative = Path("train") / action / "front" / subject / f"clip{index}.mov"
    return AugmentationJob(
        source_video=Path("source") / relative,
        source_pose=Path("poses") / relative.with_suffix(".npz"),
        relative_path=relative,
        split="train",
        action=action,
        view="front",
        subject=subject,
        start_frame=30,
        peak_frame=33,
        total_frames=90,
        fps=30.0,
    )


def test_selection_applies_ratio_per_subject_and_class() -> None:
    jobs = [
        _job(action, subject, index)
        for action in ("backhand_drive", "forehand_clear")
        for subject in ("Sub01", "Sub02")
        for index in range(10)
    ]

    selected = select_balanced_jobs(jobs, ratio=0.30, seed=7)

    groups = {(job.action, job.subject) for job in selected}
    assert groups == {
        ("backhand_drive", "Sub01"),
        ("backhand_drive", "Sub02"),
        ("forehand_clear", "Sub01"),
        ("forehand_clear", "Sub02"),
    }
    assert all(
        sum(job.action == action and job.subject == subject for job in selected) == 3
        for action, subject in groups
    )


def test_output_name_is_isolated_and_marks_variant() -> None:
    source = Path("train/backhand_drive/front/Sub01/clip.mov")
    assert output_relative_path(source) == Path(
        "train/backhand_drive/front/Sub01/clip__start_at_swing.mp4"
    )


def test_source_subdirectory_cannot_be_an_output(tmp_path: Path) -> None:
    source = tmp_path / "manual_clips"
    with pytest.raises(ValueError, match="outside the source"):
        validate_output_roots(source, source / "augmented", tmp_path / "poses")
