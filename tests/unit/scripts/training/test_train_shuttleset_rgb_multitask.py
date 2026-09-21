from pathlib import Path

import torch

from scripts.training.train_shuttleset_rgb_multitask import (
    MultiTaskR2Plus1D,
    Record,
    balanced_subset,
    class_weights,
    source_weighted_stroke_loss,
    windows_to_wsl,
)


def test_balanced_subset_caps_each_stroke_side():
    records = [
        Record(str(i), Path("video.mp4"), 0, 10, "lift", side)
        for side in ("forehand", "backhand", "aroundhead")
        for i in range(4)
    ]
    selected = balanced_subset(records, 2, seed=1)
    assert len(selected) == 6
    assert {side: sum(row.stroke_side == side for row in selected) for side in ("forehand", "backhand", "aroundhead")} == {
        "forehand": 2,
        "backhand": 2,
        "aroundhead": 2,
    }


def test_multitask_model_has_independent_heads():
    model = MultiTaskR2Plus1D()
    assert model.stroke_head.out_features == 8
    assert model.side_head.out_features == 3


def test_windows_path_is_unchanged_outside_wsl():
    converted = windows_to_wsl(r"C:\Users\ADMIN\video.mp4")
    if not Path("/mnt").is_dir():
        assert str(converted) == r"C:\Users\ADMIN\video.mp4"


def test_class_weights_allow_classes_absent_from_smoke_subset():
    records = [Record("1", Path("video.mp4"), 0, 10, "lift", "forehand")]
    weights = class_weights(records, ["lift", "serve"], "coarse_label")
    assert weights.tolist() == [0.5, 0.0]


def test_source_weight_does_not_cancel_for_single_source_batch():
    losses = torch.tensor([2.0, 4.0])
    weighted = source_weighted_stroke_loss(
        losses, ["shuttle-a", "shuttle-b"], fine_weight=1.0, shuttle_weight=0.35
    )
    assert torch.isclose(weighted, torch.tensor(1.05))


def test_source_weight_mixes_fine_and_shuttle_absolutely():
    losses = torch.tensor([2.0, 4.0])
    weighted = source_weighted_stroke_loss(
        losses, ["fine:a", "shuttle-b"], fine_weight=1.0, shuttle_weight=0.5
    )
    assert torch.isclose(weighted, torch.tensor(2.0))


def test_source_weight_can_mask_noisy_shuttle_stroke_but_keep_batch():
    losses = torch.tensor([10.0, 2.0])
    weighted = source_weighted_stroke_loss(
        losses,
        ["soft-drive", "clean-drive"],
        fine_weight=1.0,
        shuttle_weight=0.5,
        zero_weight_sample_ids={"soft-drive"},
    )
    assert torch.isclose(weighted, torch.tensor(0.5))
