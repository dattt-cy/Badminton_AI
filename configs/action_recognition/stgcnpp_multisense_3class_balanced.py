"""Balanced three-class model with legacy backhand and forehand-clear data."""

_base_ = "./stgcnpp_multisense_3class_legacy_both_augmented.py"

model = dict(
    cls_head=dict(
        type="GCNHead",
        num_classes=3,
        in_channels=256,
        dropout=0.5,
        loss_cls=dict(
            type="CrossEntropyLoss",
            class_weight=[1.0, 1.5, 0.8],
        ),
    )
)
work_dir = "./work_dirs/stgcnpp_multisense_3class_balanced"
