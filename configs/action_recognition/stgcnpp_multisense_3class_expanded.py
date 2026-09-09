"""ST-GCN++ trained on the expanded, subject-balanced MultiSense dataset."""

_base_ = "./stgcnpp_badminton.py"

model = dict(
    cls_head=dict(
        type="GCNHead",
        num_classes=3,
        in_channels=256,
        dropout=0.3,
    )
)

ann_file = "data/annotations/multisense_actions_3class_expanded.pkl"
data = dict(
    train=dict(dataset=dict(ann_file=ann_file)),
    val=dict(ann_file=ann_file),
    test=dict(ann_file=ann_file),
)

checkpoint_config = dict(interval=1)
evaluation = dict(interval=1, metrics=["top_k_accuracy"])
work_dir = "./work_dirs/stgcnpp_multisense_3class_expanded"
