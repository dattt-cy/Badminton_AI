"""ST-GCN++ for background, backhand drive, and forehand clear."""

_base_ = "./stgcnpp_badminton.py"

model = dict(cls_head=dict(type="GCNHead", num_classes=3, in_channels=256))

ann_file = "data/annotations/multisense_actions_3class.pkl"
data = dict(
    train=dict(dataset=dict(ann_file=ann_file)),
    val=dict(ann_file=ann_file),
    test=dict(ann_file=ann_file),
)
work_dir = "./work_dirs/stgcnpp_multisense_3class"
