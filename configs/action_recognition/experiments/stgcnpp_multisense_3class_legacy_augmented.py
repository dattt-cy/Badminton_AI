"""Three-class model augmented with quality-screened legacy backhand clips."""

_base_ = "./stgcnpp_multisense_3class.py"

ann_file = "data/annotations/multisense_actions_3class_legacy_augmented.pkl"
data = dict(
    train=dict(dataset=dict(ann_file=ann_file)),
    val=dict(ann_file=ann_file),
    test=dict(ann_file=ann_file),
)
work_dir = "./work_dirs/stgcnpp_multisense_3class_legacy_augmented"
