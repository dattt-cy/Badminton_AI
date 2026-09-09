"""Short sanity run for the MultiSense three-class model."""

_base_ = "./stgcnpp_multisense_3class.py"

ann_file = "data/annotations/multisense_actions_3class_overfit.pkl"
data = dict(
    videos_per_gpu=9,
    workers_per_gpu=0,
    train=dict(dataset=dict(ann_file=ann_file)),
    val=dict(ann_file=ann_file),
    test=dict(ann_file=ann_file),
)
total_epochs = 10
evaluation = dict(interval=2, metrics=["top_k_accuracy"])
checkpoint_config = dict(interval=2)
log_config = dict(interval=1, hooks=[dict(type="TextLoggerHook")])
work_dir = "./work_dirs/stgcnpp_multisense_3class_overfit"
