"""Conservative fine-tuning from the original two-class checkpoint."""

_base_ = "./stgcnpp_manual_clips_2class_motion_onset.py"

optimizer = dict(
    type="SGD",
    lr=0.0005,
    momentum=0.9,
    weight_decay=0.0005,
    nesterov=True,
)
total_epochs = 5
checkpoint_config = dict(interval=1)
work_dir = "./work_dirs/stgcnpp_manual_clips_2class_motion_onset_low_lr"
