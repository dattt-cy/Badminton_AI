"""Fine-tune the original model with balanced motion-onset variants."""

_base_ = "./stgcnpp_manual_clips_2class.py"

ann_file = "data/annotations/manual_clips_2class_motion_onset.pkl"
data = dict(
    train=dict(dataset=dict(ann_file=ann_file)),
    val=dict(ann_file=ann_file),
    test=dict(ann_file=ann_file, split="val"),
)

load_from = "models/checkpoints/action_recognition/stgcnpp_manual_clips_2class_best.pth"
optimizer = dict(
    type="SGD",
    lr=0.01,
    momentum=0.9,
    weight_decay=0.0005,
    nesterov=True,
)
total_epochs = 15
checkpoint_config = dict(interval=5)
work_dir = "./work_dirs/stgcnpp_manual_clips_2class_motion_onset"
