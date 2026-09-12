"""ST-GCN++ for the manually clipped backhand-drive/forehand-clear dataset."""

_base_ = "./stgcnpp_badminton.py"

class_names = ("backhand_drive", "forehand_clear")

model = dict(cls_head=dict(type="GCNHead", num_classes=2, in_channels=256))

ann_file = "data/annotations/manual_clips_2class.pkl"

# The original two-class dataset only had 83 training examples, so its base
# config repeated the dataset five times. The manual dataset is much larger and
# should be traversed once per epoch.
data = dict(
    videos_per_gpu=16,
    workers_per_gpu=2,
    test_dataloader=dict(videos_per_gpu=1),
    train=dict(
        times=1,
        dataset=dict(ann_file=ann_file),
    ),
    val=dict(ann_file=ann_file),
    test=dict(ann_file=ann_file),
)

checkpoint_config = dict(interval=5)
evaluation = dict(
    interval=1,
    metrics=["top_k_accuracy", "mean_class_accuracy"],
    save_best="top1_acc",
    rule="greater",
)
log_config = dict(interval=20, hooks=[dict(type="TextLoggerHook")])
total_epochs = 30
work_dir = "./work_dirs/stgcnpp_manual_clips_2class"
