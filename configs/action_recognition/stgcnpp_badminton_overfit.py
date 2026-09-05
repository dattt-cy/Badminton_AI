"""Sanity config that must memorize ten known skeleton clips."""

_base_ = "./stgcnpp_badminton.py"

data = dict(
    videos_per_gpu=10,
    workers_per_gpu=0,
    test_dataloader=dict(videos_per_gpu=1),
    train=dict(dataset=dict(ann_file="data/annotations/badminton_overfit_10.pkl")),
    val=dict(ann_file="data/annotations/badminton_overfit_10.pkl"),
    test=dict(ann_file="data/annotations/badminton_overfit_10.pkl"),
)
total_epochs = 50
evaluation = dict(interval=5, metrics=["top_k_accuracy"])
work_dir = "./work_dirs/stgcnpp_badminton_overfit"
